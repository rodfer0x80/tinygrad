from typing import Dict, List, Optional, Final, Callable, NamedTuple, Tuple, Union, DefaultDict, cast
from collections import defaultdict
import linecache
import re
import math
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, Op
from tinygrad.codegen.linearizer import  UOp, UOps
from tinygrad.helpers import ImageDType, dtypes, prod, DType, strip_parens, getenv, DEBUG, 
from triton.compiler import compile as triton_compile


triton_dtypes = {dtypes.double: "tl.float64", dtypes.float32: "tl.float32", dtypes.float16: "tl.float16", dtypes.bool: "tl.int1", dtypes.int8: "tl.int8", dtypes.uint8: "tl.uint8", dtypes.int32: "tl.int32", dtypes.int64: "tl.int64", dtypes.uint32: "tl.uint32", dtypes.uint64: "tl.uint64", dtypes.int16: "tl.int16", dtypes.uint16: "tl.uint16"}
signature_dtypes = {dtypes.double: "*fp64",dtypes.float32: "*fp32", dtypes.float16: "*fp16", dtypes.bool: "*i8", dtypes.int8: "*i1", dtypes.uint8: "*u8", dtypes._arg_int32: "i32", dtypes.int32: "*i32", dtypes.int64: "*i64", dtypes.uint32: "*u32", dtypes.uint64: "*u64", dtypes.int16: "*i16", dtypes.uint16: "*u16"}

class TritonBackend(NamedTuple):
  size_prefix: str = "int"
  generic_var_prefix: str = ""
  kernel_prefix: str = ""
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = ""
  barrier: str = ""
  xid: List[str] = []
  gid: List[str] = []
  lid: List[str] = []
  global_max: List[int] = []
  local_max: List[int] = []
  extra_args: List[str] = []
  float4: Optional[str] = None
  half_prekernel: Optional[str] = None
  uses_vload: bool = False
  external_local_bufs: bool = False
  uses_ptr_arithmetic: bool = False
  launch_bounds: bool = False
  code_for_op: Final[Dict[Op, Callable]] = {
    UnaryOps.EXP2: lambda x,: f"tl.math.exp2({x})",
    UnaryOps.LOG2: lambda x,: f"tl.math.log2({x})",
    UnaryOps.SIN: lambda x,: f"tl.sin({x})",
    UnaryOps.SQRT: lambda x,: f"tl.sqrt({x})",
    UnaryOps.NEG: lambda x,: f"-{x}",
    BinaryOps.ADD: lambda x,y,: f"({x}+{y})", BinaryOps.SUB: lambda x,y,: f"({x}-{y})",
    BinaryOps.MUL: lambda x,y,: f"({x}*{y})", BinaryOps.DIV: lambda x,y,: f"({x}/{y})" if y != '0.0' else f"{x}*tl.where({x}==0.0, float('nan'), float('inf'))",
    BinaryOps.MAX: lambda x,y,: f"tl.maximum({x},{y})",
    BinaryOps.CMPLT: lambda x,y,: f"({x}<{y})",
    BinaryOps.MOD: lambda x,y,: f"tl.abs({x})%tl.abs({y})*tl.where({x}<0,-1,1)",
    TernaryOps.MULACC: lambda x,y,z,: f"(({x}*{y})+{z})",
    TernaryOps.WHERE: lambda x,y,z,: f"tl.where({x},{y},{z})",
  }
  # returns a str expression of the casted xs with the given type
  def render_cast(self, x:List[str], var_dtype:DType) -> str:
    if len(x) == 1: return f"({var_dtype.name})({x[0]})"
    assert len(x) == var_dtype.sz, f"cast is wrong size {len(x)} != {var_dtype.sz}"
    assert self.float4 is not None, "cast is not supported on this platform"
    if var_dtype == dtypes._half16: return f"{{{','.join(f'(half){x}' for x in x)}}}"
    if var_dtype == dtypes._float8: return f"{{{','.join(x)}}}"
    if var_dtype == dtypes._float4: return f"{self.float4}({','.join(x)})"
    if var_dtype == dtypes._float2: return f"{self.float4.replace('float4', 'float2')}({','.join(x)})"
    if var_dtype == dtypes._int2: return f"{self.float4.replace('float4', 'int2')}({','.join(x)})"
    raise NotImplementedError(f"no cast for {var_dtype}")

  # returns a str expression of the const with the given type
  def render_const(self, x:Union[float,int], var_dtype) -> str:
    if math.isnan(x): val = "NAN"
    elif math.isinf(x): val = ("-" if x < 0 else "") + "INFINITY"
    else: val = f"{x}f" if dtypes.is_float(var_dtype) and isinstance(x, float) else f"{int(x)}"
    return self.render_cast([val]*var_dtype.sz, var_dtype) if var_dtype.sz > 1 else val

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert output_dtype == dtypes._float4, f"images must be float4, getting {output_dtype}"
      return f"read_imagef({buf_name}, smp, {idx})"
    if self.uses_vload and buf_dtype == dtypes.float16:
      return f"vload_half{'' if output_dtype.sz == 1 else str(output_dtype.sz)}(0, {buf_name}+{idx})"
    if output_dtype.sz > 1:
      out_val = f"*(({self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix}{buf_dtype.name}{output_dtype.sz}*)({buf_name}+{idx}))"
    else:
      out_val = f"*({buf_name}+{idx})" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}]"

    return self.render_cast([out_val], output_dtype) if output_dtype != buf_dtype else out_val

  def render_local(self, name:str, size:int):
    return self.smem_align + self.smem_prefix + f"float {name}[{size}];"

  def render_for(self, expr: str, _min:Union[int,str], _max:Union[int,str]) -> str:
    return f"for (int {expr} = {_min}; {expr} < {_max}; ++{expr}) {{"

  def render_if(self, cond: str):
    return f"if ({cond}) {{"

  def render_conditional(self, cond: str, x:str, y:str) -> str:
    return f"({cond})?({x}):{y}"

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,DType]], local_size:List[int], prekernel:List[str]) -> str:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,dtype in bufs) else ""
    buftypes = [(name,f"{'read_only' if i > 0 else 'write_only'} image2d_t" if dtype.name.startswith('image') else
                self.arg_int_prefix if dtype == dtypes._arg_int32 else
                ("const " if i > 0 else "")+self.buffer_prefix+dtype.name+"*"+self.buffer_suffix) for i,(name,dtype) in enumerate(bufs)]
    prg = ''.join([f"{self.kernel_prefix}void {f'__launch_bounds__ ({prod(local_size)}, 1) ' if self.launch_bounds else ''}{function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    if self.half_prekernel and any(dtype == dtypes.float16 for _,dtype in bufs): prg = ''.join([f"{self.half_prekernel}", "\n", prg])
    return prg

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert var_dtype == dtypes._float4, "images must be float4"
      return f"write_imagef({buf_name}, {idx}, {var_name});"
    if self.uses_vload and buf_dtype == dtypes.float16 and var_dtype != dtypes.float16:
      return f"vstore_half{'' if var_dtype.sz == 1 else str(var_dtype.sz)}({var_name}, 0, {buf_name}+{idx});"
    if var_dtype.sz > 1:
      return f"*(({self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix}{buf_dtype.name}{var_dtype.sz}*)({buf_name}+{idx})) = ({buf_dtype.name}{var_dtype.sz}){var_name};"
    return f"*({buf_name}+{idx}) = {var_name};" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}] = {var_name};"

  def uops_to_triton(self, function_name:str, uops:List[UOp]) -> Tuple[str, Dict]:
    local_size: List[int] = []
    kernel,prekernel,bufs = [],[],[]
    #pend_close = None
    depth = 1
    def kk(s): kernel.append("  "*depth+s)

    c: DefaultDict[str, int] = defaultdict(int)
    r: Dict[UOp, str] = {}
    def ssa(u, prefix="t"):
      nonlocal c, r
      c[prefix] += 1
      r[u]=f"{prefix}{c[prefix]-1}"
      return r[u]

    child_count: DefaultDict[UOp, int] = defaultdict(int)
    for ru in uops:
      for v in ru.vin:
        child_count[v] += 1

    for u in uops:
      uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
      if uop == UOps.LOOP:
        kk(self.render_for(ssa(u,'ridx'), r[vin[0]], r[vin[1]]))
        depth += 1
      elif uop == UOps.IF:
        kk(self.render_if(r[vin[0]]))
        depth += 1
      elif uop == UOps.BARRIER:
        kk(self.barrier)
      elif uop == UOps.END:
        depth -= 1
        kk("}")
      elif uop == UOps.WMMA:
        if args[0] == "METAL":
          assert dtype == dtypes._float2, "output dtype of METAL TC is _float2"
          # ((lidx2*32)+(lidx3*4)+(lidx4*16)+(lidx5*8)+(lidx6*2))
          output = ssa(u, 'wmma')
          kk(f"{self.generic_var_prefix if self.generic_var_prefix else dtype.name} {output};")
          kk("{ simdgroup_float8x8 a,b,c;")
          kk(f"a.thread_elements()[0] = {r[vin[0]]}; a.thread_elements()[1] = {r[vin[1]]};")
          kk(f"b.thread_elements()[0] = {r[vin[2]]}; b.thread_elements()[1] = {r[vin[3]]};")
          kk(f"c.thread_elements()[0] = {r[vin[4]]}; c.thread_elements()[1] = {r[vin[5]]};")
          kk("simdgroup_multiply_accumulate(c, a, b, c);")
          kk(f"{output}.x = c.thread_elements()[0]; {output}.y = c.thread_elements()[1]; }}")
        elif args[0] == "HIP":
          assert dtype == dtypes._float8, "output dtype of HIP TC is _float8"
          kk(f"{self.generic_var_prefix if self.generic_var_prefix else dtype.name} {ssa(u, 'wmma')} = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32({r[vin[0]]}, {r[vin[1]]}, {r[vin[2]]});")
        else:
          raise NotImplementedError(f"WMMA not implemented for {args}")
      elif uop == UOps.ALU:
        assert dtype is not None
        # remove parens if ALU types are the same. TODO: can do more here
        if vin[0].uop == UOps.ALU and vin[0].arg == args and args in {BinaryOps.ADD, BinaryOps.SUB, BinaryOps.MUL}:
          val = self.code_for_op[args](strip_parens(r[vin[0]]), *[r[x] for x in vin[1:]])
        elif args == BinaryOps.MAX:
          val = self.code_for_op[args](*[self.render_cast([r[x]], dtype) if x.dtype != dtype else r[x] for x in vin])
        else:
          val = self.code_for_op[args](*[r[x] for x in vin])
        assert child_count[u] != 0, f"childless ALU op found {u}"
        if (child_count[u] <= 1 or dtypes.is_int(dtype)) and args != BinaryOps.MAX:  # fix index rendering issue. fix clang nested max macro issue
          r[u] = val
        else:
          kk(f"{self.generic_var_prefix if self.generic_var_prefix else dtype.name} {ssa(u,'alu')} = {val};")
      elif uop == UOps.DEFINE_ACC:
        assert dtype is not None
        kk(f"{self.generic_var_prefix if self.generic_var_prefix else dtype.name} {ssa(u,'acc')} = {self.render_const(args, dtype)};")
      elif uop == UOps.SPECIAL:
        xid = self.gid if args[1].startswith("g") else (self.xid if args[1].startswith("i") else lang.lid)
        kk(f"{self.size_prefix} {args[1]} = {xid[args[0]]}; /* {args[2]} */")
        if args[1].startswith("l"): local_size.append(args[2])
        r[u] = args[1]
      elif uop == UOps.CONST:
        r[u] = self.render_const(args, dtype) if args >= 0 else f"({self.render_const(args, dtype)})"
      elif uop == UOps.LOAD:
        assert dtype is not None
        val = self.render_load(dtype, r[vin[0]], vin[0].dtype, strip_parens(r[vin[1]]), vin[0].uop == UOps.DEFINE_LOCAL)
        if len(vin) > 3: val = self.render_conditional(r[vin[2]], val, r[vin[3]])
        kk(f"{self.generic_var_prefix if self.generic_var_prefix else dtype.name} {ssa(u,'val')} = {val};")
      elif uop == UOps.PHI:
        kk(f"{r[vin[0]]} = {r[vin[1]]};")
        r[u] = r[vin[0]]
      elif uop == UOps.STORE:
        assert vin[0].dtype is not None and vin[2].dtype is not None
        kk(self.render_store(r[vin[0]], vin[0].dtype, r[vin[2]], vin[2].dtype, strip_parens(r[vin[1]]), vin[0].uop == UOps.DEFINE_LOCAL))
      elif uop == UOps.CAST and dtype is not None:
        val = self.render_cast([r[x] for x in vin], dtype)
        if child_count[u] <= 1: r[u] = val
        else: kk(f"{self.generic_var_prefix if self.generic_var_prefix else dtype.name} {ssa(u,'cast')} = {val};")
      elif uop == UOps.DEFINE_LOCAL:
        if self.external_local_bufs:
          prekernel.append(self.render_local(args[0], args[1]))
        else:
          kk(self.render_local(args[0], args[1]))
        r[u] = args[0]
      elif uop == UOps.DEFINE_GLOBAL:
        bufs.append(args)
        r[u] = args[0]
      elif uop == UOps.GEP:
        if cast(DType, vin[0].dtype).sz > 4:
          r[u] = f"({r[vin[0]]})[{args}]"  # this is correct for HIP
        else:
          r[u] = f"({r[vin[0]]}).{'xyzw'[args]}"
      else:
        raise RuntimeError(f"failed to render {uop}")
    return self.render_kernel(function_name, kernel, bufs, local_size, prekernel), {}

  def next_power_of_2(self, x):
    return 1 << (x - 1).bit_length()

  def render_valid(self, valid):
    return '(' * (len(valid) -1) + ') and '.join(valid) if len(valid) else 'True'

  #NOTE Triton requires matching dimensions for load/store, disable this and see TestOps::test_output_padded_conv_transpose2d fail to compile
  def fill_dims_for_idx(self, idx, dims):
    return "(" + idx + "+ (" + (f"0*({'+'.join(d for d in dims)})))") if len(dims) else idx

  def get_max(self, var):
    if isinstance(var, int): return var
    return re.sub(r'\[(.*?)\]', '', str(var))[1:-1]

  #NOTE can be removed after https://github.com/gpuocelot/gpuocelot/issues/8 gets resolved
  def remove_single_scalar_curly_braces(self, ptx_code):
    return '\n'.join([re.sub(r'\{\s*(%\w+)\s*\}', r'\1', line) for line in ptx_code.split('\n')])

  def render_const(self, args):
    return (('-' if args<0 else '') + 'tl.where(1,float("inf"),0)') if math.isinf(args) else ('tl.where(1,float("nan"),0)' if math.isnan(args) else str(args))

  def render_cast(self, x:str, dtype:DType):
    return f"{x}.to({triton_dtypes[dtype]})"

  def define_scalar(self, local_size, dtype, args):
    if len(local_size) > 0: return f"tl.full(({','.join([str(self.next_power_of_2(x)) for x in local_size])},),{self.render_const(args)}, dtype={triton_dtypes[dtype]})"
    return self.render_const(args)
  
  def kk(self, s): kernel.append("  "*depth+s)
  
  def int_div(self, x,y): return f"({x}//{y})" if y != '0' else f"{x}*tl.where({x}==0, float('nan'), float('inf'))"
    for u in uops:
      uop,dtype,vin,args = u.uop,u.dtype,u.vin,u.arg
      if uop == UOps.LOOP:
        kk(f"for {ssa(u, 'ridx')} in range({vin[0].arg}, {r[vin[1]]}):")
        depth += 1
      elif uop == UOps.END: depth -= 1
      elif uop == UOps.ALU:
        assert dtype is not None
        val = code_for_op[args](*[r[x] for x in vin])
        if child_count[u] <=1 or dtypes.is_int(dtype): r[u] = int_div(*[r[x] for x in vin]) if args == BinaryOps.DIV and dtypes.is_int(dtype) else val
        else: kk(f"{ssa(u, 'alu')} = ({val})")
      elif uop == UOps.LOAD:
        assert dtype is not None
        if len(vin) == 2: kk(f"{ssa(u, 'val')} = {render_cast(f'tl.load({r[vin[0]]} + { fill_dims_for_idx(r[vin[1]], dims)}, mask = {render_valid(valid)})', dtype)}")
        else: kk(f"{ssa(u, 'val')} = {render_cast(f'tl.where({r[vin[2]]}, tl.load({r[vin[0]]}+{fill_dims_for_idx(r[vin[1]],dims)} , mask={render_valid(valid+[r[vin[2]]])}), 0.0)', dtype)}")
      elif uop == UOps.DEFINE_ACC: kk(f"{ssa(u, 'acc')} = {define_scalar(local_size, dtype, args).replace('//', '/')}")
      elif uop == UOps.CONST: r[u] = define_scalar([], dtype, args)
      elif uop == UOps.PHI:
        kk(f"{r[vin[0]]} = {r[vin[1]].replace('//', '/')}")
        r[u] = r[vin[0]]
      elif uop == UOps.STORE:
        assert not isinstance(dtype, ImageDType), "unimplemented: image store"
        kk(f"tl.store({r[vin[0]]} + {r[vin[1]]}, {r[vin[2]].replace('//', '/')}, mask = {render_valid(valid)}) ")
      elif uop == UOps.DEFINE_GLOBAL:
        bufs.append(args)
        signatures.append(signature_dtypes[args[1]])
        r[u] = args[0]
      elif uop == UOps.SPECIAL:
        dims.append(args[1])
        valid.append(f"{args[1]}<{get_max(args[2])}")
        if args[1].startswith("g"): kk(f"{args[1]} = tl.program_id({args[0]}) # {args[2]}")
        elif args[1].startswith("l"):
          kk(f"{args[1]} = tl.arange({0}, {next_power_of_2(args[2])})")
          local_size.append(args[2])
        r[u] = args[1]
      elif uop == UOps.CAST and dtype is not None: r[u] = render_cast(r[vin[0]], dtype)
      else: raise NotImplementedError(f"unimplemented: {uop}")

    prg = f"import triton\nimport triton.language as tl\ntl.core.TRITON_MAX_TENSOR_NUMEL = float('inf')\n@triton.jit\ndef {function_name}("+','.join(f"{buf[0]}" for buf in bufs)+"):\n"
    for i, line in enumerate(list(filter(lambda line: "tl.arange" in line, kernel))): kernel[kernel.index(line)] +=  f"[{', '.join([':' if i == j else 'None' for j in range(len(local_size))])}]"
    prg += "\n".join(kernel)

    acc_local_size = 1
    for x in local_size: acc_local_size *= next_power_of_2(x)
    local_size = [acc_local_size] + [1] * (len(local_size) - 1)

    if DEBUG >= 4: print(prg)
    getlines = linecache.getlines
    linecache.getlines = lambda filename, module_globals=None: prg.splitlines(keepends=True) if "<triton>" == filename else getlines(filename, module_globals)
    exec(compile(prg, "<triton>", "exec"), globals()) # pylint: disable=W0122\
    compiled = triton_compile(globals()[function_name], signature=",".join(signatures), device_type="cuda", debug=False, cc=(35 if getenv("CUDACPU", 0) else None))
    prg = remove_single_scalar_curly_braces(compiled.asm["ptx"].split(".file")[0].split(".visible .func")[0])
    max_local_size =  [int(x) for x in prg.split(".maxntid ")[1].split("\n")[0].split(", ")]
    for i in range(len(local_size)): local_size[i] = min(local_size[i], max_local_size[i])

    return prg, {"shared":compiled.metadata["shared"], "local_size":local_size + [1]*(3-len(local_size))}
