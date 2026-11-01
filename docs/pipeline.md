# How a kernel runs: trace, dialect, C++, hipcc, launch

A kernel written in hk-dsl goes through five stages between your Python and
the GPU. Each one is a call you can make on its own, each is cached, and each
is where a different kind of mistake is caught. The reference pages go into
each stage; this page is the map.

```
your Python                   k.trace()              k.source()             k.build()               k(*tensors)
@hk.kernel / a template  -->  hk dialect (MLIR)  -->  HipKittens C++   -->  hipcc: code object -->  launch on the GPU
                              + the passes            + a module boundary   + the gates
```

## The stages

| stage | call | what it does | needs | catches |
|---|---|---|---|---|
| trace | `k.trace()` | runs the kernel body once, per set of constexprs, into a module in the `hk` MLIR dialect; `for` / `if` / `while` on device values become loops and branches | nothing | the language's own errors, at your line |
| passes | (inside `trace`) | verify the ops, set matrix-core fragments, place LDS, derive every wait, fence accumulators on CDNA, check the register budget | nothing | illegal ops, a tile read before its load landed, a slow operand layout |
| translate | `k.source()` | the module as HipKittens C++, with a module boundary around it (pybind, a torch op, or a device-only code object) | nothing | -- (it is the artifact you read when something is wrong) |
| build | `k.build()` | hipcc compiles it; the gates read the compiler's resource report and the code object | hipcc | a spill, a missing wait, an MFMA hazard |
| launch | `k(*tensors)` | loads the module and launches it on torch tensors | a GPU, torch | a wrong answer, through [`hk.check`](check.md) |

- **Trace** turns a Python function into IR without a compiler or a GPU:
  [the language](language.md) is what a body is written in, and
  [the IR](ir.md) is what it becomes.
- **The passes** are C++ (`mlir/lib/Dialect/HK/Transforms`), run by
  `k.trace()` in a fixed order ([passes](ir.md#passes)). Their errors are
  `hk.VerifyError`, ending at the kernel author's line.
- **Translation** (`mlir/lib/Target/Cpp/HKToCpp.cpp`) writes readable C++
  with the dump's value names: what you take to the disassembler
  ([translation](ir.md#translation)).
- **Build** runs the stock hipcc. A kernel that spills is refused, not
  returned, so `k.build()` needs no GPU and still catches the failure most
  likely to bite ([the gates](correctness-gates.md)).
- **Launch** caches every stage before it: the trace per set of constexprs,
  the build by a hash of its source, the loaded module by path. After the
  first call, a launch is a dictionary lookup
  ([what a launch does](runtime.md#what-a-launch-does)).

## Who does what

| | does |
|---|---|
| hk-dsl's Python layer | the language, tracing, the templates and the tuned kernels (`hk.ops`), the cache, the engine routes |
| the `hk` dialect and its passes (MLIR) | checking and deriving what C++ leaves to the author: waits, LDS offsets, fragments, budgets |
| [HipKittens](https://github.com/HazyResearch/HipKittens) | the C++ backend: tiles, matrix-core operations and waits, per architecture (`third_party/HipKittens`) |
| hipcc (LLVM) | instruction selection, register allocation, the code object |
| torch | tensors, streams, and the engines' custom-op machinery |

A kernel written in hk-dsl compiles to HipKittens C++ and runs at that C++'s
speed: the gain is in what you do not have to write or debug by hand.

## The same pipeline behind every entry point

- A **template** (`hk.templates.elementwise`, `rowwise`, `gemm`) builds an
  `@hk.kernel` from your math and a plan for the GPU, then goes through the
  same stages ([templates](templates/elementwise-rowwise.md)).
- **`hk.ops`** are kernels written in the language and shipped
  ([API reference](api/hk.ops.md)).
- An **engine route** wraps a kernel as a torch op that vLLM or SGLang calls
  ([integration](integration.md)).

Where next: [getting started](getting-started.md) runs the four calls on a kernel,
[tiles and strips](tiles-and-strips.md) are the data a kernel holds, and
[the correctness gates](correctness-gates.md) are what the build refuses.
