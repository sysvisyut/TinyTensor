# Canonical DSL and SSA Tensor IR schema

The parser accepts the DSL below. The IR printer is canonical text for golden tests; it is not yet required to parse its own output. Whitespace, value order, and type spelling are part of the contract.

## DSL grammar

```ebnf
program    = { declaration | assignment | output } ;
declaration= "input" , ident , ":" , "f32" , "[" , dims , "]" ;
dims       = integer , { "," , integer } ;
assignment = ident , "=" , expr ;
output     = "output" , ident ;
expr       = sum ; sum = product , { ("+" | "-") , product } ;
product    = factor , { ("*" | "/") , factor } ;
factor     = ident | float | call | "(" , expr , ")" ;
call       = ident , "(" , expr , { "," , expr } , ")" ;
```

Names are single-assignment. Inputs are declared before use. Tensor shapes are positive integers; rank is one or two. A scalar broadcasts to any operand shape; `[N]` broadcasts only to the trailing dimension of `[M,N]`. `matmul` only accepts `[M,K]` and `[K,N]`.

## Canonical printer grammar

```ebnf
module  = "module {\n", { input }, { constant }, { instruction }, { output }, "}\n" ;
input   = "  %", id, " = input \"", name, "\" : ", type, "\n" ;
constant= "  %", id, " = constant ", float, " : f32[]\n" ;
instruction = "  %", id, " = ", opname, " ", operands, " : ", type, "\n" ;
operands= "%", id, { ", %", id } ;
output  = "  output %", id, "\n" ;
type    = "f32[", [ integer, { ",", integer } ], "]" ;
```

Print inputs by their declaration order, constants by creation order, instructions by construction order, then outputs by declaration order. IDs are decimal and never reused. Scalar type is `f32[]`; there are no commas or spaces inside dimensions. Op names are lowercase: `add`, `sub`, `mul`, `div`, `max`, `relu`, `exp`, `neg`, `matmul`, `fused_elementwise`, `fused_matmul_epilogue`.

## Baseline example

DSL:

```text
input A: f32[1024,1024]
input B: f32[1024,1024]
input bias: f32[1024]
T1 = matmul(A, B)
T2 = T1 + bias
C = relu(T2)
unused = exp(C)
output C
```

Before optimization (the names are retained only on inputs; SSA IDs are authoritative):

```text
module {
  %0 = input "A" : f32[1024,1024]
  %1 = input "B" : f32[1024,1024]
  %2 = input "bias" : f32[1024]
  %3 = matmul %0, %1 : f32[1024,1024]
  %4 = add %3, %2 : f32[1024,1024]
  %5 = relu %4 : f32[1024,1024]
  %6 = exp %5 : f32[1024,1024]
  output %5
}
```

After DCE, only the dead `exp` instruction disappears; it does not renumber surviving values:

```text
module {
  %0 = input "A" : f32[1024,1024]
  %1 = input "B" : f32[1024,1024]
  %2 = input "bias" : f32[1024]
  %3 = matmul %0, %1 : f32[1024,1024]
  %4 = add %3, %2 : f32[1024,1024]
  %5 = relu %4 : f32[1024,1024]
  output %5
}
```

After legal matmul-epilogue fusion, the pass replaces only the `add`+`relu` chain with a group operation. `%3` has exactly one user and `%4` has exactly one user; `%3` is not externally visible, so fusion is legal:

```text
module {
  %0 = input "A" : f32[1024,1024]
  %1 = input "B" : f32[1024,1024]
  %2 = input "bias" : f32[1024]
  %7 = fused_matmul_epilogue %0, %1, %2 : f32[1024,1024]
  output %7
}
```

The fusion payload (operations and attributes) must be stored in a future `KernelGroup` side table or explicit structured attribute; do not infer it from an arbitrary operand order. For v1, `fused_matmul_epilogue(A,B,bias)` means `relu(matmul(A,B) + bias)` exactly.

## Key negative legality case

If `%3 = mul %0, %1` feeds both `add %3, %2` and `sub %3, %2`, fusing only one consumer would duplicate/reorder the producer and is rejected. The pass should leave this subgraph unchanged unless it forms a single coherent group containing **all** uses and preserves external outputs.
