#!/usr/bin/env python3
"""Inspect an athena-exported ONNX binary file."""
import sys
import numpy as np
import onnx

if len(sys.argv) < 2:
    print("Usage: python inspect_onnx.py <file.onnx>")
    sys.exit(1)

model = onnx.load(sys.argv[1])
print(f"IR version: {model.ir_version}")
print(f"Opset: {[o.version for o in model.opset_import]}")
print(f"Producer: {model.producer_name}")
print()

g = model.graph
print("Inputs:")
for inp in g.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: shape={dims}, dtype={inp.type.tensor_type.elem_type}")

print("Outputs:")
for out in g.output:
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: shape={dims}, dtype={out.type.tensor_type.elem_type}")

print()
print("Nodes:")
for n in g.node:
    print(f"  {n.name}: op={n.op_type}")
    print(f"    inputs: {list(n.input)}")
    print(f"    outputs: {list(n.output)}")
    for a in n.attribute:
        if a.ints:
            print(f"    attr {a.name}: ints={list(a.ints)}")
        elif a.floats:
            print(f"    attr {a.name}: floats={list(a.floats)}")
        elif a.s:
            print(f"    attr {a.name}: string={a.s}")

print()
print("Initializers:")
for init in g.initializer:
    data = np.array(init.float_data, dtype=np.float32)
    print(f"  {init.name}: dims={list(init.dims)}, dtype={init.data_type}, "
          f"nfloats={len(init.float_data)}, first5={data[:5]}")

print()
print("Value infos:")
for vi in g.value_info:
    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    print(f"  {vi.name}: shape={dims}")

print()
print("--- ONNX Checker ---")
try:
    onnx.checker.check_model(model)
    print("PASS: Model is valid ONNX")
except onnx.checker.ValidationError as e:
    print(f"FAIL: {e}")
