#!/usr/bin/env python3
"""
Read an athena-exported ONNX binary file and verify consistency with PyTorch.

This script demonstrates loading a binary ONNX model exported by athena,
running inference with ONNX Runtime, reconstructing the equivalent model
in PyTorch, and comparing outputs to verify correctness.

Usage:
    python3 verify_onnx.py <model.onnx> [reference_output.txt]

Requirements:
    pip install onnx onnxruntime numpy torch
"""
import sys
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn as nn
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install onnx onnxruntime numpy torch")
    sys.exit(1)


def load_and_inspect(model_path):
    """Load ONNX model and print its structure."""
    model = onnx.load(model_path)

    print(f"Model: {model_path}")
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset: {[o.version for o in model.opset_import]}")
    print(f"  Producer: {model.producer_name} {model.producer_version}")
    print()

    g = model.graph
    print("Graph structure:")
    for inp in g.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name} shape={dims}")
    for out in g.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} shape={dims}")
    print()

    print("Nodes:")
    for n in g.node:
        print(f"  {n.name}: {n.op_type}  "
              f"inputs={list(n.input)} -> outputs={list(n.output)}")
    print()

    print("Initializers:")
    for init in g.initializer:
        print(f"  {init.name}: dims={list(init.dims)} "
              f"({len(init.float_data)} floats)")
    print()

    # Validate
    try:
        onnx.checker.check_model(model)
        print("ONNX validation: PASS")
    except onnx.checker.ValidationError as e:
        print(f"ONNX validation: FAIL - {e}")
        return None

    return model


def run_onnxruntime(model_path, input_data):
    """Run inference using ONNX Runtime."""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_data})
    return result[0]


def build_pytorch_from_onnx(model):
    """
    Build a PyTorch Sequential model from the ONNX graph.
    Supports Gemm (fully connected) and common activations.
    """
    g = model.graph

    # Index initializers by name
    inits = {}
    for init in g.initializer:
        data = np.array(init.float_data, dtype=np.float32)
        dims = list(init.dims)
        if len(dims) > 1:
            data = data.reshape(dims)
        inits[init.name] = data

    layers = []
    for node in g.node:
        op = node.op_type

        if op == "Gemm":
            # Gemm: Y = alpha * A @ B + beta * C
            weight_name = node.input[1]
            weight = inits[weight_name]

            has_bias = len(node.input) >= 3 and node.input[2] in inits
            bias = inits[node.input[2]] if has_bias else None

            # Check transB attribute
            trans_b = 0
            for attr in node.attribute:
                if attr.name == "transB":
                    trans_b = attr.i

            # Weight shape from ONNX: [in_features, out_features] (transB=0)
            # or [out_features, in_features] (transB=1)
            if trans_b:
                out_features, in_features = weight.shape
            else:
                in_features, out_features = weight.shape

            linear = nn.Linear(in_features, out_features, bias=has_bias)
            with torch.no_grad():
                if trans_b:
                    linear.weight.copy_(torch.from_numpy(weight))
                else:
                    linear.weight.copy_(torch.from_numpy(weight.T))
                if has_bias:
                    linear.bias.copy_(torch.from_numpy(bias))
            layers.append(linear)

        elif op == "MatMul":
            weight_name = node.input[1]
            weight = inits[weight_name]
            in_features, out_features = weight.shape

            linear = nn.Linear(in_features, out_features, bias=False)
            with torch.no_grad():
                linear.weight.copy_(torch.from_numpy(weight.T))
            layers.append(linear)

        elif op == "Relu":
            layers.append(nn.ReLU())
        elif op == "Sigmoid":
            layers.append(nn.Sigmoid())
        elif op == "Tanh":
            layers.append(nn.Tanh())
        elif op == "Softmax":
            layers.append(nn.Softmax(dim=-1))
        else:
            print(f"  Warning: Unsupported op '{op}' - skipping")

    return nn.Sequential(*layers)


def load_reference_output(filepath):
    """Load reference output values from athena text file."""
    values = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values.append(float(line))
    return np.array(values, dtype=np.float32)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 verify_onnx.py <model.onnx> [reference_output.txt]")
        sys.exit(1)

    model_path = sys.argv[1]
    ref_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Step 1: Load and validate ONNX model
    print("=" * 60)
    print("Step 1: Load and validate ONNX model")
    print("=" * 60)
    model = load_and_inspect(model_path)
    if model is None:
        sys.exit(1)

    # Determine input shape from model
    inp = model.graph.input[0]
    input_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"\nUsing input shape: {input_shape}")
    input_data = np.ones(input_shape, dtype=np.float32)

    # Step 2: Run ONNX Runtime inference
    print()
    print("=" * 60)
    print("Step 2: ONNX Runtime inference")
    print("=" * 60)
    ort_output = run_onnxruntime(model_path, input_data)
    print(f"  Output shape: {ort_output.shape}")
    print(f"  Output: {ort_output}")

    # Step 3: Build equivalent PyTorch model
    print()
    print("=" * 60)
    print("Step 3: PyTorch model reconstruction")
    print("=" * 60)
    pytorch_model = build_pytorch_from_onnx(model)
    print(f"  Model: {pytorch_model}")

    with torch.no_grad():
        pt_input = torch.from_numpy(input_data)
        pt_output = pytorch_model(pt_input).numpy()
    print(f"  Output shape: {pt_output.shape}")
    print(f"  Output: {pt_output}")

    # Step 4: Compare results
    print()
    print("=" * 60)
    print("Step 4: Comparison")
    print("=" * 60)

    all_pass = True

    # Compare ONNX Runtime vs PyTorch
    ort_flat = ort_output.flatten()
    pt_flat = pt_output.flatten()
    max_diff = np.max(np.abs(ort_flat - pt_flat))
    match = np.allclose(ort_flat, pt_flat, atol=1e-6)
    status = "PASS" if match else "FAIL"
    print(f"  ONNX Runtime vs PyTorch:  {status}  (max diff: {max_diff:.2e})")
    if not match:
        all_pass = False

    # Compare with athena reference if provided
    if ref_path:
        ref_output = load_reference_output(ref_path)
        max_diff_ref = np.max(np.abs(ort_flat - ref_output))
        match_ref = np.allclose(ort_flat, ref_output, atol=1e-5)
        status_ref = "PASS" if match_ref else "FAIL"
        print(f"  ONNX Runtime vs Athena:   {status_ref}  (max diff: {max_diff_ref:.2e})")
        if not match_ref:
            all_pass = False

        max_diff_pt = np.max(np.abs(pt_flat - ref_output))
        match_pt = np.allclose(pt_flat, ref_output, atol=1e-5)
        status_pt = "PASS" if match_pt else "FAIL"
        print(f"  PyTorch vs Athena:        {status_pt}  (max diff: {max_diff_pt:.2e})")
        if not match_pt:
            all_pass = False

    print()
    if all_pass:
        print("All comparisons PASSED - model is consistent across frameworks!")
    else:
        print("Some comparisons FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
