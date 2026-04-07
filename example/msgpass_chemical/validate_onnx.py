"""Validate ONNX JSON export against Fortran forward pass output."""
import numpy as np
import onnx
import onnxruntime as ort

def read_first_graph(path):
    """Read first graph sample from all_graphs.txt."""
    with open(path) as f:
        num_samples = int(f.readline().strip())
        header = f.readline().strip().split()
        nv, ne, ncsr, nvf, nef = [int(x) for x in header]

        # Vertex features: nv lines x nvf values
        vf = np.zeros((nv, nvf), dtype=np.float32)
        for i in range(nv):
            vf[i] = [float(x) for x in f.readline().split()]

        # Edge features: ne lines x nef values
        ef = np.zeros((ne, nef), dtype=np.float32)
        for i in range(ne):
            ef[i] = [float(x) for x in f.readline().split()]

        # CSR adj_ia: nv+1 values on one line
        adj_ia = [int(x) for x in f.readline().split()]

        # CSR adj_ja: ncsr lines, 2 values each (neighbour_idx, edge_idx)
        adj_ja = np.zeros((ncsr, 2), dtype=np.int64)
        for i in range(ncsr):
            vals = f.readline().split()
            adj_ja[i] = [int(vals[0]), int(vals[1])]

        # Energy label
        label = float(f.readline().strip())

    return nv, ne, ncsr, nvf, nef, vf, ef, adj_ia, adj_ja, label


def build_edge_index(nv, adj_ia, adj_ja):
    """Build [3, ncsr] edge_index from CSR format.

    Row 0: source vertex (0-indexed) = adj_ja neighbour
    Row 1: edge feature index (0-indexed) = adj_ja edge idx
    Row 2: target vertex (0-indexed) = v from CSR iteration
    """
    ncsr = adj_ja.shape[0]
    edge_index = np.zeros((3, ncsr), dtype=np.int64)
    k = 0
    for v in range(nv):  # v is target (0-indexed)
        start = adj_ia[v] - 1   # convert from 1-indexed to 0-indexed
        end = adj_ia[v + 1] - 1
        for j in range(start, end):
            edge_index[0, k] = adj_ja[j, 0] - 1  # source vertex (0-indexed)
            edge_index[1, k] = adj_ja[j, 1] - 1  # edge feature index (0-indexed)
            edge_index[2, k] = v                  # target vertex (0-indexed)
            k += 1
    assert k == ncsr, f"Expected {ncsr} entries but got {k}"
    return edge_index


def compute_degree(nv, adj_ia):
    """Compute degree per vertex from CSR row pointers."""
    degree = np.zeros(nv, dtype=np.int64)
    for v in range(nv):
        degree[v] = adj_ia[v + 1] - adj_ia[v]
    return degree


def main():
    # Read Fortran output
    with open("example/msgpass_chemical/fortran_output.txt") as f:
        fortran_output = float(f.readline().strip())
    print(f"Fortran output: {fortran_output:.12e}")

    # Read first graph
    nv, ne, ncsr, nvf, nef, vf, ef, adj_ia, adj_ja, label = \
        read_first_graph("example/msgpass_chemical/all_graphs.txt")
    print(f"Graph: {nv} vertices, {ne} edges, {ncsr} CSR entries, "
          f"{nvf} vertex features, {nef} edge features")

    # Build ONNX inputs
    edge_index = build_edge_index(nv, adj_ia, adj_ja)
    degree = compute_degree(nv, adj_ia)

    print(f"Edge index shape: {edge_index.shape}")
    print(f"Degree: {degree}")

    # Run ONNX Runtime inference
    # Load JSON format with onnx library, then pass serialized bytes to ORT
    from google.protobuf import json_format

    # json_format.ParseDict(model_json, model)
    model = onnx.load("example/msgpass_chemical/model.json")
    sess = ort.InferenceSession(model.SerializeToString())
    input_names = [inp.name for inp in sess.get_inputs()]
    print(f"Model inputs: {input_names}")

    feeds = {
        "input_5_vertex": vf,
        "input_5_edge": ef,
        "input_5_edge_index": edge_index,
        "input_5_degree": degree,
    }

    results = sess.run(None, feeds)
    onnx_output = results[0]
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"ONNX output: {onnx_output.flatten()[0]:.12e}")

    # Compare
    diff = abs(onnx_output.flatten()[0] - fortran_output)
    rel_diff = diff / max(abs(fortran_output), 1e-30)
    print(f"Absolute difference: {diff:.6e}")
    print(f"Relative difference: {rel_diff:.6e}")

    if rel_diff < 1e-5:
        print("PASS: Numerical match within 1e-5 relative tolerance")
    elif rel_diff < 1e-3:
        print("WARN: Close but not exact match")
    else:
        print("FAIL: Significant numerical difference")


if __name__ == "__main__":
    main()
