"""Shared test utilities."""

import numpy as np
import torch
from pathlib import Path


def read_ply_xyz(ply_path: Path) -> torch.Tensor:
    """Simple binary PLY reader that extracts XYZ coordinates."""
    with open(ply_path, "rb") as f:
        line = f.readline().decode("ascii").strip()
        assert line == "ply", f"Not a PLY file: {line}"

        vertex_count = None
        properties = []
        format_type = None

        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                parts = line.split()
                format_type = parts[1]
            elif line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            elif line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break

        assert vertex_count is not None, "No vertex count found"
        assert format_type == "binary_little_endian", f"Unsupported format: {format_type}"

        xyz_indices = [i for i, (_, name) in enumerate(properties) if name in ["x", "y", "z"]]
        assert len(xyz_indices) == 3, f"Expected x,y,z properties, got {len(xyz_indices)}"

        dtype_map = {
            "char": 1, "uchar": 1, "short": 2, "ushort": 2,
            "int": 4, "uint": 4, "float": 4, "double": 8,
            "float32": 4, "float64": 8, "int32": 4, "uint32": 4,
        }
        prop_sizes = [dtype_map.get(dtype, 4) for dtype, _ in properties]
        vertex_stride = sum(prop_sizes)
        data = f.read()

    points = np.zeros((vertex_count, 3), dtype=np.float32)
    xyz_offsets = [sum(prop_sizes[:i]) for i in xyz_indices]

    for i in range(vertex_count):
        offset = i * vertex_stride
        for j, (prop_idx, prop_offset) in enumerate(zip(xyz_indices, xyz_offsets)):
            prop_dtype = properties[prop_idx][0]
            if prop_dtype in ["float", "float32"]:
                val = np.frombuffer(
                    data[offset + prop_offset : offset + prop_offset + 4], dtype=np.float32
                )[0]
            elif prop_dtype == "double":
                val = np.frombuffer(
                    data[offset + prop_offset : offset + prop_offset + 8], dtype=np.float64
                )[0]
            else:
                continue
            points[i, j] = val

    return torch.from_numpy(points)
