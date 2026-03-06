"""
Unit tests for Megatron-LM Tensor Parallelism primitives.
"""

import pytest
import torch
import torch.nn as nn
from dist_llm.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear


def test_column_parallel_equivalence():
    in_dim = 64
    out_dim = 128
    tp_size = 2

    # Standard full linear
    torch.manual_seed(1337)
    full_linear = nn.Linear(in_dim, out_dim, bias=False)

    # Mock column parallel rank 0 and rank 1
    split_out = out_dim // tp_size
    col_p0 = ColumnParallelLinear(in_dim, out_dim, bias=False, tp_size=tp_size)
    col_p1 = ColumnParallelLinear(in_dim, out_dim, bias=False, tp_size=tp_size)

    # Assign weight slices
    col_p0.weight.data.copy_(full_linear.weight.data[:split_out, :])
    col_p1.weight.data.copy_(full_linear.weight.data[split_out:, :])

    x = torch.randn(4, 16, in_dim)

    # Full forward
    y_expected = full_linear(x)

    # Local parallel forward
    y0 = col_p0(x)
    y1 = col_p1(x)
    y_parallel = torch.cat([y0, y1], dim=-1)

    assert torch.allclose(y_expected, y_parallel, atol=1e-5)


def test_row_parallel_equivalence():
    in_dim = 128
    out_dim = 64
    tp_size = 2

    # Standard full linear
    torch.manual_seed(1337)
    full_linear = nn.Linear(in_dim, out_dim, bias=False)

    split_in = in_dim // tp_size
    row_p0 = RowParallelLinear(in_dim, out_dim, bias=False, tp_size=tp_size)
    row_p1 = RowParallelLinear(in_dim, out_dim, bias=False, tp_size=tp_size)

    row_p0.weight.data.copy_(full_linear.weight.data[:, :split_in])
    row_p1.weight.data.copy_(full_linear.weight.data[:, split_in:])

    x = torch.randn(4, 16, in_dim)
    x0 = x[:, :, :split_in]
    x1 = x[:, :, split_in:]

    y_expected = full_linear(x)

    # Row parallel sums partial products across ranks
    y0 = row_p0(x0)
    y1 = row_p1(x1)
    y_summed = y0 + y1

    assert torch.allclose(y_expected, y_summed, atol=1e-5)
