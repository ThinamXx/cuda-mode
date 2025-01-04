import torch
import triton
import triton.language as tl


def test_ops(batch_size, num_heads, seq_len, head_dim, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    K = (
        torch.empty(
            (batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    V = (
        torch.empty(
            (batch_size, num_heads, seq_len, head_dim), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (head_dim**0.5)
    dO = torch.randn_like(Q)

    # reference implementation:
    mask = torch.tril(torch.ones((seq_len, seq_len), device="cuda"))
    P = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
    if causal:
        P[:, :, mask == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dQ, Q.grad = Q.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dV, V.grad = V.grad.clone(), None

    # triton implementation:
    tri_O = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_O.backward(dO)
    tri_dQ, Q.grad = Q.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dV, V.grad = V.grad.clone(), None

    # comparison of the outputs:
    atol = 1e-2
    rtol = 0.0
    assert torch.allclose(ref_O, tri_O, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dQ, tri_dQ, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dK, tri_dK, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dV, tri_dV, rtol=rtol, atol=atol)
