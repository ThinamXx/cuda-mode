import torch
import triton
import triton.language as tl


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        head_dim_Q, head_dim_K, head_dim_V = Q.shape[-1], K.shape[-1], V.shape[-1]
        assert head_dim_Q == head_dim_K and head_dim_K == head_dim_V

        batch_size, num_heads, seq_len, head_dim = Q.shape

        output = torch.empty_like(Q)
        stage = 3 if causal else 1

        grid = lambda meta: (
            triton.cdiv(
                seq_len, meta["BLOCK_SIZE"]
            ),  # which block of queries we are working with.
            batch_size * num_heads,  # which heads of which batch we are working with.
            1,  # Z dimension.
        )

        # used to save logsumexp for the backward pass.
        M = torch.empty(
            (batch_size, num_heads, seq_len), device=output.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            output=output,
            M=M,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_output_batch=output.stride(0),
            stride_output_head=output.stride(1),
            stride_output_seq=output.stride(2),
            stride_output_dim=output.stride(3),
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            stage=stage,
        )

        ctx.save_for_backward(Q, K, V, output, M)
        ctx.grid = grid
        ctx.head_dim = head_dim
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale

        return output


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
