import torch
from stoke_ml.models.tft.attention import InterpretableMultiHeadAttention


class TestInterpretableMHA:
    def test_output_shape(self):
        mha = InterpretableMultiHeadAttention(d_model=64, n_heads=4)
        q = torch.randn(2, 20, 64)
        k = torch.randn(2, 20, 64)
        v = torch.randn(2, 20, 64)
        out, attn = mha(q, k, v)
        assert out.shape == (2, 20, 64)
        assert attn.shape == (2, 4, 20, 20)

    def test_attention_weights_sum_to_one(self):
        mha = InterpretableMultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
        q = torch.randn(1, 10, 64)
        k = torch.randn(1, 10, 64)
        v = torch.randn(1, 10, 64)
        _, attn = mha(q, k, v)
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_mask_support(self):
        mha = InterpretableMultiHeadAttention(d_model=64, n_heads=4)
        q = torch.randn(1, 5, 64)
        k = torch.randn(1, 5, 64)
        v = torch.randn(1, 5, 64)
        mask = torch.triu(torch.ones(5, 5), diagonal=1).bool()
        out, _ = mha(q, k, v, attn_mask=mask)
        assert out.shape == (1, 5, 64)
