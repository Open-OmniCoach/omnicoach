"""Compare batched nucleus transport against SGLang's original selection."""

import pytest
import torch
from sglang.srt.layers.utils.top_p import get_top_p_token_ids_from_probs as original
from sglang.srt.sampling.sampling_params import TOP_K_ALL

from slime_ext.backends.sglang.top_p import get_top_p_token_ids_from_probs as batched


@pytest.mark.parametrize("batch,vocab", [(1, 31), (3, 152064), (8, 257)])
@pytest.mark.parametrize("need_top_p,need_min_p", [(False, False), (True, False), (True, True)])
def test_same_token_ids_and_order(batch, vocab, need_top_p, need_min_p):
    generator = torch.Generator().manual_seed(42)
    probs = torch.randn(batch, vocab, generator=generator).softmax(dim=-1)
    # Include ties, a one-token distribution, mixed requests and unfiltered rows.
    probs[0].fill_(1 / vocab)
    if batch > 1:
        probs[1].zero_()
        probs[1, 7] = 1
    args = dict(
        probs=probs,
        top_ks=torch.tensor(([20, TOP_K_ALL, 1] * batch)[:batch]),
        top_ps=torch.tensor(([0.95, 1.0, 0.5] * batch)[:batch]),
        min_ps=torch.tensor(([0.0, 0.01, 0.2] * batch)[:batch]),
        need_top_p_sampling=need_top_p,
        need_min_p_sampling=need_min_p,
        request_mask=torch.tensor(([True, True, False] * batch)[:batch]),
    )
    expected, actual = original(**args), batched(**args)
    assert len(actual) == len(expected)
    for left, right in zip(expected, actual, strict=True):
        if left is None:
            assert right is None
        else:
            assert right.dtype == left.dtype == torch.int32
            assert torch.equal(left, right)


@pytest.mark.parametrize("requested", [False, True])
def test_no_filtered_rows_returns_none(requested):
    args = dict(
        probs=torch.tensor([[0.0, 0.5, 0.5]]),
        top_ks=torch.tensor([TOP_K_ALL]),
        top_ps=torch.tensor([1.0]),
        min_ps=torch.tensor([0.0]),
        need_top_p_sampling=True,
        need_min_p_sampling=True,
        request_mask=torch.tensor([requested]),
    )
    assert original(**args) is None
    assert batched(**args) is None
