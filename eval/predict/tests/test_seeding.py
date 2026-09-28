"""Tests for deterministic, shard-independent inference seeding."""

import torch

from eval.predict.seeding import (
    DEFAULT_BASE_SEED,
    derive_rank_seed,
    resolve_base_seed,
    seed_inference,
)


def test_default_base_seed_when_env_unset():
    assert resolve_base_seed(env={}) == DEFAULT_BASE_SEED


def test_env_seed_is_widened_like_upstream():
    # Upstream _seed_procs multiplies seeds below 1000 by 1000; mirrored so a
    # given ANEMOI_BASE_SEED means the same thing everywhere in the stack.
    assert resolve_base_seed(env={"ANEMOI_BASE_SEED": "756"}) == 756_000
    assert resolve_base_seed(env={"ANEMOI_BASE_SEED": "123456"}) == 123_456


def test_blank_env_falls_back_to_default():
    assert resolve_base_seed(env={"ANEMOI_BASE_SEED": "  "}) == DEFAULT_BASE_SEED


def test_rank_seeds_are_distinct():
    seeds = {derive_rank_seed(756_000, r) for r in range(8)}
    assert len(seeds) == 8


def test_same_rank_reproduces_the_same_draw():
    seed_inference(2, base_seed=756_000)
    a = torch.randn(1024)
    seed_inference(2, base_seed=756_000)
    b = torch.randn(1024)
    assert torch.equal(a, b)


def test_different_ranks_draw_different_noise():
    """The defect this module fixes.

    The initial diffusion noise is drawn AFTER the grid is sharded, on the local
    shard. Shards are balanced to within one point and the global O1280 grid
    divides exactly by four, so seeding every rank with one shared seed made
    ranks 1..3 draw bit-identical noise. Per-rank seeds must not collide.
    """
    draws = []
    for rank in range(4):
        seed_inference(rank, base_seed=756_000)
        draws.append(torch.randn(4096))
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not torch.equal(draws[i], draws[j]), f"rank {i} and {j} collided"
