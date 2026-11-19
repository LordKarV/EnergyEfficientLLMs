"""Tests for the frozen P0 task lists. Run:  python -m pytest tests/test_tasklists.py -q"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrace.tasklists import HELDOUT_REPOS, POOL_REPOS, allocate, build_lists, stratified_sample


def fake_rows(per_repo=60):
    return [(f"img/{repo}:{i:04d}", repo) for repo in POOL_REPOS + HELDOUT_REPOS for i in range(per_repo)]


def test_allocate_is_proportional_and_exact():
    counts = {"a": 500, "b": 300, "c": 200}
    assert allocate(counts, 10) == {"a": 5, "b": 3, "c": 2}
    assert sum(allocate({"a": 7, "b": 5, "c": 1}, 6).values()) == 6


def test_allocate_never_exceeds_group_size():
    alloc = allocate({"big": 1000, "tiny": 1}, 50)
    assert alloc["tiny"] <= 1 and sum(alloc.values()) == 50


def test_stratified_sample_is_deterministic_and_order_free():
    groups = {"a": [f"a{i}" for i in range(20)], "b": [f"b{i}" for i in range(10)]}
    shuffled = {"a": list(reversed(groups["a"])), "b": groups["b"]}
    assert stratified_sample(groups, 6, seed=1) == stratified_sample(shuffled, 6, seed=1)
    assert stratified_sample(groups, 6, seed=1) != stratified_sample(groups, 6, seed=2)


def test_build_lists_are_disjoint_and_sized():
    verified = [f"repo__x-{i}" for i in range(500)]
    lists = build_lists(fake_rows(), verified, seed=7, inrepo_size=30, pool_size=20, heldout_size=10, verified_size=5)
    inrepo, pool, held, ver = (set(lists[k]["ids"]) for k in ("inrepo_test", "pool200", "heldout100", "verified100"))
    assert (len(inrepo), len(pool), len(held), len(ver)) == (30, 20, 10, 5)
    assert not inrepo & pool
    pool_repos = {tid.split("/")[1].split(":")[0] for tid in inrepo | pool}
    held_repos = {tid.split("/")[1].split(":")[0] for tid in held}
    assert pool_repos <= set(POOL_REPOS) and held_repos <= set(HELDOUT_REPOS)


def test_build_lists_rejects_duplicate_task_ids():
    rows = fake_rows() + [fake_rows()[0]]
    try:
        build_lists(rows, ["v-1"], seed=0, inrepo_size=1, pool_size=1, heldout_size=1, verified_size=1)
    except ValueError as err:
        assert "duplicate" in str(err)
    else:
        raise AssertionError("duplicate task ids were accepted")
