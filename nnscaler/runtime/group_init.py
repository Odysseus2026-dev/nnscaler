# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Create communication groups in deterministic, collision-free split rounds."""
import ctypes
import hashlib

import torch
import torch.distributed as dist


def _nccl_split_color(ranks):
    """Match PyTorch's native genNcclSplitColor, including size_t overflow.

    split_group uses this C++ hash, not distributed_c10d._process_group_color.
    Distinct disjoint memberships can hash to the same NCCL split color.
    """
    combined_hash = 0
    for rank in ranks:
        combined_hash = ctypes.c_size_t(
            combined_hash ^ (rank + 0x9e3779b9 + (combined_hash << 6) + (combined_hash >> 2))
        ).value
    return abs(ctypes.c_int64(combined_hash).value) % ((1 << 31) - 1)


def partition_groups(groups):
    """Pack memberships without reusing a rank or NCCL color in one round."""
    batches, occupied, colors = [], [], []
    for ranks in sorted(set(groups), key=lambda ranks: (-len(ranks), ranks)):
        members = set(ranks)
        color = _nccl_split_color(ranks)
        for batch, used, used_colors in zip(batches, occupied, colors):
            if members.isdisjoint(used) and color not in used_colors:
                batch.append(ranks)
                used.update(members)
                used_colors.add(color)
                break
        else:
            batches.append([ranks])
            occupied.append(members)
            colors.append({color})
    return batches


def _split_context():
    if not dist.is_initialized():
        return None
    parent = dist.group.WORLD
    device = getattr(parent, 'bound_device_id', None)
    if device is None or dist.get_backend(parent) != 'nccl':
        return None
    backend = parent._get_backend(device)
    supported = callable(getattr(dist, 'split_group', None)) and getattr(backend, 'supports_splitting', False)
    # Some PyTorch builds index an empty WORLD rank map after reinitialization.
    # Do not attempt the split: this failure is a native crash, not an exception.
    rank_map = getattr(getattr(backend, 'options', None), 'global_ranks_in_group', ())
    supported = supported and (bool(rank_map) or getattr(backend, 'uid', None) == 0)
    return device, supported


def _agree_on_split(groups):
    """Validate the shared group plan and require split support on every rank."""
    context = _split_context()
    if context is None:
        return False
    device, supported = context
    signature = int.from_bytes(hashlib.sha256(repr(groups).encode()).digest()[:8], 'big') >> 1
    # One MIN reduction computes all(supported), min(signature), -max(signature).
    # Equal minimum and maximum mean every rank supplied the same plan hash.
    agreement = torch.tensor([int(supported), signature, -signature], dtype=torch.int64, device=device)
    dist.all_reduce(agreement, op=dist.ReduceOp.MIN)
    all_supported, min_signature, neg_max_signature = agreement.tolist()
    if min_signature != -neg_max_signature:
        raise ValueError('Ranks supplied different pending communication groups')
    return bool(all_supported)


def create_groups(groups, *, timeout):
    """All WORLD ranks declare the same memberships in the same call order.

    Single groups use ordinary new_group. Multiple disjoint groups share a
    split; excluded ranks participate without creating padding communicators.
    """
    if not groups:
        return {}
    if not _agree_on_split(groups):
        return {ranks: dist.new_group(list(ranks), timeout=timeout) for ranks in groups}

    rank = dist.get_rank()
    result = {}
    for batch in partition_groups(groups):
        if len(batch) == 1:
            ranks = batch[0]
            result[ranks] = dist.new_group(list(ranks), timeout=timeout)
        else:
            group = dist.split_group(split_ranks=[list(ranks) for ranks in batch], timeout=timeout)
            for ranks in batch:
                # None from split_group means nonmember, but None in a collective
                # means WORLD. Preserve the ordinary new_group sentinel instead.
                result[ranks] = group if rank in ranks else dist.GroupMember.NON_GROUP_MEMBER
    return result
