import random
from collections import Counter
from typing import List, Tuple, Sequence, Hashable
from typing import List, Tuple, Dict, Hashable, Sequence

from typing import List, Tuple, Dict, Hashable

def build_group_ids_from_groups(
    dataset,
    groups: List[List[Tuple[Hashable, Hashable]]],
    treat_unassigned_as_singleton: bool = True,
) -> List[int]:

    pair_to_group: Dict[Tuple[Hashable, Hashable], int] = {}
    for gid, g in enumerate(groups):
        for pair in g:
            if pair in pair_to_group:
                raise ValueError()
            pair_to_group[pair] = gid


    group_ids: List[int] = []
    next_group_id = len(groups)  
    n_samples = len(dataset)
    for idx in range(n_samples):
        item = dataset[idx]  
        try:
            core_id = item[-2]
            ligand_id = item[-1]
        except Exception as e:
            raise ValueError() from e

        pair = (core_id, ligand_id)

        if pair in pair_to_group:
            gid = pair_to_group[pair]
        else:
            if treat_unassigned_as_singleton:
                gid = next_group_id
                pair_to_group[pair] = gid
                next_group_id += 1
            else:
                raise ValueError()

        group_ids.append(gid)

    return group_ids






def group_train_val_test_split(
    group_ids: Sequence[Hashable],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:

    n_samples = len(group_ids)
    group_counts = Counter(group_ids) 

    unique_groups = list(group_counts.keys())
    random.Random(seed).shuffle(unique_groups)

    target_val_size = int(n_samples * val_ratio)
    target_test_size = int(n_samples * test_ratio)

    val_groups = set()
    test_groups = set()

    val_size = 0
    test_size = 0


    for g in unique_groups:
        if val_size < target_val_size:
            val_groups.add(g)
            val_size += group_counts[g]
        elif test_size < target_test_size:
            test_groups.add(g)
            test_size += group_counts[g]
        else:
            pass


    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for idx, g in enumerate(group_ids):
        if g in val_groups:
            val_indices.append(idx)
        elif g in test_groups:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

    return train_indices, val_indices, test_indices

