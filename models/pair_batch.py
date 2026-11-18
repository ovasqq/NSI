"""CPU collation of ragged residual states before one device transfer per field."""

import torch

from models.encoding import collate


def collate_pairs(rows):
    batch = collate([row["state"] for row in rows])
    count = len(rows)
    vertices = max(len(row["vertices"]) for row in rows)
    pairs = max(len(row["pair_ids"]) for row in rows)
    actions = max(len(row["actions"]) for row in rows)
    local = max(len(row["local"]) for row in rows)
    batch.update(
        vertices=torch.zeros((count, vertices), dtype=torch.long),
        owners=torch.zeros((count, pairs), dtype=torch.long),
        pair_ids=torch.zeros((count, pairs), dtype=torch.long),
        vertex_keep=torch.zeros((count, vertices), dtype=torch.bool),
        pair_keep=torch.zeros((count, pairs), dtype=torch.bool),
        stats=torch.zeros((count, vertices, 4)),
        query_edges=torch.zeros((count, vertices, vertices), dtype=torch.bool),
        relations=torch.zeros((count, pairs, pairs, 4), dtype=torch.bool),
        device_relations=torch.zeros(count, dtype=torch.bool),
        actions=torch.zeros((count, actions, vertices), dtype=torch.bool),
        action_keep=torch.zeros((count, actions), dtype=torch.bool),
        gains=torch.zeros((count, actions)),
        local=torch.zeros((count, local), dtype=torch.long),
        local_keep=torch.zeros((count, local), dtype=torch.bool),
        teacher=torch.zeros((count, local)),
        teacher_keep=torch.zeros(count, dtype=torch.bool),
        target_image=torch.full((count,), -100, dtype=torch.long),
    )
    for i, row in enumerate(rows):
        n, k, a, c = map(
            len, (row["vertices"], row["pair_ids"], row["actions"], row["local"])
        )
        for name in ("vertices", "stats"):
            batch[name][i, :n] = row[name]
        for name in ("owners", "pair_ids"):
            batch[name][i, :k] = row[name]
        batch["vertex_keep"][i, :n] = True
        batch["pair_keep"][i, :k] = True
        batch["query_edges"][i, :n, :n] = row["query_edges"]
        if row["relations"] is None:
            batch["device_relations"][i] = True
        else:
            batch["relations"][i, :k, :k] = row["relations"]
        batch["actions"][i, :a, :n] = row["actions"]
        batch["action_keep"][i, :a] = True
        if "gains" in row:
            batch["gains"][i, :a] = row["gains"]
        batch["local"][i, :c] = torch.tensor(row["local"])
        batch["local_keep"][i, :c] = True
        if "teacher" in row:
            batch["teacher"][i, :c] = row["teacher"]
            batch["teacher_keep"][i] = True
        batch["target_image"][i] = row.get("target_image", -100)
    return batch
