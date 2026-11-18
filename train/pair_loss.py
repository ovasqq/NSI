"""Vectorized ranking, check-value and action losses for a padded batch."""

import torch
from torch.nn import functional as F


def pair_loss(ranks, scores, batch, teacher_ranks):
    keep = batch["local_keep"]
    student = ranks.gather(1, batch["local"]).masked_fill(~keep, float("-inf"))
    teacher = teacher_ranks.gather(1, batch["local"])
    teacher = torch.where(batch["teacher_keep"][:, None], batch["teacher"], teacher)
    teacher = teacher.masked_fill(~keep, float("-inf"))
    log_student = F.log_softmax(student, dim=-1).masked_fill(~keep, 0.0)
    loss = F.kl_div(log_student, F.softmax(teacher, dim=-1), reduction="none").sum(-1)
    loss = loss + F.cross_entropy(
        ranks, batch["target_image"], reduction="none", ignore_index=-100
    )
    if scores.shape[1]:
        gains = batch["gains"]
        targets = gains.sign() * gains.abs().log1p()
        actions = batch["action_keep"]
        regression = F.smooth_l1_loss(scores, targets, reduction="none") * (
            1 + 2 * (gains > 0)
        )
        loss = loss + (regression * actions).sum(-1) / actions.sum(-1).clamp_min(1)
        logits = torch.cat(
            (
                scores.masked_fill(~actions, float("-inf")),
                scores.new_zeros((len(scores), 1)),
            ),
            dim=-1,
        )
        best_gain, best = gains.masked_fill(~actions, float("-inf")).max(-1)
        target = torch.where(best_gain > 0, best, scores.shape[1])
        loss = loss + 0.1 * F.cross_entropy(logits, target, reduction="none")
    return loss.mean()
