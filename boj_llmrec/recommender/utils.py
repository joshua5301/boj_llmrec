import torch

def vae_bce_loss(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return -torch.sum(torch.nn.functional.log_softmax(pred, 1) * true, -1)

def vae_reg_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), axis=1)

def recall(true: list[list], pred: list[list]) -> float:
    recall = 0
    nonempty_cnt = 0
    for p, t in zip(pred, true):
        if len(t) == 0:
            continue
        recall += len(set(p) & set(t)) / len(t)
        nonempty_cnt += 1
    try:
        recall /= nonempty_cnt
    except ZeroDivisionError:
        recall = -1
    return recall