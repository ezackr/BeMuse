import torch
import torch.nn.functional as F


def loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the loss between a group of estimate and target vectors using
    pairwise similarity.
    :param estimate: the estimated vectors
    :param target: the true vectors
    :return: the sum of squared losses for each pairwise similarity
    """
    num_samples = len(estimate)
    estimate_norm = F.normalize(estimate, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    actual_similarity = torch.matmul(estimate_norm, target_norm.t())
    expected_similarity = torch.eye(num_samples)
    return (1 / num_samples) * torch.sum((actual_similarity - expected_similarity) ** 2)
