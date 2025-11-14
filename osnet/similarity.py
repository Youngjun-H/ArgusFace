import torch


def compute_pairwise_similarity(features: torch.Tensor, metric: str = 'cosine') -> torch.Tensor:
    """
    Compute pairwise similarity or distance matrix for embeddings.

    Args:
        features: Tensor of shape (N, D)
        metric: 'cosine' or 'euclidean'

    Returns:
        Tensor of shape (N, N):
          - cosine: cosine similarity in [-1, 1]
          - euclidean: pairwise Euclidean distance (>= 0)
    """
    if features.dim() != 2:
        raise ValueError('features must be a 2D tensor of shape (N, D)')

    metric = metric.lower()

    if metric == 'cosine':
        # Normalize to unit length then compute dot products
        feats = torch.nn.functional.normalize(features, p=2, dim=1)
        sim = feats @ feats.t()
        return sim

    if metric == 'euclidean':
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
        # then take sqrt (ensure numerical stability)
        x = features
        x2 = (x ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        dist2 = x2 + x2.t() - 2.0 * (x @ x.t())
        dist2 = torch.clamp(dist2, min=0.0)
        dist = torch.sqrt(dist2 + 1e-12)
        return dist

    raise ValueError("metric must be one of ['cosine', 'euclidean']")


def compare_images(image_list, extractor, metric: str = 'cosine') -> torch.Tensor:
    """
    Convenience wrapper that extracts features and returns pairwise matrix.

    Args:
        image_list: list of image paths/arrays supported by extractor
        extractor: OSNetFeatureExtractor
        metric: 'cosine' or 'euclidean'

    Returns:
        (N, N) tensor (similarity for 'cosine', distance for 'euclidean')
    """
    with torch.no_grad():
        feats = extractor(image_list)  # (N, D)
    return compute_pairwise_similarity(feats, metric=metric)


