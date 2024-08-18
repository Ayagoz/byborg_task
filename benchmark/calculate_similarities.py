import time
import numpy as np
import torch


def sort_similarity(similarity: torch.Tensor):
    top_value = torch.sort(similarity, dim=0, descending=False)
    return top_value


def search_function(query: np.ndarray, tags: np.ndarray, device: str = 'cpu'):
    # Simulated search function (replace with actual logic)
    # for tag in tags:
    start = time.time()
    t_tags = torch.tensor(tags, device=device)
    t_query = torch.tensor(query, device=device)

    similarity = torch.cosine_similarity(
        t_query[None],
        t_tags[:, None],
        dim=-1
    )
    similarity = similarity.squeeze()
    # for each element in batch we sort the similarity values by ascending order to get top tag similarity values
    end = time.time()
    return end - start


def benchmark_search_batch(queries: np.ndarray, tags: np.ndarray, device: str = 'cpu'):
    start_time = time.time()
    one_batch_avg_time = 0
    for query in queries:
        local_time = search_function(query, tags, device=device)
        one_batch_avg_time += local_time
    end_time = time.time()
    return end_time - start_time, one_batch_avg_time / len(queries)


def benchmark_search_offline(queries: np.ndarray, tags: np.ndarray, device: str = 'cpu'):
    start_time = time.time()
    t_tags = torch.tensor(tags, device=device)
    t_query = torch.tensor(queries, device=device)

    # Step 1: Expand dimensions
    queries_expanded = t_query.unsqueeze(1)  # Shape becomes (x, 1, y)
    tags_expanded = t_tags.unsqueeze(0)
    similarity = torch.cosine_similarity(
        queries_expanded,
        tags_expanded,
        dim=-1
    )

    end_time = time.time()
    return end_time - start_time


def benchmark_search_online(query: np.ndarray, tags: np.ndarray, device: str = 'cpu'):
    start_time = time.time()
    t_tags = torch.tensor(tags, device=device)
    t_query = torch.tensor(query, device=device)[None]

    similarity = torch.cosine_similarity(
        t_query,
        t_tags,
        dim=-1
    )

    end_time = time.time()
    return end_time - start_time
