import time
from typing import Union, List
from pathlib import Path
from argparse import ArgumentParser
from enum import Enum

import torch
import numpy as np
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings

from benchmark.create_dummy_tags import tags
from benchmark.calculate_similarities import benchmark_search_batch, benchmark_search_offline, benchmark_search_online


class InferenceTypes(Enum):
    online: str = "online"
    offline: str = "offline"
    batch: str = "batch"


def load_embedding_model(model_name: str, cache_dir: str, device: str = 'cpu') -> HuggingFaceEmbeddings:
    print("Loading embedding model...")
    start = time.time()
    model_kwargs = {'device': device}
    transformer = HuggingFaceEmbeddings(
        model_name=model_name, cache_folder=cache_dir,
        model_kwargs=model_kwargs
    )
    end = time.time()
    print(f"Embedding model loaded successfully in {end - start:.2f} seconds")
    return transformer


def convert_texts_to_embeddings(texts: List[str], transformer: HuggingFaceEmbeddings, embedding_device: str) -> Union[np.ndarray, torch.Tensor]:
    print("Convert tags to embeddings...")
    start = time.time()
    # 189 tags with X-dim embeddings (default 768-dim)
    embeddings = transformer.embed_documents(texts)
    if embedding_device == 'cpu':
        embeddings = np.array(embeddings)
    end = time.time()
    print(
        f"Converted to embeddings {embeddings.shape} in {end - start:.2f} seconds")
    return embeddings


def load_dummy_tags(dummy_tags_path: str) -> np.ndarray:
    print("Check if dummy tags are created")
    local_path = Path(dummy_tags_path)
    if not local_path.exists():
        print("Dummy tags not found. Please run create_dummy_tags.py to create dummy tags.")
        print("The command to run create_dummy_tags.py is:\npython create_dummy_tags.py")

    queries = pd.read_csv(local_path)['name'].values
    return queries


def create_batched_embeddings(embeddings: Union[np.ndarray, torch.Tensor], batch_size: int) -> np.ndarray:
    start = time.time()
    _, dim = embeddings.shape
    new_shape = (len(embeddings) // batch_size, batch_size, dim)
    max_len = new_shape[0] * batch_size
    batched_embeddings = None
    if isinstance(embeddings, torch.Tensor):
        batched_embeddings = torch.reshape(embeddings[:max_len], new_shape)
    elif isinstance(embeddings, np.ndarray):
        batched_embeddings = np.reshape(embeddings[:max_len], new_shape)
    else:
        raise ValueError("Embeddings should be either numpy array or torch tensor")
    end = time.time()
    print(
        f"Batched embeddings of shape {batched_embeddings.shape} created in {end - start:.2f} seconds")
    return batched_embeddings


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--type_inference", type=str, default="online",
                        help="Online/offline/batch inference")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache the model")
    parser.add_argument("--embedding_model", type=str,
                        default='sentence-transformers/all-mpnet-base-v2', help="Sentence transformer model to generate embeddings")
    parser.add_argument("--embedding_device", type=str, default='cpu',
                        help="Device to run the embedding model")
    parser.add_argument("--dummy_tags_path", type=str,
                        default='./dummy_tags.csv', help="Dummy tags path default local directory")
    return parser.parse_args()


if __name__ == "__main__":

    args = arg_parser()

    transformer = load_embedding_model(
        args.embedding_model, args.cache_dir, device=args.embedding_device)

    tags_embeddings = convert_texts_to_embeddings(tags, transformer, args.embedding_device)

    queries = load_dummy_tags(args.dummy_tags_path)

    query_embeddings = convert_texts_to_embeddings(queries, transformer, args.embedding_device)

    if args.type_inference == InferenceTypes.online.value:
        avg_time = 0.
        for query in query_embeddings:
            cpu_time = benchmark_search_online(
                query, tags_embeddings, device='cpu')
            avg_time += cpu_time
        print(
            f"CPU avg time for online inference: {avg_time/len(query_embeddings):.10f} seconds")

        # Benchmark on GPU (if available)
        if torch.cuda.is_available():
            avg_time = 0.
            for query in query_embeddings:
                gpu_time = benchmark_search_online(query, tags_embeddings, device='cuda')
                avg_time += gpu_time

            print(
                f"GPU avg time for online inference {avg_time/len(query_embeddings):.10f} seconds")
        else:
            print("GPU is not available.")
    elif args.type_inference == InferenceTypes.batch.value and args.batch_size:
        query_embeddings = create_batched_embeddings(query_embeddings, args.batch_size)

        # Benchmark on CPU
        cpu_time, one_batch_avg = benchmark_search_batch(
            query_embeddings, tags_embeddings, device='cpu')

        print(
            f"CPU Time on batch size 4: {cpu_time:.10f} seconds, one batch avg time: {one_batch_avg:.10f} seconds")

        # Benchmark on GPU (if available)
        if torch.cuda.is_available():
            gpu_time, one_batch_avg = benchmark_search_batch(
                query_embeddings, tags_embeddings, device='cuda')
            print(
                f"GPU Time on batch size 4: {gpu_time:.10f} seconds, one batch avg time: {one_batch_avg:.10f} seconds")
        else:
            print("GPU is not available.")

    elif args.type_inference == InferenceTypes.offline.value:
        # Benchmark on CPU
        cpu_time = benchmark_search_offline(query_embeddings, tags_embeddings, device='cpu')
        print(f"CPU Time on offline inference: {cpu_time:.10f} seconds")

        # Benchmark on GPU (if available)
        if torch.cuda.is_available():
            gpu_time = benchmark_search_offline(
                query_embeddings, tags_embeddings, device='cuda')
            print(f"GPU Time on offline inference: {gpu_time:.10f} seconds")
        else:
            print("GPU is not available.")
