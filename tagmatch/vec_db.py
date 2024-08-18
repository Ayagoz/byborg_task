import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (CollectionInfo, Distance,
                                       FieldCondition, Filter, MatchValue,
                                       PointStruct, VectorParams)
from sklearn.decomposition import PCA


def reduce_embeddings(embeddings, target_dim=100):
    pca = PCA(n_components=target_dim)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings


class Embedder:
    def __init__(self, model_name: str, cache_dir: str, target_dim: Optional[int] = None):
        self.embedding_model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.embedding_dim: int = list(self.embedding_model.embed("Test for dims"))[0].shape[0]
        self.target_dim = self.embedding_dim
        if target_dim is not None:
            self.target_dim = target_dim
            self.pca = PCA(n_components=self.target_dim)

    def embed(self, text: str) -> np.ndarray:
        # Generate the original high-dimensional embedding
        emb_generator = self.embedding_model.embed(text)
        original_embedding = list(emb_generator)[0].reshape(1, -1)

        # If the target dimension is the same as the original, return the original embedding
        if self.target_dim == self.embedding_dim:
            return original_embedding.flatten()
        # Otherwise, reduce the dimensionality of the embedding
        reduced_embedding = self.pca.transform(original_embedding)
        return reduced_embedding.flatten()

    def fit_pca(self, data: List[str]):
        # Fit the PCA model on the embeddings of the provided data
        embeddings = np.array([list(self.embedding_model.embed(text))[0] for text in data])
        self.pca.fit(embeddings)


class VecDB:
    _ALLOWED_DISTANCES = ("cosine", "euclidean")

    def __init__(self, host: str, port: int, collection: str, distance: str = "cosine", vector_size: Optional[int]=None):
        if distance not in self._ALLOWED_DISTANCES:
            raise ValueError(
                f"Distance {distance} not allowed. Allowed distances are {self._ALLOWED_DISTANCES}")

        self.distance = distance
        self.client = QdrantClient(host, port=int(port))
        self.collection = collection
        self.vector_size = vector_size

        if not self.collection_exists():
            self._create_collection()

    def collection_exists(self) -> bool:
        try:
            _ = self.client.get_collection(self.collection)
            return True
        except UnexpectedResponse:
            return False

    def _create_collection(self):
        dist = Distance.COSINE if self.distance == "cosine" else Distance.EUCLID
        self.client.create_collection(self.collection,
                                      vectors_config=VectorParams(size=self.vector_size, distance=dist))

    def remove_collection(self):
        self.client.delete_collection(self.collection)

    def find_closest(self, vector: np.ndarray, k: int) -> List[str]:
        vec_list: List[float] = vector.tolist()
        query_filter = None
        res = self.client.search(self.collection, query_vector=vec_list,
                                 limit=k, query_filter=query_filter)
        return res

    def store(self, vector: np.ndarray, payload: Dict[str, Any]) -> bool:
        vec_list: List[float] = vector.tolist()
        rnd_id = uuid.uuid4().int & (1 << 64) - 1
        try:
            self.client.upsert(self.collection, points=[PointStruct(
                id=rnd_id, vector=vec_list, payload=payload)])
            return True
        except Exception:
            return False

    def get_item_count(self) -> int:
        collection_info: CollectionInfo = self.client.get_collection(self.collection)
        nb_points = collection_info.points_count

        if nb_points is None:
            return -1

        return nb_points
