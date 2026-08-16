import numpy as np
from params import *
from sklearn.cluster import MiniBatchKMeans

class IVF:
    def __init__(self, nlist:int = 8192, max_iter: int  = KMEANS_ITER)-> None:
        self.nlist = nlist
        self.max_iter = max_iter
        self.centroids = None
    
    def train(self, chunk: np.ndarray) -> None:
        # TODO: Normalize the chunk -> Run k-means to compute centroids
        print("-----[Training IVF]-----")
        chunk = np.asarray(chunk, dtype=np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-13
        chunk_norm = chunk / norms

        KMeans_res = MiniBatchKMeans(n_clusters=min(self.nlist,chunk_norm.shape[0]), max_iter=self.max_iter,init="k-means++",random_state=123,n_init="auto" ,batch_size= min(max(1024, chunk.shape[0]//8),self.nlist))
        KMeans_res.fit(chunk_norm)
        centroids = KMeans_res.cluster_centers_.astype(np.float32)
        centroids_norm = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-13
        self.centroids = (centroids / centroids_norm).astype(np.float32)
        print("-----[IVF Training complete]------")
    
    def assign_batch(self, x_batch: np.ndarray) -> np.ndarray:
        xc = x_batch @ self.centroids.T
        return np.argmax(xc, axis=1).astype(np.int32)
    
    def assign_batch_normalized(self, x_batch: np.ndarray) -> np.ndarray:
        xc = x_batch @ self.centroids.T
        return np.argmax(xc, axis=1).astype(np.int32)

    def assign(self, x: np.ndarray) -> int:
        xc = x @ self.centroids.T
        return np.argmax(xc, axis=1).astype(np.int32)