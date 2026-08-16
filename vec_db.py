####### Builtin Modules #######
from multiprocessing import heap
from typing import List, Annotated
import os
from abc import ABC, abstractmethod
import sys
import heapq
######## Third-Party Modules #######
import numpy as np
######## Project Modules #######
from ivf import IVF
from disk import DiskIndex
from params import *
######## Constants #######
ELEMENT_SIZE = np.dtype(np.float32).itemsize
########################################



class IDB(ABC):
    @abstractmethod
    def __init__(self, database_file_path, index_file_path, new_db, db_size) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        pass


class VecDB(IDB):
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.db_size = self._get_num_records() if db_size is None else db_size
        db_sizes = np.array(list(IVF_PARAMS.keys()))
        closest_index = np.argmin(np.abs(self.db_size - db_sizes))
        self.ivf_params = IVF_PARAMS.get(db_sizes[closest_index], IVF_PARAMS[20000000])
        if new_db:
            self._build_index()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)])-> None:
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)
            return np.array([])
        
    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], 
                top_k: int = 15) -> List[int]:

        index_dir = os.path.dirname(self.index_path)
        disk_manager = DiskIndex(index_dir)
        ivf = IVF(self.ivf_params["nlist"], KMEANS_ITER)
        ivf.centroids = disk_manager.load_centroids().astype(np.float32)
        
        # Step 1: normalize query
        q = np.asarray(query, dtype=np.float32).ravel()
        q_norm = np.linalg.norm(q) + 1e-13
        q = q / q_norm

        cent_norms = np.linalg.norm(ivf.centroids, axis=1,keepdims=True) + 1e-13
        centroids_normalized = ivf.centroids / cent_norms

        # Step 2: find top_nbprobe
        dots = centroids_normalized @ q
        probe_ids = np.argsort(-dots)[:self.ivf_params["nprobe"]].astype(np.uint32)
        
        # Step 3: get candidates
        candidates = []
        for cid in probe_ids:
            ids = disk_manager.load_cluster_ids(cid)

            for start in range(0, len(ids), SEARCH_BATCH_SIZE):
                end = min(len(ids), start + SEARCH_BATCH_SIZE)
                batch_ids = ids[start:end]
                vecs = np.empty((batch_ids.shape[0], DIMENSION), dtype=np.float32)
                for i, id in enumerate(batch_ids):
                    vecs[i] = self.get_one_row(int(id))
                vec_norms = np.linalg.norm(vecs, axis=1,keepdims=True) + 1e-13
                vecs_normed = vecs / vec_norms
                scores_batch = vecs_normed @ q
                for score, id in zip(scores_batch, batch_ids):
                    if len(candidates) < top_k:
                        heapq.heappush(candidates, (score, int(id)))
                    elif score > candidates[0][0]:
                        heapq.heapreplace(candidates, (score, int(id)))
        # Step 4: Get top-K candidates
        candidates.sort(reverse=True)
        return [id for (_, id) in candidates]


    def _build_index(self)-> None:
        print("Building index...")
        X = self.get_all_rows().astype(np.float32)
        index_dir = os.path.dirname(self.index_path)
        disk_manager = DiskIndex(index_dir)

        # Step 1: IVF training
        ivf = IVF(self.ivf_params["nlist"], KMEANS_ITER)
        sample_sz = ASSIGN_BATCH_SIZE * 16
        random_ids = np.random.choice(self.db_size, size=sample_sz, replace=False)
        random_chunk = X[random_ids]

        ivf.train(random_chunk)
        disk_manager.write_centroids(ivf.centroids)
        
        # Step 2: Assignment after normalization
        clusters = [[] for _ in range(ivf.centroids.shape[0])]
        for start in range(0, X.shape[0], ASSIGN_BATCH_SIZE):
            end = min(X.shape[0], start + ASSIGN_BATCH_SIZE)
            batch = X[start:end]
            batch_norm = batch / (np.linalg.norm(batch, axis=1, keepdims=True) + 1e-12) 
            assigned = ivf.assign_batch(batch_norm)
            for i, cid in enumerate(assigned):
                clusters[cid].append(start + i)
        max_cluster_size = np.argmax([len(c) for c in clusters])
        print(f"Max cluster size: {len(clusters[max_cluster_size])}")
        disk_manager.write(clusters)
        print("Index build completed")

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity