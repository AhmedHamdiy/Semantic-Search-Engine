# vec_db/index/disk.py  (or disk_index.py depending on your project)
import os
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor
from params import *

class DiskIndex:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir or "."
        self.cluster_dir = os.path.join(self.base_dir, "clusters")
        os.makedirs(self.cluster_dir, exist_ok=True)

    def _write_single_cluster(self, cid: int, cluster_members: list[int]):
        filepath = os.path.join(self.cluster_dir, f"ids_{cid:05d}.bin")
        arr = np.asarray(cluster_members, dtype=np.uint32)
        arr.tofile(filepath)

    def write( self, clusters: List[List[int]]) -> None:
        cluster_index = []
        for cid in range(len(clusters)):
            cluster_index.append(f"ids_{cid:05d}.bin")

        with ThreadPoolExecutor(max_workers=16) as ex:
            tasks = [
                ex.submit(self._write_single_cluster, cid, clusters[cid])
                for cid in range(len(clusters))
            ]
            for t in tasks:
                t.result()


    def load_centroids(self) -> np.ndarray:
        filepath = os.path.join(self.base_dir, CENTROIDS_PATH)
        return np.load(filepath)
    

    def load_cluster_ids(self, cid: int) -> np.ndarray:
        filepath = os.path.join(self.cluster_dir, f"ids_{cid:05d}.bin")
        if not os.path.exists(filepath):
            return np.empty(0, dtype=np.uint32)
        return np.fromfile(filepath, dtype=np.uint32)

    def write_centroids(self, centroids: np.ndarray) -> None:
        filepath = os.path.join(self.base_dir, CENTROIDS_PATH)
        np.save(filepath, centroids)