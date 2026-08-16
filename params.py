KMEANS_ITER = 24
ASSIGN_BATCH_SIZE = 20480
SEARCH_BATCH_SIZE = 8192
DIMENSION = 64
# IVF
IVF_PARAMS = {
    1000000: {"nlist": 4096, "nprobe": 4, "assign_batch_size":24_576},
    10000000: {"nlist": 12700, "nprobe": 6,  "assign_batch_size":184_320},
    20000000: {"nlist": 18000, "nprobe": 12, "assign_batch_size":368_640},
}

# Filenames
CENTROIDS_PATH = "ivf_centroids.npy"
IDS_PATH = "ids.bin"