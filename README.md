# Semantic Search Engine

A vector similarity search engine that uses an Inverted File Index (IVF) to do approximate nearest-neighbor lookups over high-dimensional embeddings. It's built to handle databases in the 1M to 20M vector range, with everything stored on disk as memory-mapped binary files.


## Setup

You need Python >= 3.13.

**Using uv:**

```bash
uv sync
```

**Using pip:**

```bash
pip install -r requirements.txt
```

For generating embeddings, you'll also need `sentence-transformers` and `torch` (see the notebook in `data/`).

## Generating data

Open `data/Generate_Vector_Embeddings.ipynb` in Google Colab. It downloads a subset of OpenSubtitles (English), filters sentences, and produces 64-dimensional embeddings using the `minishlab/potion-base-2M` model. Output is a memory-mapped `.dat` file.

## Usage

```python
from vec_db import VecDB

db = VecDB(
    database_file_path="your_data.dat",
    index_file_path="your_index/",
    new_db=True,          # True = build index on first run
    db_size=1_000_000     # picks IVF params from params.py
)

# search
results = db.retrieve(query_vector, top_k=10)
print(results)  # list of row IDs
```

`new_db=True` triggers index building on init — this can take a while for large databases. After that, pass `new_db=False` and the index gets loaded from disk.

## Configuration

All the tuning knobs live in `params.py`:

| Parameter | Value | Notes |
|---|---|---|
| `DIMENSION` | 64 | Must match your embedding model |
| `KMEANS_ITER` | 24 | Max iterations for KMeans training |
| `ASSIGN_BATCH_SIZE` | 20480 | Batch size when assigning vectors to clusters |
| `SEARCH_BATCH_SIZE` | 8192 | Batch size during retrieval |

IVF parameters are set per database size:

| DB Size | `nlist` | `nprobe` | `assign_batch_size` |
|---|---|---|---|
| 1M | 4096 | 4 | 24576 |
| 10M | 12700 | 6 | 184320 |
| 20M | 18000 | 12 | 368640 |

## Evaluation

Run the local eval:

```bash
cd eval
python simple_eval.py
```

It creates a small test database, runs random queries, and compares VecDB results against brute-force cosine similarity. The score is 0 for a perfect match and goes negative when results are wrong or missing.

The full evaluation notebook (`eval/final_eval.ipynb`) is meant for Colab and tests at 1M, 10M, 15M, and 20M vector scales.

## On-disk index format

```
<index_dir>/
├── ivf_centroids.npy           centroid vectors (numpy format)
└── clusters/
    ├── ids_00000.bin           cluster 0 member IDs (uint32)
    ├── ids_00001.bin           cluster 1 member IDs
    └── ...
```

Each cluster file is a flat array of 32-bit unsigned integers — the row IDs of vectors assigned to that cluster.
