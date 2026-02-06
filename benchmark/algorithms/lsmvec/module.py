import glob
import numpy as np
import os
import shutil
import psutil
from benchmark.algorithms.base import BaseANN
import lsm_vec

def _read_fbin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n = np.fromfile(f, dtype=np.uint32, count=1)[0]
        d = np.fromfile(f, dtype=np.uint32, count=1)[0]
        x = np.fromfile(f, dtype=np.float32, count=int(n) * int(d))
    return x.reshape(int(n), int(d))

def _guess_base_file(dataset_name: str) -> str:
    data_root = os.environ.get("DATA_DIR", "data")
    cands = glob.glob(os.path.join(data_root, "**", "data_*"), recursive=True)
    norm = dataset_name.replace("-", "").lower()
    cands_pref = [p for p in cands if norm in p.replace("-", "").lower()]
    if cands_pref:
        return sorted(cands_pref)[0]
    if cands:
        return sorted(cands)[0]
    raise FileNotFoundError(f"Cannot find base vectors for dataset={dataset_name} under {data_root}")

class LsmVec(BaseANN):
    def __init__(self, metric, dimOrParams=None, paramsMaybe=None):
        self.name = "LSMVec"
        if isinstance(metric, (list, tuple)) and dimOrParams is None and paramsMaybe is None:
            argsList = list(metric)
            if len(argsList) == 0:
                raise ValueError("Empty constructor args")
            if len(argsList) == 1:
                metric = argsList[0]
                dimOrParams = None
                paramsMaybe = None
            elif len(argsList) == 2:
                metric = argsList[0]
                dimOrParams = argsList[1]
                paramsMaybe = None
            else:
                metric = argsList[0]
                dimOrParams = argsList[1]
                paramsMaybe = argsList[2]

        self.metric = metric

        if isinstance(dimOrParams, dict) and paramsMaybe is None:
            self.dim = None
            self.params = dict(dimOrParams)
        else:
            self.dim = None if dimOrParams is None else int(dimOrParams)
            self.params = {} if paramsMaybe is None else dict(paramsMaybe)

        self.ef_search = int(self.params.get("ef_search", 64))
        self.db = None
        self.db_dir = None
        self.index = None

    def set_query_arguments(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            kwargs = args[0]
            args = ()

        if "ef_search" in kwargs:
            self.ef_search = int(kwargs["ef_search"])
            return

        if len(args) == 1:
            self.ef_search = int(args[0])
            return

        if len(args) == 0:
            self.ef_search = int(self.ef_search)
            return

        raise TypeError(f"Unexpected query args: args={args}, kwargs={kwargs}")


    def _map_metric(self):
        m = str(self.metric).lower()
        if m in ("l2", "euclidean", "sqeuclidean"):
            return lsm_vec.DistanceMetric.L2
        if m in ("cosine", "angular"):
            return lsm_vec.DistanceMetric.Cosine
        if m in ("ip", "inner_product", "mips", "dot"):
            return lsm_vec.DistanceMetric.InnerProduct
        raise ValueError(f"Unsupported metric: {self.metric}")

    def _get_index_root(self) -> str:
        return os.environ.get("INDEX_DIR", os.environ.get("ANN_INDEX_DIR", "/tmp/ann_index"))

    def _apply_params_to_options(self, options):
        alias = {"M": "m", "Mmax": "m_max", "Ml": "m_level", "efc": "ef_construction"}
        for k, v in self.params.items():
            if k == "ef_search":
                continue
            optKey = alias.get(k, k)
            if hasattr(options, optKey):
                setattr(options, optKey, v)

    def _make_search_options(self, k: int):
        # LSM Vec repo example uses SearchOptions().k then db.search_knn(query, opts)
        opts = lsm_vec.SearchOptions()
        if hasattr(opts, "k"):
            opts.k = int(k)

        # Best effort mapping for ef_search, different bindings may expose different names
        for name in ["ef_search", "ef", "efSearch", "efsearch", "efs"]:
            if hasattr(opts, name):
                setattr(opts, name, int(self.ef_search))
                break

        return opts


    def fit(self, dataset):
        if isinstance(dataset, str):
            base_path = _guess_base_file(dataset)
            X = _read_fbin(base_path)
        else:
            X = np.asarray(dataset, dtype=np.float32, order="C")

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")

        if self.dim is None:
            self.dim = int(X.shape[1])
        if X.shape[1] != self.dim:
            raise ValueError(f"X shape must be (N, {self.dim}), got {X.shape}")

        index_root = self._get_index_root()
        self.db_dir = os.path.join(index_root, "lsmvec_db")

        if os.path.exists(self.db_dir):
            shutil.rmtree(self.db_dir)
        os.makedirs(self.db_dir, exist_ok=True)

        options = lsm_vec.LSMVecDBOptions()
        options.dim = self.dim
        options.metric = self._map_metric()
        self._apply_params_to_options(options)

        if hasattr(options, "vector_file_path"):
            options.vector_file_path = os.path.join(self.db_dir, "vectors.bin")
        if hasattr(options, "log_file_path"):
            options.log_file_path = os.path.join(self.db_dir, "lsmvec.log")

        self.db = lsm_vec.LSMVecDB.open(self.db_dir, options)

        # Insert vectors
        for i in range(X.shape[0]):
            self.db.insert(int(i), np.asarray(X[i], dtype=np.float32, order="C"))

        # Flush like you already did
        if hasattr(self.db, "flush"):
            self.db.flush()
        if hasattr(self.db, "sync"):
            self.db.sync()
        if hasattr(self.db, "finalize"):
            self.db.finalize()

        # Set query handle: LSM Vec binding exposes search_knn on db
        if hasattr(self.db, "search_knn"):
            self.index = self.db
        else:
            cands = [m for m in dir(self.db) if ("search" in m.lower() or "knn" in m.lower() or "query" in m.lower())]
            raise RuntimeError(f"Cannot find search interface on LSMVecDB, candidates: {cands}")


    def _queryOne(self, v, k):
        v = np.asarray(v, dtype=np.float32, order="C").reshape(-1)
        if v.shape != (self.dim,):
            raise ValueError(f"Query must be shape ({self.dim},), got {v.shape}")

        if self.index is None:
            raise RuntimeError("self.index is None, fit did not set it")

        if not hasattr(self.index, "search_knn"):
            cands = [m for m in dir(self.index) if ("search" in m.lower() or "knn" in m.lower() or "query" in m.lower())]
            raise RuntimeError(f"No supported knn method on index, candidates: {cands}")

        opts = self._make_search_options(k)

        # Repo example passes Python list, so be conservative here
        results = self.index.search_knn(v.tolist(), opts)  # :contentReference[oaicite:1]{index=1}

        # results is a list of objects with fields like .id and .distance
        ids = np.full((k,), -1, dtype=np.int64)
        dists = np.full((k,), np.inf, dtype=np.float32)

        n = min(k, len(results))
        for i in range(n):
            r = results[i]
            rid = getattr(r, "id", None)
            rdist = getattr(r, "distance", None)
            if rid is not None:
                ids[i] = int(rid)
            if rdist is not None:
                dists[i] = float(rdist)

        return ids, dists



    def query(self, X, k):
        X = np.asarray(X)

        if X.ndim == 1:
            ids, dists = self._queryOne(X, k)
            self.res = ids.reshape(1, -1)
            self.query_dists = dists.reshape(1, -1)
            return

        if X.ndim == 2:
            nq, dim = X.shape
            if dim != self.dim:
                raise ValueError(f"Query dim mismatch, expected {self.dim}, got {dim}")

            allIds = np.empty((nq, k), dtype=np.int64)
            allDists = np.empty((nq, k), dtype=np.float32)
            for i in range(nq):
                ids, dists = self._queryOne(X[i], k)
                allIds[i, :] = ids
                allDists[i, :] = dists

            self.res = allIds
            self.query_dists = allDists
            return

        raise ValueError(f"Unsupported query shape: {X.shape}")

    def get_memory_usage(self):
        return psutil.Process(os.getpid()).memory_info().rss

    def done(self):
        try:
            if self.db is not None and hasattr(self.db, "close"):
                self.db.close()
        finally:
            self.db = None

    def load_index(self, dataset):
        return False

    def save_index(self, dataset):
        return
