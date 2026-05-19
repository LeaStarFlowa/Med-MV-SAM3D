from __future__ import annotations

from collections import deque

import numpy as np


try:
    from scipy import ndimage as ndi  # type: ignore
except ImportError:

    class _SimpleNDI:
        @staticmethod
        def binary_erosion(array: np.ndarray, structure: np.ndarray | None = None, iterations: int = 1) -> np.ndarray:
            out = np.asarray(array) > 0
            structure = np.ones((3,) * out.ndim, dtype=bool) if structure is None else structure > 0
            center = tuple(s // 2 for s in structure.shape)
            offsets = np.argwhere(structure) - np.asarray(center)
            for _ in range(iterations):
                padded = np.pad(out, 1, mode="constant", constant_values=False)
                eroded = np.ones_like(out, dtype=bool)
                base = np.indices(out.shape).reshape(out.ndim, -1).T + 1
                for offset in offsets:
                    idx = tuple((base + offset).T)
                    eroded &= padded[idx].reshape(out.shape)
                out = eroded
            return out

        @staticmethod
        def binary_dilation(array: np.ndarray, structure: np.ndarray | None = None, iterations: int = 1) -> np.ndarray:
            out = np.asarray(array) > 0
            structure = np.ones((3,) * out.ndim, dtype=bool) if structure is None else structure > 0
            center = tuple(s // 2 for s in structure.shape)
            offsets = np.argwhere(structure) - np.asarray(center)
            for _ in range(iterations):
                padded = np.pad(out, 1, mode="constant", constant_values=False)
                dilated = np.zeros_like(out, dtype=bool)
                base = np.indices(out.shape).reshape(out.ndim, -1).T + 1
                for offset in offsets:
                    idx = tuple((base + offset).T)
                    dilated |= padded[idx].reshape(out.shape)
                out = dilated
            return out

        @staticmethod
        def binary_closing(array: np.ndarray, structure: np.ndarray | None = None, iterations: int = 1) -> np.ndarray:
            dilated = _SimpleNDI.binary_dilation(array, structure=structure, iterations=iterations)
            return _SimpleNDI.binary_erosion(dilated, structure=structure, iterations=iterations)

        @staticmethod
        def binary_fill_holes(array: np.ndarray) -> np.ndarray:
            arr = np.asarray(array) > 0
            padded = np.pad(arr, 1, mode="constant", constant_values=False)
            visited = np.zeros_like(padded, dtype=bool)
            q: deque[tuple[int, ...]] = deque()
            for idx in np.argwhere(~padded):
                idx_tuple = tuple(int(v) for v in idx)
                if any(v == 0 or v == padded.shape[d] - 1 for d, v in enumerate(idx_tuple)):
                    visited[idx_tuple] = True
                    q.append(idx_tuple)
            while q:
                cur = q.popleft()
                for d in range(padded.ndim):
                    for step in (-1, 1):
                        nxt = list(cur)
                        nxt[d] += step
                        nxt_tuple = tuple(nxt)
                        if 0 <= nxt[d] < padded.shape[d] and not padded[nxt_tuple] and not visited[nxt_tuple]:
                            visited[nxt_tuple] = True
                            q.append(nxt_tuple)
            holes = ~padded & ~visited
            filled = padded | holes
            slices = tuple(slice(1, -1) for _ in range(arr.ndim))
            return filled[slices]

        @staticmethod
        def distance_transform_edt(array: np.ndarray) -> np.ndarray:
            arr = np.asarray(array) > 0
            coords = np.argwhere(arr)
            boundary = np.argwhere(arr & ~_SimpleNDI.binary_erosion(arr))
            if len(coords) == 0 or len(boundary) == 0:
                return np.zeros(arr.shape, dtype=np.float32)
            dist = np.zeros(arr.shape, dtype=np.float32)
            for chunk_start in range(0, len(coords), 2048):
                chunk = coords[chunk_start : chunk_start + 2048]
                d = np.linalg.norm(chunk[:, None, :] - boundary[None, :, :], axis=-1).min(axis=1)
                dist[tuple(chunk.T)] = d
            return dist

        @staticmethod
        def label(array: np.ndarray) -> tuple[np.ndarray, int]:
            arr = np.asarray(array) > 0
            labels = np.zeros(arr.shape, dtype=np.int32)
            label_id = 0
            for start in np.argwhere(arr):
                start_tuple = tuple(int(v) for v in start)
                if labels[start_tuple] != 0:
                    continue
                label_id += 1
                q: deque[tuple[int, ...]] = deque([start_tuple])
                labels[start_tuple] = label_id
                while q:
                    cur = q.popleft()
                    for d in range(arr.ndim):
                        for step in (-1, 1):
                            nxt = list(cur)
                            nxt[d] += step
                            if 0 <= nxt[d] < arr.shape[d]:
                                nxt_tuple = tuple(nxt)
                                if arr[nxt_tuple] and labels[nxt_tuple] == 0:
                                    labels[nxt_tuple] = label_id
                                    q.append(nxt_tuple)
            return labels, label_id

        @staticmethod
        def sum(input_array: np.ndarray, labels: np.ndarray, index: np.ndarray) -> np.ndarray:
            return np.asarray([input_array[labels == idx].sum() for idx in index], dtype=np.float32)

        @staticmethod
        def generate_binary_structure(rank: int, connectivity: int) -> np.ndarray:
            del connectivity
            return np.ones((3,) * rank, dtype=bool)

        @staticmethod
        def gaussian_filter(array: np.ndarray, sigma: float) -> np.ndarray:
            del sigma
            return np.asarray(array)

    ndi = _SimpleNDI()


def nearest_neighbor_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 3)
    if len(a) == 0 or len(b) == 0:
        return np.full((len(a),), np.inf, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        distances, _ = cKDTree(b).query(a, k=1)
        return distances.astype(np.float32)
    except ImportError:
        out = np.empty(len(a), dtype=np.float32)
        for start in range(0, len(a), 2048):
            chunk = a[start : start + 2048]
            out[start : start + len(chunk)] = np.linalg.norm(chunk[:, None, :] - b[None, :, :], axis=-1).min(axis=1)
        return out

