"""Read a sample of a LAS file without downloading it: the header plus evenly spaced chunks
of points, fetched by byte range (works on local files and on HTTP servers that honour
Range requests, such as Hugging Face).

Points in a LiDAR file are stored in flight order, so a chunk is a short stretch of one
flight strip at full density; many chunks spread over the file cover the whole tile.
"""

import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import laspy
import numpy as np


class LocalSource:
    def __init__(self, path):
        self.path = path

    def get(self, start: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(start)
            return f.read(length)


class HttpSource:
    """Byte ranges over HTTP. The redirect (to a signed CDN URL) is resolved once."""

    def __init__(self, url: str, timeout: float = 60.0):
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.url = r.geturl()
        self.timeout = timeout
        self.bytes_read = 0

    def get(self, start: int, length: int) -> bytes:
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{start + length - 1}"}
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                if len(data) != length:
                    raise OSError(f"short read {len(data)} of {length}")
                self.bytes_read += len(data)
                return data
            except OSError:
                if attempt == 2:
                    raise
        raise AssertionError


def read_sample(
    src, fraction: float = 0.02, chunks: int = 2000, fields=("ground_truth",), workers: int = 8
):
    """(xyz (n, 3) float64, {field: values}, header) for ~`fraction` of the points."""
    header = laspy.LasHeader.read_from(io.BytesIO(src.get(0, 65536)))
    missing = [f for f in fields if f not in header.point_format.dimension_names]
    if missing:  # fail before fetching any points
        raise ValueError(f"no field {missing} in this file")
    rec = header.point_format.size
    total = header.point_count
    per = max(1, int(total * fraction / chunks))
    starts = np.linspace(0, max(total - per, 0), chunks).astype(np.int64)

    def fetch(i0):
        buf = src.get(header.offset_to_point_data + int(i0) * rec, per * rec)
        return laspy.PackedPointRecord.from_buffer(bytearray(buf), header.point_format, per)

    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(fetch, starts))
    xyz = np.column_stack(
        [
            np.concatenate([np.asarray(r[d], dtype=np.float64) for r in recs]) * header.scales[i]
            + header.offsets[i]
            for i, d in enumerate("XYZ")
        ]
    )
    out = {f: np.concatenate([np.asarray(r[f]) for r in recs]) for f in fields}
    return xyz, out, header
