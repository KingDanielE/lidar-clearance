import numpy as np

from clearance.io import write_las
from clearance.sample import LocalSource, read_sample


def test_sample_reads_the_right_points(tmp_path):
    rng = np.random.default_rng(0)
    n = 50_000
    pts = rng.uniform(0, 100, (n, 3)).astype(np.float32)
    labels = (np.arange(n) % 22).astype(np.int16)
    path = tmp_path / "t.las"
    write_las(path, pts, np.array([1000.0, 2000.0, 300.0]), np.zeros(n, np.uint8),
              extra={"ground_truth": labels})

    xyz, fields, header = read_sample(LocalSource(path), fraction=0.05, chunks=50, workers=4)

    per = int(n * 0.05 / 50)
    starts = np.linspace(0, n - per, 50).astype(np.int64)
    idx = np.concatenate([np.arange(s, s + per) for s in starts])
    assert header.point_count == n
    assert np.array_equal(fields["ground_truth"], labels[idx])
    expected = pts[idx].astype(np.float64) + np.array([1000.0, 2000.0, 300.0])
    assert np.allclose(xyz, expected, atol=2e-3)  # 1 mm LAS scale + float32 input
