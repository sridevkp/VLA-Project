"""Original JPEGs remain native portable image features with identical decoded pixels."""

import io

import numpy as np
import pytest
from PIL import Image

from vlarobot.vla.encoded_images import embed_recorded_jpegs
from vlarobot.vla.schema import CAMERA_KEY

pa = pytest.importorskip("pyarrow", reason="optional installed LeRobot image integration")
pq = pytest.importorskip("pyarrow.parquet")


def test_original_jpeg_embedding_preserves_rows_metadata_and_pixels(tmp_path):
    raw = tmp_path / "raw"
    images = raw / "frames/episode-00"
    images.mkdir(parents=True)
    pixels = np.random.default_rng(0).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(images / "000000.jpg", quality=85)
    original = (images / "000000.jpg").read_bytes()
    decoded = np.array(Image.open(io.BytesIO(original)))
    png = io.BytesIO()
    Image.fromarray(decoded).save(png, format="PNG")
    data = tmp_path / "native/data/chunk-000"
    data.mkdir(parents=True)
    table = pa.table(
        {
            "episode_index": [0],
            "frame_index": [0],
            CAMERA_KEY: [{"bytes": png.getvalue(), "path": "frame-000000.png"}],
            "action": [[1.0, 2.0]],
        }
    ).replace_schema_metadata({b"proof": b"unchanged"})
    path = data / "file-000.parquet"
    pq.write_table(table, path)
    report = embed_recorded_jpegs(tmp_path / "native", [raw])
    loaded = pq.read_table(path)
    assert loaded.schema == table.schema
    assert loaded.schema.metadata == table.schema.metadata
    row = loaded[CAMERA_KEY].to_pylist()[0]
    assert row["bytes"] == original
    assert row["path"].endswith(".jpg")
    np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(row["bytes"]))), decoded)
    assert loaded["action"].to_pylist() == [[1.0, 2.0]]
    assert report["frames"] == 1 and report["image_reencoding_performed"] is False
