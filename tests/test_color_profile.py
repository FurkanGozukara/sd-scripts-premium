"""Training images in sRGB (library.color_profile): load_image converts a photo with an embedded Display P3 profile and
leaves untagged images as they are."""

import struct

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PIL.ImageCms")

from library.train_util import load_image  # noqa: E402

# Display P3 colorants adapted to the D50 profile connection space, as in Apple's Display P3 profile
P3_COLORANTS = ((0.515102, 0.241196, -0.001053), (0.291965, 0.692245, 0.041885), (0.157153, 0.066561, 0.784073))
D50 = (0.964203, 1.0, 0.824905)


def _s15(value: float) -> bytes:
    return struct.pack(">i", int(round(value * 65536)))


def _xyz(values) -> bytes:
    return b"XYZ " + b"\0" * 4 + b"".join(_s15(v) for v in values)


def _curve(gamma: float) -> bytes:
    return b"curv" + b"\0" * 4 + struct.pack(">I", 1) + struct.pack(">H", int(round(gamma * 256))) + b"\0\0"


def _desc(text: str) -> bytes:
    ascii_text = text.encode("ascii") + b"\0"
    return b"desc" + b"\0" * 4 + struct.pack(">I", len(ascii_text)) + ascii_text + b"\0" * 8 + b"\0" * 3 + b"\0" * 67


def matrix_profile(colorants=P3_COLORANTS, gamma=2.2, name="Display P3 (test)") -> bytes:
    """A minimal ICC v2 RGB matrix / TRC display profile."""
    tags = [(b"desc", _desc(name)), (b"wtpt", _xyz(D50)), (b"rXYZ", _xyz(colorants[0])), (b"gXYZ", _xyz(colorants[1])),
            (b"bXYZ", _xyz(colorants[2])), (b"rTRC", _curve(gamma)), (b"gTRC", _curve(gamma)), (b"bTRC", _curve(gamma)),
            (b"cprt", b"text" + b"\0" * 4 + b"test\0")]
    offset = 128 + 4 + 12 * len(tags)
    table, body = b"", b""
    for signature, data in tags:
        while (offset + len(body)) % 4:
            body += b"\0"
        table += signature + struct.pack(">II", offset + len(body), len(data))
        body += data
    size = 128 + 4 + len(table) + len(body)
    header = (struct.pack(">I", size) + b"lcms" + struct.pack(">I", 0x02100000) + b"mntrRGB XYZ " + b"\0" * 12 + b"acsp"
              + b"\0" * 24 + struct.pack(">I", 0) + _s15(D50[0]) + _s15(D50[1]) + _s15(D50[2]) + b"\0" * 48)
    assert len(header) == 128
    return header + struct.pack(">I", len(tags)) + table + body


def test_load_image_converts_a_display_p3_photo(tmp_path):
    p3 = tmp_path / "p3.png"
    Image.new("RGB", (8, 8), (200, 100, 60)).save(p3, icc_profile=matrix_profile())
    pixel = load_image(str(p3))[0, 0].astype(int)
    assert pixel[0] > 200 and pixel[1] < 100 and pixel[2] < 60, pixel
    plain = tmp_path / "plain.png"
    Image.new("RGB", (8, 8), (200, 100, 60)).save(plain)
    assert tuple(load_image(str(plain))[0, 0]) == (200, 100, 60)
    rgba = tmp_path / "p3a.png"
    Image.new("RGBA", (8, 8), (200, 100, 60, 77)).save(rgba, icc_profile=matrix_profile())
    assert load_image(str(rgba), alpha=True)[0, 0, 3] == 77
