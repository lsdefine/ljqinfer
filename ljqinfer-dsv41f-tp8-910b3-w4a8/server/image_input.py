"""Bounded image intake. No network, filesystem paths or torch in the HTTP process."""
import base64
import binascii
import io
import time
import warnings
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps
from server.image_grid import plan_image_grid

IMAGE_ID = 129264
MAX_IMAGES = 8
MAX_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_VISION_TOKENS = 8192
CONFIG = SimpleNamespace(vision_patch_size=14, vision_downsample_ratio=3,
    vision_max_wh_ratio=None, vision_min_pixels=295936, vision_max_n_token=1024)


def decode_record(record):
    """Accept base64 only; reject SSRF/local-file sources explicitly."""
    if not isinstance(record, dict):
        raise ValueError('image must be an object')
    source = record.get('source', record)
    if not isinstance(source, dict):
        raise ValueError('image source must be an object')
    url = record.get('url') or source.get('url')
    if url is not None:
        if not isinstance(url, str) or not url.startswith('data:image/'):
            raise ValueError('only base64 image data URLs are supported; remote URLs and paths are disabled')
        header, sep, data = url.partition(',')
        if not sep or not header.endswith(';base64'):
            raise ValueError('image data URL must use base64')
    else:
        data = source.get('data')
    if not isinstance(data, str) or len(data) > (MAX_BYTES + 2) // 3 * 4:
        raise ValueError('image base64 missing or exceeds 16 MiB limit')
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('invalid image base64') from exc
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('empty image or image exceeds 16 MiB limit')
    return raw


def inspect_image(raw):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in ('PNG', 'JPEG', 'WEBP', 'BMP'):
                    raise ValueError('supported image formats: PNG, JPEG, WEBP, BMP')
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('animated images are not supported')
                w, h = image.size
                if w < 1 or h < 1 or w * h > MAX_PIXELS:
                    raise ValueError('image exceeds 40 million pixel limit')
                image.load()
                return w, h
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('invalid or unsafe image') from exc


def expand_images(ids, records):
    """Return expanded raw IDs plus a JSON-safe, bounded, verified image payload."""
    if not records:
        if IMAGE_ID in ids:
            raise ValueError('image placeholder without an image')
        return ids, None
    if len(records) > MAX_IMAGES or ids.count(IMAGE_ID) != len(records):
        raise ValueError('image count/placeholder mismatch or too many images (maximum 8)')
    begin = time.perf_counter()
    expanded, images, total_bytes, total_tokens = [], [], 0, 0
    iterator = iter(records)
    for token in ids:
        if token != IMAGE_ID:
            expanded.append(token)
            continue
        raw = decode_record(next(iterator))
        total_bytes += len(raw)
        if total_bytes > MAX_TOTAL_BYTES:
            raise ValueError('total image data exceeds 32 MiB')
        width, height = inspect_image(raw)
        nh, nw, bh, bw = plan_image_grid(width, height, CONFIG)
        count = nh * (nw + 1) + 2
        total_tokens += count
        if total_tokens > MAX_VISION_TOKENS:
            raise ValueError('total expanded image spans exceed 8192 tokens')
        images.append(dict(start=len(expanded), length=count, grid=[bh // 14, bw // 14],
                           llm_grid=[nh, nw], data=base64.b64encode(raw).decode('ascii')))
        expanded.extend([IMAGE_ID] * count)
    return expanded, dict(version=1, images=images, image_tokens=total_tokens,
                          preprocess_ms=(time.perf_counter()-begin)*1000)


def patchify(image):
    """Official RGB conversion, bicubic ImageOps.pad, [-1,1], CHW patch order.

    EXIF is intentionally NOT transposed: matches the released processor.
    """
    raw = decode_record(image)
    width, height = inspect_image(raw)
    nh, nw, bh, bw = plan_image_grid(width, height, CONFIG)
    if image['grid'] != [bh // 14, bw // 14] or image['llm_grid'] != [nh, nw]:
        raise ValueError('image grid metadata mismatch')
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert('RGB')
        if CONFIG.vision_max_wh_ratio is not None and width >= CONFIG.vision_max_wh_ratio * height:
            im = im.resize((bw, bh))
        else:
            im = ImageOps.pad(im, (bw, bh), color=(127, 127, 127))
        x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    return x.reshape(3, bh//14, 14, bw//14, 14).transpose(1, 3, 0, 2, 4).reshape(-1, 588).copy()


def validate_payload(ids, payload):
    if not isinstance(payload, dict) or payload.get('version') != 1:
        raise ValueError('unsupported image payload')
    images = payload.get('images')
    if not isinstance(images, list) or not 1 <= len(images) <= MAX_IMAGES:
        raise ValueError('invalid image count')
    end, total, raw_bytes = 0, 0, 0
    for im in images:
        if not isinstance(im, dict):
            raise ValueError('image must be an object')
        start, length = im.get('start'), im.get('length')
        if type(start) is not int or type(length) is not int or start < end or length < 4 or start+length > len(ids):
            raise ValueError('invalid image span')
        if any(t != IMAGE_ID for t in ids[start:start+length]):
            raise ValueError('image span raw token IDs mismatch')
        raw = decode_record(im)
        raw_bytes += len(raw)
        w, h = inspect_image(raw)
        nh, nw, bh, bw = plan_image_grid(w, h, CONFIG)
        if im.get('grid') != [bh//14, bw//14] or im.get('llm_grid') != [nh,nw] or length != nh*(nw+1)+2:
            raise ValueError('image span/grid mismatch')
        total += length
        end = start+length
    if total > MAX_VISION_TOKENS or raw_bytes > MAX_TOTAL_BYTES or ids.count(IMAGE_ID) != total:
        raise ValueError('image payload limits or uncovered image IDs')
    return payload
