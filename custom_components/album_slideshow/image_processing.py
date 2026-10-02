from __future__ import annotations

import io
import logging

from PIL import Image, ImageColor, ImageFilter, ImageOps, ImageSequence

from .coordinator import MediaItem

_LOGGER = logging.getLogger(__name__)

# Re-export fill mode constants so callers can import from here.
FILL_COVER = "cover"
FILL_CONTAIN = "contain"
FILL_BLUR = "blur"

# Absolute pixel ceiling. A 20000x20000 JPEG decodes to ~1.2 GB of RGB; Pillow
# raises DecompressionBombError above MAX_IMAGE_PIXELS. We set this high enough
# that 4K+ sources still decode, but reject anything absurd to protect
# low-memory devices like the Home Assistant Green.
_MAX_IMAGE_PIXELS = 80_000_000  # ~8K x 10K
Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS


def open_image(
    data: bytes,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Open image bytes, apply EXIF orientation, normalise to RGB/RGBA.

    If ``target_size`` is given, uses PIL's ``draft`` mode so libjpeg decodes
    at a reduced scale. Big speed/memory win on low-power devices when the
    source is much larger than the output canvas.
    """
    img = Image.open(io.BytesIO(data))
    
    if getattr(img, "is_animated", False):
        # Do not transpose or force a single mode yet to preserve animation
        return img

    if target_size is not None and img.format == "JPEG":
        try:
            img.draft("RGB", target_size)
        except Exception:
            pass
    img = ImageOps.exif_transpose(img)
    # Force pixel data into memory; BytesIO must stay reachable until here.
    img.load()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def safe_close(img: Image.Image | None) -> None:
    """Close a PIL image without raising. No-op on None."""
    if img is None:
        return
    try:
        img.close()
    except Exception:
        pass


def is_portrait_img(img: Image.Image) -> bool:
    try:
        w, h = img.size
        return h >= w
    except Exception:
        return False


def is_portrait_item(item: MediaItem, img: Image.Image | None = None) -> bool:
    by_meta = _is_portrait_dims(item.width, item.height)
    if by_meta is not None:
        return by_meta
    if img is not None:
        return is_portrait_img(img)
    return False


def is_portrait_item_by_metadata(item: MediaItem) -> bool | None:
    """Return portrait/landscape from item metadata only, or None if unknown."""
    return _is_portrait_dims(item.width, item.height)


def resolve_output_size(
    req_w: int | None,
    req_h: int | None,
    ratio: str,
    max_short_edge: int | None = None,
) -> tuple[int, int]:
    ratio_w, ratio_h = _parse_aspect_ratio(ratio)
    target = ratio_w / ratio_h

    if req_w is None and req_h is None:
        if ratio_w >= ratio_h:
            width = 3840
            height = max(1, int(round(width / target)))
        else:
            height = 3840
            width = max(1, int(round(height * target)))
    elif req_w is None:
        height = max(1, int(req_h or 2160))
        width = max(1, int(round(height * target)))
    elif req_h is None:
        width = max(1, int(req_w or 3840))
        height = max(1, int(round(width / target)))
    else:
        req_w = max(1, int(req_w))
        req_h = max(1, int(req_h))
        if (req_w / req_h) >= target:
            height = req_h
            width = max(1, int(round(height * target)))
        else:
            width = req_w
            height = max(1, int(round(width / target)))

    if max_short_edge is not None:
        short = min(width, height)
        if short > max_short_edge:
            scale = max_short_edge / short
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))

    return (width, height)


def render_image(
        img: Image.Image,
        fill_mode: str,
        width: int,
        height: int,
        focus: tuple[float, float] | None = None,
) -> Image.Image:
    """Render img into a (width x height) canvas using the given fill mode."""
    if getattr(img, "is_animated", False):
        frames = []
        durations = []
        for frame in ImageSequence.Iterator(img):
            # Extract the delay for this specific frame, fallback to 100ms
            durations.append(frame.info.get("duration", 100))

            frame_rgba = frame.convert("RGBA")
            if fill_mode == FILL_CONTAIN:
                processed = _resize_contain(frame_rgba, width, height)
            elif fill_mode == FILL_BLUR:
                processed = _blur_fill(frame_rgba, width, height)
            else:
                processed = _resize_cover(frame_rgba, width, height, focus)
            frames.append(processed)

        res = frames[0]
        res.info = img.info.copy()
        res.info["append_images"] = frames[1:]
        res.info["duration"] = durations
        res.is_animated = True
        return res

    if fill_mode == FILL_CONTAIN:
        return _resize_contain(img, width, height)
    if fill_mode == FILL_BLUR:
        return _blur_fill(img, width, height)
    return _resize_cover(img, width, height, focus)

def pair_images(
    img1: Image.Image,
    img2: Image.Image,
    target_w: int,
    target_h: int,
    fill_mode: str,
    portrait_canvas: bool,
    divider: int,
    divider_fill: tuple[int, int, int] | tuple[int, int, int, int],
    transparent_divider: bool,
    focus1: tuple[float, float] | None = None,
    focus2: tuple[float, float] | None = None,
) -> Image.Image:
    canvas_mode = "RGBA" if transparent_divider else "RGB"
    
    if portrait_canvas:
        top_h = max(1, (target_h - divider) // 2)
        bottom_h = max(1, target_h - divider - top_h)
        img1_rendered = render_image(img1, fill_mode, target_w, top_h, focus1)
        img2_rendered = render_image(img2, fill_mode, target_w, bottom_h, focus2)
        boxes = [((0, 0), img1_rendered), ((0, top_h + divider), img2_rendered)]
    else:
        left_w = max(1, (target_w - divider) // 2)
        right_w = max(1, target_w - divider - left_w)
        img1_rendered = render_image(img1, fill_mode, left_w, target_h, focus1)
        img2_rendered = render_image(img2, fill_mode, right_w, target_h, focus2)
        boxes = [((0, 0), img1_rendered), ((left_w + divider, 0), img2_rendered)]

    is_animated = getattr(img1_rendered, "is_animated", False) or getattr(img2_rendered, "is_animated", False)

    if not is_animated:
        canvas = Image.new(canvas_mode, (target_w, target_h), divider_fill)
        for pos, img in boxes:
            canvas.paste(img.convert(canvas_mode), pos)
            safe_close(img)
        return canvas

    frames1 = getattr(img1_rendered, "_frames", [img1_rendered])
    frames2 = getattr(img2_rendered, "_frames", [img2_rendered])
    max_frames = max(len(frames1), len(frames2))

    paired_frames = []
    for i in range(max_frames):
        canvas = Image.new(canvas_mode, (target_w, target_h), divider_fill)
        f1 = frames1[i % len(frames1)]
        f2 = frames2[i % len(frames2)]
        if portrait_canvas:
            canvas.paste(f1.convert(canvas_mode), (0, 0))
            canvas.paste(f2.convert(canvas_mode), (0, top_h + divider))
        else:
            canvas.paste(f1.convert(canvas_mode), (0, 0))
            canvas.paste(f2.convert(canvas_mode), (left_w + divider, 0))
        paired_frames.append(canvas)

    safe_close(img1_rendered)
    safe_close(img2_rendered)

    res = paired_frames[0]
    src_anim = img1_rendered if getattr(img1_rendered, "is_animated", False) else img2_rendered
    res.info = src_anim.info.copy()
    res.format = src_anim.format
    res.is_animated = True
    res._frames = paired_frames
    return res


def render_pair_photo(
    data: bytes,
    photo_position: int,
    portrait_canvas: bool,
    divider: int,
    fill_mode: str,
) -> bytes:
    """Render one safe photo from an already composed pair without source I/O."""
    if photo_position not in (0, 1):
        raise ValueError("Invalid paired-photo position")
    with open_image(data) as paired:
        width, height = paired.size
        length = height if portrait_canvas else width
        first_length = max(1, (length - divider) // 2)
        start, end = (
            (0, first_length) if photo_position == 0
            else (first_length + divider, length)
        )
        if not 0 <= start < end <= length:
            raise ValueError("Paired photo is outside the rendered canvas")
        box = (0, start, width, end) if portrait_canvas else (start, 0, end, height)
        
        if getattr(paired, "is_animated", False):
            frames = []
            for frame in ImageSequence.Iterator(paired):
                frames.append(frame.crop(box))
                
            cropped_anim = frames[0]
            cropped_anim.info = paired.info.copy()
            cropped_anim.format = paired.format
            cropped_anim.is_animated = True
            cropped_anim._frames = frames
            
            with render_image(cropped_anim, fill_mode, width, height) as rendered:
                return encode_image(rendered)

        with paired.crop(box) as photo:
            with render_image(photo, fill_mode, width, height) as rendered:
                return encode_image(rendered)


def encode_image(img: Image.Image) -> bytes:
    """Encode a PIL image to a client-compatible JPEG, PNG, WEBP, or GIF."""
    out = io.BytesIO()

    if getattr(img, "is_animated", False):
        fmt = img.info.get("original_format", "GIF")
        if fmt not in ("GIF", "WEBP", "PNG"):
            fmt = "GIF"

        img.save(
            out,
            format=fmt,
            save_all=True,
            append_images=img.info.get("append_images", []),
            loop=img.info.get("loop", 0),
            duration=img.info.get("duration", 100),
            disposal=2  # Clears the background between frames
        )
        return out.getvalue()

    if "A" in img.getbands():
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()
    rgb = img if img.mode == "RGB" else img.convert("RGB")
    rgb.save(
        out,
        format="JPEG",
        quality=88,
        optimize=True,
        progressive=False,
        subsampling=2,
    )
    if rgb is not img:
        safe_close(rgb)
    return out.getvalue()

def parse_divider_color(color: str) -> tuple[tuple[int, int, int] | tuple[int, int, int, int], bool]:
    raw = (color or "").strip().lower()
    compact = raw.replace(" ", "")
    if compact in ("transparent", "transperant", "none", "clear", "rgba(0,0,0,0)"):
        return (0, 0, 0, 0), True
    try:
        return ImageColor.getrgb(color), False
    except Exception:
        return (255, 255, 255), False


# -- Private helpers ---------------------------------------------------------

def _is_portrait_dims(width: int | None, height: int | None) -> bool | None:
    if not width or not height:
        return None
    try:
        w, h = int(width), int(height)
        if w <= 0 or h <= 0:
            return None
        return h >= w
    except Exception:
        return None


def _parse_aspect_ratio(ratio: str) -> tuple[int, int]:
    try:
        left, right = ratio.split(":", maxsplit=1)
        w, h = int(left), int(right)
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        pass
    return (16, 9)


def _resize_cover(
    img: Image.Image,
    target_w: int,
    target_h: int,
    focus: tuple[float, float] | None = None,
) -> Image.Image:
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return img.resize((target_w, target_h))
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    focus_x, focus_y = (0.5, 0.5)
    if (
        isinstance(focus, tuple)
        and len(focus) == 2
        and all(isinstance(value, (int, float)) for value in focus)
    ):
        focus_x = max(0.0, min(1.0, float(focus[0])))
        focus_y = max(0.0, min(1.0, float(focus[1])))
    max_left = max(0, new_w - target_w)
    max_top = max(0, new_h - target_h)
    left = max(0, min(max_left, int(round(focus_x * new_w - target_w / 2))))
    top = max(0, min(max_top, int(round(focus_y * new_h - target_h / 2))))
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    if cropped is not resized:
        safe_close(resized)
    return cropped


def _resize_contain(img: Image.Image, target_w: int, target_h: int, bg=(0, 0, 0)) -> Image.Image:
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return img.resize((target_w, target_h))
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    canvas_mode = "RGBA" if img.mode == "RGBA" or len(bg) == 4 else "RGB"
    canvas = Image.new(canvas_mode, (target_w, target_h), bg)
    
    img_conv = resized if resized.mode == canvas_mode else resized.convert(canvas_mode)
    canvas.paste(img_conv, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    
    if img_conv is not resized:
        safe_close(img_conv)
    safe_close(resized)
    return canvas


def _blur_fill(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    bg = _resize_cover(img, target_w, target_h).filter(ImageFilter.GaussianBlur(radius=24))
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return bg
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    fg = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    rgb_fg = fg if fg.mode == bg.mode else fg.convert(bg.mode)
    if "A" in rgb_fg.getbands():
        bg.paste(rgb_fg, ((target_w - new_w) // 2, (target_h - new_h) // 2), rgb_fg)
    else:
        bg.paste(rgb_fg, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        
    if rgb_fg is not fg:
        safe_close(rgb_fg)
    safe_close(fg)
    return bg