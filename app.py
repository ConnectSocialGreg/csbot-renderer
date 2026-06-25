"""
Connect Social proposal renderer (flat layout for easy deploy).

POST /render { "html": "...", "images": { "cover": url, "close": url, "b1": url, "b3": url, ... } }
  -> application/pdf

csBot writes the copy (with its own Claude key) and sends finished HTML plus the raw
image URLs it picked. This service treats the images (B&W + fade, identical to the
approved pipeline) and renders the PDF with weasyprint. No API key needed here.
"""
import os, io, shutil, tempfile, urllib.request
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from weasyprint import HTML

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS = ["Archivo-700.ttf", "Archivo-800.ttf", "Hanken-400.ttf",
         "Hanken-500.ttf", "Hanken-700.ttf", "Hanken-800.ttf"]
FLAT = ["proposal.css", "cs-logo-white.png", "cs_logo3_720.png"] + FONTS

app = FastAPI(title="cs-proposal-renderer")


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cs-renderer/1.0"})
    return urllib.request.urlopen(req, timeout=25).read()


def _fill(im, w, h):
    iw, ih = im.size
    s = max(w / iw, h / ih)
    im = im.resize((max(int(iw * s), w), max(int(ih * s), h)), Image.LANCZOS)
    iw, ih = im.size
    l = (iw - w) // 2; t = (ih - h) // 2
    return im.crop((l, t, l + w, t + h))


def _mono(raw, c=1.15, b=0.95):
    im = ImageOps.grayscale(Image.open(io.BytesIO(raw)))
    im = ImageEnhance.Contrast(im).enhance(c)
    return ImageEnhance.Brightness(im).enhance(b)


def _grid(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx / w, yy / h


def _save_jpg(arr, path, q=82):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path, "JPEG", quality=q, optimize=True)


def treat_cover(raw, path, W=1700, H=2200):
    im = _fill(_mono(raw, 1.16, 0.95), W, H).convert("RGB")
    a = np.asarray(im).astype(np.float32); x, y = _grid(W, H)
    rev = np.clip((x * 0.74 + (1 - y) * 0.6) ** 1.18, 0, 1)[..., None]
    a = a * (0.15 + 0.85 * rev)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); ang = np.deg2rad(-32)
    al = (xx - 1300) * np.cos(ang) + (yy - 560) * np.sin(ang)
    pe = -(xx - 1300) * np.sin(ang) + (yy - 560) * np.cos(ang)
    sh = np.exp(-al ** 2 / (2 * 520 ** 2)) * np.exp(-pe ** 2 / (2 * 64 ** 2))
    a = 255 - (255 - a) * (1 - 0.30 * sh[..., None])
    _save_jpg(a, path)


def treat_close(raw, path, W=1700, H=2200):
    im = _fill(_mono(raw, 1.14, 0.95), W, H).convert("RGB")
    a = np.asarray(im).astype(np.float32); x, y = _grid(W, H)
    rev = np.clip((y * 0.92 + 0.08) ** 1.1, 0, 1)[..., None]
    a = a * (0.12 + 0.62 * rev)
    _save_jpg(a, path)


def treat_side(raw, path, W=1000, H=2970):
    im = _fill(_mono(raw, 1.05, 1.18), W, H).convert("L")
    g = np.asarray(im).astype(np.float32)
    g = 255 - (255 - g) * 0.55
    x, y = _grid(W, H)
    horiz = np.clip(x, 0, 1) ** 1.5
    vt = np.clip(np.minimum(y / 0.15, (1 - y) / 0.15), 0, 1)
    alpha = np.clip(horiz * vt * 0.62, 0, 1)
    out = g * alpha + 255.0 * (1 - alpha)
    _save_jpg(np.stack([out] * 3, -1), path)


def treat_bottom(raw, path, W=1700, H=680):
    im = _fill(_mono(raw, 1.08, 1.16), W, H).convert("L")
    g = np.asarray(im).astype(np.float32)
    g = 255 - (255 - g) * 0.5
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); x = xx / W; y = yy / H
    vert = np.clip(y, 0, 1) ** 1.7
    hf = np.clip(np.minimum(x / 0.08, (1 - x) / 0.08), 0, 1)
    a = np.clip(vert * hf * 0.5, 0, 1)
    o = g * a + 255.0 * (1 - a)
    _save_jpg(np.stack([o] * 3, -1), path)


TREATERS = {
    "cover": ("cover_photo.jpg", treat_cover),
    "close": ("close_photo.jpg", treat_close),
    "side": ("side_bleed.jpg", treat_side),
    "b1": ("b1.jpg", treat_bottom),
    "b2": ("b2.jpg", treat_bottom),
    "b3": ("b3.jpg", treat_bottom),
}


class RenderReq(BaseModel):
    html: str
    images: dict = {}


@app.get("/health")
def health():
    return {"ok": True, "service": "cs-proposal-renderer"}


@app.post("/render")
def render(req: RenderReq):
    tmp = tempfile.mkdtemp(prefix="csr_")
    try:
        for f in FLAT:
            shutil.copy(os.path.join(APP_DIR, f), os.path.join(tmp, f))
        imgdir = os.path.join(tmp, "img"); os.makedirs(imgdir)

        for key, url in (req.images or {}).items():
            if key not in TREATERS or not url:
                continue
            fname, fn = TREATERS[key]
            try:
                fn(_fetch(url), os.path.join(imgdir, fname))
            except Exception as e:
                print(f"image '{key}' failed: {e}")

        pdf = HTML(string=req.html, base_url=tmp).write_pdf()
        return Response(content=pdf, media_type="application/pdf")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
