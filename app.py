"""
Connect Social proposal renderer (flat layout for easy deploy).

POST /render { "html": "...", "images": { "cover": url, "close": url, "b1": url, "b3": url, ... } }
  -> application/pdf

csBot writes the copy (with its own Claude key) and sends finished HTML plus the raw
image URLs it picked. This service treats the images (B&W + fade, identical to the
approved pipeline) and renders the PDF with weasyprint. No API key needed here.
"""
import os, io, re, gc, json, shutil, tempfile, urllib.request, urllib.parse
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from fastapi import FastAPI, Response, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from weasyprint import HTML

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS = ["Archivo-700.ttf", "Archivo-800.ttf", "Hanken-400.ttf",
         "Hanken-500.ttf", "Hanken-700.ttf", "Hanken-800.ttf"]
FLAT = ["proposal.css", "cs-logo-white.png", "cs_logo3_720.png"] + FONTS

PEXELS_KEY = os.environ.get("PEXELS_API_KEY", "").strip()
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
ORIENT = {"cover": "portrait", "close": "portrait", "side": "portrait",
          "b1": "landscape", "b2": "landscape", "b3": "landscape"}

app = FastAPI(title="cs-proposal-renderer")


def pexels_url(phrase, orientation):
    """Resolve a search phrase to a high-res image URL via Pexels."""
    if not PEXELS_KEY:
        return None
    try:
        u = ("https://api.pexels.com/v1/search?query=%s&per_page=12&orientation=%s"
             % (urllib.parse.quote(phrase), orientation))
        req = urllib.request.Request(u, headers={"Authorization": PEXELS_KEY, "User-Agent": UA})
        d = json.loads(urllib.request.urlopen(req, timeout=20).read())
        photos = d.get("photos") or []
        if photos:
            s = photos[0].get("src", {})
            return s.get("large2x") or s.get("original") or s.get("large")
    except Exception as e:
        print("pexels error:", e)
    return None


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=25).read()


def _resolve_to_image(url: str) -> str:
    """If url is a direct image, return it. If it's a web page, return its
    og:image / twitter:image / image_src (so 'pull a cover from his site' works)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        resp = urllib.request.urlopen(req, timeout=25)
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if ctype.startswith("image/"):
            return url
        html = resp.read(800000).decode("utf-8", "ignore")
        pats = [
            r'<meta[^>]+property=["\']og:image(?::secure_url|:url)?["\'][^>]+content=["\']([^"\']+)',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
            r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)',
            r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)',
        ]
        for p in pats:
            m = re.search(p, html, re.I)
            if m:
                return urllib.parse.urljoin(url, m.group(1).strip())
    except Exception as e:
        print("resolve image error:", e)
    return url  # fall back; _fetch will try it directly


def _fill(im, w, h):
    iw, ih = im.size
    s = max(w / iw, h / ih)
    im = im.resize((max(int(iw * s), w), max(int(ih * s), h)), Image.LANCZOS)
    iw, ih = im.size
    l = (iw - w) // 2; t = (ih - h) // 2
    return im.crop((l, t, l + w, t + h))


def _mono(raw, c=1.15, b=0.95):
    src = Image.open(io.BytesIO(raw))
    src.thumbnail((2200, 2200), Image.LANCZOS)  # cap decode size to bound memory
    im = ImageOps.grayscale(src)
    im = ImageEnhance.Contrast(im).enhance(c)
    return ImageEnhance.Brightness(im).enhance(b)


def _grid(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx / w, yy / h


def _save_jpg(arr, path, q=82):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path, "JPEG", quality=q, optimize=True)


def treat_cover(raw, path, W=1500, H=1940):
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


def treat_close(raw, path, W=1500, H=1940):
    im = _fill(_mono(raw, 1.14, 0.95), W, H).convert("RGB")
    a = np.asarray(im).astype(np.float32); x, y = _grid(W, H)
    rev = np.clip((y * 0.92 + 0.08) ** 1.1, 0, 1)[..., None]
    a = a * (0.12 + 0.62 * rev)
    _save_jpg(a, path)


def treat_side(raw, path, W=880, H=2610):
    im = _fill(_mono(raw, 1.05, 1.18), W, H).convert("L")
    g = np.asarray(im).astype(np.float32)
    g = 255 - (255 - g) * 0.55
    x, y = _grid(W, H)
    horiz = np.clip(x, 0, 1) ** 1.5
    vt = np.clip(np.minimum(y / 0.15, (1 - y) / 0.15), 0, 1)
    alpha = np.clip(horiz * vt * 0.62, 0, 1)
    out = g * alpha + 255.0 * (1 - alpha)
    _save_jpg(np.stack([out] * 3, -1), path)


def treat_bottom(raw, path, W=1500, H=600):
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


@app.get("/peek")
def peek(url: str = Query("")):
    """Fetch a page and return a short niche-signal summary (title, description, h1s)."""
    if not str(url).startswith("http"):
        return {"summary": ""}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        html = urllib.request.urlopen(req, timeout=15).read(600000).decode("utf-8", "ignore")

        def grab(pat):
            m = re.search(pat, html, re.I)
            return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

        title = grab(r"<title[^>]*>([^<]+)</title>")
        ogt = grab(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)')
        desc = grab(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)')
        ogd = grab(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)')
        h1s = [re.sub(r"<[^>]+>", "", h).strip() for h in re.findall(r"<h1[^>]*>(.*?)</h1>", html, re.I | re.S)[:3]]
        parts = [p for p in [title, ogt, desc, ogd, " / ".join([h for h in h1s if h])] if p]
        summary = " | ".join(dict.fromkeys(parts))[:600]
        return {"summary": summary}
    except Exception as e:
        print("peek error:", e)
        return {"summary": ""}


def _render_pdf(html: str, images: dict) -> bytes:
    """Resolve + treat images, then render the HTML to PDF bytes."""
    tmp = tempfile.mkdtemp(prefix="csr_")
    try:
        for f in FLAT:
            shutil.copy(os.path.join(APP_DIR, f), os.path.join(tmp, f))
        imgdir = os.path.join(tmp, "img"); os.makedirs(imgdir)
        for key, val in (images or {}).items():
            if key not in TREATERS or not val:
                continue
            fname, fn = TREATERS[key]
            if str(val).startswith("http"):
                url = _resolve_to_image(val)  # direct image, or pull og:image off a site link
            else:
                url = pexels_url(val, ORIENT.get(key, "landscape"))
            if not url:
                print(f"no image resolved for '{key}' ({val})")
                continue
            try:
                fn(_fetch(url), os.path.join(imgdir, fname))
            except Exception as e:
                print(f"image '{key}' failed: {e}")
            finally:
                gc.collect()  # release big numpy/PIL buffers between images
        return HTML(string=html, base_url=tmp).write_pdf()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/render")
def render(req: RenderReq):
    try:
        return Response(content=_render_pdf(req.html, req.images or {}), media_type="application/pdf")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------- async audit: long AI generation off the bot's 60s limit ----------------
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
SELF_URL = os.environ.get("SELF_URL", "https://csbot-renderer.onrender.com").strip()
AUDIT_RESULTS = {}  # thread_ts -> pdf bytes (fetched once by the bot, then dropped)


class AuditReq(BaseModel):
    prompt: str
    documents: list = []  # [{channel, name, b64}]
    imagery: str = ""
    images: dict = {}  # caller-supplied image phrases (edits keep the original cover/close)
    channel_id: str
    thread_ts: str
    webhook_url: str
    title: str = "Audit"
    filename: str = "audit.pdf"


def _anthropic_audit(prompt, documents):
    content = [{"type": "text", "text": prompt}]
    for d in documents or []:
        content.append({"type": "text", "text": f"Attached PDF for channel {d.get('channel', '')}, file {d.get('name', '')}:"})
        content.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": d.get("b64", "")}})
    body = json.dumps({"model": "claude-sonnet-4-6", "max_tokens": 16000, "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={"content-type": "application/json", "x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"})
    data = json.loads(urllib.request.urlopen(req, timeout=300).read())
    if data.get("error"):
        raise RuntimeError("anthropic: " + json.dumps(data["error"])[:300])
    text = (data.get("content") or [{}])[0].get("text", "")
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise RuntimeError("no JSON from AI")
    return json.loads(m.group(0))


def _post_json(url, payload):
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"content-type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        print("callback post error:", e)


def do_audit(req: "AuditReq"):
    try:
        obj = _anthropic_audit(req.prompt, req.documents)
        html = obj.get("html", "")
        images = obj.get("images", {}) or {}
        if req.images:
            images = dict(req.images)  # caller override (edits keep the original cover/close)
        if req.imagery and str(req.imagery).strip().lower().startswith("http"):
            images["cover"] = str(req.imagery).strip()
        pdf = _render_pdf(html, images)
        AUDIT_RESULTS[req.thread_ts] = {"pdf": pdf, "html": html, "images": json.dumps(images)}
        _post_json(req.webhook_url, {
            "channel_id": req.channel_id, "thread_ts": req.thread_ts,
            "pdf_url": f"{SELF_URL}/audit_pdf/{req.thread_ts}",
            "meta_url": f"{SELF_URL}/audit_meta/{req.thread_ts}",
            "title": req.title, "filename": req.filename, "error": "",
        })
    except Exception as e:
        print("do_audit error:", e)
        _post_json(req.webhook_url, {
            "channel_id": req.channel_id, "thread_ts": req.thread_ts,
            "pdf_url": "", "meta_url": "",
            "title": req.title, "filename": req.filename, "error": str(e)[:250],
        })


@app.post("/audit")
def audit(req: AuditReq, bg: BackgroundTasks):
    if not ANTHROPIC_KEY:
        return JSONResponse(status_code=500, content={"error": "ANTHROPIC_API_KEY not set on renderer"})
    bg.add_task(do_audit, req)
    return {"ok": True, "queued": True}


@app.get("/audit_pdf/{key}")
def audit_pdf(key: str):
    item = AUDIT_RESULTS.get(key)
    if not item:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return Response(content=item["pdf"], media_type="application/pdf")


@app.get("/audit_meta/{key}")
def audit_meta(key: str):
    item = AUDIT_RESULTS.pop(key, None)  # last fetch, clean up the entry
    if not item:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return {"html": item["html"], "images": item["images"]}
