# A zone's own names, kept over its neighbours (Media/Textures/MapLetters/<locale>/<uiMapID>, Data/Letters.lua).
#
# Where two zones' own land overlaps (the Barrens and Mulgore round Camp Taurajo), the one drawn later
# covers the other's art there, its names too. The masks the zones are drawn through were made round the
# English names; another language's are often longer ("Acampamento Taurajo"), and their ends went under
# the neighbour. The lettering is found as what differs between the English art and that language's
# (the same drawing, only the names change), so nothing is guessed: each zone whose names a neighbour's
# own land covers gets a mask of those names (the words a neighbour reaches, whole), and Stitch.lua draws
# its overlay art again through it, over every zone's interior.
#
# usage:
#   python -P tools/map-preview/letters.py dump <db2 dir> <texture cache> <out dir>
#       every zone's overlay art as render.py draws it, for the language OPENMAP_LOCALE names (set
#       OPENMAP_WOW on an install of it; the English art comes from wago.tools)
#   python -P tools/map-preview/letters.py bake <enUS dump> <locale dump> <locale>
#       writes Media/Textures/MapLetters/<locale>/ and that language's part of Data/Letters.lua
import json, os, re, runpy, sys
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
import clients

DIFF = 0.15        # a pixel is lettering where the two languages' art differs by this much (0..1)
GROW = 3           # art px round the letters' strokes the mask takes in (their dark outline and halo)
WORD = 7           # art px apart that strokes are still one word (a covered word is kept whole)
COVER_MIN = 12     # lettering px a neighbour's own land must cover before a zone gets a mask
SOFT = 1.2         # the mask's edge softness (art px, a Gaussian)
SIZE = 256         # the masks' size (px)


def dump(db2, cache, out):
    os.makedirs(out, exist_ok=True)
    os.environ["OPENMAP_SIZE"] = "16x16"
    sys.argv = [os.path.join(HERE, "render.py"), db2, cache, os.path.join(out, "_render"), "none=0.5,0.5,0.01"]
    g = runpy.run_path(sys.argv[0], run_name="letters")
    meta = {}
    for m in g["zones"]:
        if not g["reveal"].get(m["art"]): continue
        o = g["over"](m)
        np.save(os.path.join(out, "%d.npy" % m["id"]), (np.clip(o, 0, 1) * 255).astype(np.uint8))
        meta[m["id"]] = {"name": m["name"], "rect": m["rect"]}
        print(m["id"], m["name"], flush=True)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)


def own_land(a):
    """a zone's own land in its art px (render.py's own_land): its overlay pieces, closed and filled"""
    return ndimage.binary_fill_holes(ndimage.binary_closing(a > 127, np.ones((3, 3)), iterations=6))


def on_art(src, src_rect, dst_rect):
    """an image over one zone's art (1002 x 668) resampled onto another's (nearest)"""
    sl, sr, st, sb = src_rect
    dl, dr, dt, db = dst_rect
    xs = dl + (np.arange(1002) + 0.5) / 1002 * (dr - dl)
    ys = dt + (np.arange(668) + 0.5) / 668 * (db - dt)
    sx = np.floor((xs - sl) / (sr - sl) * 1002).astype(int)
    sy = np.floor((ys - st) / (sb - st) * 668).astype(int)
    okx, oky = (sx >= 0) & (sx < 1002), (sy >= 0) & (sy < 668)
    out = np.zeros((668, 1002), src.dtype)
    if not okx.any() or not oky.any(): return out
    out[np.ix_(oky, okx)] = src[np.ix_(sy[oky], sx[okx])]
    return out


def bake(en_dir, loc_dir, locale):
    meta = json.load(open(os.path.join(loc_dir, "meta.json")))
    texdir = os.path.join(clients.TEX, "MapLetters", locale)
    os.makedirs(texdir, exist_ok=True)
    for f in os.listdir(texdir):
        if f.endswith(".tga"): os.remove(os.path.join(texdir, f))
    arts = {int(k): np.load(os.path.join(loc_dir, "%s.npy" % k)) for k in meta}
    lands = {k: own_land(a[..., 3]) for k, a in arts.items()}
    rects = {int(k): tuple(v["rect"]) for k, v in meta.items()}
    out = {}
    for k in sorted(arts):
        p = os.path.join(en_dir, "%d.npy" % k)
        if not os.path.exists(p): continue
        a, e = arts[k].astype(np.float32) / 255, np.load(p).astype(np.float32) / 255
        diff = np.abs(a[..., :3] - e[..., :3]).max(-1) * np.maximum(a[..., 3], e[..., 3])
        ink = (diff > DIFF) & (a[..., 3] > 0.5) & (e[..., 3] > 0.5)   # (not the English art's banner holes)
        if ink.sum() < COVER_MIN: continue
        # the other zones' own land over this one's art
        others = np.zeros((668, 1002), bool)
        l, r, t, b = rects[k]
        for n, land in lands.items():
            if n == k: continue
            nl, nr, nt, nb = rects[n]
            if nr <= l or nl >= r or nb <= t or nt >= b: continue
            others |= on_art(land, rects[n], rects[k])
        if (ink & others).sum() < COVER_MIN: continue
        # the words a neighbour reaches, whole
        words, count = ndimage.label(ndimage.binary_dilation(ink, iterations=WORD))
        hit = set(np.unique(words[ink & others])) - {0}
        keep = np.isin(words, list(hit)) & ink
        if not keep.any(): continue
        grown = ndimage.binary_dilation(keep, iterations=GROW) & (a[..., 3] > 0.5)
        soft = ndimage.gaussian_filter(grown.astype(np.float32), SOFT)
        boxes = []
        for sl_ in ndimage.find_objects(np.where(np.isin(words, list(hit)), words, 0)):
            if sl_ is None: continue
            y0, y1, x0, x1 = sl_[0].start, sl_[0].stop, sl_[1].start, sl_[1].stop
            boxes.append((max(0, x0 - GROW - 2), max(0, y0 - GROW - 2), min(1002, x1 + GROW + 2), min(668, y1 + GROW + 2)))
        img = Image.fromarray((np.clip(soft, 0, 1) * 255).astype(np.uint8), "L").resize((SIZE, SIZE), Image.LANCZOS)
        rgba = Image.merge("RGBA", (Image.new("L", img.size, 255),) * 3 + (img,))
        rgba.save(os.path.join(texdir, "%d.tga" % k))
        out[k] = boxes
        print(k, meta[str(k)]["name"], len(boxes), "words", flush=True)
    write_data(locale, out)


def write_data(locale, zones):
    path = os.path.join(clients.DATA, "Letters.lua")
    have = {}
    if os.path.exists(path):
        for loc, body in re.findall(r"^ns\.MapLetters\.(\w+) = \{\n(.*?)^\}", open(path, encoding="utf-8").read(), re.M | re.S):
            have[loc] = body
    lines = []
    for k in sorted(zones):
        boxes = ", ".join("{ %d, %d, %d, %d }" % b for b in zones[k])
        lines.append("\t[%d] = { %s },\n" % (k, boxes))
    have[locale] = "".join(lines)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("-- A zone's own names, kept over its neighbours: [language][uiMapID] = { { x0, y0, x1, y1 }, ... }, the\n"
                "-- words a neighbour's own land would cover, in the zone's art px; Stitch.lua draws the zone's overlay art\n"
                "-- again through Media/Textures/MapLetters/<language>/<uiMapID>, over every zone's interior.\n"
                "-- Generated by tools/map-preview/letters.py; regenerate rather than edit.\n\n"
                "local _, ns = ...\n\nns.MapLetters = {}\n")
        for loc in sorted(have):
            f.write("\nns.MapLetters.%s = {\n%s}\n" % (loc, have[loc]))


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "dump":
        dump(*sys.argv[2:5])
    elif len(sys.argv) >= 5 and sys.argv[1] == "bake":
        bake(*sys.argv[2:5])
    else:
        sys.exit(__doc__ or "usage: letters.py dump <db2> <cache> <out> | bake <enUS dump> <locale dump> <locale>")
