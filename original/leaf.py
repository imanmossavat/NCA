"""Procedural leaf-with-veins target (PIL only, no torch, no downloads)."""
import numpy as np
import PIL.Image, PIL.ImageDraw


def make_leaf(size=48, ss=8):
    """Return RGBA float array [size,size,4]: green blade, darker midrib and side veins."""
    s = size * ss
    im = PIL.Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = PIL.ImageDraw.Draw(im)
    cx, top, bot = s / 2, s * 0.06, s * 0.94
    # blade outline: pointed ellipse-ish shape, widest a little below the middle
    pts_l, pts_r = [], []
    for t in np.linspace(0, 1, 80):
        y = top + t * (bot - top)
        w = (s * 0.36) * np.sin(np.pi * t ** 0.8) ** 0.9
        pts_l.append((cx - w, y)); pts_r.append((cx + w, y))
    d.polygon(pts_l + pts_r[::-1], fill=(76, 160, 64, 255), outline=(40, 100, 40, 255))
    # midrib
    vein = (170, 220, 110, 255)
    d.line([(cx, top), (cx, bot + s * 0.04)], fill=vein, width=int(ss * 1.6))
    # side veins: curve outward and up from the midrib
    for k in range(1, 6):
        y0 = top + (bot - top) * (0.16 + 0.13 * k)
        for sgn in (-1, 1):
            xs = [cx + sgn * (s * 0.25) * (u ** 0.85) for u in np.linspace(0, 1, 20)]
            ys = [y0 - (s * 0.11) * (u ** 1.3) for u in np.linspace(0, 1, 20)]
            d.line(list(zip(xs, ys)), fill=vein, width=int(ss * 1.1))
    im = im.resize((size, size), PIL.Image.LANCZOS)
    return np.float32(im) / 255.0


if __name__ == "__main__":
    a = make_leaf(48)
    PIL.Image.fromarray((a * 255).astype(np.uint8)).resize((384, 384), PIL.Image.NEAREST).save("leaf_preview.png")
    print("ok", a.shape, "alpha coverage", round(float((a[..., 3] > 0.5).mean()), 2))
