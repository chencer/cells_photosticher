import argparse
import os
import sys
import cv2
import numpy as np
import torch
from scipy.spatial import KDTree


# --- Argument parsing ---
def parse_args():
    p = argparse.ArgumentParser(description="Cellpose cell-center registration stitching")
    p.add_argument("input_dir", help="Directory containing tif images")
    p.add_argument("-o", "--output", default="output.png", help="Output file path")
    p.add_argument("--cols", type=int, default=21, help="Number of columns")
    p.add_argument("--rows", type=int, default=5, help="Number of rows")
    p.add_argument("--model", default="./cyto3", help="Path to cellpose model")
    p.add_argument("--diameter", type=int, default=120)
    return p.parse_args()


# --- Load images sorted by filename ---
def load_images(input_dir):
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]
    files.sort()
    print(f"Found {len(files)} tif files")
    images = []
    for f in files:
        path = os.path.join(input_dir, f)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [warn] failed to load {f}, skipping")
            continue
        images.append((f, img))
    return images


# --- Build snake-order grid ---
# even rows: images reversed (right-to-left), odd rows: left-to-right
# grid[r][c] = index into images list, or None
def build_grid(images, cols, rows):
    if len(images) < cols * rows:
        print(f"[warn] expected {cols*rows} images, got {len(images)}")
    grid = []
    idx = 0
    for r in range(rows):
        row_indices = [idx + c if idx + c < len(images) else None for c in range(cols)]
        idx += cols
        if r % 2 == 0:
            row_indices = row_indices[::-1]
        grid.append(row_indices)
    return grid


# --- Cellpose segmentation: return cell centers for one image ---
def get_centers(img, model, diameter):
    masks, flows, styles = model.eval(
        img,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=50,
        resample=False,
    )[:3]
    cell_ids = np.unique(masks)[1:]
    centers = []
    for cid in cell_ids:
        ys, xs = np.where(masks == cid)
        centers.append((float(xs.mean()), float(ys.mean())))
    return np.array(centers, dtype=np.float32) if centers else np.zeros((0, 2), dtype=np.float32)


# --- Match cell centers between two image strips and return median offset ---
# strip_a: centers in image-a coords, strip_b: centers in image-b coords
# Returns (dx, dy) offset such that: pos_in_canvas_b = pos_in_canvas_a + (dx, dy)
#   i.e. matched_b_center ≈ matched_a_center + (dx, dy)
def match_centers(centers_a, centers_b, threshold=50):
    if len(centers_a) < 3 or len(centers_b) < 3:
        return None
    tree = KDTree(centers_b)
    dists, idxs = tree.query(centers_a, k=1)
    mask = dists < threshold
    if mask.sum() < 3:
        return None
    deltas = centers_b[idxs[mask]] - centers_a[mask]
    dx = float(np.median(deltas[:, 0]))
    dy = float(np.median(deltas[:, 1]))
    return dx, dy


# --- Compute offset between two adjacent images using cellpose centers ---
# direction: 'h' (img_b is to the right of img_a) or 'v' (img_b is above img_a)
# Returns (offset_dx, offset_dy, n_matches) where offset is img_b canvas pos - img_a canvas pos
def compute_offset(img_a, img_b, centers_a, centers_b, img_w, img_h, direction):
    overlap_x = int(img_w * 0.5)
    overlap_y = int(img_h * 0.5)

    if direction == 'h':
        # img_b is to the RIGHT of img_a
        # overlap: right half of img_a, left half of img_b
        strip_a = centers_a[centers_a[:, 0] >= img_w - overlap_x] if len(centers_a) else centers_a
        strip_b = centers_b[centers_b[:, 0] < overlap_x] if len(centers_b) else centers_b
        # Shift strip_a coords to overlap frame (subtract offset from left of img_a's right edge)
        strip_a_shifted = strip_a - np.array([img_w - overlap_x, 0], dtype=np.float32)
        result = match_centers(strip_a_shifted, strip_b, threshold=50)
        if result is None:
            print(f"    [fallback] <3 matches, using fixed offset")
            return img_w - overlap_x, 0, 0
        dx_in_overlap, dy = result
        n = int((centers_a[:, 0] >= img_w - overlap_x).sum())
        # Total dx from img_a origin to img_b origin
        offset_dx = (img_w - overlap_x) + dx_in_overlap
        return offset_dx, dy, n

    else:  # 'v': img_b is ABOVE img_a (larger canvas y)
        # overlap: top half of img_a, bottom half of img_b
        strip_a = centers_a[centers_a[:, 1] < overlap_y] if len(centers_a) else centers_a
        strip_b = centers_b[centers_b[:, 1] >= img_h - overlap_y] if len(centers_b) else centers_b
        # Shift strip_b to overlap frame
        strip_b_shifted = strip_b - np.array([0, img_h - overlap_y], dtype=np.float32)
        result = match_centers(strip_a, strip_b_shifted, threshold=50)
        if result is None:
            print(f"    [fallback] <3 matches, using fixed offset")
            return 0, img_h - overlap_y, 0
        dx, dy_in_overlap = result
        n = int((centers_b[:, 1] >= img_h - overlap_y).sum())
        offset_dy = (img_h - overlap_y) + dy_in_overlap
        return dx, offset_dy, n


# --- Linear blend composite ---
def linear_blend(canvas, img, x_off, y_off, overlap_x, overlap_y):
    h, w = img.shape[:2]
    ch, cw = canvas.shape[:2]
    x0 = max(0, x_off)
    y0 = max(0, y_off)
    x1 = min(cw, x_off + w)
    y1 = min(ch, y_off + h)
    if x0 >= x1 or y0 >= y1:
        return
    img_x0 = x0 - x_off
    img_y0 = y0 - y_off
    canvas_region = canvas[y0:y1, x0:x1].astype(np.float32)
    img_region = img[img_y0:img_y0 + (y1 - y0), img_x0:img_x0 + (x1 - x0)].astype(np.float32)

    alpha = np.ones((y1 - y0, x1 - x0), dtype=np.float32)
    if overlap_x > 0 and img_x0 < overlap_x:
        bw = min(overlap_x - img_x0, x1 - x0)
        if bw > 0:
            alpha[:, :bw] *= np.linspace(0, 1, bw, dtype=np.float32)[np.newaxis, :]
    if overlap_y > 0 and img_y0 < overlap_y:
        bh = min(overlap_y - img_y0, y1 - y0)
        if bh > 0:
            alpha[:bh, :] *= np.linspace(0, 1, bh, dtype=np.float32)[:, np.newaxis]

    alpha3 = alpha[:, :, np.newaxis]
    existing = (canvas_region.sum(axis=2) > 0).astype(np.float32)[:, :, np.newaxis]
    blended = np.where(existing > 0,
                       img_region * alpha3 + canvas_region * (1 - alpha3),
                       img_region)
    canvas[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)


# --- Main stitching ---
def stitch(images, grid, cols, rows, img_h, img_w, model, diameter):
    step_x = int(img_w * 0.5)
    step_y = int(img_h * 0.5)

    # --- Run cellpose on all images, cache centers ---
    print("\n--- Segmenting cells ---")
    centers_cache = {}
    total = len(images)
    for i, (fname, img) in enumerate(images):
        print(f"  [{i+1}/{total}] {fname} ...", end=" ", flush=True)
        c = get_centers(img, model, diameter)
        centers_cache[i] = c
        print(f"{len(c)} cells")

    def get_img(r, c):
        idx = grid[r][c]
        return (idx, images[idx][0], images[idx][1]) if idx is not None else (None, None, None)

    # --- Build positions grid ---
    # Snake layout:
    #   even row r: grid[r][c] is at physical col (cols-1-c)  [right-to-left]
    #   odd  row r: grid[r][c] is at physical col c           [left-to-right]
    #
    # Vertical neighbor of grid[r][c] in row r-1 is always grid[r-1][cols-1-c]
    # (proven: physical column maps to cols-1-c in the adjacent reversed row)

    positions = [[None] * cols for _ in range(rows)]
    positions[0][0] = (0, 0)

    print("\n--- Computing positions ---")

    # Row 0: even → right-to-left → each next col is to the LEFT (negative dx)
    for c in range(1, cols):
        prev = positions[0][c - 1]
        idx_a, fname_a, img_a = get_img(0, c - 1)
        idx_b, fname_b, img_b = get_img(0, c)
        if img_a is not None and img_b is not None:
            # img_b is to the LEFT of img_a (even row, right-to-left)
            # Compute as if img_b is to the right, then negate dx
            dx, dy, n = compute_offset(img_b, img_a,
                                       centers_cache[idx_b], centers_cache[idx_a],
                                       img_w, img_h, 'h')
            dx, dy = -dx, -dy
            print(f"  row0 col{c}→col{c-1} (reversed): n={n} dx={dx:.1f} dy={dy:.1f}")
        else:
            dx, dy = -step_x, 0
        positions[0][c] = (prev[0] + int(round(dx)), prev[1] + int(round(dy)))

    # Rows 1+
    for r in range(1, rows):
        # Anchor col 0 vertically from row r-1 col cols-1
        below_c = cols - 1
        prev = positions[r - 1][below_c]
        if prev is None:
            ref = next((p for p in positions[r - 1] if p is not None), (0, r * step_y))
            prev = ref
            print(f"  [warn] row{r} col0 anchor missing, using fallback")

        idx_a, _, img_a = get_img(r - 1, below_c)
        idx_b, _, img_b = get_img(r, 0)
        if img_a is not None and img_b is not None:
            dx, dy, n = compute_offset(img_a, img_b,
                                       centers_cache[idx_a], centers_cache[idx_b],
                                       img_w, img_h, 'v')
            print(f"  row{r} col0 ← row{r-1} col{below_c}: n={n} dx={dx:.1f} dy={dy:.1f}")
        else:
            dx, dy = 0, step_y
        positions[r][0] = (prev[0] + int(round(dx)), prev[1] + int(round(dy)))

        # Fill rest of row r horizontally
        # Even row → right-to-left; odd row → left-to-right
        for c in range(1, cols):
            prev = positions[r][c - 1]
            idx_a, _, img_a = get_img(r, c - 1)
            idx_b, _, img_b = get_img(r, c)
            if img_a is not None and img_b is not None:
                if r % 2 == 0:
                    # even: img_b is to the LEFT of img_a
                    dx, dy, n = compute_offset(img_b, img_a,
                                               centers_cache[idx_b], centers_cache[idx_a],
                                               img_w, img_h, 'h')
                    dx, dy = -dx, -dy
                else:
                    # odd: img_b is to the RIGHT of img_a
                    dx, dy, n = compute_offset(img_a, img_b,
                                               centers_cache[idx_a], centers_cache[idx_b],
                                               img_w, img_h, 'h')
                print(f"  row{r} col{c-1}→col{c}: n={n} dx={dx:.1f} dy={dy:.1f}")
            else:
                dx = -step_x if r % 2 == 0 else step_x
                dy = 0
            positions[r][c] = (prev[0] + int(round(dx)), prev[1] + int(round(dy)))

    # --- Normalize canvas origin ---
    all_pos = [p for row in positions for p in row if p is not None]
    if not all_pos:
        raise RuntimeError("No positions computed")
    min_x = min(p[0] for p in all_pos)
    min_y = min(p[1] for p in all_pos)
    max_x = max(p[0] for p in all_pos) + img_w
    max_y = max(p[1] for p in all_pos) + img_h

    positions = [
        [(p[0] - min_x, p[1] - min_y) if p is not None else None for p in row]
        for row in positions
    ]

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    print(f"\nCanvas: {canvas_w}×{canvas_h} px")

    # --- Composite ---
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    print("\n--- Compositing ---")
    for r in range(rows):
        for c in range(cols):
            idx = grid[r][c]
            if idx is None:
                continue
            pos = positions[r][c]
            if pos is None:
                continue
            fname, img = images[idx]
            x_off, y_off = pos
            print(f"  place row{r} col{c} ({fname}) at ({x_off}, {y_off})")
            linear_blend(canvas, img, x_off, y_off, step_x, step_y)

    return canvas


def main():
    args = parse_args()

    images = load_images(args.input_dir)
    if not images:
        print("No images loaded.")
        sys.exit(1)

    _, sample = images[0]
    img_h, img_w = sample.shape[:2]
    print(f"Image size: {img_w}×{img_h}")

    grid = build_grid(images, args.cols, args.rows)

    print(f"\nLoading cellpose model: {args.model}")
    from cellpose import models
    import torch as _torch
    _orig = _torch.serialization.load
    _torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})
    use_gpu = _torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=args.model)
    print(f"  GPU: {use_gpu}")

    result = stitch(images, grid, args.cols, args.rows, img_h, img_w, model, args.diameter)

    cv2.imwrite(args.output, result)
    print(f"\nSaved: {args.output} ({result.shape[1]}×{result.shape[0]})")


if __name__ == "__main__":
    main()
