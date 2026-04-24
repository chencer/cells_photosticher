import argparse
import os
import sys
import cv2
import numpy as np


# --- Argument parsing ---
def parse_args():
    p = argparse.ArgumentParser(description="Phase correlation stitching for fluorescence tif images")
    p.add_argument("input_dir", help="Directory containing tif images")
    p.add_argument("-o", "--output", default="output.png", help="Output file path")
    p.add_argument("--cols", type=int, default=21, help="Number of columns")
    p.add_argument("--rows", type=int, default=5, help="Number of rows")
    return p.parse_args()


# --- Load images sorted by filename number ---
def load_images(input_dir):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]
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


# --- Snake-order grid layout ---
# Row 0 = bottom row. Odd rows go left→right, even rows go right→left.
# grid[row][col] = image index
def build_grid(images, cols, rows):
    total = cols * rows
    if len(images) < total:
        print(f"[warn] expected {total} images but only {len(images)} found")
    grid = []
    idx = 0
    for r in range(rows):
        row_indices = []
        for c in range(cols):
            row_indices.append(idx if idx < len(images) else None)
            idx += 1
        # even row index (0-based from bottom) = right-to-left
        if r % 2 == 0:
            row_indices = row_indices[::-1]
        grid.append(row_indices)
    return grid


# --- Phase correlate two images, return (dx, dy, confidence) ---
def phase_correlate_pair(img_a, img_b, init_dx, init_dy):
    h, w = img_a.shape[:2]

    # Crop ROI around expected overlap region for correlation
    if init_dx > 0:
        # img_b is to the right: right edge of a overlaps left edge of b
        overlap_w = min(init_dx + 200, w)
        roi_a = img_a[:, w - overlap_w:].astype(np.float32)
        roi_b = img_b[:, :overlap_w].astype(np.float32)
    elif init_dx < 0:
        overlap_w = min(-init_dx + 200, w)
        roi_a = img_a[:, :overlap_w].astype(np.float32)
        roi_b = img_b[:, w - overlap_w:].astype(np.float32)
    else:
        roi_a = img_a.astype(np.float32)
        roi_b = img_b.astype(np.float32)

    if init_dy > 0:
        overlap_h = min(init_dy + 200, h)
        roi_a = roi_a[h - overlap_h:, :]
        roi_b = roi_b[:overlap_h, :]
    elif init_dy < 0:
        overlap_h = min(-init_dy + 200, h)
        roi_a = roi_a[:overlap_h, :]
        roi_b = roi_b[h - overlap_h:, :]

    # Convert to grayscale for phase correlation
    if roi_a.ndim == 3:
        ga = cv2.cvtColor(roi_a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(roi_b, cv2.COLOR_BGR2GRAY)
    else:
        ga, gb = roi_a, roi_b

    # Resize to same shape if needed
    min_h = min(ga.shape[0], gb.shape[0])
    min_w = min(ga.shape[1], gb.shape[1])
    ga = ga[:min_h, :min_w]
    gb = gb[:min_h, :min_w]

    shift, confidence = cv2.phaseCorrelate(ga, gb)
    # shift = (dx_in_roi, dy_in_roi) from roi_b to roi_a
    # Translate back to full-image offset
    if init_dx > 0:
        dx = w - overlap_w + (-shift[0])
    elif init_dx < 0:
        dx = -(w - overlap_w + shift[0])
    else:
        dx = shift[0]

    if init_dy > 0:
        dy = h - overlap_h + (-shift[1])
    elif init_dy < 0:
        dy = -(h - overlap_h + shift[1])
    else:
        dy = shift[1]

    return dx, dy, confidence


# --- Linear blend two images at overlap region ---
def linear_blend(canvas, img, x_off, y_off, overlap_x, overlap_y):
    h, w = img.shape[:2]
    ch, cw = canvas.shape[:2]

    # Clip to canvas bounds
    x0 = max(0, x_off)
    y0 = max(0, y_off)
    x1 = min(cw, x_off + w)
    y1 = min(ch, y_off + h)

    if x0 >= x1 or y0 >= y1:
        return

    img_x0 = x0 - x_off
    img_y0 = y0 - y_off
    img_x1 = img_x0 + (x1 - x0)
    img_y1 = img_y0 + (y1 - y0)

    canvas_region = canvas[y0:y1, x0:x1].astype(np.float32)
    img_region = img[img_y0:img_y1, img_x0:img_x1].astype(np.float32)

    # Create alpha mask: blend in overlap zone
    alpha = np.ones((y1 - y0, x1 - x0), dtype=np.float32)

    # Horizontal blend
    if overlap_x > 0 and img_x0 < overlap_x:
        blend_w = min(overlap_x - img_x0, x1 - x0)
        ramp = np.linspace(0, 1, blend_w, dtype=np.float32)
        alpha[:, :blend_w] *= ramp[np.newaxis, :]

    # Vertical blend
    if overlap_y > 0 and img_y0 < overlap_y:
        blend_h = min(overlap_y - img_y0, y1 - y0)
        ramp = np.linspace(0, 1, blend_h, dtype=np.float32)
        alpha[:blend_h, :] *= ramp[:, np.newaxis]

    alpha3 = alpha[:, :, np.newaxis]
    existing_mask = (canvas_region.sum(axis=2) > 0).astype(np.float32)[:, :, np.newaxis]

    # Where canvas is empty, just paste; where overlap, blend
    blended = img_region * alpha3 + canvas_region * (1 - alpha3) * existing_mask + canvas_region * (1 - existing_mask) * 0
    blended = np.where(existing_mask > 0, img_region * alpha3 + canvas_region * (1 - alpha3), img_region)

    canvas[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)


# --- Main stitching ---
def stitch(images, grid, cols, rows, img_h, img_w):
    overlap_x = img_w // 2   # ~1024px
    overlap_y = img_h // 2   # ~540px

    # Compute per-image absolute positions via phase correlation
    # positions[row][col] = (x_offset, y_offset) in canvas coords
    positions = [[None] * cols for _ in range(rows)]

    # Anchor: bottom-left grid position at (0, 0)
    positions[0][0] = (0, 0)

    # Process row by row: anchor col 0 vertically, then fill row horizontally
    for r in range(rows):
        # Step 1: anchor col 0 of this row via vertical correlation from row below
        if r > 0:
            prev_idx = grid[r - 1][0]
            curr_idx = grid[r][0]
            prev_pos = positions[r - 1][0]
            if prev_idx is not None and curr_idx is not None and prev_pos is not None:
                _, prev_img = images[prev_idx]
                _, curr_img = images[curr_idx]
                init_dy = img_h - overlap_y
                dx, dy, conf = phase_correlate_pair(prev_img, curr_img, 0, init_dy)
                positions[r][0] = (prev_pos[0] + int(round(dx)), prev_pos[1] + int(round(dy)))
                print(f"  col0 row{r-1}→row{r}: dx={dx:.1f} dy={dy:.1f} conf={conf:.4f}")

        # Step 2: fill remaining cols horizontally from col 0
        for c in range(1, cols):
            prev_pos = positions[r][c - 1]
            if prev_pos is None:
                continue
            prev_idx = grid[r][c - 1]
            curr_idx = grid[r][c]
            if prev_idx is None or curr_idx is None:
                continue
            _, prev_img = images[prev_idx]
            _, curr_img = images[curr_idx]
            init_dx = img_w - overlap_x
            dx, dy, conf = phase_correlate_pair(prev_img, curr_img, init_dx, 0)
            positions[r][c] = (prev_pos[0] + int(round(dx)), prev_pos[1] + int(round(dy)))
            print(f"  row{r} col{c-1}→col{c}: dx={dx:.1f} dy={dy:.1f} conf={conf:.4f}")

    # Compute canvas size
    all_pos = [(x, y) for row in positions for (x, y) in row if (x, y) is not None and x is not None]
    min_x = min(p[0] for p in all_pos)
    min_y = min(p[1] for p in all_pos)
    max_x = max(p[0] for p in all_pos) + img_w
    max_y = max(p[1] for p in all_pos) + img_h

    # Shift all positions so canvas origin is (0, 0)
    positions = [
        [
            (positions[r][c][0] - min_x, positions[r][c][1] - min_y)
            if positions[r][c] is not None else None
            for c in range(cols)
        ]
        for r in range(rows)
    ]

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    print(f"\nCanvas size: {canvas_w} x {canvas_h}")

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
            print(f"  placing row{r} col{c} ({fname}) at ({x_off}, {y_off})")
            linear_blend(canvas, img, x_off, y_off, overlap_x, overlap_y)

    return canvas


def main():
    args = parse_args()

    images = load_images(args.input_dir)
    if not images:
        print("No images loaded. Exiting.")
        sys.exit(1)

    _, sample = images[0]
    img_h, img_w = sample.shape[:2]
    print(f"Image size: {img_w}x{img_h}")

    grid = build_grid(images, args.cols, args.rows)

    result = stitch(images, grid, args.cols, args.rows, img_h, img_w)

    cv2.imwrite(args.output, result)
    print(f"\nSaved: {args.output} ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    main()
