"""
Reconstruct whole-slide anomaly heatmaps from a per-patch score CSV.

Each patch filename encodes its grid position: "<col>_<row>-<dzlevel>.jpeg"
(e.g. 100_31-16.jpeg -> col=100, row=31). We place each patch's Ascore back at
[row, col] to form the slide heatmap. For each slide we save a figure with the
predicted anomaly-score map and, beside it, the ground-truth tumor map (from the
CSV 'label' column). No GPU / model needed.
"""
import argparse
import os
import re
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATCH_RE = re.compile(r'^(\d+)_(\d+)-(\d+)\.jpe?g$', re.IGNORECASE)


def load_reference(path):
    ref = {}
    if not path:
        return ref
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        cols = [h.strip().lower() for h in header] if header else []
        name_idx = cols.index('image') if 'image' in cols else 0
        type_idx = cols.index('type') if 'type' in cols else 1
        for row in reader:
            if len(row) <= type_idx:
                continue
            key = os.path.splitext(row[name_idx].strip())[0].lower()
            t = row[type_idx].strip().lower()
            if t in ('tumor', 'tumour'):
                ref[key] = 'tumor'
            elif t == 'normal':
                ref[key] = 'normal'
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='per-patch CSV: patch_name,slide,label,Ascore')
    ap.add_argument('--out_dir', required=True, help='folder to write the heatmap PNGs into')
    ap.add_argument('--slides', nargs='+',
                    default=[f'test_{i:03d}' for i in range(1, 11)],
                    help='which slides (default test_001..test_010)')
    ap.add_argument('--reference_csv', default=None, help='optional, for tumor/normal in the title')
    ap.add_argument('--cmap', default='jet', help='matplotlib colormap (jet, hot, viridis, ...)')
    ap.add_argument('--stride', type=int, default=1,
                    help='pixel stride between patches; filenames are x_y in pixels for Camelyon17 (use 256). '
                         'Default 1 = filenames are already grid indices (Camelyon16).')
    ap.add_argument('--per_slide_norm', action='store_true',
                    help='stretch each slide to its own min/max instead of fixed 0..1')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    want = {s.lower() for s in args.slides}
    ref = load_reference(args.reference_csv)

    # collect per-slide patch lists: (col, row, score, label)
    data = defaultdict(list)
    with open(args.csv, newline='') as f:
        for row in csv.DictReader(f):
            slide = row['slide'].strip().lower()
            if slide not in want:
                continue
            m = PATCH_RE.match(row['patch_name'].strip())
            if not m:
                continue
            col, rownum = int(m.group(1)) // args.stride, int(m.group(2)) // args.stride
            try:
                score = float(row['Ascore'])
            except ValueError:
                score = float('nan')
            try:
                label = int(row['label'])
            except (ValueError, KeyError):
                label = -1
            data[slide].append((col, rownum, score, label))

    for slide in args.slides:
        key = slide.lower()
        pts = data.get(key, [])
        if not pts:
            print(f'  {slide}: no patches found in CSV, skipped')
            continue

        cols = [p[0] for p in pts]
        rows = [p[1] for p in pts]
        min_c, max_c = min(cols), max(cols)
        min_r, max_r = min(rows), max(rows)
        H, W = max_r - min_r + 1, max_c - min_c + 1

        score_grid = np.full((H, W), np.nan, dtype=float)
        label_grid = np.full((H, W), np.nan, dtype=float)
        for col, rownum, score, label in pts:
            score_grid[rownum - min_r, col - min_c] = score
            if label in (0, 1):
                label_grid[rownum - min_r, col - min_c] = label

        cmap = plt.get_cmap(args.cmap).copy()
        cmap.set_bad('white')  # missing tiles (background) shown white

        if args.per_slide_norm:
            finite = score_grid[np.isfinite(score_grid)]
            vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        else:
            vmin, vmax = 0.0, 1.0

        slide_type = ref.get(key, '')
        n_tumor = int(np.nansum(label_grid == 1))
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        im0 = axes[0].imshow(np.ma.masked_invalid(score_grid), cmap=cmap,
                             vmin=vmin, vmax=vmax, origin='upper', interpolation='nearest')
        axes[0].set_title('Predicted anomaly score P(abnormal)')
        axes[0].set_xlabel('column'); axes[0].set_ylabel('row')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        gt_cmap = plt.get_cmap('Reds').copy(); gt_cmap.set_bad('white')
        im1 = axes[1].imshow(np.ma.masked_invalid(label_grid), cmap=gt_cmap,
                             vmin=0, vmax=1, origin='upper', interpolation='nearest')
        axes[1].set_title(f'Ground-truth tumor patches (n={n_tumor})')
        axes[1].set_xlabel('column'); axes[1].set_ylabel('row')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        title = f'{slide}' + (f'  ({slide_type})' if slide_type else '') + f'   grid {H}x{W}, {len(pts)} patches'
        fig.suptitle(title)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f'{slide}.png')
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'  {slide}: {len(pts)} patches, grid {H}x{W}, tumor patches {n_tumor} -> {out_path}')

    print(f'done -> {args.out_dir}')


if __name__ == '__main__':
    main()