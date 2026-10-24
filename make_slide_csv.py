"""
From a per-patch scores CSV (patch_name,slide,label,Ascore) + reference.csv,
build a slide-level CSV with max and top-1% mean anomaly scores.
No GPU needed.
"""
import argparse
import os
import csv
import numpy as np
from collections import defaultdict


def load_reference(path):
    ref = {}
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
                ref[key] = 1
            elif t == 'normal':
                ref[key] = 0
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch_csv', required=True,
                    help='per-patch CSV: patch_name,slide,label,Ascore')
    ap.add_argument('--reference_csv', required=True,
                    help='Camelyon16 reference.csv (image,type,class,center)')
    ap.add_argument('--out_csv', required=True,
                    help='where to write the slide-level CSV')
    ap.add_argument('--topk_frac', type=float, default=0.01,
                    help='top fraction for mean score (default 0.01 = top 1%%)')
    args = ap.parse_args()

    ref = load_reference(args.reference_csv)

    slide_scores = defaultdict(list)
    with open(args.patch_csv, newline='') as f:
        for row in csv.DictReader(f):
            try:
                sc = float(row['Ascore'])
            except (ValueError, KeyError):
                continue
            if not np.isfinite(sc):
                continue
            slide_scores[row['slide'].lower()].append(sc)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    col = f'Ascore_top{args.topk_frac*100:g}pct_mean'
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slide', 'label', 'n_patches', 'Ascore_max', col])
        for slide in sorted(slide_scores):
            scs = np.array(slide_scores[slide])
            k = max(1, int(np.ceil(len(scs) * args.topk_frac)))
            top1 = float(np.sort(scs)[::-1][:k].mean())
            w.writerow([slide,
                        ref.get(slide, ''),
                        len(scs),
                        float(scs.max()),
                        top1])
    print(f'wrote {len(slide_scores)} slides -> {args.out_csv}')


if __name__ == '__main__':
    main()
