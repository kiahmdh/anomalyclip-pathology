"""
Build a FULL test-set meta.json (no subsampling) with true per-patch labels:
  test/1_tumor/<slide>/*.jpeg -> label from tile_label.csv
  test/0_normal/<slide>/*.jpeg -> label 0 (normal slides)
Writes meta.json (test key) to --out_dir. Reads patches only; nothing written to data.
"""
import os
import csv
import json
import argparse

CLS_NAME = 'lymph node tissue'
IMG_EXT = '.jpeg'


def parse_labels(csv_path):
    labels = {}
    with open(csv_path, newline='') as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            parts = row[0].replace('\\', '/').split('/')
            slide = next((p[:-4] for p in parts if p.endswith('.tif')), None)
            if slide is None:
                continue
            try:
                labels[(slide, parts[-1])] = int(row[1])
            except ValueError:
                pass
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_root', required=True, help='.../single/fold1/test')
    ap.add_argument('--tile_label_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    labels = parse_labels(args.tile_label_csv)
    os.makedirs(args.out_dir, exist_ok=True)
    test_root = os.path.abspath(args.test_root)

    items = []
    n_pos = n_neg = missing = 0
    for class_dir in sorted(os.listdir(test_root)):
        class_path = os.path.join(test_root, class_dir)
        if not os.path.isdir(class_path):
            continue
        is_tumor = 'normal' not in class_dir.lower()
        for slide in sorted(os.listdir(class_path)):
            sp = os.path.join(class_path, slide)
            if not os.path.isdir(sp):
                continue
            for f in sorted(os.listdir(sp)):
                if not f.lower().endswith(IMG_EXT):
                    continue
                if is_tumor:
                    lab = labels.get((slide, f))
                    if lab is None:
                        missing += 1
                        continue
                else:
                    lab = 0
                items.append(dict(img_path=os.path.join(sp, f), mask_path='',
                                  cls_name=CLS_NAME, specie_name=slide, anomaly=lab))
                if lab == 1:
                    n_pos += 1
                else:
                    n_neg += 1

    meta_path = os.path.join(args.out_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        f.write(json.dumps({'test': {CLS_NAME: items}}, indent=4) + '\n')
    print(f'Wrote {meta_path}')
    print(f'  total={len(items)}  tumor(1)={n_pos}  normal(0)={n_neg}  missing_in_csv={missing}')


if __name__ == '__main__':
    main()
