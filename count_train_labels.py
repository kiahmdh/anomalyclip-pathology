import os
import csv
import argparse


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


def count_dir(root, labels, use_csv):
    n1 = n0 = nm = nf = 0
    if not os.path.isdir(root):
        return (0, 0, 0, 0)
    for slide in sorted(os.listdir(root)):
        sp = os.path.join(root, slide)
        if not os.path.isdir(sp):
            continue
        for f in os.listdir(sp):
            if not f.lower().endswith('.jpeg'):
                continue
            nf += 1
            if use_csv:
                lab = labels.get((slide, f))
                if lab is None:
                    nm += 1
                elif lab == 1:
                    n1 += 1
                else:
                    n0 += 1
            else:
                n0 += 1
    return (n1, n0, nm, nf)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tile_label_csv', required=True)
    ap.add_argument('--train_root', required=True, help='.../single/fold1/train')
    args = ap.parse_args()

    labels = parse_labels(args.tile_label_csv)
    print(f'tile_label.csv: {len(labels)} labeled (slide,patch) entries')

    t1, t0, tm, tf = count_dir(os.path.join(args.train_root, '1_tumor'), labels, True)
    n1, n0, nm, nf = count_dir(os.path.join(args.train_root, '0_normal'), labels, False)

    print('--- train/1_tumor  (true labels from tile_label.csv) ---')
    print(f'  true tumor   (label 1): {t1}')
    print(f'  true normal  (label 0): {t0}    <- hard negatives (normal tissue in tumor slides)')
    print(f'  NOT FOUND in csv:       {tm}    <- if large, csv does not cover these slides')
    print(f'  files total:            {tf}')
    print('--- train/0_normal  (normal slides, all normal) ---')
    print(f'  clean normals:          {nf}    <- clean negatives')
    print()
    print(f'BOTTLENECK = true tumor patches in train = {t1}')
    if t1 > 0:
        print(f'Your 1:2 plan -> {t1} tumor + {t1} clean-neg + {t1} hard-neg = {3 * t1} patches total')
        print(f'(down from {tf + nf} patches currently sitting in train/1_tumor + train/0_normal)')
