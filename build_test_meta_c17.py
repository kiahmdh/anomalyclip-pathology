"""
Build a Camelyon17 TEST meta.json (for generalization eval of a Camelyon16-trained model).

Pools ALL of fold_0/{train,test,validation} into one test set.
Structure:  fold_0/<split>/{0_normal,1_tumor}/patient_XXX_node_Y/<x>_<y>-1.jpeg
  - 0_normal patches -> label 0
  - 1_tumor patches  -> true per-patch label from tile_label.csv (key: "patient_XXX_node_Y/<file>")
The "slide" unit is the node (patient_XXX_node_Y), stored as specie_name.
Reads patches only; writes meta.json to --out_dir.
"""
import os
import csv
import json
import argparse

CLS_NAME = 'lymph node tissue'      # matches generate_class_info('camelyon16')
IMG_EXT = '.jpeg'


def parse_labels(csv_path):
    """tile_label.csv: header 'Tile,Label', rows 'patient_XXX_node_Y/<file>.jpeg,<0|1>'."""
    labels = {}
    with open(csv_path, newline='') as f:
        r = csv.reader(f)
        next(r, None)  # header
        for row in r:
            if len(row) < 2:
                continue
            key = row[0].replace('\\', '/').strip()   # "patient_XXX_node_Y/<file>.jpeg"
            try:
                labels[key] = int(row[1])
            except ValueError:
                pass
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold_root', required=True,
                    help='.../stratified_60_20_20_rand42/fold_0  (contains train/ test/ validation/)')
    ap.add_argument('--tile_label_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--splits', nargs='+', default=['train', 'test', 'validation'],
                    help='which split subfolders to pool (default: all three)')
    args = ap.parse_args()

    labels = parse_labels(args.tile_label_csv)
    os.makedirs(args.out_dir, exist_ok=True)
    fold_root = os.path.abspath(args.fold_root)

    items = []
    n_pos = n_neg = missing = 0
    seen_nodes = set()

    for split in args.splits:
        split_dir = os.path.join(fold_root, split)
        if not os.path.isdir(split_dir):
            print(f'  WARNING: split not found, skipped: {split_dir}')
            continue
        for class_dir in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            is_tumor = 'normal' not in class_dir.lower()   # 1_tumor -> True, 0_normal -> False
            for node in sorted(os.listdir(class_path)):
                node_path = os.path.join(class_path, node)
                if not os.path.isdir(node_path):
                    continue
                seen_nodes.add(node)
                for fn in sorted(os.listdir(node_path)):
                    if not fn.lower().endswith(IMG_EXT):
                        continue
                    if is_tumor:
                        lab = labels.get(f'{node}/{fn}')
                        if lab is None:
                            missing += 1
                            continue
                    else:
                        lab = 0
                    items.append(dict(img_path=os.path.join(node_path, fn), mask_path='',
                                      cls_name=CLS_NAME, specie_name=node, anomaly=lab))
                    if lab == 1:
                        n_pos += 1
                    else:
                        n_neg += 1

    meta_path = os.path.join(args.out_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        f.write(json.dumps({'test': {CLS_NAME: items}}, indent=4) + '\n')
    print(f'Wrote {meta_path}')
    print(f'  nodes={len(seen_nodes)}  total_patches={len(items)}  '
          f'tumor(1)={n_pos}  normal(0)={n_neg}  missing_in_csv={missing}')


if __name__ == '__main__':
    main()
