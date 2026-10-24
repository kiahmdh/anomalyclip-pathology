"""
Score every patch with the fine-tuned AnomalyCLIP (score = global-token abnormal
probability text_probs[:,0,1], identical to test.py), then:
  - write a per-patch CSV (patch_name, slide, label, Ascore)
  - report PATCH-level AUROC / AP   (Task 1 / Task 3)
  - if --reference_csv is given, aggregate to SLIDE level by max score and report
    SLIDE-level AUROC / AP            (Task 2)
Slide name is taken from each patch's parent folder.
"""
import argparse
import os
import csv
import torch
torch.backends.cudnn.enabled = False
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import get_transform


def load_reference(path):
    """slide-level reference csv -> dict slide(lower, no .tif) -> 1 tumor / 0 normal."""
    ref = {}
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            name = row[0].strip()
            if name.lower() in ('slide_name', 'case', 'image', 'filename', 'name', 'slide'):
                continue
            key = os.path.splitext(name)[0].strip().lower()
            label = None
            for cell in row[1:]:
                c = cell.strip().lower()
                if c in ('tumor', 'tumour', '1', 'positive', 'malignant'):
                    label = 1
                    break
                if c in ('normal', '0', 'negative', 'benign'):
                    label = 0
                    break
            if label is not None:
                ref[key] = label
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="folder with meta.json (test key)")
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--dataset", default="camelyon16")
    # addedd
    ap.add_argument("--vlm", default="clip", choices=["clip", "plip"])
    ap.add_argument("--dpam_layer", type=int, default=None)
    # to here
    ap.add_argument("--out_csv", default=None, help="write per-patch scores here")
    ap.add_argument("--reference_csv", default=None, help="slide labels -> enables slide-level AUROC")
    ap.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    ap.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--batch_size", type=int, default=32, help="eval batch size (image-level only, safe to raise)")
    ap.add_argument("--seed", type=int, default=111)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = {"Prompt_length": args.n_ctx,
              "learnabel_text_embedding_depth": args.depth,
              "learnabel_text_embedding_length": args.t_n_ctx}
    #added
    if args.vlm == "plip":
        import plip
        model, _ = plip.load(device=device, design_details=params)
    else:
        model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    dpam_layer = args.dpam_layer if args.dpam_layer is not None else (8 if args.vlm == "plip" else 20)
    # to here

    model.eval()
    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess,
                        target_transform=target_transform, dataset_name=args.dataset)
    loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=(8 if args.vlm == "plip" else 20))
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    rows = []
    gt_list, score_list = [], []
    text_pair = text_features[0]  # [2, 768] : index 0 normal, index 1 abnormal
    for items in tqdm(loader):
        image = items['img'].to(device)
        labels = items['anomaly'].detach().cpu().numpy().tolist()
        paths = items['img_path']
        with torch.no_grad():
            image_features, _ = model.encode_image(image, args.features_list, DPAM_layer=(8 if args.vlm == "plip" else 20))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_pair.t()          # [B, 2]
            probs = (logits / 0.07).softmax(-1)
            scores = probs[:, 1].detach().cpu().numpy().tolist()   # P(abnormal), == test.py
        for path, lab, sc in zip(paths, labels, scores):
            patch_name = os.path.basename(path)
            slide = os.path.basename(os.path.dirname(path))
            rows.append((patch_name, slide, int(lab), float(sc)))
            gt_list.append(int(lab))
            score_list.append(float(sc))

    if args.out_csv:
        d = os.path.dirname(os.path.abspath(args.out_csv))
        os.makedirs(d, exist_ok=True)
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['patch_name', 'slide', 'label', 'Ascore'])
            w.writerows(rows)
        print(f"wrote {len(rows)} patch scores -> {args.out_csv}")

    gt = np.array(gt_list)
    pr = np.array(score_list)
    print(f"PATCH-LEVEL: n={len(gt)}  tumor={int((gt == 1).sum())}  normal={int((gt == 0).sum())}")
    if (gt == 1).any() and (gt == 0).any():
        print(f"  patch-AUROC = {roc_auc_score(gt, pr) * 100:.2f}")
        print(f"  patch-AP    = {average_precision_score(gt, pr) * 100:.2f}")
    else:
        print("  only one class present -> patch AUROC undefined")

    if args.reference_csv:
        ref = load_reference(args.reference_csv)
        slide_score = {}
        for _, slide, _, score in rows:
            s = slide.lower()
            slide_score[s] = max(slide_score.get(s, -1.0), score)   # max aggregation
        sg, sp, missing = [], [], []
        for s, sc in slide_score.items():
            if s in ref:
                sg.append(ref[s])
                sp.append(sc)
            else:
                missing.append(s)
        sg, sp = np.array(sg), np.array(sp)
        print(f"SLIDE-LEVEL (max over patches): slides={len(sg)}  "
              f"tumor={int((sg == 1).sum())}  normal={int((sg == 0).sum())}")
        if missing:
            print(f"  WARNING: {len(missing)} slides not in reference.csv, e.g. {missing[:5]}")
        if (sg == 1).any() and (sg == 0).any():
            print(f"  slide-AUROC = {roc_auc_score(sg, sp) * 100:.2f}")
            print(f"  slide-AP    = {average_precision_score(sg, sp) * 100:.2f}")
        else:
            print("  only one slide class present -> slide AUROC undefined")


if __name__ == '__main__':
    main()
