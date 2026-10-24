# AnomalyCLIP for lymph node metastasis detection

This is a fork of [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) (Zhou et al.,
ICLR 2024) adapted to histopathology. AnomalyCLIP learns two text prompts, one for
"normal" and one for "anomalous", on top of a frozen CLIP model, and scores an image by
which prompt it matches better. I adapted it to CAMELYON16 patches, added PLIP (a CLIP
model trained on pathology images) as an alternative backbone, and evaluated it at patch
and slide level, both on CAMELYON16 and, without retraining, on CAMELYON17.



## What I changed

The original code targets industrial defects (MVTec, VisA) and trains a pixel-level loss
from segmentation masks. CAMELYON16 patches only have patch-level labels, so I disabled
the pixel loss and trained only the image-level prompt loss. The prompts describe
"lymph node tissue" vs "metastatic lymph node tissue". The training set uses patch labels
from tumour slides: [describe briefly, e.g. "tumour patches plus normal patches from both
tumour and normal slides, at a 1:2 ratio"]. Unlike my other anomaly detection projects,
this setup sees real tumour examples during training.

To use PLIP, I wrote a converter (`plip.py`) that maps the Hugging Face PLIP weights into
the OpenAI CLIP layout AnomalyCLIP expects. The rest of the pipeline runs unchanged. The
converter includes a check that the converted model's embeddings match the original
(cosine ≈ 1.0).

For evaluation, I score every test patch, aggregate patch scores to slide level, and
rebuild whole-slide heatmaps from patch coordinates. The CAMELYON16-trained prompts are
also applied unchanged to CAMELYON17, which comes from different hospitals.

## Results

AUROC. Slide scores use the maximum patch score.

| Backbone | CAMELYON16 patch | CAMELYON16 slide | CAMELYON17 patch | CAMELYON17 slide |
|---|---|---|---|---|
| CLIP ViT-L/14 | 93.4 | 72.6 | 93.8 | 65.6 |
| PLIP ViT-B/32 | 96.3 | 91.2 | 95.7 | 69.8 |

PLIP beat CLIP everywhere, even though it's a much smaller model, and the gap was largest
at slide level. Domain-specific pretraining mattered more than model size here. Adding the
validation set to PLIP's training data made almost no difference. Averaging the top 1% of
patch scores instead of taking the max gave lower slide AUROC for both backbones.

Patch-level performance carried over to CAMELYON17 almost unchanged, but slide-level
performance dropped. Breaking the slides down by lesion size shows why. On slides with
macro-metastases only, both backbones reach about 99 slide AUROC on CAMELYON16 and 97–98
on CAMELYON17, so the model transfers well across hospitals. The drop comes from small
lesions: CAMELYON17 includes isolated tumour cell (ITC) nodes with only about six tumour
patches each, and a single high-scoring normal patch elsewhere on the slide is enough to
outrank them. Micro-metastases cause the same problem on CAMELYON16.

## Files

Files from the original repo that I modified: `train.py`, `prompt_ensemble.py`,
`dataset.py`. The other original files are unchanged.

| File | Purpose |
|---|---|
| `plip.py` | Loads PLIP into the AnomalyCLIP pipeline (weight conversion + sanity check) |
| `build_test_meta.py`, `build_test_meta_c17.py` | Test-set index files with per-patch labels |
| `eval_scores.py` | Scores every patch, writes a per-patch CSV, reports patch and slide AUROC |
| `auroc_from_csv.py` | Recomputes metrics from a saved CSV; can exclude slide classes (e.g. micro) |
| `make_slide_csv.py` | Slide-level scores (max and top-1% mean) |
| `make_heatmaps.py` | Whole-slide heatmaps next to the ground truth |
| `new_agg.py` | Experimental slide score based on spatial clusters of high-scoring patches |

## Running it

```bash
# test index with per-patch labels
python build_test_meta.py --test_root <camelyon16>/test --tile_label_csv tile_label.csv --out_dir data/c16_test

# train the prompts (add --vlm plip for the PLIP backbone)
python train.py --train_data_path data/c16_train --dataset camelyon16 --vlm clip \
    --batch_size 32 --epoch 15 --save_path checkpoints/clip

# score the test set
python eval_scores.py --data_path data/c16_test --dataset camelyon16 --vlm clip \
    --checkpoint_path checkpoints/clip/epoch_15.pth \
    --out_csv results/clip_c16.csv --reference_csv reference.csv

# metrics and heatmaps from the saved scores
python auroc_from_csv.py --csv results/clip_c16.csv --reference_csv reference.csv
python auroc_from_csv.py --csv results/clip_c16.csv --reference_csv reference.csv --exclude_class micro
python make_heatmaps.py --csv results/clip_c16.csv --out_dir heatmaps/clip --reference_csv reference.csv
```

For CAMELYON17, build the index with `build_test_meta_c17.py`, and pass `--stride 256` to
`make_heatmaps.py`, since those patch filenames store pixel coordinates.

## Limitations

Because it trains on tumour labels, this model isn't directly comparable to normal-only
anomaly detectors. There are no pixel masks, so AnomalyCLIP's localisation branch isn't
trained, and the heatmaps come from patch-level scores only. Each setting was run with
one seed. F1 and recall use thresholds chosen on the test data, so they're optimistic;
AUROC is unaffected. The spatial clustering score in `new_agg.py` is experimental and
isn't part of the reported results.

## References

AnomalyCLIP (Zhou et al., ICLR 2024), CLIP (Radford et al., ICML 2021),
PLIP (Huang et al., Nature Medicine 2023), CAMELYON16 (Ehteshami Bejnordi et al., JAMA 2017),
CAMELYON17 (Bándi et al., IEEE TMI 2019).

## License

The original AnomalyCLIP code keeps its license (see LICENSE). My additions use the same license.
