# Evaluation summary

Detection is scored on the detector's **test** split (2349 images). Usage classification is scored on the **held-out** part of label.csv (2855 images, 1158 using a phone, 40.6%). Thresholds were chosen on the separate tune split (1224 images).

## Detection (test split)

| Model | Class | mAP50 | mAP50-95 | Mean IoU of true positives | Precision | Recall | At confidence |
|---|---|---|---|---|---|---|---|
| COCO yolo26n | phone | 0.313 | 0.103 | 0.664 | 0.272 | 0.481 | 0.05 |
| Fine-tuned | phone | 0.869 | 0.604 | 0.844 | 0.793 | 0.846 | 0.2 |
| Fine-tuned | head | 0.954 | 0.576 | 0.799 | 0.741 | 0.954 | 0.1 |

COCO has no head class, so only the fine-tuned model is scored on heads. Precision, recall and IoU are measured at each model's tuned confidence; mAP uses all detections.

### Phone mAP50 by box size

| Model | Small (< 32²) | Medium (32²–96²) | Large (> 96²) |
|---|---|---|---|
| COCO yolo26n | 0.076 (n=226) | 0.390 (n=860) | 0.517 (n=43) |
| Fine-tuned | 0.792 (n=226) | 0.889 (n=860) | 0.948 (n=43) |

## Usage classification (held-out, n=2855)

| Method | Accuracy | Precision | Recall | F1 | Specificity | Thresholds |
|---|---|---|---|---|---|---|
| A-coco | 0.679 | 0.590 | 0.680 | 0.632 | 0.678 | phone_conf 0.05 |
| A-ft | 0.876 | 0.818 | 0.893 | 0.854 | 0.864 | phone_conf 0.3 |
| A2-authors | 0.862 | 0.780 | 0.918 | 0.843 | 0.823 | phone_conf 0.2, head_conf 0.1 |
| B-rules | 0.873 | 0.809 | 0.900 | 0.852 | 0.855 | rules.json |
| C-mobilenet | 0.736 | 0.643 | 0.782 | 0.706 | 0.704 | threshold 0.5 |

Positive class = using phone. Every number was also recomputed with the FPI-Det authors' `calculate_metrics.py`; all 5 methods agree.
`A2-authors` uses the authors' decision logic (a head and a phone both found = using) on our detections. Their script returns 0 for "using" while label.csv marks "using" as 1 (checked against the boxes in Phase 1), so their numeric encoding is flipped here; the logic is unchanged. label.csv has no behaviour categories, so there is no per-category recall.

**Sanity check (B-rules must beat A-ft on accuracy and F1): FAIL** (accuracy 0.873 vs 0.876, F1 0.852 vs 0.854).

## Speed (CPU: 13th Gen Intel Core i7-13700HX, 640 px, batch of 1, mean of 200 images after 20 warm-up runs)

| Model | Mean ms | Median ms | 95th percentile ms |
|---|---|---|---|
| COCO yolo26n | 24.5 | 22.5 | 34.5 |
| Fine-tuned | 24.3 | 23.0 | 33.2 |

## Failure analysis of B-rules (held-out)

- False positives: 246 ({'ON_CALL': 95, 'ON_PHONE': 151}).
- False negatives: 116, by cause: no phone detected: 93; head and phone found, rules say not using: 21; phone found, no head detected: 2.
- Grids (`failures_fp.png`, `failures_fn.png`) show the most confident mistakes. Ranking: false positives: confidence of the phone that triggered 'using'; false negatives: confidence of the most confident phone detected (0 if none).

### Observations

Written after looking at both grids (`failures_fp.png`, `failures_fn.png`).

- **The two grids show the same kind of scene with opposite labels.** In the false-positive grid (label says "not using") many people are plainly holding a phone in front of them or looking at it. In the false-negative grid (label says "using") people hold a phone at hip height or beside them, much like the first group. The rules cannot separate what the labels separate, so some of these "mistakes" are probably label noise or a label definition we cannot see.
- **A phone held low is the main rule miss.** Standing people checking a phone at waist height sit more than 2 head-heights below the head centre, so ON_PHONE does not fire. Raising that limit fixes these but adds false positives, which is why tuning settled on 2 head-heights.
- **Seated people with the phone to the side are missed too.** The phone is level with the shoulder or desk and far to the side of the head, so it is neither "at the ear" nor "in front".
- **Most missed "using" images have no phone detected at all (93 of 116).** These are mostly small or partly hidden phones. The detector's small-phone mAP50 is 0.792 against 0.948 for large ones, so detection is the bigger limit, not the geometry.
- **Crowded scenes confuse the head choice.** A phone close to a neighbour's head can be matched to the wrong person.

## Low-light experiment (Unit II)

Does enhancing a dark frame before detection help? Dim and dark copies of the test images were made by lowering brightness, applying gamma and adding noise (`ml/make_lowlight.py`). The "real dark photos" are the 15% of original test images with the lowest brightness. The CLAHE settings and the method were chosen on 400 val images darkened the same way, not on test.

### Detector mAP50 on the test split (phone / head)

| Condition (images) | none | global HE | CLAHE | CLAHE + blur |
|---|---|---|---|---|
| original (2349) | 0.869 / 0.954 | 0.844 / 0.950 | 0.860 / 0.953 | 0.852 / 0.952 |
| dim (2349) | 0.699 / 0.837 | 0.600 / 0.745 | 0.672 / 0.811 | 0.737 / 0.864 |
| dark (2349) | 0.000 / 0.015 | 0.000 / 0.000 | 0.000 / 0.001 | 0.000 / 0.027 |
| real dark photos (352) | 0.924 / 0.920 | 0.908 / 0.903 | 0.924 / 0.914 | 0.914 / 0.913 |

Chosen on val: CLAHE clip limit 2.0, tiles (8, 8). Phone mAP50 on the val sample (mean of dim and dark): none 0.311, global HE 0.271, CLAHE 0.296, CLAHE + blur 0.343. Full CLAHE grid in `results/lowlight/lowlight.json`.

### Usage classification with B-rules (held-out, n=2855)

| Lighting / enhancement | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| original / none | 0.873 | 0.809 | 0.900 | 0.852 |
| original / CLAHE + blur | 0.872 | 0.806 | 0.902 | 0.851 |
| dark / none | 0.594 | 0.000 | 0.000 | 0.000 |
| dark / CLAHE + blur | 0.594 | 0.000 | 0.000 | 0.000 |

Mean brightness (0 to 255) of test images: original median 109, dim median 16, dark median 4; the real dark photos reach up to 75.

**App setting chosen from this** (`shared/rules.json`, `app`): enhancement `clahe_smooth`, used when mean frame brightness is below 29.

### Observations

- **Enhancement helps only in a narrow band.** On the "dim" copies (median brightness 16 of 255) CLAHE with a light blur raised phone mAP50 from 0.699 to 0.737 and head mAP50 from 0.837 to 0.864. Plain CLAHE (0.672) and global histogram equalization (0.600) made phone detection worse, because they also boost the noise, and the detector reads noise as texture.
- **The "dark" copies are too dark for any method.** At a median brightness of 4 (out of 255) the picture information is gone, and phone mAP50 is 0.000 without enhancement and 0.000 with the best method. Enhancement cannot bring back detail that is not there. A real app should tell the user the room is too dark instead of pretending to detect.
- **Real dark photos and normal photos do not need it.** The darkest real test photos (brightness up to 75) scored 0.924 with no enhancement and slightly less with any enhancement. On normal images the same enhancement changes phone mAP50 from 0.869 to 0.852 (CLAHE + blur). So enhancement is only worth turning on for genuinely dim frames, which is why the app setting is a brightness threshold (below 29), not "always on".
- **Usage accuracy follows detection.** With normal lighting B-rules scores 0.873 accuracy with or without enhancement. At the dark level it predicts "not using" for everything (0.594 accuracy, which is just the share of "not using" images), with or without enhancement.
- **Limits of this test.** The darkening is a recipe (scale, gamma, noise), not a real dim room, and the CLAHE settings and method were chosen on the same recipe applied to val images. The real-dark result is the check that does not depend on the recipe, and it says enhancement is not needed there.

## MobileNetV2 baseline (Unit IV)

A MobileNetV2 (ImageNet weights) looks at the whole image shrunk to 224 x 224 and answers "using phone or not". It was trained on the tune split only (979 train / 245 validation images; class ratio 1.47:1, so class weights were used). Stage 1 trained a new head on the frozen backbone (15 epochs), stage 2 fine-tuned the top 30 layers (10 epochs). Decision threshold 0.5 (best on validation: 0.65, 0.771 vs 0.763 at 0.5). Its row is C-mobilenet in the classification table above; curves are in `results/mobilenet/training_curves.png`.

MobileNet and B-rules disagree on 778 held-out images: MobileNet alone is right on 193, B-rules alone on 585 (`results/mobilenet/examples.png`).

### Observations

- **Shrinking the image hides small phones.** The whole picture is squeezed to 224 x 224 pixels, so a small phone in a wide scene is only a few pixels. In the bottom row of the grid the detector clearly sees the phone and MobileNet gives it a very low probability. The detector works at 640 pixels and can still find it. This is the main reason B-rules is right far more often (585 images against 193).
- **MobileNet judges the whole scene, the rules judge one relationship.** The cases MobileNet wins (top row) are pictures where the detector finds a head and something phone-like in a misleading place, such as an extreme close-up face or a phone at the side of a seated person. The rules follow the geometry and call it "using"; MobileNet looks at everything and says "not using", which matches the label.
- **MobileNet learned from far less.** It saw 979 images with one yes/no label each, while the detector learned from the full training set with a box around every head and phone. The curves show the gap starting to open (final training accuracy 0.812, validation 0.763) and validation was still improving when the epoch limit stopped it, so it could gain a little with more data or epochs, but it is unlikely to close a gap of 14 accuracy points (0.736 against 0.873).

## Focus Coach app speed

CPU: 13th Gen Intel Core i7-13700HX. 200 webcam-sized frames (1280 x 720) after 20 warm-up frames, through the whole app pipeline: enhancement check, YOLO26n on the CPU, rules, smoothing and drawing the HUD. Camera capture and the window are not included. The target is at least 8 FPS.

| Detector size | Mean ms per frame | Median ms | 95th percentile ms | FPS (mean) |
|---|---|---|---|---|
| 640 | 26.5 | 26.5 | 29.7 | 37.7 |
| 480 | 22.4 | 22.3 | 24.5 | 44.7 |

## Caveats

- `best.pt` was chosen on the detector's val split, and about 42% of label.csv are val images, so held-out numbers are slightly optimistic.
- FPI-Det is mostly surveillance-angle crops; results say little about a frontal laptop webcam.
