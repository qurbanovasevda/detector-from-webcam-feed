# Real-Time Pen and Pencil Detector

A lightweight webcam object detector built with **YOLOv8n, Ultralytics, and OpenCV**. The model was fine-tuned on a self-collected dataset to detect pens and pencils as one custom class: `writing_tool`.

## Features

- Continuous webcam inference on a laptop CPU.
- Live bounding boxes, class labels, and confidence scores.
- Measured live FPS, averaged over recent frame durations.
- Adjustable confidence threshold through an on-screen slider.
- Image capture with **S** and video recording with **R**.

Confidence thresholding is implemented. Temporal box smoothing is not part of the baseline; smoothing the FPS display does not smooth detections.

## Dataset

The dataset contains **127 self-collected phone images**, manually annotated with bounding boxes in Roboflow.

| Item | Details |
|---|---|
| Custom class | `writing_tool`: pens and pencils |
| Training images | 108 |
| Validation images | 19 |
| Separate static test split | None; additional evaluation uses live webcam scenes |
| Validation instances | 25 |
| Backgrounds | White, black, blue, and cluttered scenes |
| Other variations | Hand-held objects, different angles, and different lighting |
| Negative examples | Images without pens or pencils |

Glasses, sharpeners, tape, mice, and rulers are background objects and are not labeled. Every visible target object receives its own box. Similar images were grouped within the same split to reduce leakage between training and validation.

Preprocessing uses auto-orientation and aspect-ratio-preserving resizing to 640 × 640. The dataset is exported in YOLOv8 format. Augmentation is applied during training rather than generating an augmented Roboflow version.

## Training

Training was performed in Google Colab using a **Tesla T4 GPU** and **Ultralytics 8.4.183**. Both experiments started from pretrained `yolov8n.pt` weights and fine-tuned them on the custom dataset. The second experiment did not resume from the first experiment's weights.

| Setting | Experiment 1 | Experiment 2 |
|---|---|---|
| Maximum epochs | 60 | 100 |
| Completed epochs | 16 | 100 |
| Early-stopping patience | 15 | 35 |
| Image size | 640 | 640 |
| Batch size | 8 | 8 |
| Optimizer | Library default | AdamW |
| Initial learning rate | Library-selected | 0.001 |
| Mosaic probability | Library default | 0.5 |
| Mosaic disabled for final epochs | Library default | 10 |
| MixUp | Library default | 0.0 |
| Seed | 42 | 42 |

Experiment 1 stopped after 15 epochs without exceeding its best validation fitness, achieved at epoch 1. Its later curves showed recovery, motivating a longer second experiment with explicitly configured optimization. Multiple settings changed together, so the improvement cannot be attributed to one parameter alone.

## Validation Results

Metrics below were reported on the same **19-image validation split**, not on an independent test set.

| Metric | Experiment 1 | Experiment 2 — selected model |
|---|---:|---:|
| Precision | 0.477 | **0.853** |
| Recall | 0.520 | **0.960** |
| mAP@50 | 0.378 | **0.946** |
| mAP@50–95 | 0.259 | **0.815** |

The deployed weights are **Experiment 2's `best.pt`**. High validation performance on this small split does not establish performance on all webcam conditions.

The final Colab validation log reported 3.4 ms preprocessing, 3.4 ms inference, and 1.3 ms postprocessing per image on the T4. These are GPU validation timings, not laptop CPU webcam FPS.

## Project Layout

```text
detector-from-webcam-feed/
├── webcam.py
├── pen_detector/
│   └── best.pt
├── notebooks/
│   └── training.ipynb
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
├── results/
│   └── results.png
├── requirements.txt
├── .gitignore
└── README.md
```

This is the intended submission layout. Place the training notebook, exported dataset, training plots, and captured live evidence in the corresponding folders before submission. Captures can be copied from the location configured in `webcam.py` into `results/`.

## Installation and Usage

Use Python with a working webcam. The current capture configuration targets Windows.

Open PowerShell in the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe webcam.py
```

The runtime requirements are:

```text
ultralytics==8.4.183
opencv-python
```

The model path in `webcam.py` must point to:

```python
model_path = Path(__file__).parent / "pen_detector" / "best.pt"
```

| Control | Action |
|---|---|
| Confidence % slider | Adjust the minimum detection confidence |
| S | Save an image |
| R | Control video recording as implemented in `webcam.py` |
| Q | Exit |

The baseline uses camera index `0`, capture resolution 640 × 480, inference size 640, and `device="cpu"`. If the camera fails to open, close other camera applications and check the camera index.

FPS includes camera reading, inference, drawing, and display handling. It is computed from recent completed frame durations; it is not a theoretical model benchmark. Verify that saved images and videos contain the annotated output, including the FPS and threshold overlay.

## Live Evaluation

Reported qualitative observations from live testing:

- Both pens and pencils were detected.
- Angled objects were detected more readily than upright/straight orientations, where boxes sometimes appeared later.
- Tested additional objects were generally not detected as writing tools, but some dark background regions produced false positives.
- Raising the confidence threshold sometimes delayed the appearance of a box, while detection remained possible. This is a confidence-filtering effect and does not by itself demonstrate slower inference.

Uneven orientation coverage and limited color/background diversity are possible explanations, not confirmed causes. Checking the dataset distribution would be necessary to substantiate them.

### Test Conditions and Threshold Comparison

Live checks covered a normal/simple background, a cluttered background, and different confidence thresholds. Saved screenshots and video recordings document the tested scenes. The application displays FPS and the selected confidence threshold during inference.

| Test | Observed result |
|---|---|
| Pen and pencil detection | Both pens and pencils could be detected. |
| Object orientation | Angled objects were detected more readily; upright/straight orientations sometimes took longer to produce a detection. |
| Background clutter | Tested additional objects were generally not identified as writing tools. Some dark background regions nevertheless produced false positives. |
| Higher confidence threshold | Pens and pencils remained detectable, but boxes sometimes appeared later because detections needed to exceed the higher threshold. |


## Limitations and Improvements

- Small dataset and validation split; further independent testing is needed.
- Orientation-dependent detection and false positives on some dark regions.
- Phone training images differ from webcam frames in sharpness, perspective, and exposure.
- CPU speed depends on hardware and must be measured locally.
- No temporal box smoothing is implemented in the baseline.

Future work includes more upright-object examples, more pen colors, negative examples containing dark non-target objects, webcam-collected training images, and temporal smoothing to reduce flicker.

