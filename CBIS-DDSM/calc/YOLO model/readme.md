```
YOLO_MODEL/
├─ data/
│  ├─ CBIS_YOLO_TRAIN_DATASET/
│  │  ├─ images/                      # Original mammography images used by YOLO
│  │  └─ labels/                      # YOLO .txt labels: `cls xc yc w h` (normalized)
│  ├─ YOLO_CROPS/                     # Detector crops generated from source images
│  ├─ YOLO_CROPS_CSV_VIZ/             # PNG visualizations produced from yolo_crops.csv
│  ├─ calc_preprocessed_clean.csv     # Master table for source images (+ metadata)
│  ├─ yolo_crops.csv                  # One row per saved crop with paths/metadata
│  ├─ yolo_crops_validation_report.csv# Per-image crop/pred validation summary
│  └─ cbis.yaml                       # Ultralytics dataset config (path/train/val/nc/names)
│
├─ models/
│  ├─ yolo_best_model.pt              # Selected YOLO weights for inference
│  ├─ yolov8s.pt                      # base pretrained weights
│  └─ yolov11n.pt                     # base pretrained weights
│
├─ CBIS_YOLO_PROJECT_output/
│  └─ yolov8s_optimized3/             # Example training run directory (Ultralytics auto)
│     ├─ weights/
│     │  ├─ best.pt                   # Best checkpoint for this run (use for inference)
│     │  └─ last.pt                   # Last epoch checkpoint (if present)
│     ├─ results.csv                  # Per-epoch metrics (losses/AP/etc.)
│     ├─ results.png                  # Summary curves (loss/precision/recall)
│     ├─ confusion_matrix.png
│     ├─ confusion_matrix_normalized.png
│     ├─ train_batch*.jpg             # Training batch previews with labels
│     ├─ val_batch*_labels.jpg        # Val previews (labels)
│     ├─ val_batch*_pred.jpg          # Val previews (predictions)
│     └─ args.yaml                    # Run configuration captured by Ultralytics
│
├─ notebooks/
│  ├─ YOLO_mode_train.ipynb           # Train YOLO using `data/cbis.yaml`
│  ├─ yolo_data_generation.ipynb      # Build dataset, generate crops & CSVs
│  └─ yolo_performance_analysis.ipynb # Evaluate, visualize, and report metrics


