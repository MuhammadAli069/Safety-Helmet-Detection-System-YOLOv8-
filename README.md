# Helmet Detection Dataset (Cleaned)

Derived from the Roboflow "Construction Site Safety" dataset (v30, CC BY 4.0,
https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/30),
filtered down to 3 classes for a Safety Helmet Detection project.

## Classes (3)
| id | name       |
|----|------------|
| 0  | Hardhat    |
| 1  | NO-Hardhat |
| 2  | Person     |

The original dataset had 25 classes (vehicles, machinery, PPE, etc.). All boxes
for other classes were removed and label files were rewritten with the 3
classes above re-indexed to 0/1/2.

## Selection method
- Current active dataset is copied from `cv_project_dataset_cleaned/cv_project/dataset`.
- Splits are used as provided by the cleaned dataset source.

## Final counts
- Train: 140 images
- Val: 40 images
- Test: 20 images
- Total: 200 images

## Class balance (images containing each class)
- Hardhat: 102 images / 274 boxes
- NO-Hardhat: 90 images / 136 boxes
- Person: 200 images / 444 boxes

## Structure
```
dataset/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── data.yaml
```

## Annotation format
- Labels use standard YOLO text format per object line:
  class_id x_center y_center width height
- x_center, y_center, width, and height are normalized numeric values in [0, 1].
- There is no separate "normalized" folder. Normalization is encoded directly in each .txt label line.

## Notes for training
- Update `path:` in `data.yaml` to your actual Google Drive path once uploaded
  (e.g. `/content/drive/MyDrive/cv_project/dataset`).
- This is a subset filtered from an existing labeled dataset, not manually
  annotated from scratch. If your task requires "at least 100 manually
  annotated images," check with your supervisor whether reusing/re-labeling
  a public dataset counts, or whether you need to manually verify/re-draw a
  portion of these boxes yourself and note that in your report.
