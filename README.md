# Plant Disease Detection Using YOLO26

This project uses computer vision and object detection to detect strawberry plant diseases. The model is trained with YOLO26 and evaluated by comparing how different hyperparameters affect detection performance.

## Project Overview

Plant diseases can reduce crop health and agricultural productivity. This project applies a YOLO26 object detection model to identify and classify strawberry plant diseases from labeled images.

YOLO26 uses a one-stage object detection architecture. It extracts image features, combines features across different scales, and predicts bounding boxes, class labels, and confidence scores.

The default model used in this project is:

```text
yolo26s.pt
```

The baseline training configuration is:

```python
epochs = 50
batch = 16
imgsz = 640
lr0 = 0.01
```

The project compares the baseline with different hyperparameter settings, primarily changes to:

- `epochs`
- `batch`
- `imgsz`
- `lr0`

## Dataset

This project uses a strawberry disease detection dataset imported from Kaggle. The original dataset contains image files and JSON annotation files. A preprocessing script converts the JSON polygon annotations into YOLO bounding box label format.

The dataset contains:

- Approximately 2,500 labeled strawberry disease images
- 7 disease categories
- Images resized to `419 x 419`
- Object detection annotations

The disease categories are:

1. Angular Leafspot
2. Anthracnose Fruit Rot
3. Blossom Blight
4. Gray Mold
5. Leaf Spot
6. Powdery Mildew Fruit
7. Powdery Mildew Leaf

After preprocessing, the dataset is saved in YOLO format:

```text
/kaggle/working/strawberry_yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── train.cache
├── val.cache
└── data.yaml
```

## Running the Project on Kaggle

### 1. Import the Notebook

1. Go to [Kaggle](https://www.kaggle.com/).
2. Sign in to your account.
3. Click **Create**.
4. Select **New Notebook**.
5. Open the notebook menu.
6. Select **File > Upload Notebook**.
7. Upload the project `.ipynb` notebook.
8. In the notebook settings, enable GPU acceleration.

Recommended accelerator:

```text
GPU T4 x2
```

### 2. Add the Dataset

1. Open the Kaggle notebook.
2. Click **Add Input** on the right-side panel.
3. Search for the strawberry disease detection dataset.
4. Select the dataset.
5. Click **Add**.

The dataset will be available under:

```text
/kaggle/input/
```

For this project, the source dataset path used in the notebook is:

```python
SRC = Path("/kaggle/input/datasets/usmanafzaal/strawberry-disease-detection-dataset")
```

The converted YOLO dataset is written to:

```python
DST = Path("/kaggle/working/strawberry_yolo")
```

### 3. Convert the Dataset to YOLO Format

Run the preprocessing cell in the notebook. This cell:

- Creates the YOLO dataset folders
- Copies images into `images/train`, `images/val`, and `images/test`
- Converts JSON polygon annotations into YOLO `.txt` labels
- Creates `data.yaml`

The output file used for training is:

```text
/kaggle/working/strawberry_yolo/data.yaml
```

### 4. Load the YOLO26 Model

The notebook loads the YOLO26 model with:

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")
```

Other model sizes, such as `yolo26n.pt`, may appear in the working directory if they were previously downloaded or tested. The active model is determined by the file passed into `YOLO(...)`.

### 5. Train the Model

Example training command:

```python
results = model.train(
    data="/kaggle/working/strawberry_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.01,
    project="/kaggle/working/runs",
    name="strawberry_yolo26s_baseline"
)
```

Training outputs are saved to:

```text
/kaggle/working/runs/strawberry_yolo26s_baseline/
```

### 6. Validate the Model

```python
metrics = model.val(data="/kaggle/working/strawberry_yolo/data.yaml")
print(metrics)
```

### 7. Run Predictions

```python
model.predict(
    source="/kaggle/working/strawberry_yolo/images/test",
    save=True,
    conf=0.25,
    project="/kaggle/working/runs",
    name="strawberry_preds"
)
```

Prediction outputs are saved to:

```text
/kaggle/working/runs/strawberry_preds/
```

## Kaggle Working Directory

After running the notebook, the Kaggle working directory should look similar to this:

```text
/kaggle/working/
├── runs/
│   ├── detect/
│   ├── strawberry_preds/
│   └── strawberry_yolo26s_baseline/
├── strawberry_yolo/
│   ├── images/
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   ├── train.cache
│   ├── val.cache
│   └── data.yaml
└── yolo26s.pt
```
