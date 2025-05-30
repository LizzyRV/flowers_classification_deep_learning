# 🌸 Flower Image Classifier with PyTorch

This project is part of the [AI Programming with Python Nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089) by Udacity. It consists of an **image classifier** trained to recognize different species of flowers using deep learning with **PyTorch**.

---

## Project Structure

```
flowers_classification/
│
├── cat_to_name.json
├── checkpoint.pth            # Not included in this repository.
├── predict.py
├── train.py
├── image_classifier.ipynb
├── image_classifier.html
├── .gitignore
└── README.md
```

---

## Features

* Trains an image classifier using a **pre-trained MobileNetV2** model.
* Fine-tuned with a custom classifier for **102 flower classes**.
* Supports training on GPU (if available).
* Saves and loads model checkpoints.
* Predicts the top **K** most probable classes for a given image.
* Provides command-line interfaces for training and prediction.

---

## Installation

To run this project locally, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/LizzyRV/flowers_classification
cd flowers_classification
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# Or: source venv/bin/activate  # On Mac/Linux
```

### 3. Install required packages

```bash
pip install torch torchvision matplotlib numpy pillow
```

---

## Dataset

This project uses the [102 Category Flower Dataset](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).

**⚠️ The dataset is NOT included in this repository.** To use the project, manually download and extract the dataset into a folder named `flowers/` with the following structure:

```
flowers/
├── train/
├── valid/
└── test/
```

Add `flowers/` to your project folder before running training or prediction.

---

## Model Architecture

* **Base model**: `mobilenet_v2` pretrained on ImageNet
* **Classifier**: Fully connected layers + ReLU + Dropout + LogSoftmax
* **Loss**: Negative Log Likelihood (`NLLLoss`)
* **Optimizer**: Adam

---

## How to Use

### Train the model

```bash
python train.py flowers --gpu --epochs 3 --learning_rate 0.003 --save_dir .
```

### Predict with the trained model

```bash
python predict.py flowers/test/1/image_06743.jpg --checkpoint checkpoint.pth --category_names cat_to_name.json --top_k 5 --gpu
```

---

## Note about `checkpoint.pth`

The `checkpoint.pth` file is **NOT included in this GitHub repository**.

To generate it, run the training script as described above.


---

## Deliverables Summary

* `image_classifier.ipynb` — Notebook with full model development
* `image_classifier.html` — HTML version of the notebook (required by Udacity)
* `train.py` — CLI script for training
* `predict.py` — CLI script for prediction
* `cat_to_name.json` — Label to name mapping
* `checkpoint.pth` — Trained model checkpoint (excluded from GitHub)

---

## 👩‍💻 Author

**Elizabeth Rojas Vargas**
Microbiologist and Bioanalyst | Junior Data Scientist
[LinkedIn](https://www.linkedin.com/in/elizabethrojasvargas/)
[GitHub](https://github.com/LizzyRV)

---

## 🏁 License

This project is part of the Udacity AI Programming with Python Nanodegree and is intended for educational purposes only.
