# CV Knowledge Distillation

Compact computer vision classification using offline response-based knowledge distillation with PyTorch.

---

# Project Overview

This project was developed as a real-world machine learning engineering workflow rather than a standalone notebook experiment.

The goal is to build a compact deployable image classification model under **500,000 parameters** for a production-line computer vision system.

The project uses:

* A compact deployable student CNN
* A large pretrained teacher model
* Offline response-based knowledge distillation
* Labeled and unlabeled image data
* Config-driven experimentation
* Modular project structure

Only the compact student model is submitted for deployment.

---

# Repository Structure

```text
cv-knowledge-distillation/
│
├── configs/
│   ├── student.yml
│   ├── teacher.yml
│   └── distill.yml
│
├── src/
│   ├── datasets.py
│   ├── models.py
│   └── utils.py
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── predictions/
│
├── docs/
├── train_baby.py
├── train_teacher.py
├── distill.py
├── requirements.txt
├── report_notes.md
└── README.md
```

---

# Machine Learning Pipeline

The project follows this workflow:

```text
1. Train baseline compact student
2. Train pretrained teacher model
3. Generate teacher logits on unlabeled images
4. Distill teacher knowledge into compact student
5. Compare multiple temperature values
6. Compare architecture and regularization strategies
7. Select best deployable student model
```

---

# Student Architecture

The deployable student model uses a compact MobileNet-style CNN.

Features:

* Convolution blocks
* Depthwise separable convolutions
* Batch normalization
* ReLU activations
* Dropout
* Global average pooling

The final deployable student model remains below the assignment limit:

```text
317,479 parameters
```

---

# Teacher Architecture

Teacher model:

* EfficientNet-B0
* ImageNet pretrained
* Fine-tuned for 7-class classification

Teacher improvements:

* AdamW optimizer
* Weight decay
* Label smoothing
* Longer training schedule

The teacher model is used only during training and is NOT submitted.

---

# Distillation Type

This project uses:

```text
Offline response-based knowledge distillation
```

The student learns from:

1. Hard labels using cross-entropy loss
2. Teacher soft logits using KL divergence loss

Combined objective:

```text
total_loss =
(1 - alpha) * CE_loss
+
alpha * KD_loss
```

---

# Training Environment

The project was developed using:

* PyTorch
* Apple Silicon local development (MPS)
* Google Colab GPU acceleration
* GitHub version control

Due to computational requirements for the pretrained teacher model and temperature sweep experiments, large experiments were executed using Google Colab.

---

# Experiment Log

---

# Experiment 1

* Distillation temperature: T = 4
* Alpha: 0.7
* Student parameters: 317,479
* Validation accuracy: 36.71%
* Validation loss: 2.0879

Goal:
Evaluate whether temperature-scaled soft targets improve validation accuracy over the baseline student and verify the end-to-end offline knowledge distillation pipeline.

Result:
The full offline knowledge distillation pipeline executed successfully. However, validation accuracy remained relatively low.

---

# Experiment 2

* Distillation temperature: T = 2
* Alpha: 0.5
* Student parameters: 317,479
* Validation accuracy: 36.71%
* Validation loss: 2.0130

Goal:
Evaluate whether a lower temperature and reduced distillation weight improve learning stability.

Result:
Validation accuracy remained unchanged, but validation loss improved slightly.

---

# Experiment 3

* Distillation temperature: T = 1
* Alpha: 0.5
* Student parameters: 317,479
* Validation accuracy: 36.71%
* Validation loss: 2.0422

Goal:
Evaluate whether sharper teacher targets improve student performance.

Result:
Sharper teacher distributions did not improve validation accuracy.

---

# Experiment 4

* Distillation temperature: T = 2
* Alpha: 0.5
* Student parameters: 317,479
* Validation accuracy: 45.57%
* Validation loss: 1.8958
* Additional change: Data augmentation enabled

Goal:
Evaluate whether data augmentation improves generalization during distillation.

Result:
Adding augmentation significantly improved validation accuracy and reduced validation loss.

---

# Experiment 5

* Distillation temperature: T = 2
* Alpha: 0.5
* Student parameters: 431,447
* Validation accuracy: 44.30%
* Validation loss: 1.7872
* Additional change: Increased student capacity

Goal:
Evaluate whether increasing student capacity improves validation accuracy.

Result:
Increasing model capacity improved loss calibration slightly but did not improve validation accuracy over the smaller student.

---

# Experiment 6

* Distillation temperature: T = 2
* Alpha: 0.5
* Student parameters: 317,479
* Validation accuracy: 45.57%
* Validation loss: 1.8469
* Additional change: Improved teacher training with AdamW, weight decay, label smoothing, and longer training

Goal:
Evaluate whether improving teacher calibration improves student distillation quality.

Result:
Validation accuracy matched the best previous experiment while validation loss improved further.

---

# Experiment 7

* Distillation temperatures: T = [2, 4, 6, 8]
* Alpha: 0.7
* Student parameters: 317,479
* Additional change: Increased KD weighting with improved teacher

Goal:
Evaluate whether giving more weight to teacher soft targets improves student performance after teacher calibration improvements.

Results:

| Temperature | Validation Accuracy | Validation Loss |
| ----------- | ------------------: | --------------: |
| 2           |              40.51% |          1.7418 |
| 4           |              39.24% |          1.7486 |
| 6           |              36.71% |          1.7391 |
| 8           |              36.71% |          1.7324 |

Result Summary:
Increasing alpha from 0.5 to 0.7 reduced validation accuracy across all tested temperatures. The best result achieved was 40.51% at T = 2, which remained below the previous best result of 45.57% obtained using alpha = 0.5.

Observation:
The compact student benefited more from a balanced combination of hard labels and teacher supervision rather than relying too heavily on teacher soft targets.

---

# Temperature Analysis

Multiple temperature values were evaluated:

| Temperature | Alpha | Validation Accuracy | Validation Loss |
| ----------- | ----: | ------------------: | --------------: |
| 1           |   0.5 |              36.71% |          2.0422 |
| 2           |   0.5 |              45.57% |          1.8469 |
| 2           |   0.7 |              40.51% |          1.7418 |
| 4           |   0.7 |              39.24% |          1.7486 |
| 6           |   0.7 |              36.71% |          1.7391 |
| 8           |   0.7 |              36.71% |          1.7324 |

Observations:

* Lower temperatures produced sharper teacher distributions.
* Higher temperatures produced softer probability targets.
* T = 2 produced the strongest balance between calibration and classification accuracy.
* Temperature scaling affected confidence calibration more than raw accuracy.
* Increasing alpha too aggressively reduced validation accuracy, suggesting the compact student still benefits strongly from hard-label supervision.

---

# Best Configuration

Best observed configuration:

```yaml
Student:
  Parameters: 317,479
  Architecture: MobileNet-style compact CNN

Distillation:
  temperature: 2
  alpha: 0.5

Teacher:
  EfficientNet-B0
  AdamW
  label_smoothing: 0.1
  epochs: 20

Training:
  Data augmentation enabled
```

---

# Running the Project

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Train baseline student

```bash
python train_baby.py
```

---

## 3. Train teacher

```bash
python train_teacher.py
```

Outputs:

```text
outputs/checkpoints/teacher_efficientnet_b0.pth
outputs/predictions/teacher_soft_labels.npy
outputs/predictions/teacher_filenames.txt
```

---

## 4. Run distillation

```bash
python distill.py
```

Outputs:

```text
model.pt
outputs/logs/temperature_results.csv
```

---

# Key Learnings

Important observations from experimentation:

* Data augmentation produced the largest performance improvement.
* Increasing student capacity alone did not improve validation accuracy.
* Teacher calibration improved validation loss.
* Offline response-based knowledge distillation successfully transferred knowledge into a compact deployment model.
* Small datasets require strong regularization and careful experimentation.

---

# Future Improvements

Potential future work:

* Stronger augmentation policies
* MixUp / CutMix
* EMA teacher averaging
* Better validation splits
* Feature-based distillation
* Automated hyperparameter sweeps

---

# Final Deployment Model

Final deployable model:

```text
Compact TorchScript student model
< 500,000 parameters
```

Generated as:

```bash
model.pt
```
