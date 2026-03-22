# Postoperative Facial Outcome Simulation using Pix2Pix GAN

## Overview
This project focuses on generating realistic postoperative facial images using a Pix2Pix-based Generative Adversarial Network (GAN). The goal is to simulate surgical outcomes from preoperative facial images to assist in surgical planning and visualization.

---

## Model Architecture
The model is based on the Pix2Pix conditional GAN framework:

- **Generator**: U-Net architecture (Encoder-Decoder with Skip Connections)
- **Discriminator**: PatchGAN (evaluates local image patches)
- **Loss Functions**:
  - Adversarial Loss
  - L1 Reconstruction Loss

---

## Dataset
The model was trained using:

- Botox Dataset

Dataset consist of paired images:
- Preoperative (input)
- Postoperative (target)

---

## Technologies Used
- Python
- PyTorch
- NumPy
- OpenCV
- Matplotlib

---

## How to Run

### 1. Clone the Repository
```bash
git clone <your-repo-link>
cd <repo-folder>