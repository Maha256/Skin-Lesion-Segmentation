# 🧬 Skin Lesion Segmentation using U-Net

This project implements a **U-Net-based convolutional neural network (CNN)** for binary segmentation of skin lesions in medical images.

The model architecture follows the U-Net design, featuring encoder-decoder paths with skip connections. Custom evaluation metrics such as **IoU**, **Dice Coefficient**, **Precision**, **Recall**, and **Accuracy** are used to assess performance. A **Jaccard Distance** loss function is employed to optimize segmentation quality.

To improve generalization, the training data is augmented using **random rotations** and **horizontal flipping**. Predicted masks are visualized alongside ground truth to assess the model's ability to identify lesion regions accurately.

---

> **Note:** This project is currently under development, with training and validation experiments in progress.
