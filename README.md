# FaceID-CNN

This project implements a **high-accuracy facial recognition system** using a **convolutional neural network (CNN)** trained with **triplet loss** to generate 256-dimensional face embeddings. It includes:

- A model training pipeline using labeled face images
- A live camera-based recognition demo
- Face detection using **MTCNN**
- Embedding matching with a customizable threshold

---

## Project Files

| File                        | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `FaceID_classes_and_functions.py` | Defines the CNN model, training agent, dataset loader, and face embedding logic |
| `FaceID_live.py`            | Loads the trained model and runs live webcam-based face recognition        |
| `face_database.pkl`         | Serialized face embedding database (generated during training)             |
| `FaceID_Agent.pth`          | Serialized trained model (saved and loaded via `torch.save`/`load`)        |
| `LICENSE`                   | MIT License                                                                 |

---
