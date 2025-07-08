# opencv_ocr

# 🚗 Indian License Plate Detection using YOLOv8 + OpenCV

A real-time vehicle license plate detection system using **YOLOv8** and **OpenCV**, specifically optimized for **Indian license plates**. This project demonstrates a full pipeline from video input to detection, extraction, and visualization of license plates using object detection.

---

## 🎯 Objectives

- 🎥 Detect license plates from real-time video or images.
- 🧠 Use YOLOv8 for high-accuracy detection.
- 🔍 Use OpenCV to read, preprocess, draw bounding boxes, and crop detected plates.
- 🇮🇳 Focused on Indian plate patterns and region styles.

---

## 🏗️ Pipeline Overview

```mermaid
flowchart LR
A[Video Frame or Image] --> B[Preprocessing (Resize, Color)]
B --> C[YOLOv8 Detection]
C --> D[Bounding Box Extraction]
D --> E[Crop License Plates]
E --> F[Save or OCR (Optional)]
