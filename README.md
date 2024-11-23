# Fatty-Liver-Level-Recognition-Using-Particle-Swarm-optimization-PSO-Image-Segmentation-and-Analysi
%% Fatty Liver Level Recognition Using Particle Swarm optimization (PSO) Image Segmentation and Analysis

- Link to the paper:
- https://ieeexplore.ieee.org/document/9960108
- DOI: [10.1109/ICCKE57176.2022.9960108](https://doi.org/10.1109/ICCKE57176.2022.9960108)

## Please cite below:
Mousavi, Seyed Muhammad Hossein, et al. "Fatty liver level recognition using particle swarm optimization (PSO) image segmentation and analysis." 2022 12th International Conference on Computer and Knowledge Engineering (ICCKE). IEEE, 2022.
# Fatty Liver Level Recognition Using Particle Swarm Optimization (PSO) Image Segmentation

This repository contains the implementation and details of the **Fatty Liver Level Recognition** system using **Particle Swarm Optimization (PSO)** and other image segmentation techniques. The proposed system efficiently segments microscopic liver images to detect fat deposits and classify fatty liver levels.

---

## Introduction
Fatty liver disease, caused by excessive fat deposits in the liver, is a significant health issue. This system leverages **Particle Swarm Optimization (PSO)** and various image segmentation techniques to detect fatty liver and classify its severity into different levels. The implementation focuses on high-resolution microscopic images with a zoom level of 200x or more.

---

## Features
- **Segmentation Methods**:
  - Particle Swarm Optimization (PSO)
  - Otsu's Thresholding
  - Watershed Algorithm
  - K-Means Clustering
- **Performance Metrics**:
  - Accuracy
  - F-Score
  - Intersection over Union (IoU)
- **Visualization**:
  - Segmented liver images
  - Fat deposit recognition and classification levels.

---

## Workflow
1. **Preprocessing**:
   - Intensity adjustment
   - Histogram equalization
   - Canny edge detection
2. **Segmentation**:
   - Segment images using PSO, Otsu, Watershed, and K-Means.
3. **Fatty Liver Level Recognition**:
   - Analyze segmented images to detect fat deposits and classify fatty liver levels based on predefined markers.

---

## Performance
The system achieved remarkable results when compared with traditional segmentation methods:
- **Average Accuracy**: 92.2%
- **Average F-Score**: 87.2%
- **Average IoU**: 90.7%

The PSO algorithm consistently outperformed Otsu, Watershed, and K-Means in all metrics.

---

## Segmentation Techniques
1. **Otsu's Thresholding**: Minimizes interclass variance for binary segmentation.
2. **Watershed Algorithm**: Region-based segmentation inspired by drainage patterns.
3. **K-Means Clustering**: Clusters image pixels based on intensity or color.
4. **Particle Swarm Optimization (PSO)**: Nature-inspired optimization algorithm, providing superior segmentation results.

---

![icon](https://user-images.githubusercontent.com/11339420/206430469-69dde48b-7787-4fb7-a1c5-e9838d86ab86.jpg)
- Link to the paper:
- https://ieeexplore.ieee.org/document/9960108
- DOI: [10.1109/ICCKE57176.2022.9960108](https://doi.org/10.1109/ICCKE57176.2022.9960108)
