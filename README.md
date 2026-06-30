# Aditya Hegde

MS Artificial Intelligence @ SJSU · Computer Vision · Agentic Systems · Edge AI

---

## Work

### [SpaceDebrisResearch](https://github.com/AdityaHegde712/SpaceDebrisResearch)
Probabilistic orbital risk framework over 13.59M VCM snapshots across 19,300 tracked objects. Transitions debris tracking from deterministic Keplerian propagation to uncertainty-aware 3D probability volumes. RK4 J2 propagator + TCN sequence model predicts residual trajectory errors rather than absolute coordinates, feeding a Dynamic Probabilistic Occupancy Map for launch window optimization. Pipelines in Polars; 41 engineered features across AMR, decay rates, Harris-Priester atmospheric density, and solar flux indices. Built and diagnosed three baselines before the physics-ML hybrid pivot — including a tautological RF classifier and a TCN with GAP signal dilution.

`PyTorch` `Polars` `RK4 numerical integration` `TCNs` `irregular time series`

---

### [SimaAI_Industry_Safety](https://github.com/AdityaHegde712/Industrial-Safety-Detection)
Real-time PPE compliance and zone-based safety enforcement on SiMa.ai Modalix edge hardware (50 TOPS, 8× Cortex-A65). 10-class YOLOv8 detector with inverse-frequency class weighting to handle severe imbalance (person:vehicle ≈ 1:7.57). ByteTrack multi-object tracker enables per-track dwell-time measurement inside safety zones defined as camera-space polygons. ONNX export with static 640×640 input shape for MLSoC deployment. 18-run hyperparameter search (model size × LR × optimizer) tracked in W&B; best model: yolov8m, mAP@0.5 = 0.705.

`YOLOv8` `ByteTrack` `ONNX` `edge inference` `RTSP` `W&B`

---

### [Autoencoders-for-Compression](https://github.com/AdityaHegde712/Autoencoders-for-Compression)
Asymmetric autoencoder for learned video compression targeting aerial surveillance. Shallow encoder (depthwise separable convs) runs on-device; heavy decoder (residual conv blocks) reconstructs server-side. Factorized entropy bottleneck with learned Laplacian priors for BPP estimation. GOP=10 with I-frame direct reconstruction and P-frame residual coding; temporal coherence loss stabilizes latent space across frames. Rate-distortion training: SSIM + L1 weighted with lambda-scheduled BPP. Trained on VIRAT aerial video.

`PyTorch` `entropy bottleneck` `rate-distortion optimization` `SSIM` `depthwise separable convs`

---

## Capstone

**Agentic Perception Co-Pilot for Autonomous Vehicles** — LangGraph ReAct agent reasoning over uncertain YOLOv8/DETR detections on nuScenes data. Advising: Prof. Kaikai Liu, SJSU CMPE.

---

## Publications

- *Machine Learning-Based Space Risk Management* — IEEE ICEPES 2024 · [ieeexplore.ieee.org/document/10653497](https://ieeexplore.ieee.org/document/10653497)
- *Deep Learning Based Dementia Detection on MRI Data* — Springer ICETSS 2024 · [link.springer.com/chapter/10.1007/978-3-032-11488-4_15](https://link.springer.com/chapter/10.1007/978-3-032-11488-4_15)

---

## Stack

Python · PyTorch · FastAPI · LangGraph · Docker · Postgres · Redis · YOLOv8 · ONNX · Polars · Airflow · dbt · DuckDB

---

[adityahegde712.github.io](https://adityahegde712.github.io) · [linkedin](https://linkedin.com/in/aditya-hegde712/) · [aditya.hegde@sjsu.edu](mailto:aditya.hegde@sjsu.edu)
