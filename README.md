# Performance and Generalizability Impacts of Incorporating Location Encoders into Deep Learning for Dynamic PM2.5 Estimation

> **Paper (GIScience & Remote Sensing):** Performance and generalizability impacts of incorporating location encoders into deep learning for dynamic PM2.5 estimation
> **DOI:** [https://doi.org/10.1080/15481603.2025.2594797](https://doi.org/10.1080/15481603.2025.2594797)
> **Preprint:** [https://arxiv.org/abs/2505.18461](https://arxiv.org/abs/2505.18461)
> **Code base:** Fork/extension of [`geohai/PM2.5_CONUS_LSTM`](https://github.com/geohai/PM2.5_CONUS_LSTM)
> **Authors:** Morteza Karimzadeh · Zhongying Wang · James L. Crooks
> **Status:** Published (2025)
> **Last updated:** Feb 7, 2026

---

## 1 · Project summary

This repository contains code to reproduce experiments from our paper on **how geolocation should be incorporated in deep learning** for a **temporally dynamic, spatially heterogeneous** geospatial task: **daily surface PM2.5 estimation** over CONUS using a strong Bi-LSTM + Attention baseline (remote sensing + meteorology + ancillary variables).

A central contribution is a **systematic evaluation of three ways to incorporate location**, showing that:

* **Raw coordinates can help within-region interpolation**, but often **hurt out-of-region generalization**.
* **Pretrained location encoders (e.g., GeoCLIP)** can improve **both** accuracy and **geographic generalizability**, especially under rigorous spatial disjoint evaluation.
* Embedding fusion effectively acts like **conditioning inference on geographic priors** (“what is typical about this place?”) while the dynamic inputs (AOD, met, smoke, etc.) capture **day-to-day variation**.

![loc\_encoder\_fusion](docs/_static/Bi-LSTM_Fusion.png)

---

## 2 · Why this matters: Earth embeddings as priors for dynamic inference

Recent “**geospatial foundation models**” increasingly aim to learn reusable representations of the Earth. Location encoders like **GeoCLIP** and **SatCLIP** can be seen as producing **Earth embeddings**: a compact vector representation of a place derived from large-scale pretraining.

In our setting, those embeddings behave like a **static geographic prior**:

* The **time-series branch** learns relationships from dynamic observations (aerosols, meteorology, smoke).
* The **Earth embedding** injects stable contextual information (built environment / land use / infrastructure signals, etc., depending on the pretraining data).
* Fusion (especially **Hadamard product**) allows the model to **gate/modulate** temporal representations by place-specific context—i.e., *conditioning inference on priors* rather than memorizing raw lat/lon → PM2.5 mappings.

---

## 3 · What we tested (geolocation variants)

We compare four geolocation integration strategies in a Bi-LSTM + Attention PM2.5 estimation pipeline:

1. **No geolocation** (baseline): model must learn a global mapping from dynamic predictors → PM2.5
2. **Raw lat/lon** appended as features
3. **Sinusoidal lat/lon** (`sin/cos`) appended as features
4. **Pretrained location encoder embeddings** (GeoCLIP; plus ablations with SatCLIP), fused with the temporal representation

   * Fusion methods: **Hadamard product** vs **concatenation** (ablation)

---

## 4 · Main findings (from the paper)

### Within-Region (WR) performance: location adds interpolative value

When test data are spatially in-distribution (random split; spatial holdout split), adding location generally helps:

* **Raw lat/lon** and **sin/cos** improve performance vs. no-location in WR settings.
* **GeoCLIP embeddings + Hadamard fusion** yield the strongest WR improvements, including improved stability across splits.

### Out-of-Region (OoR) performance: naive coordinates can harm generalization

Under **checkerboard** spatial partitions (disjoint train/test regions), we find a clear pattern:

* **Raw lat/lon (and sin/cos)** often *degrade* OoR generalization, consistent with models overfitting to region-specific location–target associations.
* **GeoCLIP embeddings** (Earth embeddings) are consistently competitive and often best, because they provide transferable geographic context without letting the downstream model “cheat” via direct coordinates.

### Fusion matters

Ablations show:

* **Hadamard fusion** outperforms concatenation for location encoders in this task.
* **GeoCLIP > SatCLIP** in our experiments, plausibly reflecting differences in what each encoder’s pretraining data captures (human-centric Flickr imagery vs. Sentinel-2 imagery).

### Qualitative behavior (important nuance)

GeoCLIP-enhanced maps show:

* Better spatial coherence in some sparsely monitored regions and stronger recovery of known urban hotspots and wildfire-related extremes,
* But also potential artifact/noise patterns in some under-sampled regions—consistent with uneven upstream pretraining coverage + high-frequency basis functions in positional encoders.

---

## 5 · Location encoders used

| Encoder     | What it is                                                                                        | Code                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **GeoCLIP** | CLIP-style location encoder aligned with geotagged Flickr imagery; outputs 512-D Earth embeddings | [https://github.com/VicenteVivan/geo-clip](https://github.com/VicenteVivan/geo-clip) |
| **SatCLIP** | CLIP-style location encoder aligned with Sentinel-2 imagery; outputs 256-D Earth embeddings       | [https://github.com/microsoft/satclip](https://github.com/microsoft/satclip)         |

---

## 6 · Reproducibility notes

* Code is organized to support training/evaluation across multiple spatial partitioning schemes (WR and OoR).
* We keep location encoders **frozen** during downstream training to evaluate them as **general-purpose priors** rather than task-adapted encoders.
* Deterministic seeds and environment management are included for repeatability.

---

## 7 · Citation

### APA (as shown on the journal page)

Karimzadeh, M., Wang, Z., & Crooks, J. L. (2025). *Performance and generalizability impacts of incorporating location encoders into deep learning for dynamic PM2.5 estimation.* **GIScience & Remote Sensing, 62(1)**. [https://doi.org/10.1080/15481603.2025.2594797](https://doi.org/10.1080/15481603.2025.2594797)

### BibTeX

```bibtex
@article{karimzadeh2025locationencoderspm25,
  title   = {Performance and generalizability impacts of incorporating location encoders into deep learning for dynamic PM2.5 estimation},
  author  = {Karimzadeh, Morteza and Wang, Zhongying and Crooks, James L.},
  journal = {GIScience \& Remote Sensing},
  volume  = {62},
  number  = {1},
  year    = {2025},
  doi     = {10.1080/15481603.2025.2594797}
}
```

