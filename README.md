# Solar Eclipse Image Classification

Automatic classification of solar eclipse phases (total, partial, annular) from **Eclipse Megamovie** images (Great American Eclipse, 2017). The project implements two complementary approaches: a **histogram-of-gradients–style retrieval/classifier** and a **PyTorch CNN**, supported by a parallel **pre-processing** pipeline.

---

## Goal

Build a system that recognises the eclipse phase—or flags non-eclipse images—despite **varying conditions, noise, and framing** typical of “in-the-wild” data. The project is aligned with the Kaggle challenge “Categorising Solar Eclipse Phases” (Megamovie, 2017).

---

## Dataset

* **Eclipse Megamovie (2017)**: diverse images contributed by volunteers and scientists during the Great American Eclipse. The split used here comprises **495 training** and **140 test** images with a label↔class mapping (CSV/JSON).
* Key challenges: **heterogeneous quality and devices**, **off-topic images**, and **varying sizes/orientations**.

---

## Method

### 1) Pre-processing pipeline

To reduce variability and noise prior to feature extraction and training:

1. **Grayscale conversion**
2. **Noise reduction** (Gaussian filter)
3. **Normalisation** to [0, 1]
4. **Eclipse localisation** (Otsu threshold to isolate the solar disc)
5. **Crop** around the detected region
6. **Resize** to a fixed input size

The pipeline is **parallelised** with `concurrent.futures`, reducing end-to-end time from about **379.42 s** to **97.14 s** on the processed set.

### 2) Histogram-based baseline (image retrieval → classification)

* Compute **Dx/Dy** histograms of **Gaussian derivatives** along x and y to capture Sun/Moon edges. Compare images via distance/similarity metrics (Intersection, L2, Chi-square).
* Label assignment: for each image, take the label of the **most similar** training image (excluding self-matches). With **Intersection** and **16 bins**, the best accuracy observed is **76.36%**.

### 3) CNN (PyTorch)

* Architecture inspired by the official PyTorch tutorial, **extended** with an extra convolutional block and an additional **fully connected** layer; softmax over classes at the output.
* Without pre-processing: longer training (~**77 min** on a GPU) and weaker performance. With the pipeline enabled and a reduced **batch size** (**64 → 4**), we observe **better generalisation**, **lower times** (≈ **2m46s** vs **7m12s**), and **more balanced** confusion matrices.

---

## Results (summary)

* **Histogram baseline (Dx/Dy)**: **76.36%** accuracy with Intersection @ 16 bins; other settings are stable but slightly lower (see the report’s table).
* **CNN**: stronger results after pre-processing, with clear **overfitting reduction** (see classification report and confusion matrix in the report).

For full details, figures, and metrics, see **Report_long.pdf**.

---

## Known limitations

* The dataset includes **irrelevant images** and pronounced **heterogeneity**; pre-processing mitigates but does not fully remove corner cases.
* The **CNN** benefits strongly from a GPU; without pre-processing performance degrades and training time increases.

---

## References and credits

* **Project report**: “Solar Eclipse Image Classification”, *Sapienza University of Rome* — see **Report_long.pdf** (includes bibliography, dataset link, PyTorch/OpenCV references).
* Dataset/context: **Eclipse Megamovie / Kaggle**; libraries: **PyTorch** and **OpenCV** as cited in the report.

**Authors**: Luca Moresca, Nicholas Suozzi, Valerio Santini — contact details in the report.

---

### Possible extensions

* Targeted data augmentation (rotations, cutout near the limb).
* Focal loss / label smoothing for class imbalance.
* Pre-trained backbones (ResNet/MobileNet) with fine-tuning on Megamovie.
* Repeated splits and cross-validation; metrics beyond accuracy (macro-F1).
