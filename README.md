# TITLE??????

## Project Overview

This project develops a 3D variational autoencoder (VAE)-based framework for neuroimaging analysis of dementia progression using structural MRI scans (T1-weighted and T2-weighted) and the Clinical Dementia Rating – Sum of Boxes (CDR-SB) scores.

The model learns a low-dimensional latent representation of brain MRI volumes while incorporating demographic conditioning (age, sex) and stochastic regularization mechanisms. These learned representations are then used for severity-aware anomaly detection and disease progression analysis.

Unlike standard supervised classification pipelines, this approach leverages representation learning and probabilistic modeling to capture the subtle structural changes associated with early and progressive cognitive decline.
---

## Objectives

### 1. Learn a structured latent space of brain anatomy
Train a 3D VAE to encode MRI volumes into a compact latent representation that preserves clinically meaningful variation.

### 2. Incorporate demographic conditioning
Integrate age and sex information into the generative process to reduce confounding effects and improve subject-specific modeling.

### 3. Model uncertainty in feature propagation
Use stochastic (Bayesian) skip connections to capture variability in anatomical reconstruction and improve model robustness.

### 4. Capture disease-related structure without direct supervision
Learn representations that naturally separate healthy and diseased populations using reconstruction and distributional constraints rather than explicit classification labels.

### 5. Quantify disease severity in latent space
Evaluate whether learned representations correlate with clinical severity (CDR-SB) using:
- kNN-based anomaly scoring
- Group-wise severity stratification
- Monotonic progression analysis

### 6. Validate clinical relevance of latent embeddings
Assess whether anomaly scores reflect disease progression using:
- ROC-AUC (binary and ordinal formulations)
- Cohen’s d effect size
- One-vs-rest separability
---

## Dataset

**Source:** OASIS-3 dataset
> OASIS-3 is a retrospective dataset of 1378 participants collected over 30 years, including cognitively normal individuals and patients at various stages of cognitive decline.

**MRI Modalities Used:**

* T1-weighted MRI (T1w)
* T2-weighted MRI (T2w)

**Additional Data Used:**

* Clinical Dementia Rating—Sum of Boxes (CDR-SB) Scores
* Demographics (age, sex)
  
### Age and Clinical Label Alignment

Participant age at each MRI session was calculated by adding the participant's age at study entry and the number of days elapsed from study entry to the MRI session. Because CDR assessments were performed throughout the study, the CDR-SB score used for each MRI was selected as the closest available assessment to the MRI session date. This ensures that the clinical label is as temporally aligned as possible with the imaging data, even if the MRI and CDR assessment did not occur on the same day.

### Inclusion Criteria

* 3 Tesla (3T) field strength MRI T1w and T2w
* Isotropic voxel sizes

### Exclusion Criteria

* MRIs with incorrect voxel spacing
* 1.5 Tesla (1.5T) field strength MRI
* Raw NIfTI files with anisotropic voxels, since Synthstrip frequently fails even on resized volumes

---

## Data Processing

### Pipeline Overview

All MRI volumes were processed using a standardized pipeline to ensure consistent intensity distribution, spatial resolution, and anatomical alignment across MRIs. This is critical for learning meaningful representations in 3D deep learning models.

Each MRI underwent the following steps:

1. Skull stripping using SynthStrip  
2. Orientation normalization to RAS  
3. Bias field correction (N4)  
4. Resampling to uniform voxel spacing  
5. Intensity normalization  
6. Center cropping / padding to fixed shape  
7. Conversion to NumPy format  

---

### 1. Skull Stripping

Non-brain tissue (e.g., skull, scalp, neck) was removed using **SynthStrip** from FreeSurfer to isolate brain tissue and reduce irrelevant variation. Synthstrip takes a NIfTI volume as input and outputs a skull-stripped NIfTI volume. Scans that failed skull stripping were excluded from further processing, and their paths were saved.

---

### 2. Orientation Normalization

All MRIs were converted to **RAS (Right–Anterior–Superior)** orientation using `nibabel.as_closest_canonical` to ensure consistent anatomical alignment across subjects
and prevent axis inconsistencies during model training.

---

### 3. Bias Field Correction

We applied **N4 bias field correction** using SimpleITK to correct low-frequency intensity inhomogeneities. This is important for MRI data due to scanner field artifacts, as it improves intensity consistency across scans.

---

### 4. Resampling to Uniform Voxel Spacing

All volumes were resampled to a fixed voxel spacing: `(160, 192, 160)`. If the volume was larger, it was center-cropped; if it was smaller, it was zero-padded. This ensures consistent tensor dimensions and compatibility with 3d CNNs training. 

---

### 7. Final Formatting

Each processed MRI is stored as:

- Shape: `(1, 160, 192, 160)`  
- Data type: `float32`  
- Format: `.npy`  

The channel dimension is added to support PyTorch 3D convolutional models.

---

### Quality Control

- Failed preprocessing steps (e.g., skull stripping errors) were logged and excluded  
- Visual inspection was performed on sample scans to verify anatomical integrity  
- Central slices (sagittal, coronal, axial) were used for manual sanity checks
  
---

## 📊 Dataset Overview

The dataset consists of structural MRI scans from the OASIS-3 cohort, preprocessed into `.npy` volumes. Data is split into T1-weighted (T1w) and T2-weighted (T2w) modalities, with subject-level grouping to prevent leakage.

---

## Dataset Statistics

### T1-weighted (T1w)
- Number of subjects: [TODO]
- Number of scans: [TODO]
- CDR-SB distribution: [TODO]
- Age distribution: [TODO]
- Sex distribution: [TODO]

### T2-weighted (T2w)
- Number of subjects: [TODO]
- Number of scans: [TODO]
- CDR-SB distribution: [TODO]
- Age distribution: [TODO]
- Sex distribution: [TODO]

---

## Dataset Splits

### Training Dataset (Healthy Cohort)
Healthy control subjects used for unsupervised representation learning.

- T1w healthy training set: 501
- T2w healthy training set: 478
- One scan per subject selected to avoid subject-level bias
- Used for VAE training (reconstruction + KL + contrastive objectives)

---

### Max-CDR Test Dataset (Disease Severity Evaluation)
One MRI per subject was selected based on the maximum recorded CDR-SB score.

- T1w max-CDR test set: 459
- T2w max-CDR test set: 450
- Used for:
  - Anomaly scoring evaluation
  - Disease severity separation
  - ROC-AUC and group-wise comparisons

---

### Progression Dataset (Longitudinal Analysis)
Subjects with multiple timepoints used for disease progression modeling.

- T1w progression set: 50
- T2w progression set: 52
- Used for:
  - Temporal consistency of latent embeddings
  - Progression trajectory analysis
  - Monotonicity of anomaly scores over time

---

## Data Loading Pipeline

All MRI data is loaded using a custom PyTorch Dataset class (`MRIDataset`) that handles:
- MRI volume loading from `.npy` files
- Intensity normalization
- Metadata integration (age, sex)
- Optional clinical label loading (CDR-SB)

This dataset is used for both training (unsupervised VAE learning) and evaluation (CDR-SB-based analysis).

---

### Input Data Structure

Each sample consists of:

- MRI volume: `.npy` file containing a 3D brain scan
- Age: scalar float
- Sex: binary encoding (0 = male, 1 = female)
- CDR-SB (optional): clinical severity score

---

### Normalization

Each MRI volume is normalized **per-scan** using min-max scaling:

- If `max - min > EPS`: x = (x - min) / (max - min)
- Else: x = 0

This ensures all inputs are scaled to `[0, 1]`.

---

### Dataset Behavior

The dataset behaves differently depending on whether labels are provided:

#### Unlabeled mode (training)
Returns: (volume, age, sex)

#### Labeled mode (evaluation/analysis)
Returns: (volume, age, sex, cdr)


---

### Class Definition Summary

The dataset implements:

- `__len__`: returns number of MRI scans
- `__getitem__`: loads and processes a single subject sample

All outputs are converted to `torch.float32` tensors for compatibility with the VAE training pipeline.

---

## Model Architecture

This work implements a 3D Variational Autoencoder (VAE) for structural MRI reconstruction with demographic conditioning and stochastic skip connections. The architecture is fully convolutional and consists of four tightly coupled components: a demographic embedding network, a 3D CNN encoder, a probabilistic latent space, and a 3D CNN decoder with Bayesian skip connections. 

---

### 1. Demographic Embedding

Demographic variables are used to condition the generative process.

Inputs:
- age ∈ [B] 
- sex ∈ [B]

These are combined into:
- demo ∈ [B, 2] via stacking `[age, sex]`

The stacked vector is passed through an MLP:

- Linear(2 → 32)
- ReLU
- Linear(32 → embed_dim)
- ReLU

Output:
- demographic embedding ∈ [B, embed_dim] (default embed_dim = 32)

This embedding is later injected into all stochastic skip connections in the decoder.

---

### 2. Encoder (3D Convolutional VAE Encoder)

The encoder maps MRI volumes into a latent Gaussian representation.

Input:
- x ∈ [B, 1, D, H, W]

The convolutional backbone performs progressive downsampling:

- Conv3D(1 → 32, stride=2) → BatchNorm3D → ReLU
- Conv3D(32 → 64, stride=2) → BatchNorm3D → ReLU
- Conv3D(64 → 128, stride=2) → BatchNorm3D → ReLU
- Conv3D(128 → 256, stride=2) → BatchNorm3D → ReLU

Skip features are stored:
- s1, s2, s3

Final feature map:
- s4 ∈ [B, 256, 10, 12, 10]

This is flattened into:
- 256 × 10 × 12 × 10

Two linear layers produce the latent distribution:
- μ ∈ [B, latent_dim]
- logvar ∈ [B, latent_dim]

Stability constraint:
- logvar is clamped to [-4, 4]

---

### 3. Latent Space (Reparameterization)

Latent sampling follows the VAE reparameterization trick:

σ = exp(0.5 × logvar)  
ε ~ N(0, I)  
z = μ + ε ⊙ σ  

Post-processing:
- NaNs are replaced with 0
- L2 normalization is applied:

z = z / (||z|| + 1e-8)

This stabilizes latent magnitude during training.

---

### 4. Bayesian Skip Connections

Skip connections inject uncertainty into decoder feature propagation.

Each skip connection takes:
- feature map x ∈ [B, C, D, H, W]
- demographic embedding demo ∈ [B, demo_dim]

Processing steps:

1. Expand demographic embedding:
- demo → [B, demo_dim, D, H, W]

2. Concatenate:
- [x, demo] along channel dimension

3. Predict distribution:
- μ = Conv3D(C + demo_dim → C)
- logvar = Conv3D(C + demo_dim → C)
- logvar clamped to [-4, 4]

4. Sample stochastic output:
σ = exp(0.5 × logvar)  
ε ~ N(0, I)  
out = μ + ε × σ  

5. Numerical stability:
- NaNs replaced with 0

---

### 5. Decoder (3D Convolutional Decoder)

The decoder reconstructs MRI volumes from latent and skip information.

#### 5.1 Latent expansion
z ∈ [B, latent_dim] is projected via:

Linear(latent_dim → 256 × 10 × 12 × 10)  
→ reshape to [B, 256, 10, 12, 10]

---

#### 5.2 Upsampling pipeline (mirrors encoder)

- ConvTranspose3D(256 → 128) → BN → ReLU + BayesianSkip(s3, demo)
- ConvTranspose3D(128 → 64) → BN → ReLU + BayesianSkip(s2, demo)
- ConvTranspose3D(64 → 32) → BN → ReLU + BayesianSkip(s1, demo)
- ConvTranspose3D(32 → 1) → Sigmoid

Output:
- reconstructed MRI ∈ [B, 1, D, H, W] in range [0, 1]

---

### 6. Full Forward Pass

1. Encode MRI:
- μ, logvar, s1/s2/s3, bottleneck = Encoder(x)

2. Sample latent:
- z = μ + ε ⊙ σ (with normalization)

3. Compute demographics:
- demo = MLP(age, sex)

4. Decode:
- recon = Decoder(z, skips, demo)

Outputs:
- recon
- μ
- logvar
- z
- skips
- bottleneck

---

### 7. Loss Functions

#### 7.1 Reconstruction Loss
Voxel-wise L1 loss:

L_recon = ||x − recon||₁

---

#### 7.2 KL Divergence
Regularization toward standard normal prior:

L_KL = -0.5 × mean(1 + logvar − μ² − exp(logvar))

---

#### 7.3 Patch Contrastive Loss

Encourages structured similarity in feature maps.

Steps:
- AdaptiveAvgPool3D → [B, C, 5, 6, 5]
- Flatten into patch descriptors
- L2 normalization
- Compute similarity matrix: sim = featᵀ × feat
- Remove identity bias
- Penalize deviation: L_patch = mean((sim − I)²)

---

### 8. Total Objective

Final training loss:

L = L_recon + β × L_KL + λ × L_patch

Where:
- β controls KL regularization strength
- λ controls patch-level contrastive regularization strength

---

## VAE Training Procedure

The Variational Autoencoder (VAE) is trained using a multi-objective optimization framework that combines reconstruction learning, probabilistic regularization, and representation consistency constraints. The training process is implemented in the `train_vae` function, which also returns a per-epoch loss history for downstream analysis and convergence tracking.

---

### Objective

The model is optimized to learn a structured latent representation of 3D MRI brain volumes while incorporating:
- Reconstruction fidelity
- Latent space regularization via KL divergence
- Stochastic representation consistency via contrastive patch-level constraints

---

### Loss Function

For each batch, the total loss is computed as:

- **Reconstruction loss**: measures voxel-level similarity between input MRI and reconstruction  
- **KL divergence loss**: enforces latent distribution regularization toward a standard normal prior  
- **Patch-level contrastive loss**: encourages consistent anatomical representation across latent bottleneck features  

The final objective is: L = L_recon + β * L_KL + λ_patch * L_contrastive

where:
- β controls latent regularization strength
- λ_patch controls patch-level stochastic consistency

---

### Training Procedure

The model is trained over multiple epochs using minibatch optimization:

1. Input MRI volumes along with demographic variables (age, sex)
2. Forward pass through the VAE:
   - Produces reconstruction
   - Produces latent mean (μ) and variance (log σ²)
   - Produces latent sample z and intermediate bottleneck features
3. Compute composite loss (reconstruction + KL + contrastive)
4. Backpropagate gradients
5. Apply gradient clipping to stabilize training
6. Update model parameters using Adam optimizer

---

### Stability Mechanisms

To ensure stable optimization in high-dimensional 3D MRI space, the following techniques are used:
- Gradient clipping (`grad_clip = 5.0`) to prevent exploding gradients
- KL weighting (`β`) to control posterior collapse
- Patch-level contrastive loss to stabilize anatomical representation learning

---

### Outputs

The function returns:
- A list of average epoch losses
- Console logs of training progress per epoch

Each epoch loss is computed as: avg_loss = total_loss / number_of_batches

---

## Evaluation


---

## Limitations

* **Class imbalance across disease severity**  
  The dataset is heavily skewed toward cognitively normal subjects (CDR-SB = 0).

* **Weak supervision for disease structure learning**  
  Disease separation emerges implicitly from reconstruction loss, KL regularization, and latent structure constraints rather than explicit diagnostic labels. While this supports unsupervised discovery of structure, it may limit precise clinical boundary formation in latent space.

* **Residual confounding despite demographic conditioning**  
  Age and sex are incorporated into the generative process; however, other confounders (scanner differences, acquisition protocols, and site effects) may still influence latent representations.

* **Stochasticity in skip connections and latent sampling**  
  The use of stochastic (Bayesian-style) components improves robustness but introduces variance in reconstructions and downstream embeddings, which may affect reproducibility and stability of individual-level interpretations.

* **Patch-level contrastive objective sensitivity**  
  The patch contrastive loss improves anatomical consistency but introduces an additional hyperparameter-sensitive objective that may affect training stability and requires careful tuning across datasets.

* **Proxy-based clinical evaluation**  
  Clinical relevance is assessed indirectly using CDR-SB correlation, kNN-based anomaly scoring, and group-wise separability. These proxies may not fully reflect diagnostic decision-making or clinical progression pathways.

---

## References

* OASIS-3 dataset!!!!
* SynthStrip (FreeSurfer)!!!!

---
