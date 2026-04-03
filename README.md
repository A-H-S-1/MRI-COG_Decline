 TITLE (TBD)

## 3D Stochastic VAE with Demographic Conditioning and Severity-Aware Representation Learning for Alzheimer’s Progression Modeling from Structural MRI ?????????

---

## Project Overview

This project develops a 3D variational autoencoder (VAE) based multi-task deep learning framework for modeling neuroanatomical structure and dementia severity using structural MRI scans (T1-weighted and T2-weighted) trained on the OASIS-3 dataset sourced from Washinton University in St. Louis.

Unlike standard VAEs which focus solely on reconstruction, this framework jointly learns:

- A probabilistic latent representation of brain anatomy  ????
- A severity-aware regression signal for the Clinical Dementia Rating–Sum of Boxes (CDR-SB) scores  
- A stochastic feature propagation mechanism via Bayesian skip connections  
- A demographic conditioned generative process on age and sex

The model is designed to capture both anatomical variation and disease progression in a shared latent space, while incorporating clinical supervision through the CDR-SB regression.

---

## Key Contributions

### 1. Multi-objective representation learning
The model jointly optimizes:
- MRI reconstruction fidelity  
- KL divergence regularization 
- CDR-SB regression  
- Patch-level representation consistency  

---

### 2. Bayesian skip connection framework
Utilizes stochastic skip connections to model uncertainty in intermediate anatomical features by learning a Gaussian distribution over skip features conditioned on both encoder activations and demographic embedding (age and sex). By conditioning on age and sex we are able to

---

### 3. Demographic-conditioned generative modeling
Age and sex are embedded via a learned MLP and injected into decoder skip pathways. This conditioning reduces confounding effects and improves subject-specific reconstruction.

---

### 4. Severity-aware latent space structuring
A dedicated CDR prediction head enforces the latent representations of each mri to encode clinically meaningful disease severity signals.

---

### 5. Patch-level anatomical consistency regularization
A patch-wise contrastive loss enforces similarity across spatial bottleneck representations to stabilize learned anatomical embeddings.

---


## Dataset

**Source:** OASIS-3 dataset
OASIS-3 is a retrospective dataset of 1378 participants collected over 30 years, including cognitively normal individuals and patients at various stages of cognitive decline.

**MRI Modalities Used:**

* T1-weighted MRI (T1w)
* T2-weighted MRI (T2w)

**Additional Data Used:**

* Clinical Dementia Rating—Sum of Boxes (CDR-SB) Scores
* Demographics (age, sex)

**Severity Groups** 
For both the T1w and T2w MRI modalities, the CDR-SB scores are heavily skewed towards 0 (cognitivly normal) and 0.5 (questionable imparement). Additionally the data experiences sparsity throughout CDR-SB scores greater than 0.5 and some higher scores are not represented in the data. Because of this, while the model is trained using the original CDR-SB scores, it is evaluated using scoring structure defined in Wyman-Chick et al.

The severity groups are as follows:

* Questionable: 0.5 to 4
* Mild: 4.5 to 9 
* Moderate: 9.5 to 15.5 
* Severe: 16 to 18 


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
### 5. Final Formatting

Each processed MRI is stored as:

- Shape: `(1, 160, 192, 160)`  
- Data type: `float32`  
- Format: `.npy`  

The channel dimension is added to support PyTorch 3D convolutional models.

---

### Quality Control and Outlier Removal

- Failed preprocessing steps (e.g., skull stripping errors) were logged and excluded  
- Filtering pipeline that identified and removed corrupted, misaligned, or anomalous MRI volumes prior to model training

---
# 1. Overview of filtering

Each MRI file was evaluated using multiple independent criteria:

1. File integrity 
2. Global intensity statistics 
3. Spatial alignment 
4. Boundary artifacts 

A scan was removed if it failed any of the above criteria.

---

# 2. File Integrity Check

Each `.npy` file was loaded and inspected for:

- loading errors (corrupt or unreadable files)  
- presence of NaN values  
- presence of infinite values  

---

# 3. Statistical Outlier Detection


- abnormally bright or dark scans  
- scans with extremely low or high contrast  
- preprocessing failures affecting intensity scaling
To identify files that were blurry, had poor contrast, or had abnormally bright or dark scans, we computed basic intensity statistics for each scan:

- mean intensity  
- standard deviation  

These were then normalized across the dataset using a Z-score:

z = (x - mean) / std

A scan is flagged as an outlier if:

- |z_mean| > threshold  
- OR |z_std| > threshold  

After testing different thresholds, we found a threshold of 2 that balanced accuracy, flagging erroneous files while ignoring MRIs with large anatomical deviations. 

---

# 4. Center-of-Mass Shift (Spatial Misalignment)

We estimate the distance of the 'mass' in the brain image from the center to avoid training or testing on images that were misaligned, had cropping issues, or failed preprocessing.

---

## Procedure

1. Convert the 3D volume into a 2D projection (mean across slices)  
2. Normalize intensities
3. Compute the center of mass of the image  
4. Measure its distance from the image center to find its shift value
5. Scans with a shift value above the 95th percentile were removed

---

# 5. Edge Artifact Detection

We measured the fraction of the image boundary that consists of zero-valued pixels in order to identify images with excessive padding, cropping errors, or failed skull stripping.

---

## Procedure

1. Extract the middle slice of the MRI  
2. Collect pixel values along the image border  
3. Compute the fraction of border pixels equal to zero
4. Removed scans with an edge ratio > threshold

Through visual testing, we found a threshold value of 0.2 was able to identify which files had the mri partially cut off by the edge. 

---

## Dataset Summary Statistics

This section summarizes the cleaned OASIS-3 T1w and T2w datasets used in this work.

---

# 1. T1-weighted MRI Dataset (T1w)

## Overview

| Metric | Value |
|--------|------:|
| Total MRIs | 2,196 |
| Age (mean ± std) | 70.62 ± 9.24 |
| Age range | 42.69 – 95.70 |

---

## Sex Distribution

| Sex | Count | Percentage |
|-----|------:|-----------:|
| Female | 1,281 | 58.3% |
| Male | 915 | 41.7% |

---

## Severity Distribution (Aggregated)

| Severity Group | Count | Percentage |
|----------------|------:|-----------:|
| 0 (Normal) | 1,639 | 74.6% |
| 1 (Mild) | 340 | 15.5% |
| 2 (Moderate) | 125 | 5.7% |
| 3 (Severe) | 10 | 0.5% |

---

## Raw CDR-SB Distribution (T1w)

| CDR-SB | Count |
|--------|------:|
| 0.0 | 1,639 |
| 0.5 | 115 |
| 1.0 | 74 |
| 1.5 | 59 |
| 2.0 | 40 |
| 2.5 | 37 |
| 3.0 | 34 |
| 3.5 | 24 |
| 4.0 | 37 |
| 4.5 | 33 |
| 5.0 | 28 |
| 5.5 | 17 |
| 6.0 | 19 |
| 7.0 | 10 |
| 8.0 | 14 |
| 8.5 | 1 |
| 9.0 | 3 |
| 10.0 | 2 |
| 11.0 | 2 |
| 12.0 | 4 |
| 13.0 | 2 |

---

# 2. T2-weighted MRI Dataset (T2w)

## Overview

| Metric | Value |
|--------|------:|
| Total MRIs | 1,362 |
| Age (mean ± std) | 70.54 ± 9.23 |
| Age range | 42.69 – 95.70 |

---

## Sex Distribution

| Sex | Count | Percentage |
|-----|------:|-----------:|
| Female | 779 | 57.2% |
| Male | 583 | 42.8% |

---

## Severity Distribution (Aggregated)

| Severity Group | Count | Percentage |
|----------------|------:|-----------:|
| 0 (Normal) | 1,013 | 74.4% |
| 1 (Mild) | 284 | 20.9% |
| 2 (Moderate) | 103 | 7.6% |
| 3 (Severe) | 2 | 0.1% |

---

## Raw CDR-SB Distribution (T2w)

| CDR-SB | Count |
|--------|------:|
| 0.0 | 1,013 |
| 0.5 | 72 |
| 1.0 | 48 |
| 1.5 | 39 |
| 2.0 | 24 |
| 2.5 | 25 |
| 3.0 | 21 |
| 3.5 | 15 |
| 4.0 | 21 |
| 4.5 | 23 |
| 5.0 | 19 |
| 5.5 | 10 |
| 6.0 | 10 |
| 7.0 | 6 |
| 8.0 | 9 |
| 9.0 | 1 |
| 10.0 | 1 |
| 11.0 | 1 |
| 12.0 | 2 |
| 13.0 | 1 |

---

# 3. Key Dataset Characteristics

- Both of these datasets may have multiple MRIs per individual; however, most have either 1 or 2, making it unfavorable for longitudinal analysis
- Strong class imbalance toward cognitively normal subjects 
- Severe dementia cases are rare for both T1w and T2w modalities
- In the Mild Severity group, the majority of T1w and T2w MRIs have a CDR-SB score of 0.5

---

# 4. Interpretation

The datasets exhibit long-tailed clinical distributions, where:
- early-stage and healthy subjects dominate
- advanced dementia stages are sparse and underrepresented

This motivates:
- stratified sampling
- severity-aware loss weighting
- latent-space regularization techniques ??????

---

## Data Splitting and Sampling Strategy

To prevent bi
??????????????????/
---

# 1. High-Level Strategy

Instead of performing a simple random train/test split at the scan level, we split the dataset using **subject-aware and severity-aware grouping**.

This ensures:

- No MRI scans from the same subject appear in both training and test sets (prevents leakage)
- Severity distributions are preserved across splits
- Rare disease stages are handled safely during stratification
- Longitudinal structure is preserved for progression analysis

---

# 2. Input Table Structure

Each row in the dataset corresponds to one MRI session and contains:

- OASIS3_id → subject identifier  
- fixed_path → preprocessed MRI volume (.npy file)  
- age_at_session → age at scan time  
- sex → binary encoding (0/1)  
- CDRSUM → clinical dementia score (CDR-SB proxy)  

From this table, we construct derived groupings used for splitting.

---

# 3. Severity Group Construction

To enable stratified sampling, we discretize continuous CDR-SB scores into categorical severity groups.

### Mapping function:

- 0 → cognitively normal  
- 0 < CDR ≤ 4 → mild impairment  
- 4 < CDR ≤ 9 → moderate impairment  
- 9 < CDR ≤ 15.5 → severe impairment  
- CDR > 15.5 → very severe impairment  

### Output:
Each sample receives:

- severity_group ∈ {0, 1, 2, 3, 4}

This is used for stratified sampling.

---

# 4. Subject-Level Grouping

Because each subject can have multiple MRI sessions, we enforce subject-level consistency.

We group all rows by:

- OASIS3_id

This allows us to:
- isolate progression trajectories
- avoid leakage across timepoints
- construct specialized evaluation subsets

---

# 5. Table Decomposition Strategy

We split the full dataset into three mutually exclusive subsets:

---

## 5.1 Progression Table (Longitudinal Subset)

This subset contains subjects who exhibit **both healthy and diseased states over time**.

### Construction steps:

For each subject:
- check if they have at least one scan with CDR = 0
- AND at least one scan with CDR > 0

If both conditions are satisfied:
- include:
  - one baseline scan (CDR = 0)
  - one maximum severity scan (highest CDR for that subject)

### Purpose:
This dataset is used for:
- disease progression analysis
- temporal consistency evaluation
- latent trajectory studies

---

## 5.2 Max-CDR Dataset (Severity Representative Set)

This subset contains **one scan per subject representing maximum disease severity**.

### Construction steps:

For each subject:
- identify scan with maximum CDR-SB score
- select that single scan

### Purpose:
This dataset is used for:
- severity-based evaluation
- anomaly detection benchmarks
- cross-sectional disease separation

---

## 5.3 Remaining Pool (General Dataset)

All scans not included in the two subsets above are placed into a residual dataset.

This pool is later used for:
- training/test splitting
- additional evaluation samples
- balancing severity distribution

---

# 6. Stratified Train/Test Split

The remaining dataset (after Table 1 and Table 2 extraction) is split into training and test sets.

---

## 6.1 Severity-aware stratification

We perform stratified splitting based on:

- severity_group

This ensures that:
- all severity levels are represented in both training and test sets
- distribution shift is minimized

---

## 6.2 Handling rare classes

Some severity groups may have very few samples.

If a severity group has:

- fewer than 5 samples

then:

- all samples from that group are assigned to the training set
- they are excluded from test stratification

This prevents:
- unstable stratified sampling
- missing-class issues in test evaluation

---

## 6.3 Final split procedure

1. Separate rare and common severity groups  
2. Apply stratified train/test split only to common groups  
3. Add rare-group samples back into training set  
4. Concatenate final training dataset  
5. Keep test dataset strictly stratified and leakage-free  

---

# 7. Final Dataset Composition

After all splitting steps, we obtain:

---

## Training Set

Includes:
- majority of MRI scans
- all rare severity cases
- subject-disjoint from test set
- used for:
  - VAE training
  - representation learning
  - CDR head training

---

## Test Set

Includes:
- stratified severity distribution
- unseen subjects
- used for:
  - reconstruction evaluation
  - severity prediction evaluation
  - anomaly scoring

---

## Progression Set

Includes:
- paired baseline + disease scans
- subjects with longitudinal progression
- used for:
  - trajectory analysis in latent space
  - monotonicity of predicted severity

---

## Max-CDR Set

Includes:
- one scan per subject at highest disease severity
- used for:
  - severity separation benchmarking
  - cross-sectional evaluation

---

# 8. Data Loading Pipeline

Each dataset is wrapped in a PyTorch Dataset (`MRIDataset`), which:

For each sample:
1. Loads `.npy` MRI volume
2. Applies per-scan min-max normalization
3. Loads demographic variables (age, sex)
4. Optionally loads CDR-SB label
5. Optionally loads severity group

---

## Output formats

### Training mode:
Returns:
- MRI volume
- age
- sex
- CDR (optional supervision)
- severity group (for stratification tracking)

### Evaluation mode:
Returns same structure, used for:
- reconstruction evaluation
- severity prediction
- latent analysis

---

# 9. Key Design Rationale

This splitting strategy is designed to ensure:

### 1. No subject leakage
All scans from a single subject remain in a single split.

### 2. Stable severity representation
All severity levels are preserved across training and testing.

### 3. Robust rare-class handling
Rare disease stages are never excluded from training.

### 4. Longitudinal compatibility
Progression analysis is explicitly separated from cross-sectional evaluation.

---

# 10. Summary

Overall, the dataset pipeline produces three complementary datasets:

- Training set → representation learning  
- Test set → evaluation and generalization  
- Progression set → longitudinal disease modeling  
- Max-CDR set → severity benchmarking  

This structure allows the model to learn both:
- cross-sectional disease structure  
- longitudinal progression dynamics  
---

### Input Data Structure

Each sample consists of:

- MRI volume: `.npy` file containing a 3D brain scan
- Age: scalar float
- Sex: binary encoding (0 = male, 1 = female)
- CDR-SB: clinical severity score
- Group: severity score based on

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

This project implements a 3D Variational Autoencoder (VAE) extended with:
- demographic conditioning (age, sex)
- stochastic (Bayesian) skip connections
- a supervised disease severity prediction head (CDR-SB regression)

The model is designed to both:
1. reconstruct 3D brain MRI scans
2. learn a structured latent space that reflects dementia severity

---

# 1. High-Level Overview

The model has four main components:

1. **Demographic Embedding Network**
   → converts age + sex into a learned conditioning vector

2. **Encoder (3D CNN)**
   → compresses MRI into a latent representation

3. **Latent Space (VAE sampling)**
   → converts encoder output into a stochastic latent vector

4. **Decoder (3D CNN + Bayesian Skip Connections)**
   → reconstructs MRI using latent vector + stochastic skip features

5. **CDR Prediction Head**
   → predicts disease severity from latent space

---

# 2. Input Format

Each input sample contains:

- MRI volume:  
  `x ∈ [1, 160, 192, 160]`

- age: scalar (float)

- sex: scalar (0 or 1)

---

# 3. Demographic Embedding

Demographics are converted into a learned feature vector.

### Input:
- age ∈ [B]
- sex ∈ [B]

### Step 1: Stack inputs
We combine them into a single vector:

- demo_input ∈ [B, 2]  
  = [age, sex]

### Step 2: MLP projection

This vector is passed through:

- Linear(2 → 32)
- ReLU
- Linear(32 → 32)
- ReLU

### Output:
- demo_embedding ∈ [B, 32]

### Purpose:
This vector is used to condition the decoder so that reconstruction depends on patient-specific factors like age and sex.

---

# 4. Encoder (3D Convolutional Network)

The encoder compresses the MRI volume into a compact representation.

---

## 4.1 Convolutional Downsampling Path

We apply 4 layers of 3D convolutions:

### Layer 1
- Conv3D: 1 → 32
- Stride: 2
- Output: [B, 32, 80, 96, 80]

### Layer 2
- Conv3D: 32 → 64
- Stride: 2
- Output: [B, 64, 40, 48, 40]

### Layer 3
- Conv3D: 64 → 128
- Stride: 2
- Output: [B, 128, 20, 24, 20]

### Layer 4
- Conv3D: 128 → 256
- Stride: 2
- Output: [B, 256, 10, 12, 10]

Each layer uses:
- BatchNorm3D
- ReLU activation

---

## 4.2 Skip Features

We store intermediate feature maps:

- s1 → output of layer 1
- s2 → output of layer 2
- s3 → output of layer 3

These are later reused in the decoder.

---

## 4.3 Flattening

Final encoder output:

- s4 ∈ [B, 256, 10, 12, 10]

We flatten it:

- flatten → [B, 256 × 10 × 12 × 10]

This becomes the input to the latent projection layer.

---

## 4.4 Latent Projection

We compute:

- μ (mean)
- logvar (log variance)

via linear layers:

- Linear(flatten → latent_dim)
- Linear(flatten → latent_dim)

Output:
- μ ∈ [B, latent_dim]
- logvar ∈ [B, latent_dim]

We clamp logvar to stabilize training:
- logvar ∈ [-10, -2]

---

# 5. Latent Space (Reparameterization Trick)

We sample the latent vector using:

### Step 1: Convert logvar → std
σ = exp(0.5 × logvar)

### Step 2: Sample noise
ε ~ N(0, I)

### Step 3: Reparameterize
z = μ + ε × σ

---

## Stability step:
- any NaNs → replaced with 0

---

## Output:
- z ∈ [B, latent_dim]

This vector represents a compressed “summary” of the brain scan.

---

# 6. Bayesian Skip Connections

This is a key part of the model.

Instead of directly passing encoder features to the decoder, we treat them as **uncertain distributions**.

---

## 6.1 Input

Each skip connection takes:

- feature map x ∈ [B, C, D, H, W]
- demographic embedding demo ∈ [B, 32]

---

## 6.2 Expand demographics

We reshape demographics into 3D form:

- [B, 32] → [B, 32, D, H, W]

Then concatenate:

- x_cat ∈ [B, C + 32, D, H, W]

---

## 6.3 Predict distribution

We compute:

- μ_skip = Conv3D(x_cat)
- logvar_skip = Conv3D(x_cat)

Then clamp:
- logvar ∈ [-10, -2]

---

## 6.4 Sampling

We sample:

σ = exp(0.5 × logvar)  
ε ~ N(0, I)  
out = μ + ε × σ  

---

## 6.5 Output

Each skip produces:

- stochastic feature map ∈ [B, C, D, H, W]

---

## Intuition:
Instead of saying:
> “this feature is fixed”

we say:
> “this feature has uncertainty, and it depends on demographics”

---

# 7. Decoder (3D Reconstruction Network)

The decoder reconstructs the MRI from:

- latent vector z
- skip connections
- demographic embedding

---

## 7.1 Latent expansion

We project z:

Linear(latent_dim → 256 × 10 × 12 × 10)

Then reshape:

- z → [B, 256, 10, 12, 10]

---

## 7.2 Upsampling path

We reverse the encoder using ConvTranspose3D layers:

---

### Stage 1
- ConvTranspose3D: 256 → 128
- Output: [B, 128, 20, 24, 20]
- Add: BayesianSkip(s3, demo)

---

### Stage 2
- ConvTranspose3D: 128 → 64
- Output: [B, 64, 40, 48, 40]
- Add: BayesianSkip(s2, demo)

---

### Stage 3
- ConvTranspose3D: 64 → 32
- Output: [B, 32, 80, 96, 80]
- Add: BayesianSkip(s1, demo)

---

### Stage 4 (final reconstruction)
- ConvTranspose3D: 32 → 1
- Activation: tanh

---

## Output:
- reconstructed MRI ∈ [B, 1, 160, 192, 160]

---

# 8. CDR Severity Prediction Head

This is a separate branch used for clinical prediction.

---

## 8.1 Inputs

We concatenate:

- latent vector z ∈ [B, latent_dim]
- pooled bottleneck features ∈ [B, 256]

Final input:
- [B, latent_dim + 256]

---

## 8.2 Network

- Linear → 128
- ReLU
- Linear → 64
- ReLU
- Linear → 1

---

## Output:
- predicted CDR-SB score ∈ [B, 1]

---

## Purpose:
This forces the latent space to encode clinically meaningful disease severity information.

---

# 9. Bottleneck Feature Pooling

We also extract global structure from encoder output:

- AdaptiveAvgPool3D → [B, 256, 1, 1, 1]
- Flatten → [B, 256]

This is used for:
- CDR prediction
- contrastive regularization

---

# 10. Full Forward Pass

For each MRI:

### Step 1: Encode
- μ, logvar, s1/s2/s3, bottleneck = Encoder(x)

### Step 2: Sample latent
- z = μ + εσ

### Step 3: Demographics
- demo = MLP(age, sex)

### Step 4: Decode
- x̂ = Decoder(z, skips, demo)

### Step 5: Severity prediction
- cdr_pred = CDRHead(z, pooled_bottleneck)

---

## Final outputs:

- reconstructed MRI
- predicted CDR score
- latent mean μ
- latent variance logvar
- sampled latent z
- bottleneck features

---

# 11. Key Intuition (Plain Language)

- The encoder “compresses” the brain scan into numbers  
- The latent space stores a noisy compressed brain representation  
- The decoder “rebuilds” the brain from those numbers  
- Skip connections preserve fine anatomical detail  
- Demographics guide reconstruction toward expected anatomy  
- CDR head forces the model to understand disease severity  

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
Wyman-Chick, K. A., & Scott, B. J. (2015). DEVELOPMENT OF CLINICAL DEMENTIA RATING SCALE CUTOFF SCORES FOR PATIENTS WITH PARKINSON'S DISEASE. Movement disorders clinical practice, 2(3), 243–248. https://doi.org/10.1002/mdc3.12163---
