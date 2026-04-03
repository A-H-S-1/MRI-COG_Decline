# 3D Stochastic VAE with Demographic Conditioning and Severity-Aware Representation Learning for Cognitive Decline Progression Modeling from Structural MRI

---

## Project Overview

A 3D variational autoencoder (VAE) based multi-task deep learning framework is developed to model neuroanatomical structure and dementia severity using structural MRI scans (T1-weighted and T2-weighted) trained on the OASIS-3 dataset from Washington University in St. Louis.

Unlike standard VAEs, which focus solely on reconstruction, the proposed framework jointly learns to:

* A probabilistic latent representation of brain anatomy
* A severity-aware regression signal for the Clinical Dementia Rating–Sum of Boxes (CDR-SB) scores
* A stochastic feature propagation mechanism via Bayesian skip connections
* A demographic conditioned generative process on age and sex

The model is designed to capture both anatomical variation and disease progression within a shared latent space, incorporating clinical supervision through CDR-SB regression.

---

## Key Contributions

### Multi-objective representation learning

The model jointly optimizes:

* MRI reconstruction fidelity
* KL divergence regularization
* CDR-SB regression
* Patch-level representation consistency

### Bayesian skip connection framework

Stochastic skip connections are utilized to model uncertainty in intermediate anatomical features by learning a Gaussian distribution over skip features conditioned on both encoder activations and demographic embeddings (age and sex). Conditioning on age and sex enables the model to account for demographic variability in anatomical representations.

### Demographic-conditioned generative modeling

Age and sex are embedded using a learned multilayer perceptron (MLP) and incorporated into the decoder skip pathways. This conditioning reduces confounding effects and enhances subject-specific reconstruction.

### Severity-aware latent space structuring

A dedicated CDR-SB prediction head ensures that the latent representations of each MRI encode clinically meaningful disease severity signals.

---

## Dataset

### Source: 

OASIS-3 is a retrospective dataset of 1378 participants collected over 30 years, including cognitively normal individuals and patients at various stages of cognitive decline.

### MRI Modalities Used:

* T1-weighted MRI (T1w)
* T2-weighted MRI (T2w)

### Additional Data Used:

* Clinical Dementia Rating—Sum of Boxes (CDR-SB) Scores
* Demographics (age, sex)

### Severity Groups
For both T1w and T2w MRI modalities, CDR-SB scores are heavily skewed toward 0 (cognitively normal) and 0.5 (questionable impairment). The dataset exhibits sparsity in CDR-SB scores above 0.5, with some higher scores missing. Consequently, although the model is trained using the original CDR-SB scores, evaluation is performed using the scoring structure defined in O’Bryant et al.

### The severity groups are as follows:

* Questionable: 0.5 to 4
* Mild: 4.5 to 9
* Moderate: 9.5 to 15.5
* Severe: 16 to 18

### Age and Clinical Label Alignment

Participant age at each MRI session was calculated by adding the age at study entry to the number of days elapsed from study entry to the MRI session. Since CDR assessments were conducted throughout the study, the CDR-SB score assigned to each MRI corresponds to the closest available assessment date. This approach maximizes temporal alignment between clinical labels and imaging data, even when MRI and CDR assessments were not performed on the same day.

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
6. Center cropping / padding to a fixed shape
7. Conversion to NumPy format

#### Skull Stripping

Non-brain tissue (e.g., skull, scalp, neck) was removed using SynthStrip from FreeSurfer to isolate brain tissue and reduce irrelevant variation. Synthstrip takes a NIfTI volume as input and outputs a skull-stripped NIfTI volume. Scans that failed skull stripping were excluded from further processing, and their paths were saved.

#### Orientation Normalization

All MRIs were converted to RAS (Right–Anterior–Superior) orientation using nibabel.as_closest_canonical to ensure consistent anatomical alignment across subjects and prevent axis inconsistencies during model training.

#### Bias Field Correction

We applied N4 bias field correction using SimpleITK to correct low-frequency intensity inhomogeneities. This is important for MRI data because scanner field artifacts can cause inconsistent intensities across scans.

#### Resampling to Uniform Voxel Spacing

All volumes were resampled to a fixed voxel spacing: (160, 192, 160). If the volume was larger, it was center-cropped; if it was smaller, it was zero-padded. This ensures consistent tensor dimensions and compatibility with 3d CNNs training.

#### Final Formatting

Each processed MRI is stored as:

* Shape: (1, 160, 192, 160)
* Data type: float32
* Format: .npy

The channel dimension is added to support PyTorch 3D convolutional models.

### Quality Control and Outlier Removal

* Failed preprocessing steps (e.g., skull stripping errors) were logged and excluded.
* Filtering pipeline that identified and removed corrupted, misaligned, or anomalous MRI volumes prior to model training

#### Overview of filtering

Each MRI file was evaluated using multiple independent criteria:

1. File integrity
2. Global intensity statistics
3. Spatial alignment
4. Boundary artifacts

A scan was removed if it failed any of the above criteria.

#### File Integrity Check

Each .npy file was loaded and inspected for:

* loading errors (corrupt or unreadable files)
* presence of NaN values
* presence of infinite values

#### Statistical Outlier Detection

* abnormally bright or dark scans
* scans with extremely low or high contrast
* preprocessing failures affecting intensity scaling
To identify files that were blurry, had poor contrast, or had abnormally bright or dark scans, we computed basic intensity statistics for each scan:
* mean intensity
* standard deviation

These were then normalized across the dataset using a Z-score:

* z = (x - mean) / std

A scan is flagged as an outlier if:

* |z_mean| > threshold
* OR |z_std| > threshold

After testing different thresholds, we found a threshold of 2 that balanced accuracy, flagging erroneous files while ignoring MRIs with large anatomical deviations.

#### Center-of-Mass Shift (Spatial Misalignment)

We estimate the distance of the ‘mass’ in the brain image from the center to avoid training or testing on images that were misaligned, had cropping issues, or failed preprocessing.

##### Procedure

1. Convert the 3D volume into a 2D projection (mean across slices)
2. Normalize intensities
3. Compute the center of mass of the image
4. Measure its distance from the image center to find its shift value
5. Scans with a shift value above the 95th percentile were removed

#### Edge Artifact Detection

We measured the fraction of the image boundary that consists of zero-valued pixels in order to identify images with excessive padding, cropping errors, or failed skull stripping.

##### Procedure

1. Extract the middle slice of the MRI
2. Collect pixel values along the image border
3. Compute the fraction of border pixels equal to zero
4. Removed scans with an edge ratio > threshold

Through visual testing, we found a threshold value of 0.2 was able to identify which files had the mri partially cut off by the edge.

---

## Dataset Summary Statistics

This section summarizes the cleaned OASIS-3 T1w and T2w datasets used in this work.

### T1-weighted MRI Dataset (T1w)

#### Overview

Column 1	Column 2
Metric	Value
Total MRIs	2,196
Age (mean ± std)	70.62 ± 9.24
Age range	42.69 – 95.70

#### Sex Distribution

Column 1	Column 2	Column 3
Sex	Count	Percentage
Female	1,281	58.3%
Male	915	41.7%

#### Severity Distribution (Aggregated)

Column 1	Column 2	Column 3
Severity Group	Count	Percentage
0 (Normal)	1,639	74.6%
1 (Mild)	340	15.5%
2 (Moderate)	125	5.7%
3 (Severe)	10	0.5%

#### Raw CDR-SB Distribution (T1w)

Column 1	Column 2
CDR-SB	Count
0.0	1,639
0.5	115
1.0	74
1.5	59
2.0	40
2.5	37
3.0	34
3.5	24
4.0	37
4.5	33
5.0	28
5.5	17
6.0	19
7.0	10
8.0	14
8.5	1
9.0	3
10.0	2
11.0	2
12.0	4
13.0	2

### T2-weighted MRI Dataset (T2w)

#### Overview

Column 1	Column 2
Metric	Value
Total MRIs	1,362
Age (mean ± std)	70.54 ± 9.23
Age range	42.69 – 95.70

#### Sex Distribution

Column 1	Column 2	Column 3
Sex	Count	Percentage
Female	779	57.2%
Male	583	42.8%

#### Severity Distribution (Aggregated)

Column 1	Column 2	Column 3
Severity Group	Count	Percentage
0 (Normal)	1,013	74.4%
1 (Mild)	284	20.9%
2 (Moderate)	103	7.6%
3 (Severe)	2	0.1%

#### Raw CDR-SB Distribution (T2w)

Column 1	Column 2
CDR-SB	Count
0.0	1,013
0.5	72
1.0	48
1.5	39
2.0	24
2.5	25
3.0	21
3.5	15
4.0	21
4.5	23
5.0	19
5.5	10
6.0	10
7.0	6
8.0	9
9.0	1
10.0	1
11.0	1
12.0	2
13.0	1

### Key Dataset Characteristics

* Both datasets may contain multiple MRIs per individual; however, most subjects have only 1 or 2 scans, limiting their suitability for longitudinal analysis.
* Strong class imbalance toward cognitively normal subjects
* Severe dementia cases are rare for both T1w and T2w modalities.
* In the Mild Severity group, the majority of T1w and T2w MRIs have a CDR-SB score of 0.5

### Interpretation

The datasets exhibit long-tailed clinical distributions, where:

* Early-stage and healthy subjects dominate moderate and severe stages
* The advanced cognitive decline stages are sparse and underrepresented

This motivates:

* stratified sampling
* severity-aware loss weighting

---

## Train–Test Splitting Strategy

The dataset splitting strategy was designed to prevent subject-level data leakage, preserve clinically meaningful severity distributions, and increase test set robustness. This was accomplished via a multi-stage pipeline that first selected representative scans per subject, performed a stratified split, and then selectively augmented the test set.

### Step 1: Select Maximum Severity Scan per Subject

Since the OASIS-3 dataset was initially created for longitudinal analysis, each subject could have multiple MRI scans of the same modality over time. To create a subject-level representation to prevent bias, we select the scan with the highest CDR-SB score for each subject. Which produced two tables:

* The Max-CDR table contained one scan per subject, recorded at the subject's highest CDR-SB score.
* The unused table contained all other MRI scans.

### Step 2:  Stratified Train–Test Split

Each scan was assigned to a severity group based on its CDR-SB score. We then stratified the Max-CDR table to obtain training and test sets.

#### Procedure

1. Compute severity_group for each subject.
2. Count the number of samples per group.
3. Identify rare groups, defined as groups with fewer than 5 samples.

#### Handling Rare Groups

If a severity group met the rare group criteria, all samples from that group were placed into the training set and excluded from the test stratification.

#### Stratified Split

For the remaining (non-rare) samples, we perform an 80/20 split and stratify by severity_group.

#### Outputs

* train_df -> training set (subject-level, one scan per subject)
* intermediate_test_df -> initial test set (one scan per subject)

### Step 3: Test Set Augmentation

After splitting, we expand the test set using additional scans. We identified the subjects who appeared in the test set and added their corresponding MRIs from the unused table. This resulted in a final test set consisting of the original stratified test subjects and their additional scans. This increased the number of test samples, which was particularly important to better evaluate the trained model on higher-severity CDR ranges that would otherwise be poorly represented due to class imbalance and data sparsity.

### Final Dataset Structure

#### Training Set

* One scan per subject (max CDR)
* Stratified by severity
* Includes all rare classes
* Used for:
  * model training
  * representation learning
  * severity prediction

#### Test Set

* One max-CDR scan per subject (from stratified split)
* PLUS additional scans from the same subjects
* Used for:
  * evaluation

---

## Data Loading Pipeline

Each dataset is wrapped in a PyTorch Dataset (MRIDataset), which:

For each sample:

1. Loads .npy MRI volume
2. Applies per-scan min-max normalization
3. Loads demographic variables (age, sex)
4. Optionally loads the CDR-SB label.
5. Optionally loads the severity group.

### Input Data Structure

Each sample consists of:

* MRI volume: .npy file containing a 3D brain scan
* Age: scalar float
* Sex: binary encoding (0 = male, 1 = female)
* CDR-SB: clinical severity score
* Group: severity score

### Normalization

Each MRI volume is normalized per-scan using min-max scaling:

* If max - min > EPS: x = (x - min) / (max - min)
* Else: x = 0

This ensures all inputs are scaled to [0, 1].

### Class Definition Summary

The dataset implements:

* __len__: returns number of MRI scans
* __getitem__: loads and processes a single subject sample

All outputs are converted to torch.float32 tensors for compatibility with the VAE training pipeline.

---

## Model Architecture

This project implemented a 3D Variational Autoencoder (VAE) extended with:

* Demographic conditioning (age, sex)
* Stochastic (Bayesian) skip connections
* Supervised disease severity prediction head (CDR-SB regression)

The model is designed to:

1. Reconstruct 3D brain MRI scans.
2. Learn a latent representation that encodes disease severity.

Note: Due to sparsity and class imbalance, this model is trained on CDR-SB scores and produces a predicted CDR-SB score; it is evaluated using severity groups.

### High-Level Overview

1. Demographic Embedding Network: Encodes age and sex into a learned conditioning vector
2. Encoder (3D CNN): Compresses MRI volumes into a latent representation
3. Latent Space (VAE Sampling): Produces a stochastic latent vector via reparameterization
4. Decoder (3D CNN + Bayesian Skip Connections): Reconstructs MRI volumes using latent features and stochastic skip connections
5. CDR Prediction Head: Predicts CDR-SB from learned features

### Input Format

Each input sample consists of:

* MRI volume: x ∈ [1, 160, 192, 160]
* age: scalar (float)
* sex: scalar
  * 0 = male
  * 1 = female

### Demographic Embedding

Demographic variables are converted into a learned embedding vector. This embedding conditions the decoder and skip connections, allowing the model to account for anatomical variation due to sex and, more importantly, age.

#### Input

* age ∈ [B]
* sex ∈ [B]

#### Step 1: Concatenation

* demo_input ∈ [B, 2] = [age, sex]

#### Step 2: MLP Projection

* Linear(2 → 32)
* ReLU
* Linear(32 → 32)
* ReLU

#### Output

* demo_embedding ∈ [B, 32]

### Encoder (3D Convolutional Network)

The encoder compresses the MRI volume into a compact representation.

#### Convolutional Downsampling

Four 3D convolutional layers progressively reduce spatial resolution:

Column 1	Column 2	Column 3
Layer	Channels	Output Shape
Conv1	1 → 32	[B, 32, 80, 96, 80]
Conv2	32 → 64	[B, 64, 40, 48, 40]
Conv3	64 → 128	[B, 128, 20, 24, 20]
Conv4	128 → 256	[B, 256, 10, 12, 10]

#### Each layer uses:

* BatchNorm3D
* ReLU activation

#### Skip Features

Intermediate activations are stored for later use:

* s1 → output of Conv1 → BatchNorm → ReLU
* s2 → output of Conv2 → BatchNorm → ReLU
* s3 → output of Conv3 → BatchNorm → ReLU

These are post-activation feature maps used in the decoder skip refinement.

#### Flattening

Final encoder output before latent projection:

* s4 ∈ [B, 256, 10, 12, 10]

Flattened to:

* [B, 256 × 10 × 12 × 10]

#### Latent Projection

The flattened bottleneck is mapped into a probabilistic latent space.

##### Dynamic Initialization

Linear layers are created on the first forward pass:

* μ = Linear(flatten_dim → latent_dim)
* logvar = Linear(flatten_dim → latent_dim)

These layers are initialized dynamically during the first forward pass, once the flattened feature dimension is known.

##### Output

* μ ∈ [B, latent_dim]
* logvar ∈ [B, latent_dim]

#### Variance Stabilization

To ensure numerical stability, the log-variance is constrained to prevent unstable sampling and to control stochasticity in the latent space:

* logvar ∈ [-10, -2]

##### Interpretation

μ represents the deterministic encoding of the input MRI, while logvar represents the uncertainty of each latent dimension. Together, they define:

z ~ N(μ, σ² I), where σ = exp(0.5 × logvar)

5. Latent Space Reparameterization

The latent vector is sampled as:

* σ = exp(0.5 × logvar)
* ε ~ N(0, I)
* z = μ + ε × σ

##### Output:

* z ∈ [B, latent_dim]

### Bayesian Skip Connections

Skip connections are stochastic residual feature generators conditioned on demographics. Instead of deterministic skip connections, the model injects uncertainty-aware anatomical refinements conditioned on demographics at multiple spatial scales.

#### Inputs

* Feature map x ∈ [B, C, D, H, W]
* Demographic embedding ∈ [B, 32]

#### Conditioning

* demo_embedding expanded → [B, 32, D, H, W]
* Concatenated with feature map:
  * x_cat ∈ [B, C + 32, D, H, W]

#### Distribution Modeling

Two 1×1×1 convolution layers produce a Gaussian distribution:

* μ = Conv3D(C + 32 → C)
* logvar = Conv3D(C + 32 → C)

logvar is clamped to [-10, -2].

#### Sampling

* σ = exp(0.5 × logvar)
* ε ~ N(0, I)
* output = μ + ε × σ
* NaNs are replaced with 0 for stability.

#### Integration into Decoder

Each stochastic skip output is added as a residual refinement to the decoder feature maps at the corresponding resolution stage.

#### Output

* Stochastic feature map ∈ [B, C, D, H, W]

### Decoder (3D Reconstruction Network)

The decoder reconstructs MRI volumes using latent features, demographic conditioning, and stochastic skip refinements.

#### Latent Expansion

* Linear(latent_dim → 256 × 10 × 12 × 10)
* Reshape → [B, 256, 10, 12, 10]

#### Upsampling Pipeline

Column 1	Column 2	Column 3
Stage	Operation	Output Shape
1	Deconv 256 → 128 + BayesianSkip(s3)	[B, 128, 20, 24, 20]
2	Deconv 128 → 64 + BayesianSkip(s2)	[B, 64, 40, 48, 40]
3	Deconv 64 → 32 + BayesianSkip(s1)	[B, 32, 80, 96, 80]
4	Deconv 32 → 1	[B, 1, 160, 192, 160]

#### Each intermediate stage uses:

* ConvTranspose3D
* BatchNorm3D
* ReLU

#### Final activation:

* Sigmoid function

### Global Bottleneck Pooling

This function extracts a global anatomical representation from the encoder bottleneck for the CDR-SB prediction task. Instead of using the full 3D feature map, it compresses spatial information into a single vector.

#### Input

* Bottleneck ∈ [B, 256, 10, 12, 10]

#### Operation

* AdaptiveAvgPool3D → [B, 256, 1, 1, 1]
* Reshape → [B, 256]

#### Output

* skip_feat ∈ [B, 256]

This vector represents global structural brain features and is used to:

* Stabilize clinical prediction
* Provide a non-latent auxiliary signal.
* Complement stochastic latent vector z

### CDR Severity Prediction Head

Predicts clinical severity score (CDR-SB) from learned representations.

#### Input

* concatenation of:
  * latent vector z ∈ [B, latent_dim]
  * pooled bottleneck ∈ [B, 256]

#### MLP Network

* Linear → 128
* ReLU
* Linear → 64
* ReLU
* Linear → 1

#### Output

* predicted CDR-SB ∈ [B, 1]

### Full VAE Model

This is the primary model combining:

* Generative modeling (VAE)
* Supervised learning (CDR prediction)
* Self-supervised learning (contrastive loss)
* Demographic conditioning

#### Forward Pass

For each sample:

1. Encode MRI:
  * μ, logvar, skips, bottleneck = Encoder(x)
2. Sample latent:
  * z = μ + εσ
3. Compute demographics:
  * demo = MLP(age, sex)
4. Decode:
  * recon = Decoder(z, skips, demo)
5. Pool bottleneck:
  * skip_feat = pooled bottleneck
6. Predict severity:
  * cdr_pred = CDRHead(z, skip_feat)

#### Outputs

* Reconstructed MRI
* Predicted CDR score
* μ, logvar
* Latent vector z
* Bottleneck features

---

## Loss Function

The model is trained using a multi-objective loss that combines reconstruction, probabilistic regularization, supervised learning, and constraints on the representation structure.\

### Reconstruction Loss

Voxel-wise L1 loss:

* L_rec = ||x − x̂||₁

### KL Divergence

* L_KL = -0.5 × mean(1 + logvar − μ² − exp(logvar))

### CDR Regression Loss

Supervised mean squared error:

* L_CDR = MSE(cdr_pred, cdr_true)

### Contrastive Representation Loss (InfoNCE)

A batch-wise contrastive learning objective is applied to pooled bottleneck features.

#### Feature extraction

* The 3D bottleneck feature map is spatially pooled to (2, 2, 2)
* Features are L2-normalized across the feature dimension.

#### Similarity computation

S_ij = (f_i^T f_j) / τ

where:

* f_i, f_j are normalized feature vectors
* τ = 0.1 is the temperature parameter

#### Objective

Each sample is treated as its own class within the batch:

L_contrast = CrossEntropy(S, diag labels)

#### Effect

This objective encourages:

* Intra-subject compactness
* Inter-subject separation
* Structured latent representations

### Final Objective

L = 2.5 · L_rec + β · L_KL + L_CDR + γ · L_contrast

where:

* β controls latent regularization
* γ controls contrastive strength

---

## Training Procedure

The model is trained using minibatch stochastic optimization over multiple epochs. Each training step applies the previously defined loss functions in a weighted multi-objective framework.

### Batch Processing

Each batch consists of MRI volumes, demographic variables (age, sex), and clinical labels (CDR). MRI inputs are clamped and normalized before being passed through the model.

### Forward Pass

The model outputs:

* MRI reconstruction
* Latent variables (μ, logvar, z)
* Bottleneck features
* Clinical prediction (CDR)

### Contrastive Features

Bottleneck features are spatially pooled, flattened, and normalized to compute batch-wise similarity for contrastive learning.

### Optimization

The total loss is computed as L = 2.5 · L_rec + β · L_KL + L_CDR + γ · L_contrast. We chose β to be 1 and γ as 0.2

Model parameters are updated using Adam optimization with:

* Gradient clipping (max norm = 5.0)
* Skipping of non-finite losses

### Output

Each epoch reports average values of all loss components over valid batches.

---

## Evaluation

The model is evaluated using both regression-based and classification-based metrics to assess clinical prediction performance and ordinal consistency.

### Prediction Outputs

The model outputs continuous CDR predictions, which are then converted to severity classes using predefined groupings.

### Classification Metrics

Standard classification metrics are computed on severity groups:

* Accuracy
* Macro and weighted F1-score
* Macro precision and recall
* Balanced accuracy
* Confusion matrix

Class-wise sensitivity and specificity are also reported.

### Ordinal Metrics

To account for the ordinal nature of the severity groups:

* Ordinal AUC is computed based on pairwise ranking consistency.
* Cohen’s Kappa and Quadratic Weighted Kappa are used to measure agreement.

### ROC / PR Analysis

One-vs-rest ROC-AUC and PR-AUC are computed for each class, along with macro-averaged scores.

### Separation Score

A variance-based separation metric is used to quantify how well predicted scores distinguish between severity groups.

---

## Limitations

* Class imbalance across disease severity
The dataset is heavily skewed toward cognitively normal subjects (CDR-SB = 0), which may bias both reconstruction and downstream prediction performance.
* Indirect supervision of disease structure
Disease-related structure is learned implicitly through reconstruction, KL regularization, and auxiliary CDR regression rather than explicit diagnostic supervision. While this enables flexible representation learning, it may limit sharp separation between clinical stages.
* Residual confounding effects
Although age and sex are incorporated into the model, additional confounders such as scanner variability, acquisition protocols, and site effects may still influence learned representations.
* Stochasticity in latent sampling
Variational sampling introduces inherent randomness into latent representations, leading to variability in reconstructions and downstream predictions across runs.
* Contrastive learning sensitivity
The contrastive objective improves the representation structure but introduces additional sensitivity to hyperparameters (e.g., temperature and loss weighting), potentially affecting training stability across datasets.
* Proxy-based clinical evaluation
Clinical performance is evaluated using proxy measures such as CDR-SB prediction, ordinal metrics, and group separability. These do not fully capture clinical decision-making or the dynamics of disease progression.

---

## References

* OASIS-3 Dataset
LaMontagne, P. J., Benzinger, T. L. S., Morris, J. C., Keefe, S., Hornbeck, R., Xiong, C., Grant, E., Hassenstab, J., Moulder, K., Vlassenko, A. G., et al. (2019).
OASIS-3: Longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and Alzheimer disease.
Journal of Cognitive Neuroscience.
https://doi.org/10.1162/jocn_a_01231
* SynthStrip (FreeSurfer)
Hoopes, A., Mora, J. S., Dalca, A. V., Fischl, B., & Hoffmann, M. (2022).
SynthStrip: Skull-stripping for any brain image.
NeuroImage, 260, 119474.
https://doi.org/10.1016/j.neuroimage.2022.119474
https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/
* Validation of Clinical Dementia Rating (CDR) Sum of Boxes interpretive guidelines
O’Bryant, S. E., Lacritz, L. H., Hall, J., et al. (2010).
Validation of the new interpretive guidelines for the Clinical Dementia Rating Scale sum of boxes score in the National Alzheimer’s Coordinating Center Database.
Archives of Neurology, 67(6), 746–749.
https://doi.org/10.1001/archneurol.2010.132

