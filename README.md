# Smudged Fingerprints: A Systematic Evaluation of the Robustness of AI Image Fingerprints

Public code release for the paper accepted at IEEE SaTML 2026.

Paper: https://arxiv.org/abs/2512.11771

Authors:
- Kai Yao (kai.yao@ed.ac.uk), School of Informatics, University of Edinburgh
- Marc Juarez (marc.juarez@ed.ac.uk), School of Informatics, University of Edinburgh

Abstract:
Model fingerprint detection has shown promise to trace the provenance of AI-generated images in forensic applications. However, despite the inherent adversarial nature of these applications, existing evaluations rarely consider adversarial settings. We present the first systematic security evaluation of these techniques, formalizing threat models that encompass both white- and black-box access and two attack goals: fingerprint removal, which erases identifying traces to evade attribution, and fingerprint forgery, which seeks to cause misattribution to a target model. We implement five attack strategies and evaluate 14 representative fingerprinting methods across RGB, frequency, and learned-feature domains on 12 state-of-the-art image generators. Our experiments reveal a pronounced gap between clean and adversarial performance. Removal attacks are highly effective, often achieving success rates above 80 percent in white-box settings and over 50 percent under black-box access. While forgery is more challenging than removal, its success varies significantly across targeted models. We also observe a utility-robustness trade-off: accurate attribution methods are often vulnerable to attacks and, although some techniques are robust in specific settings, none achieves robustness and accuracy across all evaluated threat models. These findings highlight the need for techniques that balance robustness and accuracy, and we identify the most promising approaches toward this goal.

## What this repo provides

- Train attribution models for a chosen fingerprint method.
- Run robustness attacks for a chosen attack type and attack goal.
- Evaluate performance under removal or forgery threats.

## Setup

### Option A: Docker (recommended for full environment)

The Dockerfile matches the environment used in our experiments.

```bash
docker build -t fingerprint-robustness .
# Requires nvidia-container-toolkit for GPU usage
docker run --gpus all -it --rm -v "$(pwd)":/workspace -w /workspace fingerprint-robustness bash
```

### Option B: Python environment

1) Install torch, torchvision, and torchaudio (CPU or CUDA) from https://pytorch.org/get-started/locally/
2) Install the remaining dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- Some model samplers auto-install extra dependencies and clone third-party repos into `libs/` on first use.
- A few generators (e.g., NCSN++ and ADM) require heavier dependencies (tensorflow, jax). See `src/models/` for details.

## Data layout

`--data-dir` controls where training outputs and cached data are stored. The expected layout is:

```
<data_dir>/
  images/<model_name>/*.png
  features_<fingerprint_method>/<model_name>/*.npy
  models_<fingerprint_method>/
    true_attribution_model.pth
    surrogate_attribution_model.pth
    surrogate_extractor_model.pth
  attacked_images/  # optional, created when --save-attacked-images is used
```

## Train attribution models

Train the three models needed for attacks for a single fingerprint method:

```bash
python train_models.py \
  --fingerprint-method nataraj19 \
  --data-dir ./data \
  --images-per-model 100 \
  --device cuda
```

Notes:
- The training script generates images for all available generators in `src/models/`.
- For Song24 methods, you can optionally provide real data for manifold estimation:

```bash
python train_models.py \
  --fingerprint-method song24rgb \
  --data-dir ./data \
  --real-data-path /path/to/ffhq_real_data.pkl \
  --cache-dir ./cache
```

## Run attacks

### Removal attacks

```bash
python run_attacks.py \
  --data-dir ./data \
  --fingerprint-method nataraj19 \
  --attack-types w2 \
  --attack-goal removal \
  --test-images-per-model 100
```

### Forgery attacks

```bash
python run_attacks.py \
  --data-dir ./data \
  --fingerprint-method nataraj19 \
  --attack-types w1,w3,b1 \
  --attack-goal forgery \
  --test-images-per-model 100
```

Note: Forgery targets are assigned randomly per sample in the current release.

### B2 (image perturbation) removal attacks

```bash
python run_attacks.py \
  --data-dir ./data \
  --fingerprint-method nataraj19 \
  --attack-types b2 \
  --attack-goal removal \
  --perturbation-types gaussian_noise blur jpeg resize \
  --epsilon-linf 0.05
```

Optional flags:
- `--attack-data-dir` to load images/features from a separate folder while keeping models in `--data-dir`.
- `--save-attacked-images` to save successful attacks under `<data_dir>/attacked_images/`.

## Attack types and goals

Attack goals:
- `removal`: make the attribution model incorrect on images that were originally correct.
- `forgery`: force misattribution to a (random) target model.

Attack types:
- `w1`: white-box gradient-based attack on the fingerprint function.
- `w2`: white-box analytic approximation attack for non-differentiable methods.
- `w3`: white-box surrogate fingerprint extractor.
- `b1`: black-box surrogate attribution classifier.
- `b2`: black-box image perturbations (removal only).

Applicability notes:
- `w1` requires differentiability of the fingerprint method.
- `w2` requires an analytic approximation.
- `w3`, `b1`, `b2` are available for all methods.

## Fingerprint methods (14)

`mccloskey18`, `marra19a`, `nataraj19`, `durall20`, `dzanic20`, `qian20`, `wang20`, `giudice21`, `nowroozi22`, `corvi23r`, `corvi23s`, `song24rgb`, `song24freq`, `song24sl`

## Generative models (12)

`r3gan-ffhq-256`, `cips-ffhq-256`, `stylegan2-ffhq-256`, `stylegan3-ffhq-256`, `ganformer-ffhq-256`, `styleswin-ffhq-256`, `vdvae-ffhq-256`, `ncsnpp-ffhq-256`, `adm-ffhq-256`, `ldm-ffhq-256`, `nvae-ffhq-256`, `vqvae-ffhq-256`

## Citation

If you use this code, please cite the arXiv preprint listed above.
