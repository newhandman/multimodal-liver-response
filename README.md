# Two-stage MRI response training

Stage 1 estimates treatment propensity from baseline covariates with cross-fitting.
Stage 2 trains a multimodal 3D ViT using treatment-conditioned FiLM fusion and
fixed Stage-1 weights. The highest validation AUC selects the saved checkpoint.

## Installation

Use Python 3.10:

```bash
pip install -r requirements.txt
```

## Prepare data

Training and validation labels are JSON dictionaries keyed by anonymous sample
keys. Each record contains binary treatment and response columns, plus baseline
covariates for Stage 1. Specify the column names in your configuration.

Preprocessed images follow this layout:

```text
<image_root>/<sample_key>/<modality>_transformed.nii.gz
```

Image dimensions must match `vit_img_size`.

## Configuration and training

Create a JSON configuration for each stage, filling all fields listed below.
Use a new output directory for each run. Parameters have no experiment defaults.

```bash
python train.py --stage 1 --config stage1.json
python train.py --stage 2 --config stage2.json
```

### Stage 1 configuration fields

| Group | Fields |
|---|---|
| Files | `output_dir`, `train_json` |
| Covariates | `treatment_col`, `numerical_features`, `categorical_features` |
| Cross-fitting | `folds`, `seed` |
| Logistic regression | `propensity_c`, `propensity_max_iter` |
| Weight settings | `propensity_clip`, `weight_clip_min`, `weight_clip_max`, `stabilized`, `normalize_weights` |

Feature lists are arrays of pretreatment column names; use `[]` for an unused
feature type. `folds` is an integer (5 for five-fold cross-fitting).
`stabilized` and `normalize_weights` are booleans. Set `propensity_clip` between
0 and 0.5, with positive ordered weight bounds.

Outputs: `train_weights.csv`, `propensity_model.joblib`, `training_config.json`.

### Stage 2 configuration

Replace the `null` placeholders below with your settings. `scheduler` and
`grad_clip_max_norm` may remain `null` to disable these options.

```json
{
  "output_dir": null,
  "train_json": null,
  "val_json": null,
  "image_root": null,
  "outcome_col": null,
  "treatment_col": null,
  "stage1_weight_csv": null,
  "seed": null,
  "batch_size": null,
  "val_batch_size": null,
  "num_workers": null,
  "num_epochs": null,
  "pos_weight": null,
  "grad_clip_max_norm": null,
  "optimizer": {"name": null, "kwargs": null},
  "scheduler": null,
  "model": {
    "modalities": null,
    "num_classes": null,
    "extra_dim": null,
    "cuda_device": null,
    "cuda_device2": null,
    "vit_img_size": null,
    "vit_patch_size": null,
    "vit_hidden_size": null,
    "vit_mlp_dim": null,
    "vit_num_layers": null,
    "vit_num_heads": null,
    "vit_dropout_rate": null,
    "token_layers": null,
    "token_num_heads": null,
    "token_ffn_dim": null,
    "token_dropout": null,
    "classifier_dropout_1": null,
    "classifier_dropout_2": null
  }
}
```

- `stage1_weight_csv` points to the Stage-1 training weights.
- Set `num_classes=1` and `extra_dim=1` for binary response and treatment.
- `modalities` is an array of names; image and patch sizes are three-integer arrays.
- For one GPU, set both device fields to `"cuda:0"`; for CPU, use `"cpu"`.
  Two different device strings distribute modality encoders across two GPUs.
- An optimizer example is `{"name": "AdamW", "kwargs": {"lr": 0.000001,
  "weight_decay": 0.0000001}}`.
- A scheduler is `null` or an object containing `name` and `kwargs`.
  `ReduceLROnPlateau` receives validation AUC and should use `mode: "max"`.

The loss is the sum of sample-weighted BCE divided by the current batch's weight
sum. `pos_weight` scales the positive-label term. Validation AUC is unweighted.
A strictly higher validation AUC replaces the saved checkpoint; ties keep the
earlier epoch.

Outputs: `best_val.pth`, `best_val.json`, `training_log.json`, `training_config.json`.
