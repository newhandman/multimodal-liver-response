"""Table propensity cross-fitting and validation-selected MRI response training."""
import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def prepare_covariates(frame, numerical, categorical):
    covariates = frame[numerical + categorical].copy()
    for column in numerical:
        covariates[column] = pd.to_numeric(covariates[column], errors='coerce')
    for column in categorical:
        covariates[column] = covariates[column].map(lambda value: normalize_categorical_value(column, value))
    return covariates

def normalize_categorical_value(column, value):
    """Canonicalize known equivalent encodings before imputation/one-hot encoding."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.casefold() in {'nan', 'none', 'null'}:
        return np.nan
    integer_categories = {'ECOG', 'ALBI_grade', 'up_to_7', 'macrovascular_invasion', 'extrahepatic_metastasis'}
    if column in integer_categories:
        try:
            number = float(text)
            if np.isfinite(number) and number.is_integer():
                return str(int(number))
        except ValueError:
            pass
        if column == 'ECOG' and text.casefold() == 'p':
            return np.nan
    if column == 'CNLC':
        aliases = {'4': 'IV', 'IVA': 'IVa', 'IVB': 'IVb', 'IIIA': 'IIIa', 'IIIB': 'IIIb', 'IIA': 'IIa', 'IIB': 'IIb', 'IA': 'Ia', 'IB': 'Ib'}
        return aliases.get(text.upper(), text)
    if column == 'BCLC':
        return {'3': 'C'}.get(text.upper(), text.upper())
    if column == 'CP':
        return text.upper()
    if column == 'Sex':
        return text.casefold()
    return text

def make_propensity_pipeline(numerical, categorical, c_value, max_iter, seed):
    transformers = []
    if numerical:
        transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical))
    if categorical:
        transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    classifier = LogisticRegression(C=c_value, max_iter=max_iter, solver='lbfgs', random_state=seed)
    return Pipeline([('preprocessor', preprocessor), ('classifier', classifier)])

def make_weights(propensity, treatment, treatment_prevalence, propensity_clip, weight_clip_min, weight_clip_max, stabilized, normalize):
    propensity = np.asarray(propensity, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    clipped_propensity = np.clip(propensity, propensity_clip, 1.0 - propensity_clip)
    observed_probability = np.where(treatment == 1, clipped_propensity, 1.0 - clipped_propensity)
    if stabilized:
        numerator = np.where(treatment == 1, treatment_prevalence, 1.0 - treatment_prevalence)
    else:
        numerator = np.ones_like(observed_probability)
    raw_weight = numerator / observed_probability
    weight = np.clip(raw_weight, weight_clip_min, weight_clip_max)
    if normalize:
        weight = weight / weight.mean()
    return (clipped_propensity, raw_weight, weight)

def read_frame(path):
    return pd.DataFrame.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), orient="index")


def stage1(c, output):
    frame = read_frame(c['train_json'])
    numerical, categorical = c['numerical_features'], c['categorical_features']
    x = prepare_covariates(frame, numerical, categorical)
    treatment = frame[c['treatment_col']].to_numpy(dtype=int)
    folds = StratifiedKFold(n_splits=c['folds'], shuffle=True, random_state=c['seed'])
    propensity = np.empty(len(frame))
    fold_ids = np.empty(len(frame), dtype=int)
    for fold, (fit, heldout) in enumerate(folds.split(x, treatment)):
        pipeline = make_propensity_pipeline(numerical, categorical, c['propensity_c'], c['propensity_max_iter'], c['seed'])
        pipeline.fit(x.iloc[fit], treatment[fit])
        propensity[heldout] = pipeline.predict_proba(x.iloc[heldout])[:, 1]
        fold_ids[heldout] = fold
    clipped, raw, weight = make_weights(propensity, treatment, float(treatment.mean()),
        c['propensity_clip'], c['weight_clip_min'], c['weight_clip_max'], c['stabilized'], c['normalize_weights'])
    pd.DataFrame(dict(case_id=frame.index, treatment=treatment, propensity=propensity,
        propensity_clipped=clipped, raw_weight=raw, weight=weight, cross_fit_fold=fold_ids)).to_csv(output/'train_weights.csv', index=False)
    final = make_propensity_pipeline(numerical, categorical, c['propensity_c'], c['propensity_max_iter'], c['seed'])
    final.fit(x, treatment)
    joblib.dump(final, output/'propensity_model.joblib')


def stage2(c, output):
    import nibabel as nib
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from MultiModalFusionClassifier import MultiModalFusionClassifier

    random.seed(c['seed'])
    np.random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c['seed'])

    class MRIDataset(Dataset):
        def __init__(self, labels):
            self.frame = read_frame(labels)
        def __len__(self):
            return len(self.frame)
        def __getitem__(self, index):
            case = str(self.frame.index[index])
            row = self.frame.iloc[index]
            images = {m: torch.from_numpy(nib.load(str(Path(c['image_root'])/case/f'{m}_transformed.nii.gz')).get_fdata().astype(np.float32)).unsqueeze(0)
                      for m in c['model']['modalities']}
            return case, images, float(row[c['outcome_col']]), float(row[c['treatment_col']])

    train = DataLoader(MRIDataset(c['train_json']), batch_size=c['batch_size'], shuffle=True, num_workers=c['num_workers'])
    validation = DataLoader(MRIDataset(c['val_json']), batch_size=c['val_batch_size'], shuffle=False, num_workers=c['num_workers'])
    weights = pd.read_csv(c['stage1_weight_csv'], dtype={'case_id': str}).set_index('case_id')['weight']
    model = MultiModalFusionClassifier(**c['model'])
    device = c['model']['cuda_device']
    optimizer_class = getattr(torch.optim, c['optimizer']['name'])
    optimizer = optimizer_class(model.parameters(), **c['optimizer']['kwargs'])
    scheduler = None
    if c['scheduler'] is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, c['scheduler']['name'])
        scheduler = scheduler_class(optimizer, **c['scheduler']['kwargs'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([c['pos_weight']], device=device), reduction='none')
    history = []
    best_auc = float('-inf')
    for epoch in range(1, c['num_epochs']+1):
        model.train()
        losses = []
        for cases, images, labels, treatment in train:
            labels = labels.to(device=device, dtype=torch.float32).reshape(-1, 1)
            treatment = treatment.to(device=device, dtype=torch.float32).reshape(-1, 1)
            logits = model(images, treatment)
            batch_weights = torch.tensor(weights.loc[list(cases)].to_numpy(), device=device, dtype=torch.float32).reshape(-1, 1)
            loss = (criterion(logits, labels)*batch_weights).sum()/batch_weights.sum().clamp_min(1e-6)
            optimizer.zero_grad()
            loss.backward()
            if c['grad_clip_max_norm'] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip_max_norm'])
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        observed, predicted = [], []
        with torch.no_grad():
            for _, images, labels, treatment in validation:
                treatment = treatment.to(device=device, dtype=torch.float32).reshape(-1, 1)
                predicted.extend(torch.sigmoid(model(images, treatment)).cpu().numpy().reshape(-1).tolist())
                observed.extend(labels.tolist())
        auc = float(roc_auc_score(observed, predicted))
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_auc=auc)
        history.append(record)
        (output/'training_log.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), output/'best_val.pth')
            (output/'best_val.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(auc)
            else:
                scheduler.step()
        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['1', '2'], required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    (output/'training_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (stage1 if args.stage == '1' else stage2)(config, output)


if __name__ == '__main__':
    main()
