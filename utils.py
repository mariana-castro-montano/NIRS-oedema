import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scatteringcorrection import preprocess_data
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
import itertools
from sklearn.impute import SimpleImputer

N_THRESH   = 200

SPEC_ORIGIN_NM     = 899.084
SPEC_STEP_NM       = 1.5925
FULL_SAFE_START_NM = 940
FULL_SAFE_END_NM   = 1650


def get_threshold(y_train_pred: np.ndarray,
                  y_train: np.ndarray,
                  n_candidates: int = N_THRESH) -> float:
    lo = y_train_pred.min() - 1e-6
    hi = y_train_pred.max() + 1e-6
    candidates = np.linspace(lo, hi, n_candidates)

    best_thresh = 0.5
    best_ba     = -1.0

    for thr in candidates:
        y_cls = (y_train_pred >= thr).astype(int)
        if len(np.unique(y_cls)) < 2:
            continue
        ba = balanced_accuracy_score(y_train, y_cls)
        if ba > best_ba:
            best_ba     = ba
            best_thresh = thr

    return best_thresh

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_score: np.ndarray) -> dict:

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan

    return {
        'accuracy'    : accuracy_score(y_true, y_pred),
        'bal_accuracy': balanced_accuracy_score(y_true, y_pred),
        'sensitivity' : sensitivity,
        'specificity' : specificity,
        'f1'          : f1_score(y_true, y_pred, zero_division=0),
        'auc'         : auc,
    }

def jackknife_metrics(outer_true, outer_pred, outer_scores):

    n = len(outer_true)
    outer_true   = np.array(outer_true)
    outer_pred   = np.array(outer_pred)
    outer_scores = np.array(outer_scores)

    fold_metrics = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        if len(np.unique(outer_true[mask])) < 2:
            continue
        m = compute_metrics(outer_true[mask],
                            outer_pred[mask],
                            outer_scores[mask])
        fold_metrics.append(m)

    summary = {}
    for k in fold_metrics[0]:
        vals = np.array([fm[k] for fm in fold_metrics])
        summary[k] = {'mean': np.mean(vals),
                      'std' : np.std(vals, ddof=1),
                      'min' : np.min(vals),
                      'max' : np.max(vals)}
    return summary

def inner_cv_lv_selection(train_patients, patient_windows, patient_window_labels,
                          patient_labels, lv_range=range(1, 21), n_splits=5, random_state=42):

    train_labels_patient = np.array([patient_labels[p] for p in train_patients])
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    inner_val_scores = {lv: [] for lv in lv_range}

    for inner_train_idx, inner_val_idx in inner_cv.split(train_patients, train_labels_patient):
        inner_train_patients = [train_patients[i] for i in inner_train_idx]
        inner_val_patients = [train_patients[i] for i in inner_val_idx]

        # Aggregate windows for inner train
        X_inner_train = np.vstack([patient_windows[p] for p in inner_train_patients])
        y_inner_train = np.concatenate([patient_window_labels[p] for p in inner_train_patients])

        # Preprocessing fit on inner train only
        X_inner_train_t =  preprocess_data(X_inner_train, method="snv", reference=None)
        std_scaler = StandardScaler()
        X_inner_train_t = std_scaler.fit_transform(X_inner_train_t)

        for lv in lv_range:
            model = PLSRegression(n_components=lv)
            model.fit(X_inner_train_t, y_inner_train)

            val_preds, val_true = [], []
            for p in inner_val_patients:
                X_p = preprocess_data(patient_windows[p], method="snv", reference=None)
                X_p = std_scaler.transform(X_p)
                p_score = model.predict(X_p).mean()
                val_preds.append(p_score)
                val_true.append(patient_labels[p])

            val_preds_bin = [1 if s > 0.5 else 0 for s in val_preds]
            ba = balanced_accuracy_score(val_true, val_preds_bin)
            inner_val_scores[lv].append(ba)


    lv_list = list(lv_range)
    means = np.array([np.mean(inner_val_scores[lv]) for lv in lv_list])
    stds = np.array([np.std(inner_val_scores[lv], ddof=1) for lv in lv_list])
    n_inner = n_splits
    se = stds / np.sqrt(n_inner)

    best_idx = np.argmax(means)
    threshold = means[best_idx] - se[best_idx]


    within_1se = np.where(means >= threshold)[0]
    best_lv = lv_list[within_1se[0]]

    return best_lv



def inner_cv_lr_selection(train_patients, patient_clinical, patient_labels,
                           n_splits=5, random_state=42):

    param_grid = {
        'l1_ratio':      [0.0, 0.5, 1.0],
        'C':            [0.001, 0.01, 0.1, 1, 10, 100, np.inf],
        'class_weight': [None, 'balanced'],
        'solver':       ['saga'],  # saga supports l1, l2 and no penalty
    }

    train_labels_patient = np.array([patient_labels[p] for p in train_patients])
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)


    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v))
                          for v in itertools.product(*values)]

    param_scores = {i: [] for i in range(len(param_combinations))}

    for inner_train_idx, inner_val_idx in inner_cv.split(train_patients,
                                                          train_labels_patient):
        inner_train_patients = [train_patients[i] for i in inner_train_idx]
        inner_val_patients   = [train_patients[i] for i in inner_val_idx]


        X_inner_train = np.vstack([patient_clinical[p]
                                   for p in inner_train_patients])
        y_inner_train = np.array([patient_labels[p]
                                  for p in inner_train_patients])
        X_inner_val = np.vstack([patient_clinical[p]
                                 for p in inner_val_patients])

        imputer = SimpleImputer(strategy='median')
        X_inner_train_s = imputer.fit_transform(X_inner_train)
        X_inner_val_s = imputer.transform(X_inner_val)


        scaler = StandardScaler()
        X_inner_train_s = scaler.fit_transform(X_inner_train_s)
        X_inner_val_s = scaler.transform(X_inner_val_s)

        y_inner_val   = np.array([patient_labels[p]
                                  for p in inner_val_patients])

        for i, params in enumerate(param_combinations):
            if params['C'] == np.inf:
                lr = LogisticRegression(
                    penalty=None,
                    class_weight=params['class_weight'],
                    solver=params['solver'],
                    max_iter=5000,
                    random_state=random_state
                )
            else:
                lr = LogisticRegression(
                    penalty='elasticnet',
                    l1_ratio=params['l1_ratio'],
                    C=params['C'],
                    class_weight=params['class_weight'],
                    solver=params['solver'],
                    max_iter=5000,
                    random_state=random_state
                )
            lr.fit(X_inner_train_s, y_inner_train)
            val_probs = lr.predict_proba(X_inner_val_s)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            ba = balanced_accuracy_score(y_inner_val, val_preds)
            param_scores[i].append(ba)

    mean_scores = {i: np.mean(param_scores[i]) for i in range(len(param_combinations))}
    best_idx    = max(mean_scores, key=mean_scores.get)
    best_params = param_combinations[best_idx]

    return best_params