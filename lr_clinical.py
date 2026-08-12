from sklearn.linear_model import LogisticRegression
from utils import compute_metrics, get_threshold, jackknife_metrics,inner_cv_lr_selection
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import time
import os
from joblib import Parallel, delayed
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.impute import SimpleImputer


df = pd.read_csv(r'C:\Users\adgk768\OneDrive - City, University of London\NeonatesGOSH2\clinical_data.csv',sep=';')
CHECKPOINT_DIR = r'C:\Users\adgk768\OneDrive - City, University of London\NeonatesGOSH2\perm_LR_new'
os.makedirs(CHECKPOINT_DIR,exist_ok=True)
print(df)
print(df[df['day'] == 'day1'].isnull().sum())
N_PERM     = 1000
df_day1 = df[df['day'] == 'day1'].copy()
patients = df['patient_id'].unique()
n_patients = len(patients)


assert set(np.unique(df['label'].values)) == {0, 1}, \
    "Labels must be binary 0/1"

patient_labels = np.array([
    df[df['patient_id'] == p]['label'].values[0]
    for p in patients
])

print(f"Patients: {n_patients}  "
      f"(oedema={patient_labels.sum()}, "
      f"no-oedema={n_patients - patient_labels.sum()})")

print(df.groupby('day')['weight'].apply(lambda x: x.isnull().sum()))

def run_nested_lopo_lr(df_clinical: pd.DataFrame,
                       patients_arr: np.ndarray,
                       shuffled_labels: np.ndarray | None = None
                       ) -> tuple:

    clinical_cols = [c for c in df_clinical.columns
                     if c not in ('patient_id', 'label')]


    patient_clinical = {
        p: df_clinical[df_clinical['patient_id'] == p][clinical_cols].values.squeeze()
        for p in patients_arr
    }
    patient_labels = {
        p: int(df_clinical[df_clinical['patient_id'] == p]['label'].iloc[0])
        for p in patients_arr
    }

    outer_true    = []
    outer_pred    = []
    outer_scores  = []
    outer_thres   = []
    chosen_params = []
    outer_scalers = []
    outer_models  = []

    for i, test_patient in enumerate(patients_arr):
        train_df = df_clinical[df_clinical['patient_id'] != test_patient].copy()
        test_df  = df_clinical[df_clinical['patient_id'] == test_patient].copy()

        if shuffled_labels is not None:
            label_map = {p: shuffled_labels[j]
                         for j, p in enumerate(patients_arr)}
            train_df['label'] = train_df['patient_id'].map(label_map)

        patient_labels_inner = {
            p: int(train_df[train_df['patient_id'] == p]['label'].iloc[0])
            for p in patients_arr if p != test_patient
        }

        inner_patients = list(train_df['patient_id'].unique())

        #  Inner CV: tune hyperparameters
        best_params = inner_cv_lr_selection(
            train_patients   = inner_patients,
            patient_clinical = patient_clinical,
            patient_labels   = patient_labels_inner,
        )
        chosen_params.append(best_params)

        #  Retrain on all N-1 outer training patients
        X_train = np.vstack([patient_clinical[p] for p in inner_patients])
        y_train = np.array([patient_labels_inner[p] for p in inner_patients])

        imputer = SimpleImputer(strategy='median')
        X_train_s = imputer.fit_transform(X_train)
        X_test_s = imputer.transform(patient_clinical[test_patient].reshape(1, -1))

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_s)
        X_test_s = scaler.transform(X_test_s)

        if best_params['C'] == np.inf:
            lr = LogisticRegression(
                penalty=None,
                class_weight=best_params['class_weight'],
                solver=best_params['solver'],
                max_iter=10000,
                random_state=42
            )
        else:
            lr = LogisticRegression(
                penalty='elasticnet',
                l1_ratio=best_params['l1_ratio'],
                C=best_params['C'],
                class_weight=best_params['class_weight'],
                solver=best_params['solver'],
                max_iter=10000,
                random_state=42
            )
        lr.fit(X_train_s, y_train)

        # Threshold optimisation on training fold
        train_probs  = lr.predict_proba(X_train_s)[:, 1]
        train_patient_probs = np.array([
            float(np.mean(lr.predict_proba(
                scaler.transform(imputer.transform(patient_clinical[p].reshape(1, -1)))
            )[:, 1]))
            for p in inner_patients
        ])
        train_patient_labels = np.array([patient_labels_inner[p]
                                         for p in inner_patients])
        final_thresh = get_threshold(train_patient_probs, train_patient_labels)

        patient_prob  = float(lr.predict_proba(X_test_s)[0, 1])
        patient_label = int(patient_prob >= final_thresh)
        true_label    = int(df_clinical[
            df_clinical['patient_id'] == test_patient]['label'].iloc[0])

        if shuffled_labels is not None:
            true_label = int(shuffled_labels[i])

        outer_true.append(true_label)
        outer_pred.append(patient_label)
        outer_scores.append(patient_prob)
        outer_thres.append(final_thresh)
        outer_scalers.append(scaler)
        outer_models.append(lr)

    metrics = compute_metrics(
        np.array(outer_true),
        np.array(outer_pred),
        np.array(outer_scores)
    )

    return (metrics, chosen_params, outer_true,
            outer_scores, outer_pred, outer_thres,
            outer_scalers, outer_models)


clinical_cols = ['fluid_balance', 'gestational_age', 'birth_weight', 'age', 'weight', 'serum_albumin']
df_day1['fluid_balance'] = df_day1['fluid_input'] - df_day1['fluid_output']
df_clinical = df_day1[['patient_id', 'label'] + clinical_cols].copy()

print("\n--- Running nested LOPO-CV: Logistic Regression ---")
(lr_metrics, lr_params, lr_true, lr_scores, lr_pred, lr_thres, lr_scalers, lr_models) = run_nested_lopo_lr(df_clinical, patients)

jk = jackknife_metrics(lr_true, lr_pred, lr_scores)

print("\n--- Metrics mean ± std (jackknife over outer folds) ---")
for k, v in jk.items():
    print(f"  {k:15s}: {v['mean']:.3f} ± {v['std']:.3f}  "
          f"[{v['min']:.3f} – {v['max']:.3f}]")

# Print chosen parameters per fold
print("\n--- LR hyperparameters chosen per outer fold ---")
print(f"  {'Fold':<6} {'Patient':<10} {'l1_ratio':>10} {'C':>10} {'class_weight':>14}")
print(f"  {'-'*52}")
for i, (p, params) in enumerate(zip(patients, lr_params)):
    print(f"  {i+1:<6} {p:<10} {params['l1_ratio']:>10.2f} "
          f"{params['C']:>10.4f} {str(params['class_weight']):>14}")

# Summary most frequent parameter values
from collections import Counter

l1_ratios    = [p['l1_ratio']     for p in lr_params]
cs           = [p['C']            for p in lr_params]
class_weights = [str(p['class_weight']) for p in lr_params]

print(f"\n--- Most frequent hyperparameters across folds ---")
print(f"  l1_ratio     : {Counter(l1_ratios).most_common(1)[0][0]:.2f}  "
      f"(counts: {dict(Counter(l1_ratios))})")
print(f"  C            : {Counter(cs).most_common(1)[0][0]}  "
      f"(counts: {dict(Counter(cs))})")
print(f"  class_weight : {Counter(class_weights).most_common(1)[0][0]}  "
      f"(counts: {dict(Counter(class_weights))})")

print(f"\n--- Permutation test (n={N_PERM}, parallel) ---")
t1 = time.time()

def run_one_permutation(seed):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'perm_{seed:04d}.npy')
    warnings.filterwarnings("ignore")
    if os.path.exists(checkpoint_path):
        return np.load(checkpoint_path, allow_pickle=True).item()
    local_rng = np.random.default_rng(seed)
    shuffled  = local_rng.permutation(patient_labels)
    m, *_= run_nested_lopo_lr(df_clinical, patients, shuffled_labels=shuffled)
    np.save(checkpoint_path, m)
    return m

existing = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.npy')]
print(f"Checkpoints already saved: {len(existing)} / {N_PERM}")

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_one_permutation)(seed)
    for seed in range(N_PERM)
)

perm_metrics = {k: np.array([r[k] for r in results]) for k in lr_metrics}

elapsed_p = time.time() - t1
print(f"Permutation test done in {elapsed_p/3600:.2f}h")

p_values = {
    k: float((np.sum(perm_metrics[k] >= lr_metrics[k]) + 1) / (len(perm_metrics[k]) + 1))
    for k in lr_metrics
}

print("\n--- Permutation test p-values ---")
for k in lr_metrics:
    print(f"  {k:15s}: observed={lr_metrics[k]:.4f}  "
          f"p={p_values[k]:.4f}  "
          f"({'*' if p_values[k] < 0.05 else 'ns'})")

# Build final LR model on all Day 1 patients
patient_clinical_day1 = {
    p: df_clinical[df_clinical['patient_id'] == p][clinical_cols].values.squeeze()
    for p in patients
}
X_all = np.vstack([patient_clinical_day1[p] for p in patients])
y_all = np.array([
    int(df_clinical[df_clinical['patient_id'] == p]['label'].iloc[0])
    for p in patients
])

imputer_day1  = SimpleImputer(strategy='median')
X_all_imp     = imputer_day1.fit_transform(X_all)

lr_scaler_day1 = StandardScaler()
X_all_s        = lr_scaler_day1.fit_transform(X_all_imp)

best_params_final = lr_params[
    [str(p) for p in lr_params].index(
        Counter([str(p) for p in lr_params]).most_common(1)[0][0]
    )
]

if best_params_final['C'] == np.inf:
    lr_final = LogisticRegression(
        penalty=None,
        solver='saga',
        class_weight=best_params_final['class_weight'],
        max_iter=10000,
        random_state=42
    )
else:
    lr_final = LogisticRegression(
        penalty='elasticnet',
        l1_ratio=best_params_final['l1_ratio'],
        C=best_params_final['C'],
        class_weight=best_params_final['class_weight'],
        solver='saga',
        max_iter=10000,
        random_state=42
    )
lr_final.fit(X_all_s, y_all)

final_lr_threshold = np.mean(lr_thres)

print(f"Final LR model: {best_params_final}")
print(f"Final LR threshold: {final_lr_threshold:.4f}")


# External validation function
def validate_external_lr(df_val, lr_model, imputer, scaler, threshold, clinical_cols):

    patients_val = df_val['patient_id'].unique()
    outer_true, outer_pred, outer_scores = [], [], []

    for p in patients_val:
        pat_clin = df_val[df_val['patient_id'] == p][clinical_cols].values.squeeze()
        X_p = imputer.transform(pat_clin.reshape(1, -1))
        X_p = scaler.transform(X_p)
        p_prob  = float(lr_model.predict_proba(X_p)[0, 1])
        p_pred  = int(p_prob >= threshold)
        p_true  = int(df_val[df_val['patient_id'] == p]['label'].iloc[0])
        outer_true.append(p_true)
        outer_pred.append(p_pred)
        outer_scores.append(p_prob)

    metrics = compute_metrics(
        np.array(outer_true),
        np.array(outer_pred),
        np.array(outer_scores)
    )
    return metrics, outer_true, outer_pred, outer_scores


# Run external validation
for day in ['day2', 'day3']:
    df_day = df[df['day'] == day].copy()
    df_day['fluid_balance'] = df_day['fluid_input'] - df_day['fluid_output']
    df_val = df_day[['patient_id', 'label'] + clinical_cols].copy()

    print(f"\n{day}: {len(df_val)} patients found")

    lr_ext_metrics, lr_ext_true, lr_ext_pred, lr_ext_scores = validate_external_lr(
        df_val        = df_val,
        lr_model      = lr_final,
        imputer       = imputer_day1,
        scaler        = lr_scaler_day1,
        threshold     = final_lr_threshold,
        clinical_cols = clinical_cols,
    )

    print(f"\n--- {day} LR external validation ---")
    for k, v in lr_ext_metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    jk = jackknife_metrics(
        outer_true   = lr_ext_true,
        outer_pred   = lr_ext_pred,
        outer_scores = lr_ext_scores
    )
    print(f"\n--- {day} LR metrics mean ± std (jackknife) ---")
    for k, v in jk.items():
        print(f"  {k:15s}: {v['mean']:.3f} ± {v['std']:.3f}")