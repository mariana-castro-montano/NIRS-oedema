from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from utils import compute_metrics, get_threshold, jackknife_metrics, inner_cv_lr_selection, inner_cv_lv_selection
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob
from scatteringcorrection import preprocess_data
from sklearn.cross_decomposition import PLSRegression
import warnings
import time
import os
from joblib import Parallel, delayed
from collections import Counter

warnings.filterwarnings("ignore", message="A single label was found")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


DATA_PATH  = r'C:\Users\adgk768\OneDrive - City, University of London\NeonatesGOSH2\Windowed_3days/*.csv'
df_clinical = pd.read_csv(r'C:\Users\adgk768\OneDrive - City, University of London\NeonatesGOSH2\clinical_data.csv',sep=';')
CHECKPOINT_DIR = r'C:\Users\adgk768\OneDrive - City, University of London\NeonatesGOSH2\perm_fusion_new'
os.makedirs(CHECKPOINT_DIR,exist_ok=True)

SPEC_ORIGIN_NM     = 899.084
SPEC_STEP_NM       = 1.5925
START_WV   =  round((1050 - SPEC_ORIGIN_NM) / SPEC_STEP_NM)
END_WV     =  round((1650 - SPEC_ORIGIN_NM) / SPEC_STEP_NM)
MAX_LV     = 20
N_PERM     = 1000
DAY        = 'day1'
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

files  = glob.glob(DATA_PATH)
df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df_all = df_all[df_all['day']== DAY].reset_index(drop=True)
df_clinical_day1 = df_clinical[df_clinical['day'] == 'day1'].copy()

patients = df_all['patient_id'].unique()
n_patients = len(patients)

assert set(np.unique(df_all['label'].values)) == {0, 1}, \
    "Labels must be binary 0/1"

# Patient-level ground truth (one label per patient)
patient_labels = np.array([
    df_all[df_all['patient_id'] == p]['label'].values[0]
    for p in patients
])

print(f"Patients: {n_patients}  "
      f"(oedema={patient_labels.sum()}, "
      f"no-oedema={n_patients - patient_labels.sum()})")

def run_nested_lopo_fusion(df_spectral: pd.DataFrame,
                           df_clinical: pd.DataFrame,
                           patients_arr: np.ndarray,
                           shuffled_labels: np.ndarray | None = None
                           ) -> tuple:

    clinical_cols = [c for c in df_clinical.columns
                     if c not in ('patient_id', 'label', 'day')]

    feature_cols = df_spectral.iloc[:, START_WV:END_WV].columns
    patient_windows = {
        p: df_spectral[df_spectral['patient_id'] == p].iloc[:, START_WV:END_WV].values
        for p in patients_arr
    }
    patient_window_labels = {
        p: df_spectral[df_spectral['patient_id'] == p]['label'].values
        for p in patients_arr
    }
    patient_clinical = {
        p: df_clinical[df_clinical['patient_id'] == p][clinical_cols].values.squeeze()
        for p in patients_arr
    }
    patient_labels = {
        p: int(df_spectral[df_spectral['patient_id'] == p]['label'].iloc[0])
        for p in patients_arr
    }

    outer_true    = []
    outer_pred    = []
    outer_scores  = []
    outer_thres   = []
    outer_pls_norm = []
    outer_lr_probs = []
    chosen_lvs = []
    chosen_params_list = []

    for i, test_patient in enumerate(patients_arr):
        train_patients = [p for p in patients_arr if p != test_patient]

        if shuffled_labels is not None:
            label_map = {p: shuffled_labels[j]
                         for j, p in enumerate(patients_arr)}
            patient_labels_inner = {p: int(label_map[p])
                                    for p in train_patients}
        else:
            patient_labels_inner = {p: patient_labels[p]
                                    for p in train_patients}

        # PLS-DA branch

        if shuffled_labels is not None:
            pls_window_labels_train = {
                p: np.full(len(patient_window_labels[p]), int(label_map[p]))
                for p in train_patients
            }
        else:
            pls_window_labels_train = {p: patient_window_labels[p]
                                       for p in train_patients}

        best_lv = inner_cv_lv_selection(
            train_patients        = train_patients,
            patient_windows       = patient_windows,
            patient_window_labels = pls_window_labels_train,
            patient_labels        = patient_labels_inner,
            n_splits              = 5,
        )

        # Retrain PLS-DA on all training patients
        X_pls_train_list = []
        for p in train_patients:
            X_p = preprocess_data(patient_windows[p], method="snv", reference=None)
            X_pls_train_list.append(X_p)
        X_pls_train = np.vstack(X_pls_train_list)
        y_pls_train = np.concatenate([pls_window_labels_train[p]
                                      for p in train_patients])

        pls_scaler = StandardScaler()
        X_pls_train_s = pls_scaler.fit_transform(X_pls_train)

        pls = PLSRegression(n_components=best_lv)
        pls.fit(X_pls_train_s, y_pls_train)

        pls_train_scores = np.array([
            float(np.mean(pls.predict(
                pls_scaler.transform(
                    preprocess_data(patient_windows[p], method="snv", reference=None)
                )
            )))
            for p in train_patients
        ])

        # Min-max normalise PLS scores using training fold range
        pls_min = pls_train_scores.min()
        pls_max = pls_train_scores.max()

        pls_train_norm = (pls_train_scores - pls_min) / (pls_max - pls_min + 1e-8)
        X_test_pls = preprocess_data(patient_windows[test_patient],
                                     method="snv", reference=None)
        X_test_pls_s = pls_scaler.transform(X_test_pls)
        pls_test_score = float(np.mean(pls.predict(X_test_pls_s)))
        pls_test_norm  = (pls_test_score - pls_min) / (pls_max - pls_min + 1e-8)

        # LR branch

        best_params = inner_cv_lr_selection(
            train_patients   = train_patients,
            patient_clinical = patient_clinical,
            patient_labels   = patient_labels_inner,
        )

        X_lr_train = np.vstack([patient_clinical[p] for p in train_patients])
        y_lr_train = np.array([patient_labels_inner[p] for p in train_patients])

        imputer = SimpleImputer(strategy='median')
        X_lr_train_imp = imputer.fit_transform(X_lr_train)

        lr_scaler = StandardScaler()
        X_lr_train_s = lr_scaler.fit_transform(X_lr_train_imp)

        lr = LogisticRegression(
            l1_ratio=best_params['l1_ratio'],
            C=best_params['C'] if best_params['l1_ratio'] is not None else 1.0,
            class_weight=best_params['class_weight'],
            solver=best_params['solver'],
            penalty='elasticnet',
            max_iter=5000,
            random_state=42
        )
        lr.fit(X_lr_train_s, y_lr_train)

        # LR probabilities for training patients
        lr_train_probs = np.array([
            float(lr.predict_proba(
                lr_scaler.transform(
                    imputer.transform(patient_clinical[p].reshape(1, -1))
                )
            )[0, 1])
            for p in train_patients
        ])

        # LR probability for test patient
        X_test_lr_imp = imputer.transform(
            patient_clinical[test_patient].reshape(1, -1))
        X_test_lr_s = lr_scaler.transform(X_test_lr_imp)
        lr_test_prob = float(lr.predict_proba(X_test_lr_s)[0, 1])


        # Late fusion

        fused_train_scores = 0.5 * pls_train_norm + 0.5 * lr_train_probs
        fused_train_labels = np.array([patient_labels_inner[p]
                                       for p in train_patients])
        final_thresh = get_threshold(fused_train_scores, fused_train_labels)

        fused_test_score  = 0.5 * pls_test_norm + 0.5 * lr_test_prob
        fused_test_label  = int(fused_test_score >= final_thresh)

        true_label = patient_labels[test_patient]
        if shuffled_labels is not None:
            true_label = int(shuffled_labels[i])

        outer_true.append(true_label)
        outer_pred.append(fused_test_label)
        outer_scores.append(fused_test_score)
        outer_thres.append(final_thresh)
        outer_pls_norm.append(pls_test_norm)
        outer_lr_probs.append(lr_test_prob)
        chosen_lvs.append(best_lv)
        chosen_params_list.append(best_params)

    metrics = compute_metrics(
        np.array(outer_true),
        np.array(outer_pred),
        np.array(outer_scores)
    )

    return metrics, outer_true, outer_scores, outer_pred, outer_thres, outer_pls_norm, outer_lr_probs, chosen_lvs, chosen_params_list

def quantify_fusion_contributions(outer_pls_scores_norm, outer_lr_probs,
                                   outer_fused_scores, outer_true):

    pls_arr    = np.array(outer_pls_scores_norm)
    lr_arr     = np.array(outer_lr_probs)
    fused_arr  = np.array(outer_fused_scores)
    true_arr   = np.array(outer_true)

    pls_corr   = np.corrcoef(pls_arr,   true_arr)[0, 1]
    lr_corr    = np.corrcoef(lr_arr,    true_arr)[0, 1]
    fused_corr = np.corrcoef(fused_arr, true_arr)[0, 1]

    pls_fused_corr = np.corrcoef(pls_arr, fused_arr)[0, 1]
    lr_fused_corr  = np.corrcoef(lr_arr,  fused_arr)[0, 1]

    var_pls   = np.var(pls_arr)
    var_lr    = np.var(lr_arr)
    cov_pl    = np.cov(pls_arr, lr_arr)[0, 1]
    var_fused = 0.25 * var_pls + 0.25 * var_lr + 2 * 0.25 * cov_pl

    pls_contribution = (0.25 * var_pls + 0.25 * cov_pl) / var_fused * 100
    lr_contribution  = (0.25 * var_lr  + 0.25 * cov_pl) / var_fused * 100

    pls_oedema    = pls_arr[true_arr == 1].mean()
    pls_no_oedema = pls_arr[true_arr == 0].mean()
    lr_oedema     = lr_arr[true_arr == 1].mean()
    lr_no_oedema  = lr_arr[true_arr == 0].mean()

    print("\n--- Fusion modality contributions ---")
    print(f"\n  Correlation with true label:")
    print(f"    PLS-DA : {pls_corr:.3f}")
    print(f"    LR     : {lr_corr:.3f}")
    print(f"    Fusion : {fused_corr:.3f}")

    print(f"\n  Correlation with fused score:")
    print(f"    PLS-DA : {pls_fused_corr:.3f}")
    print(f"    LR     : {lr_fused_corr:.3f}")

    print(f"\n  Variance contribution to fused score:")
    print(f"    PLS-DA : {pls_contribution:.1f}%")
    print(f"    LR     : {lr_contribution:.1f}%")

    print(f"\n  Mean score by class:")
    print(f"    {'':10} {'PLS-DA':>10} {'LR':>10}")
    print(f"    {'Oedema':10} {pls_oedema:>10.3f} {lr_oedema:>10.3f}")
    print(f"    {'No oedema':10} {pls_no_oedema:>10.3f} {lr_no_oedema:>10.3f}")
    print(f"\n  Score separation (oedema - no oedema):")
    print(f"    PLS-DA : {pls_oedema - pls_no_oedema:.3f}")
    print(f"    LR     : {lr_oedema  - lr_no_oedema:.3f}")

    return {
        'pls_corr_true':    pls_corr,
        'lr_corr_true':     lr_corr,
        'fused_corr_true':  fused_corr,
        'pls_var_contrib':  pls_contribution,
        'lr_var_contrib':   lr_contribution,
        'pls_separation':   pls_oedema - pls_no_oedema,
        'lr_separation':    lr_oedema  - lr_no_oedema,
    }


df_clinical_model = df_clinical_day1[['patient_id', 'label']].drop_duplicates().copy()
df_clinical_model['fluid_balance']    = df_clinical_day1['fluid_input'] - df_clinical_day1['fluid_output']
df_clinical_model['gestational_age']  = df_clinical_day1['gestational_age']
df_clinical_model['birth_weight']     = df_clinical_day1['birth_weight']
df_clinical_model['age']              = df_clinical_day1['age']
df_clinical_model['weight']           = df_clinical_day1['weight']
df_clinical_model['serum_albumin']    = df_clinical_day1['serum_albumin']

print("\n--- Running nested LOPO-CV: Late Fusion ---")
(fusion_metrics, fusion_true, fusion_scores, fusion_pred, fusion_thres,  fusion_pls_norm, fusion_lr_probs,  fusion_lvs, fusion_params) = run_nested_lopo_fusion(df_all, df_clinical_model, patients)

jk = jackknife_metrics(fusion_true, fusion_pred, fusion_scores)

print("\n--- Metrics mean ± std (jackknife over outer folds) ---")
for k, v in jk.items():
    print(f"  {k:15s}: {v['mean']:.3f} ± {v['std']:.3f}  "
          f"[{v['min']:.3f} – {v['max']:.3f}]")

print(f"\n  Mean  : {np.mean(fusion_thres):.4f}")
print(f"  Std   : {np.std(fusion_thres, ddof=1):.4f}")

contrib = quantify_fusion_contributions(
    outer_pls_scores_norm = fusion_pls_norm,
    outer_lr_probs        = fusion_lr_probs,
    outer_fused_scores    = fusion_scores,
    outer_true            = fusion_true
)

print(f"\n--- Permutation test (n={N_PERM}, parallel) ---")
t1 = time.time()

def run_one_permutation(seed):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'perm_{seed:04d}.npy')
    warnings.filterwarnings("ignore")
    if os.path.exists(checkpoint_path):
        return np.load(checkpoint_path, allow_pickle=True).item()
    local_rng = np.random.default_rng(seed)
    shuffled  = local_rng.permutation(patient_labels)
    m, *_= run_nested_lopo_fusion(df_all, df_clinical_model, patients, shuffled_labels=shuffled)
    np.save(checkpoint_path, m)
    return m

existing = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.npy')]
print(f"Checkpoints already saved: {len(existing)} / {N_PERM}")

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_one_permutation)(seed)
    for seed in range(N_PERM)
)

perm_metrics = {k: np.array([r[k] for r in results]) for k in fusion_metrics}

elapsed_p = time.time() - t1
print(f"Permutation test done in {elapsed_p/3600:.2f}h")

p_values = {
    k: float((np.sum(perm_metrics[k] >= fusion_metrics[k]) + 1) / (len(perm_metrics[k]) + 1))
    for k in fusion_metrics
}

print("\n--- Permutation test p-values ---")
for k in fusion_metrics:
    print(f"  {k:15s}: observed={fusion_metrics[k]:.4f}  "
          f"p={p_values[k]:.4f}  "
          f"({'*' if p_values[k] < 0.05 else 'ns'})")

# Build final models on all Day 1 patients
clinical_cols = [c for c in df_clinical_model.columns
                 if c not in ('patient_id', 'label', 'day')]

patient_clinical_day1 = {
    p: df_clinical_model[df_clinical_model['patient_id'] == p][clinical_cols].values.squeeze()
    for p in patients
}

# PLS-DA final model
X_pls_all = []
for p in patients:
    X_p = preprocess_data(
        df_all[df_all['patient_id'] == p].iloc[:, START_WV:END_WV].values,
        method="snv", reference=None
    )
    X_pls_all.append(X_p)
X_pls_all   = np.vstack(X_pls_all)
y_pls_all   = df_all['label'].values

pls_scaler_final = StandardScaler()
X_pls_all_s      = pls_scaler_final.fit_transform(X_pls_all)

final_lv = Counter(fusion_lvs).most_common(1)[0][0]
print(f"Most frequent LV across fusion folds: {final_lv} "
      f"(counts: {dict(Counter(fusion_lvs))})")

pls_final = PLSRegression(n_components=final_lv)
pls_final.fit(X_pls_all_s, y_pls_all)

pls_day1_scores = np.array([
    float(np.mean(pls_final.predict(
        pls_scaler_final.transform(
            preprocess_data(
                df_all[df_all['patient_id'] == p].iloc[:, START_WV:END_WV].values,
                method="snv", reference=None
            )
        )
    )))
    for p in patients
])
pls_day1_min = pls_day1_scores.min()
pls_day1_max = pls_day1_scores.max()

# LR final model
X_lr_all = np.vstack([patient_clinical_day1[p] for p in patients])
y_lr_all = np.array([
    int(df_clinical_model[df_clinical_model['patient_id'] == p]['label'].iloc[0])
    for p in patients
])

imputer_final  = SimpleImputer(strategy='median')
X_lr_all_imp   = imputer_final.fit_transform(X_lr_all)
lr_scaler_final = StandardScaler()
X_lr_all_s     = lr_scaler_final.fit_transform(X_lr_all_imp)

best_params_final = Counter(
    [str(p) for p in fusion_params]
).most_common(1)[0][0]
best_params_final = fusion_params[
    [str(p) for p in fusion_params].index(best_params_final)
]
print(f"Most frequent LR params across fusion folds: {best_params_final}")

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
lr_final.fit(X_lr_all_s, y_lr_all)

final_fusion_threshold = np.mean(fusion_thres)
print(f"\nFinal fusion threshold: {final_fusion_threshold:.4f}")


#External temporal validation: Days 2 & 3
def validate_external_fusion(df_all_spec, df_clinical_full, day,
                              pls_model, pls_scaler, pls_min, pls_max,
                              lr_model, lr_imputer, lr_scaler,
                              threshold, clinical_cols,
                              patient_clinical_day1, patients_arr):

    df_spec_day  = df_all_spec[df_all_spec['day'] == day]
    df_clin_day  = df_clinical_full[df_clinical_full['day'] == day]
    patients_day = df_spec_day['patient_id'].unique()

    print(f"\n{day}: {len(patients_day)} patients found")

    outer_true, outer_pred, outer_scores = [], [], []
    outer_pls_norm, outer_lr_probs       = [], []

    for p in patients_day:

        pat_spec = df_spec_day[df_spec_day['patient_id'] == p]
        X_spec   = pat_spec.iloc[:, START_WV:END_WV].values
        X_spec   = preprocess_data(X_spec, method="snv", reference=None)
        X_spec_s = pls_scaler.transform(X_spec)

        pls_score = float(np.mean(pls_model.predict(X_spec_s)))
        pls_norm  = (pls_score - pls_min) / (pls_max - pls_min + 1e-8)

        pat_clin = df_clin_day[df_clin_day['patient_id'] == p]
        if len(pat_clin) == 0:
            X_clin = patient_clinical_day1[p].reshape(1, -1)
        else:
            X_clin = pat_clin[clinical_cols].iloc[0].values.reshape(1, -1)

        X_clin_imp = lr_imputer.transform(X_clin)
        X_clin_s   = lr_scaler.transform(X_clin_imp)
        lr_prob    = float(lr_model.predict_proba(X_clin_s)[0, 1])

        fused_score = 0.5 * pls_norm + 0.5 * lr_prob
        fused_pred  = int(fused_score >= threshold)
        true_label  = int(pat_spec['label'].iloc[0])

        outer_true.append(true_label)
        outer_pred.append(fused_pred)
        outer_scores.append(fused_score)
        outer_pls_norm.append(pls_norm)
        outer_lr_probs.append(lr_prob)

    metrics = compute_metrics(
        np.array(outer_true),
        np.array(outer_pred),
        np.array(outer_scores)
    )
    return metrics, outer_true, outer_pred, outer_scores, outer_pls_norm, outer_lr_probs


#  Load full spectral data for all days
df_all_days = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

#  Build day-specific clinical dataframes
def build_clinical_day(df_clinical_raw, day):
    df_day = df_clinical_raw[df_clinical_raw['day'] == day].copy()
    df_out = df_day[['patient_id', 'label']].drop_duplicates().copy()
    df_out['fluid_balance']   = df_day['fluid_input'] - df_day['fluid_output']
    df_out['gestational_age'] = df_day['gestational_age']
    df_out['birth_weight']    = df_day['birth_weight']
    df_out['age']             = df_day['age']
    df_out['weight']          = df_day['weight']
    df_out['serum_albumin']   = df_day['serum_albumin']
    df_out['day']             = day
    return df_out

df_clinical_all_days = pd.concat([
    build_clinical_day(df_clinical, day)
    for day in ['day1', 'day2', 'day3']
], ignore_index=True)


df_clinical_all_days = df_clinical_all_days.sort_values(['patient_id', 'day'])
df_clinical_all_days['weight'] = (df_clinical_all_days
                                   .groupby('patient_id')['weight']
                                   .ffill())

# Run external validation
for day in ['day2', 'day3']:
    (ext_metrics, ext_true, ext_pred,
     ext_scores, ext_pls, ext_lr) = validate_external_fusion(
        df_all_spec           = df_all_days,
        df_clinical_full      = df_clinical_all_days,
        day                   = day,
        pls_model             = pls_final,
        pls_scaler            = pls_scaler_final,
        pls_min               = pls_day1_min,
        pls_max               = pls_day1_max,
        lr_model              = lr_final,
        lr_imputer            = imputer_final,
        lr_scaler             = lr_scaler_final,
        threshold             = final_fusion_threshold,
        clinical_cols         = clinical_cols,
        patient_clinical_day1 = patient_clinical_day1,
        patients_arr          = patients
    )

    print(f"\n--- {day} fusion external validation ---")
    for k, v in ext_metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    jk = jackknife_metrics(ext_true, ext_pred, ext_scores)
    print(f"\n--- {day} fusion metrics mean ± std (jackknife) ---")
    for k, v in jk.items():
        print(f"  {k:15s}: {v['mean']:.3f} ± {v['std']:.3f}")

    contrib = quantify_fusion_contributions(
        outer_pls_scores_norm = ext_pls,
        outer_lr_probs        = ext_lr,
        outer_fused_scores    = ext_scores,
        outer_true            = ext_true
    )
