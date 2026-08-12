import glob
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scatteringcorrection import preprocess_data
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed
import os
from utils import compute_metrics, get_threshold, jackknife_metrics,inner_cv_lv_selection


# Configuration

DATA_PATH  = r'C:\Users\Usuario\OneDrive - City, University of London\NeonatesGOSH2\Windowed_3days/*.csv'
SAVE_DIR   = r'C:\Users\Usuario\OneDrive - City, University of London\NeonatesGOSH2'
CHECKPOINT_DIR = r'C:\Users\Usuario\OneDrive - City, University of London\NeonatesGOSH2\perm_checkpoints_stratified_3days'

SPEC_ORIGIN_NM     = 899.084
SPEC_STEP_NM       = 1.5925
START_WV   =  round((1050 - SPEC_ORIGIN_NM) / SPEC_STEP_NM)
END_WV     =  round((1650 - SPEC_ORIGIN_NM) / SPEC_STEP_NM)
MAX_LV     = 20
N_PERM     = 1000
DAY        = 'day1'
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

print(START_WV, END_WV)

os.makedirs(CHECKPOINT_DIR,exist_ok=True)

files  = sorted(glob.glob(DATA_PATH))
df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df_all = df_all.sort_values('patient_id', kind='stable').reset_index(drop=True)
# df_all = df_all[df_all['day']== DAY].reset_index(drop=True)
df_all = df_all.drop(columns=['day'])

print(df_all.columns.tolist())
print(df_all.shape)

patients =sorted( df_all['patient_id'].unique())
n_patients = len(patients)

assert set(np.unique(df_all['label'].values)) == {0, 1}, \
    "Labels must be binary 0/1"

patient_labels = np.array([
    df_all[df_all['patient_id'] == p]['label'].values[0]
    for p in patients
])

print(f"Patients: {n_patients}  "
      f"(oedema={patient_labels.sum()}, "
      f"no-oedema={n_patients - patient_labels.sum()})")



# Core functions

def run_nested_lopo(df: pd.DataFrame,
                    patients_arr: np.ndarray,
                    shuffled_labels: np.ndarray | None = None
                    ) -> tuple[dict, list, list, list]:

    outer_true   = []
    outer_pred   = []
    outer_scores = []
    outer_thres  = []
    chosen_lvs   = []
    patient_windows = {
        p: df[df['patient_id'] == p].iloc[:, START_WV:END_WV].values
        for p in patients_arr
    }
    patient_window_labels = {
        p: df[df['patient_id'] == p]['label'].values
        for p in patients_arr
    }
    patient_labels = {
        p: int(df[df['patient_id'] == p]['label'].iloc[0])
        for p in patients_arr
    }

    for i, test_patient in enumerate(patients_arr):
        train_df = df[df['patient_id'] != test_patient].copy()
        test_df  = df[df['patient_id'] == test_patient].copy()

        if shuffled_labels is not None:
            label_map = {p: shuffled_labels[j]
                         for j, p in enumerate(patients_arr)}
            train_df['label'] = train_df['patient_id'].map(label_map)

            patient_window_labels_inner = {
                p: train_df[train_df['patient_id'] == p]['label'].values
                for p in patients_arr if p != test_patient
            }
            patient_labels_inner = {
                p: int(train_df[train_df['patient_id'] == p]['label'].iloc[0])
                for p in patients_arr if p != test_patient
            }
        else:
            patient_window_labels_inner = patient_window_labels
            patient_labels_inner = patient_labels

        inner_patients = sorted(train_df['patient_id'].unique())

        # Inner CV
        best_lv = inner_cv_lv_selection(
            train_patients=inner_patients,
            patient_windows=patient_windows,
            patient_window_labels=patient_window_labels_inner,
            patient_labels=patient_labels_inner,
        )
        chosen_lvs.append(best_lv)

        #  Retrain on all N-1 inner patients
        X_train = train_df.iloc[:, START_WV:END_WV].values
        X_test  = test_df.iloc[:,  START_WV:END_WV].values
        y_train = train_df['label'].values
        y_test  = test_df['label'].values

        X_train_f = preprocess_data(X_train, method="snv", reference=None)
        X_test_f  = preprocess_data(X_test,  method="snv", reference=None)
        scaler = StandardScaler()
        X_train_f = scaler.fit_transform(X_train_f)
        X_test_f = scaler.transform(X_test_f)

        pls = PLSRegression(n_components=best_lv)
        pls.fit(X_train_f, y_train)

        # Re-optimise threshold on full training fold
        y_train_pred = pls.predict(X_train_f).ravel()
        final_thresh = get_threshold(y_train_pred, y_train)

        # Patient level prediction
        y_test_pred   = pls.predict(X_test_f).ravel()
        patient_score = float(np.mean(y_test_pred))
        patient_label = int(patient_score >= final_thresh)

        true_label = int(y_test[0])
        if shuffled_labels is not None:
            true_label = int(shuffled_labels[i])

        outer_true.append(true_label)
        outer_pred.append(patient_label)
        outer_scores.append(patient_score)
        outer_thres.append(final_thresh)

    metrics = compute_metrics(
        np.array(outer_true),
        np.array(outer_pred),
        np.array(outer_scores)
    )

    return metrics, chosen_lvs, outer_true, outer_scores, outer_pred, outer_thres

print(df_all.iloc[:, START_WV:END_WV].columns.tolist())


print("\n--- Running stratified LOPO-CV ---")
t0 = time.time()

obs_metrics, chosen_lvs, outer_true, outer_scores, outer_pred, outer_thres = run_nested_lopo(df_all, patients)

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
print(f"\nChosen LVs per outer fold: {chosen_lvs}")
print(f"Most common LV: {max(set(chosen_lvs), key=chosen_lvs.count)}")
print("\n--- Observed metrics ---")
for k, v in obs_metrics.items():
    print(f"  {k:15s}: {v:.4f}")

print("\n--- Threshold per outer fold ---")
for i, (p, thr) in enumerate(zip(patients, outer_thres)):
    print(f"  Fold {i+1:2d}  patient={p}  threshold={thr:.4f}")

print(f"\n  Mean  : {np.mean(outer_thres):.4f}")
print(f"  Std   : {np.std(outer_thres, ddof=1):.4f}")
print(f"  Min   : {np.min(outer_thres):.4f}")
print(f"  Max   : {np.max(outer_thres):.4f}")

jk = jackknife_metrics(outer_true, outer_pred, outer_scores)

print("\n--- Metrics mean ± std (jackknife over outer folds) ---")
for k, v in jk.items():
    print(f"  {k:15s}: {v['mean']:.3f} ± {v['std']:.3f}  "
          f"[{v['min']:.3f} – {v['max']:.3f}]")


# Permutation test

print(f"\n--- Permutation test (n={N_PERM}, parallel) ---")
t1 = time.time()

def run_one_permutation(seed):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'perm_{seed:04d}.npy')
    if os.path.exists(checkpoint_path):
        return np.load(checkpoint_path, allow_pickle=True).item()
    local_rng = np.random.default_rng(seed)
    shuffled  = local_rng.permutation(patient_labels)
    m, _, _, _, _, _ = run_nested_lopo(df_all, patients, shuffled_labels=shuffled)
    np.save(checkpoint_path, m)
    return m

existing = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.npy')]
print(f"Checkpoints already saved: {len(existing)} / {N_PERM}")

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_one_permutation)(seed)
    for seed in range(N_PERM)
)

perm_metrics = {k: np.array([r[k] for r in results]) for k in obs_metrics}

elapsed_p = time.time() - t1
print(f"Permutation test done in {elapsed_p/3600:.2f}h")

p_values = {
    k: float((np.sum(perm_metrics[k] >= obs_metrics[k]) + 1) / (len(perm_metrics[k]) + 1))
    for k in obs_metrics
}

print("\n--- Permutation test p-values ---")
for k in obs_metrics:
    print(f"  {k:15s}: observed={obs_metrics[k]:.4f}  "
          f"p={p_values[k]:.6f}  "
          f"({'*' if p_values[k] < 0.05 else 'ns'})")

# Plots

metric_names   = list(obs_metrics.keys())
metric_labels  = ['Accuracy', 'Balanced\nAccuracy', 'Sensitivity',
                  'Specificity', 'F1 Score', 'AUC']
n_metrics = len(metric_names)

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

for idx, (mname, mlabel) in enumerate(zip(metric_names, metric_labels)):
    ax  = fig.add_subplot(gs[idx // 3, idx % 3])
    obs = obs_metrics[mname]
    pv  = p_values[mname]
    null_dist = perm_metrics[mname]

    ax.hist(null_dist, bins=40, color='steelblue', alpha=0.7,
            edgecolor='white', label='Null distribution')
    ax.axvline(obs, color='crimson', linewidth=2,
               label=f'Observed = {obs:.3f}')

    p95 = np.percentile(null_dist, 95)
    ax.axvline(p95, color='orange', linewidth=1.5, linestyle='--',
               label=f'95th pct = {p95:.3f}')

    sig_str = f'p = {pv:.6f}' + (' *' if pv < 0.05 else ' ns')
    ax.set_title(f'{mlabel}\n{sig_str}', fontsize=10)
    ax.set_xlabel('Metric value', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

fig.suptitle(
    f'PLS-DA Nested LOPO-CV — Permutation Test (n={N_PERM})',
    fontsize=13, fontweight='bold'
)

plt.show()


# LV selection frequency bar chart
fig2, ax2 = plt.subplots(figsize=(7, 3))
lv_counts = np.bincount(chosen_lvs, minlength=MAX_LV + 1)[1:]
ax2.bar(range(1, MAX_LV + 1), lv_counts, color='steelblue', edgecolor='white')
ax2.set_xlabel('Number of latent variables')
ax2.set_ylabel('Times selected (outer folds)')
ax2.set_xticks(range(1, MAX_LV + 1))
ax2.set_title(f'LV selection frequency across outer folds ')
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.show()

# Summary table

summary = pd.DataFrame({
    'Metric'          : metric_labels,
    'Observed'        : [obs_metrics[k] for k in metric_names],
    'Null mean'       : [perm_metrics[k].mean() for k in metric_names],
    'Null 95th pct'   : [np.percentile(perm_metrics[k], 95) for k in metric_names],
    'p-value'         : [p_values[k] for k in metric_names],
    'Significant'     : ['Yes' if p_values[k] < 0.05 else 'No'
                         for k in metric_names],
})
summary = summary.round(4)
print("\n--- Summary table ---")
print(summary.to_string(index=False))


#  Final model: train on ALL Day 1 patients
df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df_day1 = df_all[df_all['day'] == 'day1'].reset_index(drop=True)

X_day1 = df_day1.iloc[:, START_WV:END_WV].values
y_day1 = df_day1['label'].values

X_day1_snv    = preprocess_data(X_day1, method="snv", reference=None)
scaler_day1   = StandardScaler()
X_day1_scaled = scaler_day1.fit_transform(X_day1_snv)

# Train final model with LV selected by nested LOPO
final_lv        = max(set(chosen_lvs), key=chosen_lvs.count)
final_threshold = np.mean(outer_thres)

pls_final = PLSRegression(n_components=final_lv)
pls_final.fit(X_day1_scaled, y_day1)

print(f"Final model: LV={final_lv}, threshold={final_threshold:.4f}")

#  External validation function
def validate_external_day(df_all, day, pls_model, scaler,
                           threshold, start_wv, end_wv):

    df_day = df_all[df_all['day'] == day].reset_index(drop=True)
    patients_day = df_day['patient_id'].unique()

    print(f"\n {day}: {len(patients_day)} patients found")

    outer_true   = []
    outer_pred   = []
    outer_scores = []

    for p in patients_day:
        pat_df = df_day[df_day['patient_id'] == p]

        X_pat = pat_df.iloc[:, start_wv:end_wv].values
        y_pat = pat_df['label'].values

        # Apply Day 1 preprocessing — scaler fitted on Day 1 only
        X_pat_snv    = preprocess_data(X_pat, method="snv", reference=None)
        X_pat_scaled = scaler.transform(X_pat_snv)  # transform only, no fit

        # Predict
        y_scores     = pls_model.predict(X_pat_scaled).ravel()
        patient_score = float(np.mean(y_scores))
        patient_label = int(patient_score >= threshold)
        true_label    = int(y_pat[0])

        outer_true.append(true_label)
        outer_pred.append(patient_label)
        outer_scores.append(patient_score)

    metrics = compute_metrics(
        np.array(outer_true),
        np.array(outer_pred),
        np.array(outer_scores)
    )

    return metrics, outer_true, outer_pred, outer_scores

# Run for Day 2 and Day 3
for day in ['day2', 'day3']:
    metrics, true, pred, scores = validate_external_day(df_all, day, pls_final, scaler_day1,final_threshold, START_WV, END_WV)

    print(f"\n--- Day {day} external validation metrics ---")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    jk = jackknife_metrics(outer_true= true, outer_pred=pred,outer_scores=scores)
    print(f"\n--- Day {day} metrics mean ± std (jackknife) ---")
    for k, v in jk.items():
        print(f"  {k:15s}: {v['mean']:.3f} ± {v['std']:.3f}  ")



def label_starts(ax, labels, ys, colors, xs=None,
                 fontsize=7, min_gap_frac=0.045, x_pad=0.08):
    n = len(labels)
    if n == 0:
        return
    ys = np.asarray(ys, dtype=float)
    xs = np.zeros(n) if xs is None else np.asarray(xs, dtype=float)
    if isinstance(colors, str):
        colors = [colors] * n

    order = np.argsort(ys)
    ymin, ymax = ax.get_ylim()
    min_gap = (ymax - ymin) * min_gap_frac

    placed = ys.copy()
    for i in range(1, n):
        cur, prev = order[i], order[i - 1]
        if placed[cur] - placed[prev] < min_gap:
            placed[cur] = placed[prev] + min_gap

    for lab, x0, y_true, y_lab, c in zip(labels, xs, ys, placed, colors):
        moved = abs(y_lab - y_true) > 1e-9
        ax.annotate(
            lab,
            xy=(x0, y_true), xycoords='data',
            xytext=(x0 - x_pad, y_lab), textcoords='data',
            ha='right', va='center',
            fontsize=fontsize, color=c, fontweight='bold',
            arrowprops=dict(arrowstyle='-', color=c, lw=0.5, alpha=0.6)
                       if moved else None,
        )
    ax.set_xlim(left=float(xs.min()) - x_pad * 4)

def plot_pls_scores_across_days(df_all, pls_model, scaler, threshold,
                                 start_wv, end_wv, patient_labels):

    days     = ['day1', 'day2', 'day3']
    day_labels = ['Day 1', 'Day 2', 'Day 3']

    patient_scores = {}

    all_patients = df_all['patient_id'].unique()

    for p in all_patients:
        scores = []
        for day in days:
            pat_df = df_all[(df_all['day'] == day) & (df_all['patient_id'] == p)]
            if len(pat_df) == 0:
                scores.append(None)
                continue
            X_p = pat_df.iloc[:, start_wv:end_wv].values
            X_p = preprocess_data(X_p, method="snv", reference=None)
            X_p = scaler.transform(X_p)
            scores.append(float(np.mean(pls_model.predict(X_p))))
        patient_scores[p] = scores


    oedema     = [p for p in all_patients
                         if patient_labels[p] == 1 ]
    no_oedema         = [p for p in all_patients
                         if patient_labels[p] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(8,5), sharey=True,dpi=300)

    subgroups = [
        (oedema,     'Oedema ',       'firebrick',   's'),
        (no_oedema,         'No Oedema',             'steelblue',   '^'),
    ]

    x = [0, 1, 2]

    for ax, (patients, title, color, marker) in zip(axes, subgroups):
        labels, anchor_x, anchor_y = [], [], []

        for p in patients:
            scores = patient_scores[p]
            x_avail = [x[i] for i, s in enumerate(scores) if s is not None]
            s_avail = [s for s in scores if s is not None]
            if not x_avail:
                continue

            ax.plot(x_avail, s_avail, color=color, marker=marker,linewidth=1, markersize=4, alpha=0.7)

            labels.append(p)
            anchor_x.append(x_avail[0])
            anchor_y.append(s_avail[0])

        ax.axhline(y=threshold, color='black', linestyle='--',
                   linewidth=1.2, label=f'Threshold ({threshold:.3f})')
        ax.axhspan(threshold,
                  threshold + 1.2,
                   alpha=0.05, color='red', label='Oedema region')

        label_starts(ax, labels, anchor_y, color, xs=anchor_x)

        ax.set_title(title, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(day_labels)
        ax.tick_params(labelsize=5)
        ax.set_ylabel('PLS Score' if ax == axes[0] else '', fontsize=6)
        ax.set_xlabel('Day', fontsize=6)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


df_day1 = df_all[df_all['day'] == 'day1']
patient_labels_dict = {
    p: int(df_day1[df_day1['patient_id'] == p]['label'].iloc[0])
    for p in df_all['patient_id'].unique()
}

plot_pls_scores_across_days(
    df_all              = df_all,
    pls_model           = pls_final,
    scaler              = scaler_day1,
    threshold           = final_threshold,
    start_wv            = START_WV,
    end_wv              = END_WV,
    patient_labels      = patient_labels_dict)
    
