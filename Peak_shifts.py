import glob
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import mannwhitneyu
from scatteringcorrection import preprocess_data
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

DATA_PATH = r'C:\Users\adgk768\OneDrive - City, University of London\NeonatesGOSH2\Windowed_3days/*.csv'

all_files = glob.glob(DATA_PATH)

records = []
for fpath in all_files:
    df = pd.read_csv(fpath)
    patient_id = df['patient_id'].iloc[0]
    day        = df['day'].iloc[0]
    label      = df['label'].iloc[0]
    spectra    = df.iloc[:, 1:513].values     # spectral columns
    filt       = preprocess_data(spectra, method='snv', reference=None)
    mean_sp    = np.mean(filt, axis=0)
    records.append({
        'patient_id': patient_id,
        'day':        day,
        'label':      label,
        'mean_sp':    mean_sp,
    })

records_df = pd.DataFrame(records)
print(records_df)


wavelengths = np.linspace(900, 1700, 512)

WATER_PEAKS = {
    '970':  (940, 1010),
    '1200': (1150, 1260),
    '1450': (1400, 1520),
}


def find_peak_position(wavelengths, spectrum, window_nm):
    mask = (wavelengths >= window_nm[0]) & (wavelengths <= window_nm[1])
    wl_win = wavelengths[mask]
    sp_win = spectrum[mask]

    spline = UnivariateSpline(wl_win, sp_win, s=0, k=4)
    wl_dense = np.linspace(wl_win[0], wl_win[-1], 10_000)
    sp_dense = spline(wl_dense)

    idx_max = np.argmax(sp_dense)
    return wl_dense[idx_max], sp_dense[idx_max]

def band_moments(wavelengths, spectrum, window_nm, n_dense=10_000):
    mask   = (wavelengths >= window_nm[0]) & (wavelengths <= window_nm[1])
    wl_win, sp_win = wavelengths[mask], spectrum[mask]

    spline   = UnivariateSpline(wl_win, sp_win, s=0, k=4)
    wl_dense = np.linspace(wl_win[0], wl_win[-1], n_dense)
    sp_dense = spline(wl_dense)

    baseline = np.interp(wl_dense, [wl_dense[0], wl_dense[-1]],
                                   [sp_dense[0], sp_dense[-1]])
    band = np.clip(sp_dense - baseline, 0, None)

    area = np.trapezoid(band, wl_dense)
    if area <= 0:
        return np.nan, np.nan, np.nan

    centroid = np.trapezoid(band * wl_dense, wl_dense) / area
    width    = np.sqrt(np.trapezoid(band * (wl_dense - centroid)**2,
                                wl_dense) / area)
    half = band.max() / 2.0
    idx  = np.where(band >= half)[0]
    fwhm = (wl_dense[idx[-1]] - wl_dense[idx[0]]) if idx.size else np.nan
    return centroid, width, fwhm

peak_rows = []
for _, row in records_df.iterrows():
    entry = {'patient_id': row['patient_id'],
             'day':        row['day'],
             'label':      row['label']}
    for label_nm, window in WATER_PEAKS.items():
        peak_nm, _ = find_peak_position(wavelengths, row['mean_sp'], window)
        centroid, width, fwhm = band_moments(wavelengths, row['mean_sp'], window)
        entry[f'peak_{label_nm}'] = peak_nm
        entry[f'centroid_{label_nm}'] = centroid
        entry[f'width_{label_nm}'] = width
        entry[f'fwhm_{label_nm}'] = fwhm
    peak_rows.append(entry)

peak_df = pd.DataFrame(peak_rows)
print("--Peak dataframe--")
peak_df['patient_num'] = peak_df['patient_id'].str.extract(r'(\d+)').astype(int)
peak_df = peak_df.sort_values(['patient_num', 'day']).drop(columns='patient_num')

print(peak_df.to_string(index=False))

summary = (peak_df.groupby(['label','day'])[['peak_970', 'peak_1200', 'peak_1450']] .agg(['mean', 'std']).round(3))
print("\n--Summary--")
summary.to_excel("Summarypeaks.xlsx")
print(summary)


day1_peaks = (peak_df[peak_df['day'] == 'day1']
              .set_index('patient_id')
              [['peak_970', 'peak_1200', 'peak_1450']])

shift_rows = []
for _, row in peak_df[peak_df['day'] != 'day1'].iterrows():
    pid = row['patient_id']
    entry = {'patient_id': pid,
             'day':        row['day'],
             'label':      row['label']}
    for col in ['peak_970', 'peak_1200', 'peak_1450']:
        entry[col.replace('peak', 'shift')] = row[col] - day1_peaks.loc[pid, col]
    shift_rows.append(entry)

shift_df = pd.DataFrame(shift_rows)
print(shift_df)
shift_df.to_excel("shifts.xlsx")
summaryShift = (shift_df.groupby(['label','day'])[['shift_970', 'shift_1200', 'shift_1450']] .agg(['mean', 'std']).round(3))
print("\n--Summary shift--")
summaryShift.to_excel("Summaryshift.xlsx")
print(summaryShift)

days = ['day1','day2', 'day3']
peak_cols = ['peak_970', 'peak_1200', 'peak_1450']

peak_df['Condition'] = peak_df['label'].map({0: 'No_Oedema', 1: 'Oedema'})
peak_df['day_num']   = peak_df['day'].map({'day1': 1, 'day2': 2, 'day3': 3})

results = []
for col in peak_cols:
    formula = f"{col} ~ Condition + day_num"
    model   = smf.mixedlm(formula, data=peak_df, groups=peak_df['patient_id'])
    fit     = model.fit()
    results.append({
        'Feature':      col,
        'p_condition':  fit.pvalues.get('Condition[T.Oedema]', None)
    })

results_df_peaks = pd.DataFrame(results)
rejected, p_adj, _, _ = multipletests(results_df_peaks['p_condition'], method='fdr_bh')
results_df_peaks['p_adj']      = p_adj
results_df_peaks['significant'] = rejected

print('\n----MIXED MODEL - PEAKS----')
print(results_df_peaks)

shift_cols = ['shift_970', 'shift_1200', 'shift_1450']

shift_df['Condition'] = shift_df['label'].map({0: 'No_Oedema', 1: 'Oedema'})
shift_df['day_num']   = shift_df['day'].map({'day2': 2, 'day3': 3})

results_shift = []
for col in shift_cols:
    formula = f"{col} ~ Condition + day_num"
    model   = smf.mixedlm(formula, data=shift_df, groups=shift_df['patient_id'])
    fit     = model.fit()
    results_shift.append({
        'Feature':     col,
        'p_condition': fit.pvalues.get('Condition[T.Oedema]', None)
    })

results_shift_df = pd.DataFrame(results_shift)
rejected, p_adj, _, _ = multipletests(results_shift_df['p_condition'], method='fdr_bh')
results_shift_df['p_adj']       = p_adj
results_shift_df['significant'] = rejected

print('\n----MIXED MODEL - SHIFTS----')
print(results_shift_df)


test_results_day1 = []
sub1 = peak_df[peak_df['day'] == 'day1']
peak_cols = ['peak_970', 'peak_1200', 'peak_1450']
for col in peak_cols:
    grp0 = sub1.loc[sub1['label'] == 0, col].dropna()
    grp1 = sub1.loc[sub1['label'] == 1, col].dropna()
    U, p = mannwhitneyu(grp0, grp1, alternative='two-sided')
    n0, n1 = len(grp0), len(grp1)
    r_rb = 1 - (2 * U) / (n0 * n1)
    test_results_day1.append({
        'peak': col,
        'U': U, 'p': p,
        'n_label0': n0, 'n_label1': n1,
        'rank_biserial_r': r_rb
    })

results_df_day1 = pd.DataFrame(test_results_day1)


rejected, p_corr, _, _ = multipletests(results_df_day1['p'], alpha=0.05, method='fdr_bh')
results_df_day1['p_corrected'] = p_corr
results_df_day1['significant']  = rejected

print("\n----Day1 test----")
print(results_df_day1[['peak','U','n_label0','n_label1',
                   'p','p_corrected','rank_biserial_r','significant']].to_string(index=False))


day_map = {'day1': 1, 'day2': 2, 'day3': 3}
label_map = {0: 'No Oedema', 1: 'Oedema'}
colors = { 1: '#FF508A', 0: '#4DBEEE'}

peak_df['day_num'] = peak_df['day'].map(day_map)

fig, axes = plt.subplots(2, 3, figsize=(9, 4), dpi=300)

for ax, col in zip(axes[0], peak_cols):
    for lbl, grp in peak_df.groupby('label'):
        stats = grp.groupby('day_num')[col].agg(['mean', 'std'])
        ax.errorbar(stats.index, stats['mean'], yerr=stats['std'],
                    label=label_map[lbl], color=colors[lbl],
                    marker='o', capsize=3, linewidth=1.2)

    res = results_df_peaks.loc[results_df_peaks['Feature'] == col].iloc[0]
    ax.text(0.03, 0.97,
            f"p = {res['p_condition']:.3f}\np corr = {res['p_adj']:.3f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=5,
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='gray', alpha=0.9))

    ax.set_title(f'Peak ~ {col.split("_")[1]} nm', fontsize=6, fontweight='bold')
    ax.set_ylabel('Peak position (nm)', fontsize=6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Day 1', 'Day 2', 'Day 3'], fontsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(loc="upper right", fontsize=5)

shift_cols = ['shift_970', 'shift_1200', 'shift_1450']
shift_df['day_num'] = shift_df['day'].map(day_map)

for ax, col in zip(axes[1], shift_cols):
    for lbl, grp in shift_df.groupby('label'):
        stats = grp.groupby('day_num')[col].agg(['mean', 'std'])
        ax.errorbar(stats.index, stats['mean'], yerr=stats['std'],
                    label=label_map[lbl], color=colors[lbl],
                    marker='o', capsize=3, linewidth=1.2)

    res = results_shift_df.loc[results_shift_df['Feature'] == col].iloc[0]
    ax.text(0.03, 0.97,
            f"p = {res['p_condition']:.3f}\np corr = {res['p_adj']:.3f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=5,
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='gray', alpha=0.9))

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_title(f'Shift ~ {col.split("_")[1]} nm', fontsize=6, fontweight='bold')
    ax.set_ylabel('Shift from Day 1 (nm)', fontsize=6)
    ax.set_xticks([2, 3])
    ax.set_xticklabels(['Day 2', 'Day 3'], fontsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(loc="upper right", fontsize=5)

plt.tight_layout(h_pad=3.0)
plt.show()

BANDS = ['970', '1200', '1450']
day_num = {'day1': 1, 'day2': 2, 'day3': 3}

def run_mixedlm(df, feature_cols):
    df = df.copy()
    df['Condition'] = df['label'].map({0: 'No_Oedema', 1: 'Oedema'})
    df['day_num']   = df['day'].map(day_num)
    out = []
    for col in feature_cols:
        fit = smf.mixedlm(f"{col} ~ Condition + day_num", df,
                          groups=df['patient_id']).fit()
        out.append({'Feature': col,
                    'p_condition': fit.pvalues.get('Condition[T.Oedema]', np.nan)})
    res = pd.DataFrame(out)
    rej, padj, _, _ = multipletests(res['p_condition'], method='fdr_bh')
    res['p_adj'], res['significant'] = padj, rej
    return res

for metric in ['centroid', 'width', 'fwhm']:
    cols = [f'{metric}_{b}' for b in BANDS]

    print(f"\n--- MIXED MODEL: {metric} ---")
    print(run_mixedlm(peak_df, cols))

    d1 = peak_df[peak_df['day'] == 'day1'].set_index('patient_id')[cols]
    rows = []
    for _, r in peak_df[peak_df['day'] != 'day1'].iterrows():
        e = {'patient_id': r['patient_id'], 'day': r['day'], 'label': r['label']}
        for c in cols:
            e[c] = r[c] - d1.loc[r['patient_id'], c]
        rows.append(e)
    print(f"--- MIXED MODEL: {metric} shift ---")
    print(run_mixedlm(pd.DataFrame(rows), cols))