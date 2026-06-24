import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import numpy as np
from matplotlib.lines import Line2D

warnings.simplefilter("ignore", ConvergenceWarning)

df = pd.read_excel(
    r'C:\Users\adgk768\OneDrive - City, University of London\NeonatesGOSH2\features_snv.xlsx',
    keep_default_na=False
)
pd.set_option("display.max_columns", None)

df['Condition'] = df['Condition'].astype('category')
df['Days'] = df['Days'].astype('category')
feature_cols = df.columns.difference(['Patient', 'Condition', 'Days'])

condition_map = {'None': 0, 'Oedema': 1}
df['condition_num'] = df['Condition'].map(condition_map)

titles_map = {
    col: title for col, title in zip(
        [c for c in df.columns if c in feature_cols],
        ["Amplitude 970 nm", "Ratio 970/1050", "Amplitude 1200 nm",
         "Ratio 1200/1050", "Amplitude 1450 nm", "Ratio 1450/1050", "Area 1400-1480nm"]
    )
}

cond_order = ['None', 'Oedema']
cond_labels = {'None': 'No oedema', 'Oedema': 'Oedema'}
box_colors = {'None': '#4DBEEE', 'Oedema': '#FF508A'}
SIG, NS = "#FF508A", "#4B5563"


def fmt_p(val):
    return "< 0.001" if f"{val:.3f}" == "0.000" else f"{val:.3f}"


def lollipop_plot(plot_df, raw_col, adj_col, title):
    plot_df = plot_df.sort_values(adj_col, ascending=False).reset_index(drop=True)
    plot_df["nlp_raw"] = -np.log10(plot_df[raw_col])
    plot_df["nlp_adj"] = -np.log10(plot_df[adj_col])

    sig_line = -np.log10(0.05)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for yi, r, a in zip(y, plot_df["nlp_raw"], plot_df["nlp_adj"]):
        ax.plot([r, a], [yi, yi], color="#C9CDD3", lw=2.5, zorder=1)

    ax.scatter(plot_df["nlp_raw"], y, s=75, marker="o",
               color=np.where(plot_df[raw_col] < 0.05, SIG, NS), zorder=2)
    ax.scatter(plot_df["nlp_adj"], y, s=85, marker="D",
               color=np.where(plot_df[adj_col] < 0.05, SIG, NS), zorder=3)

    ax.axvline(sig_line, ls="--", color="black", lw=1)
    ax.text(sig_line + 0.04, 0.95, "p = 0.05",
            transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=9)

    for yi, a, p in zip(y, plot_df["nlp_adj"], plot_df[adj_col]):
        ax.text(a + 0.08, yi, fmt_p(p), va="center", fontsize=8, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Feature"])
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.set_xlim(0, max(plot_df["nlp_adj"].max(), plot_df["nlp_raw"].max()) + 0.7)
    ax.set_title(title, fontsize=10)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SIG, markersize=9, label='p-value < 0.05'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NS,  markersize=9, label='p-value ≥ 0.05'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=SIG, markersize=8, label='$p_{corr}$ < 0.05'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=NS,  markersize=8, label='$p_{corr}$ ≥ 0.05'),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    plt.tight_layout()
    plt.show()


def boxplot_grid(df_src, order, pmap, raw_key, adj_key):
    n = len(order)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.4 * nrows),
                             dpi=300, constrained_layout=True)
    axes = axes.flatten()

    for ax, feat in zip(axes, order):
        t = titles_map.get(feat, feat)
        data = [df_src.loc[df_src['Condition'] == c, feat].dropna().values for c in cond_order]
        lw = 0.8
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        boxprops=dict(linewidth=lw),
                        whiskerprops=dict(linewidth=lw),
                        capprops=dict(linewidth=lw),
                        medianprops=dict(color='black', linewidth=lw),
                        flierprops=dict(markeredgewidth=lw, markersize=5))
        for patch, c in zip(bp['boxes'], cond_order):
            patch.set_facecolor(box_colors[c])
            patch.set_alpha(0.75)
            patch.set_edgecolor('#444444')

        ax.set_xticks([1, 2])
        ax.set_xticklabels([cond_labels[c] for c in cond_order], fontsize=5)

        p = pmap[feat][raw_key]
        padj = pmap[feat][adj_key]
        sig = padj < 0.05
        ax.set_title(t, fontsize=6, fontweight='bold')
        ax.text(0.5, 0.985, f"p = {fmt_p(p)}\n$p_{{corr}}$ = {fmt_p(padj)}",
                transform=ax.transAxes, ha='center', va='top', fontsize=4,
                linespacing=1.3, color='#555555',
                fontweight='bold' if sig else 'normal')
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.28 * (y1 - y0))
        ax.tick_params(labelsize=5)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.show()


#Mixed-effects model (all days)

mixed_rows = []
for feature in feature_cols:
    fit = smf.mixedlm(f"{feature} ~ Condition + Days", data=df,
                      groups=df['Patient']).fit()
    mixed_rows.append({
        "Feature": feature,
        "p_condition": fit.pvalues.get("Condition[T.Oedema]", None)
    })

results_mixed = pd.DataFrame(mixed_rows)
results_mixed["p_adj"] = multipletests(results_mixed["p_condition"], method="fdr_bh")[1]

print("MIXED MODEL (all days)")
print(results_mixed)

lollipop_plot(results_mixed, raw_col="p_condition", adj_col="p_adj",
              title="Mixed-effects model — all days")

order_mixed = [c for c in df.columns if c in results_mixed['Feature'].values]
pmap_mixed = results_mixed.set_index('Feature')[['p_condition', 'p_adj']].to_dict('index')
boxplot_grid(df, order_mixed, pmap_mixed, raw_key='p_condition', adj_key='p_adj')


#  Mann-Whitney on Day1→Day3 delta
tidy = df.melt(
    id_vars=['Patient', 'Condition', 'condition_num', 'Days'],
    value_vars=df.columns[3:],
    var_name='Feature',
    value_name='Value'
)

subset_days = tidy[tidy['Days'].isin(['Day 1', 'Day 3'])].copy()
pivot = subset_days.pivot_table(
    index=['Patient', 'condition_num', 'Feature'],
    columns='Days',
    values='Value',
    observed=False
).reset_index()
pivot['Delta'] = pivot['Day 3'] - pivot['Day 1']

delta_rows = []
for feature in pivot['Feature'].unique():
    subset = pivot[pivot['Feature'] == feature]
    group_none   = subset[subset['condition_num'] == 0]['Delta']
    group_oedema = subset[subset['condition_num'] == 1]['Delta']
    if len(group_none) > 0 and len(group_oedema) > 0:
        u_stat, p_val = mannwhitneyu(group_none, group_oedema, alternative='two-sided')
        n1, n2 = len(group_none), len(group_oedema)
        delta_rows.append({
            "Feature": feature,
            "p_value": p_val,
            "rank_biserial_r": 1 - (2 * u_stat) / (n1 * n2)
        })

results_delta = pd.DataFrame(delta_rows)
results_delta["p_adj"] = multipletests(results_delta["p_value"], method="fdr_bh")[1]

print("\nMANN-WHITNEY — Day1→Day3 delta")
print(results_delta)


#  Day 1 only  Mann-Whitney

df_day1 = df[df['Days'] == 'Day 1'].copy()

tidy_day1 = df_day1.melt(
    id_vars=['Patient', 'Condition', 'condition_num', 'Days'],
    value_vars=df_day1.columns[3:],
    var_name='Feature',
    value_name='Value'
)

day1_rows = []
for feature in tidy_day1['Feature'].unique():
    subset = tidy_day1[tidy_day1['Feature'] == feature]
    group_none   = subset[subset['condition_num'] == 0]['Value']
    group_oedema = subset[subset['condition_num'] == 1]['Value']
    if len(group_none) > 0 and len(group_oedema) > 0:
        u_stat, p_val = mannwhitneyu(group_none, group_oedema, alternative='two-sided')
        n1, n2 = len(group_none), len(group_oedema)
        day1_rows.append({
            "Feature": feature,
            "p_value": p_val,
            "rank_biserial_r": 1 - (2 * u_stat) / (n1 * n2)
        })

results_day1 = pd.DataFrame(day1_rows)
results_day1["p_adj"] = multipletests(results_day1["p_value"], method="fdr_bh")[1]

print("\nMANN-WHITNEY — Day 1 only")
print(results_day1)

lollipop_plot(results_day1, raw_col="p_value", adj_col="p_adj",
              title="Mann-Whitney U — Day 1 only")

order_day1 = [c for c in df_day1.columns if c in results_day1['Feature'].values]
pmap_day1 = results_day1.set_index('Feature')[['p_value', 'p_adj']].to_dict('index')
boxplot_grid(df_day1, order_day1, pmap_day1, raw_key='p_value', adj_key='p_adj')
