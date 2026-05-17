import argparse
import re
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, mean_squared_error, roc_auc_score, confusion_matrix
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import pearsonr, spearmanr
from datetime import datetime
import matplotlib.pyplot as plt


# Helper function to get the name part without the last character and extension
def get_name_without_last_char(filename):
    return filename[:-6]  # Remove the last letter and ".mrxs"


def add_col_to_patient_df(file_path: str, my_file_path: str, save_file: str, add_cols: list[str], from_cols: list[str]):
    block_id_pattern = r'\d+-\d+_\d+_\d+'
    patient_df = pd.read_excel(file_path)
    her2_df = pd.read_csv(my_file_path)
    for add_col in add_cols:
        patient_df[add_col] = ''

    # for i, gil_row in gil_df.iterrows():
    for i, her2_row in her2_df.iterrows():
        her2_block_id = re.match(block_id_pattern, her2_row['SlideName'])[0]
        patient_matches = patient_df[patient_df['BlockID'].str.replace('/', '_') == her2_block_id]

        # if not her2_matches.empty:
        if not patient_matches.empty:
            for add_col, from_col in zip(add_cols, from_cols):
                patient_df.at[patient_matches.index[0], add_col] = her2_row[from_col]
        else:
            print(her2_block_id)

    patient_df.to_excel(save_file, index=False)

    print(f"Updated file '{save_file}' has been created.")


def mark_matching_blocks(file_path: str, my_file_path: str, save_file: str):
    add_col = 'has_pair'
    block_id_pattern = r'\d+-\d+_\d+_\d+'
    gil_df = pd.read_excel(file_path)
    her2_df = pd.read_csv(my_file_path)
    her2_df['MatchedPattern'] = her2_df['SlideName'].str.extract(f'({block_id_pattern})', expand=False)
    gil_df[add_col] = ''
    counter = 0
    pattern_counts = her2_df['MatchedPattern'].value_counts()

    # for i, gil_row in gil_df.iterrows():
    for i, her2_row in her2_df.iterrows():
        her2_block_id = re.match(block_id_pattern, her2_row['SlideName'])[0]
        gil_matches = gil_df[gil_df['BlockID'].str.replace('/', '_') == her2_block_id]
        # block_id = gil_row['BlockID'].replace('/', '_')
        # her2_matches = her2_df[her2_df['SlideName'].str.startswith(block_id)]

        # if not her2_matches.empty:
        if not gil_matches.empty:
            if gil_df.at[i, add_col] == 1:
                print(f'{her2_block_id} again')
            gil_df.at[i, add_col] = 1
            counter += 1
        else:
            print(her2_block_id)

    print(counter)
    # gil_df.to_excel(save_file, index=False)

    print(f"Updated file '{save_file}' has been created.")


def update_excel_file(file_path: str, save_file: str, output_dirs: list):
    # Load the Excel files into DataFrames
    if file_path.endswith('xlsx'):
        her2_df = pd.read_excel(file_path)
    else:
        her2_df = pd.read_csv(file_path)

    # dirs = ['IHC_to_Her2_score', 'IHC_to_Her2_status', 'HE_to_Her2_score', 'HE_to_Her2_status']
    # batch_path = 'slides_data_HER2'
    batch_dfs = {}
    for output_dir in output_dirs:
        her2_df[output_dir] = ''
        batch_dfs[output_dir] = []
        full_output_dir = os.path.join(os.getcwd(), f'outputs/mpp{args.target_mpp}', output_dir)
        
        if 'disc' in output_dir:
            config = 'her2_multi_class'
        elif 'score' in output_dir:
            config = 'her2'
        else:
            config = 'her2_status'
        op_dir_w_cfg = os.path.join(full_output_dir, config)
        infer_dirs = [d for d in os.listdir(op_dir_w_cfg) if 'infer' in d and not d.endswith('.out')]
        print(f'infer_dirs = {infer_dirs}')
        for idir in infer_dirs:
            csv_path = os.path.join(op_dir_w_cfg, idir, f'eval_pretrained_{config}', 'inference_results',
                                    'slide_scores.csv')
            batch_dfs[output_dir].append(pd.read_csv(csv_path))
            print(f'batch_dfs[output_dir] = {batch_dfs[output_dir]}')
        batch_dfs[output_dir] = pd.concat(batch_dfs[output_dir], ignore_index=True)[['slide_name', 'score']]
    # her2_df.dropna(inplace=True)

    # Iterate over each file in the Her2 DataFrame
    for i, her2_row in her2_df.iterrows():
        # Find all matching HE slides with the same base name
        for key, batch_df in batch_dfs.items():
            slide_key = "SlideName" if 'IHC' in key else "Matched_HE_SlideName"
            her2_slidename = her2_row[slide_key]
            if not type(her2_slidename) == str:
                continue  # Skip if the slide name is not a string
            batch_matches = batch_df[batch_df['slide_name'].str.startswith(her2_slidename.split('.')[0])]

            if not batch_matches.empty:
                her2_df.at[i, key] = batch_matches['score'].values[0]

    her2_df.to_csv(save_file, index=False)

    print(f"Updated file '{save_file}' has been created.")


def compute_ci_pval(baseline_dfs, local_dfs, pred_tile_y, use_gt, only_equivocal=False, per_slide=False, n_boot=1000, **kwargs):
    add_necessary_columns(kwargs.get('file_paths', False), baseline_dfs, local_dfs, per_slide=per_slide)
    metrics = [roc_auc_score, mean_squared_error, spearmanr]
    remove_HER2_2_cases(baseline_dfs, local_dfs)

    if only_equivocal:
        for i in range(len(baseline_dfs)):
            baseline_dfs[i] = baseline_dfs[i][(baseline_dfs[i]['Her2 score'] < 3) & (baseline_dfs[i]['Her2 score'] > 1)]
            local_dfs[i] = local_dfs[i][(local_dfs[i]['Her2 score'] < 3) & (local_dfs[i]['Her2 score'] > 1)]


    if not pred_tile_y:
        group_dfs_by_patient(baseline_dfs, local_dfs, save_dfs=kwargs.get('save_dfs', False), save_dir=kwargs.get('save_dir', False), 
                             per_slide=per_slide)

    if kwargs.get('test', False):
        baseline_dfs, local_dfs = ensemble_prediction(baseline_dfs, local_dfs, save_dir=kwargs.get('save_dir', False))

    if kwargs.get('save_dir', False):
        save_diff_df(baseline_dfs, local_dfs, save_dir=kwargs.get('save_dir', None), save_qt=kwargs.get('save_qt', False), qt_dir=kwargs.get('qt_dir', None))

    if not kwargs.get('only_plot', False):
        for metric_func in metrics:
            diff, diff_ci, pe_baseline_mean, pe_local_mean, baseline_ci, local_ci, pval = bootstrap_diff_cv(
                metric_func, baseline_dfs, local_dfs, pred_tile_y=pred_tile_y, use_gt=use_gt, only_equivocal=only_equivocal, n_boot=n_boot
            )

            print(
                f'Tile-level {metric_func.__name__}: '
                f'baseline CI = {pe_baseline_mean:.4f} [{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}], '
                f'local CI = {pe_local_mean:.4f} [{local_ci[0]:.4f}, {local_ci[1]:.4f}], '
                f'diff = {diff:.4f}, 95% CI = [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}], '
                f'p-value = {pval:.4f}'
            )
    
    return baseline_dfs, local_dfs

def remove_HER2_2_cases(baseline_dfs, local_dfs):
    for i in range(len(baseline_dfs)):
        baseline_dfs[i] = baseline_dfs[i][baseline_dfs[i]['Her2 score'] != 2]
        local_dfs[i] = local_dfs[i][local_dfs[i]['Her2 score'] != 2]

def add_necessary_columns(csv_path, baseline_dfs, local_dfs, per_slide=False):
    read_columns = ['file', 'patient barcode', 'Her2_status', 'Her2 score', 'fold']
    text_cols = ['file'] # 'slide_name'
    if not per_slide:
        text_cols.append('slide_name')
    her2_df_full = pd.read_csv(csv_path)
    her2_df = her2_df_full[read_columns]
    her2_df['slide_name'] = her2_df['file'].apply(lambda x: x.split('.')[0].rsplit("_", 1)[0] + '_')
    for i in range(len(baseline_dfs)):
        baseline_dfs[i]['slide_name'] = baseline_dfs[i]['slide_name'].apply(lambda x: x.rsplit("_", 1)[0] + '_')
        local_dfs[i]['slide_name'] = local_dfs[i]['slide_name'].apply(lambda x: x.rsplit("_", 1)[0] + '_')
        baseline_dfs[i] = baseline_dfs[i].merge(her2_df, left_on='slide_name', right_on='slide_name', how='left').drop(columns=text_cols)
        local_dfs[i] = local_dfs[i].merge(her2_df, left_on='slide_name', right_on='slide_name', how='left').drop(columns=text_cols)


def save_diff_df(baseline_dfs, local_dfs, save_dir=None, save_qt=False, qt_dir=None):
    diff_dfs = []
    for i in range(len(baseline_dfs)):
        diff_df = local_dfs[i].copy()
        diff_df['baseline_score'] = baseline_dfs[i]['score']
        diff_df['score_diff'] = (local_dfs[i]['score'] - baseline_dfs[i]['score']).abs()
        diff_df['baseline_label_diff'] = (baseline_dfs[i]['score'] - baseline_dfs[i]['Her2_status']).abs()
        diff_df['local_label_diff'] = (local_dfs[i]['score'] - local_dfs[i]['Her2_status']).abs()
        diff_dfs.append(diff_df)
    diff_df_all = pd.concat(diff_dfs, ignore_index=True)
    scores = diff_df_all['score'].values.reshape(-1, 1)

    qt_path = os.path.join(qt_dir, 'score_quantile_transformer.pkl')
    if save_qt:
        qt = QuantileTransformer(output_distribution='uniform', n_quantiles=len(scores))
        scaled_scores = qt.fit_transform(scores)
        with open(qt_path, 'wb') as f:
            pickle.dump(qt, f)
    else:
        with open(qt_path, 'rb') as f:
            qt = pickle.load(f)
            scaled_scores = qt.transform(scores)

    diff_df_all['score_scaled'] = scaled_scores
    if save_dir is not None:
        diff_df_all.to_csv(os.path.join(save_dir, 'score_differences.csv'), index=False)
    


def ensemble_prediction(baseline_dfs, local_dfs, save_dir=None):
    # drop the patient_barcode column and return it after averaging
    patient_barcode_df = baseline_dfs[0][['slide_name', 'patient barcode']]
    ensemble_baseline_df = pd.concat(baseline_dfs, ignore_index=True)
    ensemble_baseline_df = ensemble_baseline_df.drop(columns=['patient barcode']).groupby('slide_name').mean().reset_index()
    ensemble_baseline_df = ensemble_baseline_df.merge(patient_barcode_df, left_on='slide_name', right_on='slide_name', how='inner')
    ensemble_local_df = pd.concat(local_dfs, ignore_index=True).drop(columns=['patient barcode']).groupby('slide_name').mean().reset_index()
    ensemble_local_df = ensemble_local_df.merge(patient_barcode_df, left_on='slide_name', right_on='slide_name', how='inner')
    if save_dir is not None:
        ensemble_baseline_df.to_csv(os.path.join(save_dir, f'baseline_ensemble.csv'), index=False)
        ensemble_local_df.to_csv(os.path.join(save_dir, f'local_ensemble.csv'), index=False)
    return [ensemble_baseline_df], [ensemble_local_df]


def group_dfs_by_patient(baseline_dfs, local_dfs, save_dfs=False, save_dir=None, per_slide=False):
    for i in range(len(baseline_dfs)):
        df_b = baseline_dfs[i]
        df_l = local_dfs[i]
        if not per_slide:
            per_pat_b_df = df_b.groupby('patient barcode').mean()
            per_pat_l_df = df_l.groupby('patient barcode').mean()
            per_pat_b_df['Her2_status'] = (per_pat_b_df['Her2_status'] >= 0.5).astype(int)
            per_pat_l_df['Her2_status'] = (per_pat_l_df['Her2_status'] >= 0.5).astype(int)
            df_b = per_pat_b_df.reset_index()
            df_l = per_pat_l_df.reset_index()
        
        baseline_dfs[i] = df_b
        local_dfs[i] = df_l
    
    if save_dfs and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(len(baseline_dfs)):
            baseline_dfs[i].to_csv(os.path.join(save_dir, f'baseline_per_patient_fold{i+1}.csv'), index=False)
            local_dfs[i].to_csv(os.path.join(save_dir, f'local_per_patient_fold{i+1}.csv'), index=False)
    

def bootstrap_diff_cv(metric_func, baseline_dfs, local_dfs, pred_tile_y, use_gt, only_equivocal=False, n_boot=1000):
    rng = np.random.default_rng(42)
    K = len(baseline_dfs)

    # Bootstrap within-fold
    boot_baseline = []
    boot_local = []
    boot_diff = []
    pe_baseline = []
    pe_local = []

    if pred_tile_y:
        gt_col = 'tile_y'
        gt_bin_col = 'tile_y_bin'
        pred_col = 'tile_pred'
        if use_gt:
            gt_col = 'tile_gt'
            gt_bin_col = 'tile_gt_bin'
            pred_col = 'map_tile_pred'
    else:
        gt_col = 'label'
        gt_bin_col = 'Her2_status'
        pred_col = 'score'

    for _ in range(n_boot):
        fold_bs_baseline = []
        fold_bs_local = []

        # bootstrap each fold independently
        for i in range(K):
            df_b = baseline_dfs[i]
            df_l = local_dfs[i]
            # patient barcode column has duplicate values, create a new column with unique values for bootstrapping
            df_b['patient_index'] = df_b.groupby('patient barcode').ngroup()
            df_l['patient_index'] = df_l.groupby('patient barcode').ngroup()

            gt = df_b[gt_col].values
            gt_bin = df_b[gt_bin_col].values
            pred_b = df_b[pred_col].values
            pred_l = df_l[pred_col].values

            # n_i = len(gt)
            unique_patients = df_b['patient_index'].unique()
            n_i = len(unique_patients)
            idx_p = rng.choice(n_i, size=n_i, replace=True)
            # get the corresponding indices for the original dataframe based on the patient_index
            idx = df_b['patient_index'].isin(unique_patients[idx_p])
            while len(np.unique(gt_bin[idx])) < 2:  # Ensure both classes are present for AUC
                idx = rng.choice(n_i, size=n_i, replace=True)

            if metric_func == roc_auc_score:
                m_b = metric_func(gt_bin[idx], pred_b[idx])
                m_l = metric_func(gt_bin[idx], pred_l[idx])
            else:
                m_b = metric_func(gt[idx], pred_b[idx])
                m_l = metric_func(gt[idx], pred_l[idx])
                if metric_func == spearmanr:
                    m_b = m_b[0]
                    m_l = m_l[0]

            fold_bs_baseline.append(m_b)
            fold_bs_local.append(m_l)

        # Average metrics across folds for this bootstrap sample
        boot_baseline.append(np.mean(fold_bs_baseline))
        boot_local.append(np.mean(fold_bs_local))
        boot_diff.append(np.mean(fold_bs_local) - np.mean(fold_bs_baseline))

    boot_baseline = np.array(boot_baseline)
    boot_local = np.array(boot_local)
    boot_diff = np.array(boot_diff)

    # Confidence intervals
    baseline_ci = np.percentile(boot_baseline, [2.5, 97.5])
    local_ci = np.percentile(boot_local, [2.5, 97.5])
    diff_ci = np.percentile(boot_diff, [2.5, 97.5])

    # p-value based on sign of difference
    if np.mean(boot_diff) > 0:
        pval = np.mean(boot_diff <= 0)
    else:
        pval = np.mean(boot_diff >= 0)

    # Point-estimate
    for i in range(K):
        df_b = baseline_dfs[i]
        df_l = local_dfs[i]
        
        gt = df_b[gt_col].values
        gt_bin = df_b[gt_bin_col].values
        pred_b = df_b[pred_col].values
        pred_l = df_l[pred_col].values

        if metric_func == roc_auc_score:
            m_b = metric_func(gt_bin, pred_b)
            m_l = metric_func(gt_bin, pred_l)
        else:
            m_b = metric_func(gt, pred_b)
            m_l = metric_func(gt, pred_l)
            if metric_func == spearmanr:
                m_b = m_b[0]
                m_l = m_l[0]
        
        pe_baseline.append(m_b)
        pe_local.append(m_l)
    
    pe_baseline_mean = np.mean(np.array(pe_baseline))
    pe_local_mean = np.mean(np.array(pe_local))

    return np.mean(boot_diff), diff_ci, pe_baseline_mean, pe_local_mean, baseline_ci, local_ci, pval

# -----------------------------
# Core metric computation
# -----------------------------
def compute_clinical_metrics(y_true, y_scores, thresholds):
    sens, spec, ppv, npv, impact = [], [], [], [], []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        ppv.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        npv.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
        impact.append(np.mean(y_pred))

    return (
        np.array(sens),
        np.array(spec),
        np.array(ppv),
        np.array(npv),
        np.array(impact),
    )

# -----------------------------
# Bootstrap CI
# -----------------------------
def bootstrap_metrics(y_true, y_scores, thresholds, n_boot=1000):
    rng = np.random.default_rng(42)
    n = len(y_true)

    sens_all, spec_all, ppv_all, npv_all, impact_all = [], [], [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)

        # ensure both classes exist
        if len(np.unique(y_true[idx])) < 2:
            continue

        sens, spec, ppv, npv, impact = compute_clinical_metrics(
            y_true[idx], y_scores[idx], thresholds
        )

        sens_all.append(sens)
        spec_all.append(spec)
        ppv_all.append(ppv)
        npv_all.append(npv)
        impact_all.append(impact)

    def ci(arr):
        arr = np.array(arr)
        return np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)

    return {
        "sens": ci(sens_all),
        "spec": ci(spec_all),
        "ppv": ci(ppv_all),
        "npv": ci(npv_all),
        "impact": ci(impact_all),
    }

def get_point(y_true, y_scores, chosen_t=0.6):
    y_pred = (y_scores >= chosen_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    impact = np.mean(y_pred)
    return sens, spec, ppv, npv, impact

# -----------------------------
# Plot function
# -----------------------------
def plot_threshold_analysis(
    y_true,
    baseline_scores,
    local_scores,
    chosen_t=0.6,
    n_boot=1000,
    c_dict=None, 
    save_dir=None
):
    num_thresholds = 200
    chosen_t_ind = int(chosen_t * (num_thresholds)) -1  # index of the chosen threshold
    thresholds = np.linspace(0, 1, num_thresholds)

    # Compute clinical metrics
    # sens_b, spec_b, ppv_b, impact_b = compute_clinical_metrics(y_true, baseline_scores, thresholds)
    sens_l, spec_l, ppv_l, npv_l, impact_l = compute_clinical_metrics(y_true, local_scores, thresholds)

    # Bootstrap CI (only for local model to keep plot clean)
    ci_l = bootstrap_metrics(y_true, local_scores, thresholds, n_boot=n_boot)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(8,6))

    # --- LOCAL MODEL (solid + CI) ---
    plt.plot(thresholds, spec_l, color=c_dict["Specificity"], linewidth=2, label="Specificity")
    plt.fill_between(thresholds, ci_l["spec"][0], ci_l["spec"][1], color=c_dict["Specificity"], alpha=0.1)

    plt.plot(thresholds, sens_l, color=c_dict["Sensitivity"], linewidth=2, label="Sensitivity")
    plt.fill_between(thresholds, ci_l["sens"][0], ci_l["sens"][1], color=c_dict["Sensitivity"], alpha=0.1)

    plt.plot(thresholds, ppv_l, color=c_dict["PPV"], linewidth=2, label="PPV")
    plt.fill_between(thresholds, ci_l["ppv"][0], ci_l["ppv"][1], color=c_dict["PPV"], alpha=0.1)

    plt.plot(thresholds, npv_l, color=c_dict["NPV"], linewidth=2, label="NPV")
    plt.fill_between(thresholds, ci_l["npv"][0], ci_l["npv"][1], color=c_dict["NPV"], alpha=0.1)

    plt.plot(thresholds, impact_l, color=c_dict["Impacted patients"], linewidth=2, label="Impacted patients")
    plt.fill_between(thresholds, ci_l["impact"][0], ci_l["impact"][1], color=c_dict["Impacted patients"], alpha=0.1)

    # Vertical threshold line
    plt.axvline(chosen_t, color="gray", linestyle="--")

    # -----------------------------
    # Annotate chosen threshold
    # -----------------------------
    point = get_point(y_true, local_scores, chosen_t=chosen_t)
    sens, spec, ppv, npv, impact = point

    print(f"Specificity: {spec} (95% CI: [{ci_l['spec'][0][chosen_t_ind]}, {ci_l['spec'][1][chosen_t_ind]}])")
    print(f"Sensitivity: {sens} (95% CI: [{ci_l['sens'][0][chosen_t_ind]}, {ci_l['sens'][1][chosen_t_ind]}])")
    print(f"PPV: {ppv} (95% CI: [{ci_l['ppv'][0][chosen_t_ind]}, {ci_l['ppv'][1][chosen_t_ind]}])")
    print(f"NPV: {npv} (95% CI: [{ci_l['npv'][0][chosen_t_ind]}, {ci_l['npv'][1][chosen_t_ind]}])")
    print(f"Impacted patients: {impact} (95% CI: [{ci_l['impact'][0][chosen_t_ind]}, {ci_l['impact'][1][chosen_t_ind]}])")

    plt.scatter([chosen_t], [spec], color=c_dict["Specificity"])
    plt.scatter([chosen_t], [sens], color=c_dict["Sensitivity"])
    plt.scatter([chosen_t], [ppv], color=c_dict["PPV"])
    plt.scatter([chosen_t], [npv], color=c_dict["NPV"])
    plt.scatter([chosen_t], [impact], color=c_dict["Impacted patients"])

    plt.text(chosen_t + 0.11, spec + 0.01, f"{spec:.3f}", color=c_dict["Specificity"], fontsize=15) # chosen_t + 0.12, spec + 0.01
    plt.text(chosen_t - 0.005, sens + 0.01, f"{sens:.3f}", color=c_dict["Sensitivity"], fontsize=15) # sens - 0.03
    plt.text(chosen_t + 0.11, ppv - 0.05, f"{ppv:.3f}", color=c_dict["PPV"], fontsize=15) # ppv - 0.1
    plt.text(chosen_t - 0.005, npv + 0.01, f"{npv:.3f}", color=c_dict["NPV"], fontsize=15)
    plt.text(chosen_t - 0.005, impact + 0.01, f"{impact:.3f}", color=c_dict["Impacted patients"], fontsize=15)

    # -----------------------------
    # Styling
    # -----------------------------
    plt.gca().invert_xaxis()  # match paper
    plt.xlabel("Threshold", fontsize=15)
    plt.ylabel("Performance & Impacted patients fraction", fontsize=15)
    plt.title("HER2 Status Threshold Analysis", fontsize=15)

    plt.legend(fontsize=15)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_dir, f"threshold_analysis_{chosen_t}.png"), dpi=300, bbox_inches='tight') if save_dir else None
    plt.close()

    # plot_impact_analysis(sens_l, spec_l, ppv_l, npv_l, impact_l, ci_l, chosen_t, chosen_t_ind, point, save_dir=save_dir)

def plot_impact_analysis(y_true, local_scores, chosen_t=[0.6], n_boot=1000, c_dict=None, save_dir=None):
    # sort local_scores and corresponding y_true by local_scores
    sorted_indices = np.argsort(local_scores)
    local_scores_sorted = local_scores[sorted_indices]
    y_true_sorted = y_true[sorted_indices]
    thresholds = []

    # iteratively increase impacted patient fraction and compute metrics
    sens_l, spec_l, ppv_l, npv_l, impact_l = [], [], [], [], []
    for i in range(len(local_scores_sorted)):
        # y_scores = local_scores_sorted[:i+1]
        # y_true_subset = y_true_sorted[:i+1]
        curr_threshold = local_scores_sorted[i]
        sens, spec, ppv, npv, impact = compute_clinical_metrics(y_true, local_scores, thresholds=[curr_threshold])
        thresholds.append(curr_threshold)
        sens_l.append(sens[0])
        spec_l.append(spec[0])
        ppv_l.append(ppv[0])
        npv_l.append(npv[0])
        impact_l.append(impact[0])
    
    # Bootstrap CI (only for local model to keep plot clean)
    ci_l = bootstrap_metrics(y_true, local_scores, thresholds=thresholds, n_boot=n_boot)

    # plot the metrics as a function of impacted patient fraction
    # --- LOCAL MODEL (solid + CI) ---
    plt.plot(impact_l, spec_l, color=c_dict["Specificity"], linewidth=2, label="Specificity")
    plt.fill_between(impact_l, ci_l["spec"][0], ci_l["spec"][1], color=c_dict["Specificity"], alpha=0.1)

    plt.plot(impact_l, sens_l, color=c_dict["Sensitivity"], linewidth=2, label="Sensitivity")
    plt.fill_between(impact_l, ci_l["sens"][0], ci_l["sens"][1], color=c_dict["Sensitivity"], alpha=0.1)

    plt.plot(impact_l, ppv_l, color=c_dict["PPV"], linewidth=2, label="PPV")
    plt.fill_between(impact_l, ci_l["ppv"][0], ci_l["ppv"][1], color=c_dict["PPV"], alpha=0.1)

    plt.plot(impact_l, npv_l, color=c_dict["NPV"], linewidth=2, label="NPV")
    plt.fill_between(impact_l, ci_l["npv"][0], ci_l["npv"][1], color=c_dict["NPV"], alpha=0.1)

    for t in chosen_t:
        point = get_point(y_true, local_scores, chosen_t=t) # chosen_t
        sens, spec, ppv, npv, impact = point

        # Vertical threshold line
        plt.axvline(impact, color="gray", linestyle="--")
        # show the x-axis value at the chosen impact point
        plt.text(impact + 0.005, 0.02, f"{impact:.3f}", color="gray", fontsize=13)

        plt.scatter([impact], [spec], color=c_dict["Specificity"])
        plt.scatter([impact], [sens], color=c_dict["Sensitivity"])
        plt.scatter([impact], [ppv], color=c_dict["PPV"])
        plt.scatter([impact], [npv], color=c_dict["NPV"])

        plt.text(impact + 0.015, spec + 0.01, f"{spec:.2f}", color=c_dict["Specificity"], fontsize=13) # chosen_t + 0.12, spec + 0.01
        # plt.annotate(f"{spec:.3f}", xy=(impact, spec), xytext=(0.015, 0.01), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.8) # Masks the line)
        plt.text(impact - 0.09, sens - 0.01, f"{sens:.2f}", color=c_dict["Sensitivity"], fontsize=13) # sens - 0.03
        plt.text(impact - 0.09, ppv + 0.04, f"{ppv:.2f}", color=c_dict["PPV"], fontsize=13) # ppv - 0.1 , + 0.06
        plt.text(impact - 0.09, npv - 0.09, f"{npv:.2f}", color=c_dict["NPV"], fontsize=13)

    # -----------------------------
    # Styling
    # -----------------------------
    plt.xlabel("Impacted patients fraction", fontsize=15)
    plt.ylabel("Performance", fontsize=15)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title("HER2 Status Impact Analysis", fontsize=15)

    plt.legend(fontsize=15, loc='center right')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    t_str = "_".join([str(t) for t in chosen_t])
    plt.savefig(os.path.join(save_dir, f"impact_analysis_{t_str}_equal_impact_sampling.png"), dpi=300, bbox_inches='tight') if save_dir else None
    # plt.savefig(os.path.join(save_dir, f"impact_analysis_{chosen_t}_equal_impact_sampling.png"), dpi=300, bbox_inches='tight') if save_dir else None


def compute_metrics(her2_file_path, score_cols, label_col, plot_labels, folds=[], save_dir=None):
    read_columns = score_cols + ['patient barcode', label_col, 'fold']
    her2_df_full = pd.read_csv(her2_file_path)
    her2_df = her2_df_full[read_columns]
    aucs_dict = {plot_label: [] for plot_label in plot_labels}
    today = datetime.today().date()
    if save_dir is not None:
        save_dir = os.path.join(save_dir, f'{score_cols[0]} {today}')
    for i in folds:
        temp_her2_df = her2_df[her2_df['fold'] == i]
        temp_her2_df.dropna(inplace=True)
        temp_her2_df = temp_her2_df.reset_index()
        print(f'len(temp_her2_df) = {len(temp_her2_df)}')
        y_trues2_3, y_trues3, y_trues_bin, y_scores = [], [], [], []
        for score_col in score_cols:
            valid_indices = temp_her2_df.index[
                (temp_her2_df[label_col] != "Missing Data") & (~temp_her2_df[score_col].isna())].tolist()
            per_pat_her2_df = temp_her2_df.iloc[valid_indices].groupby('patient barcode').mean()

            y_true = per_pat_her2_df[label_col].values.astype(float)
            # multiclass
            if "score" in label_col: # HER2 score
                # Case 1: [0, 0.5, 1] vs [2, 3]
                binary_labels_case1 = np.isin(y_true, [2, 3]).astype(int)
                y_trues2_3.append(binary_labels_case1)
                # Case 2: [0, 0.5, 1, 2] vs [3]
                binary_labels_case2 = np.isin(y_true, [3]).astype(int)
                y_trues3.append(binary_labels_case2)
            else: # HER2_status
                y_true_binary = (y_true >= 0.5).astype(int)  # Convert to 0 or 1
                y_trues_bin.append(y_true_binary)
            y_score = per_pat_her2_df[score_col].values
            y_scores.append(y_score)
        patient_num = len(per_pat_her2_df)
        
        if len(y_trues3) > 0:
            aucs2_3 = show_roc_and_calc_auc(y_trues=y_trues2_3, y_scores=y_scores,
                                  score_title=f'[0, 0.5, 1] vs [2, 3] {score_cols[0]} {today} ({patient_num} patients)',
                                  plot_labels=plot_labels, save_dir=save_dir)
            aucs3 = show_roc_and_calc_auc(y_trues=y_trues3, y_scores=y_scores,
                                  score_title=f'[0, 0.5, 1, 2] vs [3] {score_cols[0]} {today} ({patient_num} patients)',
                                  plot_labels=plot_labels, save_dir=save_dir)
        else:
            aucs = show_roc_and_calc_auc(y_trues=y_trues_bin, y_scores=y_scores,
                                  score_title=f'Her2 positive vs negative {score_cols[0]} fold {i} {today} ({patient_num} patients)',
                                  plot_labels=plot_labels, save_dir=save_dir)
            for plot_label in plot_labels:
                aucs_dict[plot_label] += aucs[plot_label]

    row_num = her2_df.shape[0]
    for score_col, plot_label in zip(score_cols, plot_labels):
        mean_auc = np.mean(np.array(aucs_dict[plot_label]))
        print(f"Mean AUC for {plot_label}: {mean_auc:.4f}")
        her2_df_full.at[row_num - 1, score_col] = mean_auc
    her2_df_full.to_csv(her2_file_path, index=False)
        

def show_roc_and_calc_auc(y_trues, y_scores, score_title, plot_labels, save_dir=None):
    plt.figure(figsize=(8, 6))
    aucs = {plot_label: [] for plot_label in plot_labels}
    for y_true, y_score, plot_label in zip(y_trues, y_scores, plot_labels):
        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # Calculate AUROC
        roc_auc = auc(fpr, tpr)
        aucs[plot_label].append(roc_auc)

        # Plot ROC Curve
        plt.plot(fpr, tpr, label=f'{plot_label} ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{score_title} ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{score_title} ROC Curve.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return aucs


def show_pca_per_class(args):
    from sklearn.decomposition import PCA

    her2_df = pd.read_csv(args.file_paths) #['SlideName', 'label', 'fold']

    if 'HE_to_Her2' in args.output_dirs[0] or 'CAT' in args.output_dirs[0] or 'reg' in args.output_dirs[0]:
        slide_key = 'Matched_HE_SlideName'
    else:
        slide_key = 'SlideName'

    if len(args.folds) < 5:
        her2_df = her2_df[her2_df['fold'] == int(args.folds)]
    
    if args.slide_name is not None:
        args.output_dirs[0] = os.path.join(args.output_dirs[0], args.slide_name)
        args.tile_score_dir = os.path.join(args.tile_score_dir, args.slide_name) if args.tile_score_dir is not None else None
        her2_df = her2_df[her2_df[slide_key] == f"{args.slide_name}.mrxs"]
        print(f'*******her2_df.shape = {her2_df.shape}')

    embeddings = None
    pca = PCA(n_components=2)
    rows_to_drop = []
    for i, her2_row in her2_df.iterrows():
        slide_name = her2_row[slide_key].split('.')[0]
        if args.slide_name is not None:
            if args.comp_embeds is not None:
                embed_file = f"{args.output_dirs[0].split('/')[-2].split(f'_mpp{args.target_mpp}')[0]}"
            else:
                embed_file = f"tile_embeds_{slide_name}"  # original tile embeddings
        else:
            embed_file = slide_name  # slide embedding in inference
        full_file_path = os.path.join(args.output_dirs[0], f'{embed_file}.npy')
        if embeddings is None:
            embeddings = np.load(full_file_path)
        else:
            try:
                embeddings = np.hstack([embeddings, np.load(full_file_path)])
            except BaseException as e:
                # print(e)
                rows_to_drop.append(i)

    her2_df = her2_df.drop(rows_to_drop)
    if not args.slide_name:
        embeddings = embeddings.reshape(len(her2_df), -1)

    print(f'*******embeddings.shape = {embeddings.shape}')
    while(len(embeddings.shape)) > 2:
        embeddings = embeddings.squeeze(0)
    embeddings_2d = pca.fit_transform(embeddings)  # Reduce to 2D
    print(f'*******embeddings_2d.shape = {embeddings_2d.shape}')

    # ===== Plot the PCA Results =====
    plt.figure(figsize=(8, 6))
    if args.tile_score_dir is not None:
        scores_path = os.path.join(args.tile_score_dir, 'tile_scores.npy')
        labels = np.load(scores_path)
    else:
        labels = her2_df['label']
    print(f'*******labels.shape = {labels.shape}')
    colors = {0: 'red', 0.5: 'orange', 1: 'yellow', 2: 'blue', 3: 'purple'}
    unique_labels = list(colors.keys())

    for idx, label in enumerate(unique_labels):
        if not args.tile_score_dir:
            mask = labels == label
        else:
            next_label = unique_labels[idx + 1] if idx != len(unique_labels) - 1 else 10  # big value
            mask = (label < labels) & (labels <= next_label)
        # print(f'embeddings_2d.shape = {embeddings_2d.shape}')
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    label=f'Her2 Class {label}', alpha=0.7, color=colors[label])

    prefix = ''
    if len(args.plot_labels) > 0:
        prefix = f'{args.plot_labels[0]}'
    if args.slide_name is not None:
        prefix += f'_{args.slide_name}'
        

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"2D PCA Projection of {prefix} Embeddings")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(args.save_dir, f'{prefix}_embeds_PCA.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure at {save_path}")


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Search and replace file names and content based on Excel mapping.")
    # parser.add_argument('-r', '--root', required=True, help='Root directory to search for files.')
    parser.add_argument('-u',   '--update_file', action='store_true', default=False, help='Whether to update the file')
    parser.add_argument('-bp',  '--base_path', type=str, help='Base path for outputs', required=True)
    parser.add_argument('-sp',  '--save_path', type=str, help='Path to save the updated file')
    parser.add_argument('-od',  '--output_dirs', nargs='+', help='Output dirs from which to take output files')
    parser.add_argument('-f',   '--file_paths', type=str, help='Excel file/s to work with')
    parser.add_argument('-sf',  '--second_file_path', type=str, help='Patient excel file to update')
    parser.add_argument('-sd',  '--save_dir', type=str, help='Directory to save metrics plots')
    parser.add_argument('-qtd',  '--qt_dir', type=str, help='Directory to save quantile transformer')
    parser.add_argument('-sqt',  '--save_qt', action='store_true', default=False, help='Save quantile transformer')
    parser.add_argument('-sdfs','--save_dfs', action='store_true', default=False, help='Save dfs used for CI and p-val calculation')
    parser.add_argument('-c',   '--compute_metrics', action='store_true', default=False, help='Whether to compute metrics')
    parser.add_argument('-pl',  '--plot_labels', nargs='+', help='Curve labels to show in the metrics plot')
    parser.add_argument('-l',   '--label_column', type=str, help='Name of the label column in the file')
    parser.add_argument('-s',   '--score_columns', nargs='+', help='Name of the predicted score column')
    # parser.add_argument('-a', '--add_columns', nargs='+', help='Names of columns to add')
    parser.add_argument('-fr',  '--from_columns', nargs='+', help='Names of columns to take values from')
    parser.add_argument('-m',   '--mark_matches', action='store_true', default=False, help='Whether to mark existing blocks')
    parser.add_argument('-p',   '--patient_df_update', action='store_true', default=False, help='Whether to update patient df')
    parser.add_argument('-pca', '--pca', action='store_true', help='Whether to compute and show embeddings pca')
    parser.add_argument('-fld', '--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='Folds number to filter by')
    parser.add_argument('-sld', '--slide_name', type=str, default=None, help='Slide name to filter by')
    parser.add_argument('-tsd', '--tile_score_dir', type=str, default=None, help='Dir from which to take tile scores')
    parser.add_argument('-ce',  '--comp_embeds', action='store_true', default=False, help='Use compressed tile embeddings')
    parser.add_argument('-ccp', '--compute_ci_pval', action='store_true', default=False, help='Compute 95% CI and p-val')
    parser.add_argument('-cm', '--clinical_metrics', action='store_true', default=False, help='Compute clinical metrics')
    # parser.add_argument('-ct', '--chosen_threshold', type=float, help='Chosen threshold for clinical metrics', default=0.6)
    parser.add_argument('-ct', '--chosen_threshold', type=float, nargs='+', help='Chosen thresholds for clinical metrics', default=[0.6])
    parser.add_argument('-test', '--test', action='store_true', default=False, help='Compute metrics for test set')
    parser.add_argument('-nb',  '--n_boot', type=int, default=1000, help='Number of bootstrap samplings')
    parser.add_argument('-gt',  '--use_gt', action='store_true', default=False, help='Compute 95% CI and p-val for tile predictions vs gt')
    parser.add_argument('-pty', '--pred_tile_y', action='store_true', default=False, help='Use predicted tile y for computing CI and p-val')
    parser.add_argument('-oev', '--only_equivocal', action='store_true', default=False, help='Use only equivocal cases for computing CI and p-val')
    parser.add_argument('-op', '--only_plot', action='store_true', default=False, help='Do not compute CIs, only get dfs')
    parser.add_argument('-ps', '--per_slide', action='store_true', default=False, help='Compute CI and p-val per slide instead of per patient')
    parser.add_argument('-y_type', '--y_type', type=str, help='Type of y map to use', default='regional', choices=['regional', 'local'])
    parser.add_argument('-tmpp', '--target_mpp', type=float, help='Target tiles MPP', default=0.5, choices=[0.5, 1, 2], required=True)

    # example command: -f ./excel_files/Her2_slides_matched_HE_folds_infer.csv -sf ./excel_files/carmel_per_block_marked.xlsx -fr IHC_to_Her2_score IHC_to_Her2_status HE_to_Her2_score HE_to_Her2_status -p
    args = parser.parse_args()
    print(f'args = {args}')

    # her2_csv_path = os.path.join(os.getcwd(), 'workspace', 'WSI', 'metadata_csvs', 'Her2_slides_matched_HE_folds.csv')
    her2_csv_path = args.file_paths
    if args.pca:
        show_pca_per_class(args=args)

    if args.update_file:
        update_excel_file(file_path=her2_csv_path, save_file=args.save_path, output_dirs=args.output_dirs)

    # her2_csv_path = os.path.join('excel_files', 'Her2_slides_matched_HE_folds_infer.csv')
    if args.compute_metrics:
        if args.label_column is not None and args.score_columns is not None:
            compute_metrics(her2_file_path=her2_csv_path, label_col=args.label_column, score_cols=args.score_columns,
                            plot_labels=args.plot_labels, save_dir=args.save_dir, folds=args.folds)
        else:
            print(f'Please specify label_column and score_column for metrics calculation.\n'
                  f'label_column = {args.label_column}, score_column = {args.score_column}')

    if args.mark_matches:
        if args.second_file_path is not None:
            mark_matching_blocks(file_path=args.second_file_path, my_file_path=her2_csv_path,
                                 save_file=f'{args.second_file_path.split(".xlsx")[0]}_marked.xlsx')

    if args.patient_df_update:
        if args.second_file_path is not None:
            add_cols = [f'mpp{args.target_mpp}_{fr_col}' for fr_col in args.from_columns]
            add_col_to_patient_df(file_path=args.second_file_path, my_file_path=her2_csv_path,
                                  save_file=f'{args.second_file_path.split("_marked.xlsx")[0]}.xlsx',
                                  add_cols=add_cols, from_cols=args.from_columns)

    if args.compute_ci_pval:
        if len(args.output_dirs) >= 2:
            baseline_dfs = []
            local_dfs = []
            base_path = os.path.join(os.getcwd(), args.base_path)
            bl_dir = os.path.join(base_path, args.output_dirs[0], "her2" if 'score' in args.output_dirs[0] else "her2_status")
            model_dir = os.path.join(base_path, args.output_dirs[1], "her2" if 'score' in args.output_dirs[1] else "her2_status")
            for i in range(1, 6):
                if args.pred_tile_y:
                    bl_preds_dir = next(filter(lambda x: 'infer' in x and x.endswith(f'{i}'), os.listdir(bl_dir)))
                    model_preds_dir = next(filter(lambda x: 'infer' in x and x.endswith(f'{i}'), os.listdir(model_dir)))
                    bl_preds_file = f'tumor_tile_valid_y_preds_mpp{args.target_mpp}_val{i}.csv' if not args.use_gt else f'tile_preds_vs_gt_mpp{args.target_mpp}_val{i}.csv'
                    model_preds_file = f'tumor_tile_valid_{args.y_type}_y_preds_mpp{args.target_mpp}_val{i}.csv' if not args.use_gt else f'tile_preds_vs_gt_mpp{args.target_mpp}_val{i}.csv'
                else:
                    bl_preds_dir = next(filter(lambda x: 'infer' in x and x.endswith(f'{i}'), os.listdir(bl_dir)))
                    model_preds_dir = next(filter(lambda x: 'infer' in x and x.endswith(f'{i}'), os.listdir(model_dir)))
                    bl_preds_file = 'slide_scores.csv'
                    model_preds_file = 'slide_scores.csv'
                
                bl_csv = os.path.join(bl_dir, bl_preds_dir, "eval_pretrained_her2" if 'score' in bl_dir else "eval_pretrained_her2_status", "inference_results/", bl_preds_file)
                model_csv = os.path.join(model_dir, model_preds_dir, "eval_pretrained_her2" if 'score' in model_dir else "eval_pretrained_her2_status", "inference_results/", model_preds_file)

                baseline_dfs.append(pd.read_csv(bl_csv))
                local_dfs.append(pd.read_csv(model_csv))

            if args.save_qt:
                args.qt_dir = args.save_dir if args.save_dir is not None else os.getcwd()
            baseline_dfs, local_dfs = compute_ci_pval(baseline_dfs=baseline_dfs, local_dfs=local_dfs, pred_tile_y=args.pred_tile_y, use_gt=args.use_gt, only_equivocal=args.only_equivocal, 
                                                      per_slide=args.per_slide, file_paths=args.file_paths, n_boot=args.n_boot, save_dfs=args.save_dfs, save_dir=args.save_dir, test=args.test,
                                                      save_qt=args.save_qt, qt_dir=args.qt_dir, only_plot=args.only_plot)

            if args.clinical_metrics:
                y_true = np.asarray(pd.concat([df['Her2_status'] for df in local_dfs]))
                baseline_scores = np.asarray(pd.concat([df['score'] for df in baseline_dfs]))
                local_scores = np.asarray(pd.concat([df['score'] for df in local_dfs]))

                # COLORS dict
                c_dict = {"Specificity": "orange", "Sensitivity": "cyan", "PPV": "green", "NPV": "purple","Impacted patients": "navy"}

                # plot_threshold_analysis(y_true, baseline_scores, local_scores, chosen_t=args.chosen_threshold, n_boot=1000, c_dict=c_dict, save_dir=args.save_dir)
                plot_impact_analysis(y_true, local_scores, chosen_t=args.chosen_threshold, n_boot=1000, c_dict=c_dict, save_dir=args.save_dir)

        else:
            print(f'Please provide exactly two output dirs to compute 95% CI and p-val.\n'
                f'output_dirs = {args.output_dirs}')


if __name__ == '__main__':
    main()
