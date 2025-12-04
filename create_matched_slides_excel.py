import argparse
import re
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, mean_squared_error, roc_auc_score
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
        full_output_dir = os.path.join(os.getcwd(), 'outputs/mpp2', output_dir)
        
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


# def compute_ci_pval(baseline_dfs, local_dfs):
#     for i in range(len(baseline_dfs)):
#         df_baseline = baseline_dfs[i]
#         df_local = local_dfs[i]
#         tile_y = df_baseline['tile_y'].values
#         tile_y_bin = df_baseline['tile_y_bin'].values
#         tile_pred_baseline = df_baseline['tile_pred'].values
#         tile_pred_local = df_local['tile_pred'].values
#         for metric_func in [roc_auc_score, mean_squared_error, spearmanr]:
#             if metric_func in [spearmanr, mean_squared_error]:
#                 # if metric_func == spearmanr:
#                 #     baseline_metric = metric_func(tile_y, tile_pred_baseline)[0]
#                 #     local_metric = metric_func(tile_y, tile_pred_local)[0]
#                 # else:
#                     # baseline_metric = metric_func(tile_y, tile_pred_baseline)
#                     # local_metric = metric_func(tile_y, tile_pred_local)
#                 diff, diff_ci, baseline_ci, local_ci, pval = bootstrap_diff(metric_func, tile_y, tile_pred_baseline, tile_pred_local)
#             else:
#                 # baseline_metric = metric_func(tile_y_bin, tile_pred_baseline)
#                 # local_metric = metric_func(tile_y_bin, tile_pred_local)
#                 diff, diff_ci, baseline_ci, local_ci, pval = bootstrap_diff(metric_func, tile_y_bin, tile_pred_baseline, tile_pred_local)
#             print(f'Tile-level {metric_func.__name__}: baseline CI = [{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}], local CI = [{local_ci[0]:.4f}, {local_ci[1]:.4f}], '
#                 f'diff = {diff:.4f}, 95% CI = [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}], p-value = {pval:.4f}')


# def bootstrap_diff(metric_func, y_true, y_pred1, y_pred2, n_boot=1000):
#     rng = np.random.default_rng(42)
#     n = len(y_true)
#     print(f"n = {n}")
#     diffs = []
#     m1s, m2s = [], []
#     for _ in range(n_boot):
#         idx = rng.choice(n, size=n, replace=True)
#         m1 = metric_func(y_true[idx], y_pred1[idx])
#         m2 = metric_func(y_true[idx], y_pred2[idx])
#         if metric_func == spearmanr:
#             m1 = m1[0]
#             m2 = m2[0]
#         diffs.append(m2 - m1)  # local - baseline
#         m1s.append(m1)
#         m2s.append(m2)
#     diffs = np.array(diffs)
#     m1s, m2s = np.array(m1s), np.array(m2s)
#     diff_ci = np.percentile(diffs, [2.5, 97.5])
#     m1_ci, m2_ci = np.percentile(m1s, [2.5, 97.5]), np.percentile(m2s, [2.5, 97.5])
#     pval = (np.sum(diffs <= 0) / n_boot) if np.mean(diffs) > 0 else (np.sum(diffs >= 0) / n_boot)
#     return np.mean(diffs), diff_ci, m1_ci, m2_ci, pval

def compute_ci_pval(baseline_dfs, local_dfs, pred_tile_y, n_boot=1000):
    group_dfs_by_patient(baseline_dfs, local_dfs)
    metrics = [roc_auc_score, mean_squared_error, spearmanr]

    for metric_func in metrics:
        diff, diff_ci, baseline_ci, local_ci, pval = bootstrap_diff_cv(
            metric_func, baseline_dfs, local_dfs, pred_tile_y, n_boot=n_boot
        )

        print(
            f'Tile-level {metric_func.__name__}: '
            f'baseline CI = {baseline_ci.mean():.4f} [{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}], '
            f'local CI = {local_ci.mean():.4f} [{local_ci[0]:.4f}, {local_ci[1]:.4f}], '
            f'diff = {diff:.4f}, 95% CI = [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}], '
            f'p-value = {pval:.4f}'
        )

def group_dfs_by_patient(baseline_dfs, local_dfs):
    her2_file_path = './workspace/WSI/metadata_csvs/Her2_slides_matched_HE_folds_HE.csv'
    read_columns = ['file', 'patient barcode', 'Her2_status']
    text_cols = ['file', 'slide_name']
    her2_df_full = pd.read_csv(her2_file_path)
    her2_df = her2_df_full[read_columns]
    her2_df['slide_name'] = her2_df['file'].apply(lambda x: x.split('.')[0])
    for i in range(len(baseline_dfs)):
        df_b = baseline_dfs[i]
        df_l = local_dfs[i]
        df_b = df_b.merge(her2_df, left_on='slide_name', right_on='slide_name', how='left').drop(columns=text_cols)
        df_l = df_l.merge(her2_df, left_on='slide_name', right_on='slide_name', how='left').drop(columns=text_cols)
        per_pat_b_df = df_b.groupby('patient barcode').mean()
        per_pat_l_df = df_l.groupby('patient barcode').mean()
        per_pat_b_df['Her2_status'] = (per_pat_b_df['Her2_status'] >= 0.5).astype(int)
        per_pat_l_df['Her2_status'] = (per_pat_l_df['Her2_status'] >= 0.5).astype(int)
        df_b = per_pat_b_df.reset_index()
        df_l = per_pat_l_df.reset_index()
        baseline_dfs[i] = df_b
        local_dfs[i] = df_l
    

def bootstrap_diff_cv(metric_func, baseline_dfs, local_dfs, pred_tile_y, n_boot=1000):
    rng = np.random.default_rng(42)
    K = len(baseline_dfs)

    # Bootstrap within-fold
    boot_baseline = []
    boot_local = []
    boot_diff = []

    if pred_tile_y:
        gt_col = 'tile_y'
        gt_bin_col = 'tile_y_bin'
        pred_col = 'tile_pred'
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
            
            gt = df_b[gt_col].values
            gt_bin = df_b[gt_bin_col].values
            pred_b = df_b[pred_col].values
            pred_l = df_l[pred_col].values

            n_i = len(gt)
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

    return np.mean(boot_diff), diff_ci, baseline_ci, local_ci, pval


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
                embed_file = f"{args.output_dirs[0].split('/')[-2].split('_mpp1')[0]}"
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
    parser.add_argument('-u', '--update_file', action='store_true', default=False, help='Whether to update the file')
    parser.add_argument('-sp', '--save_path', type=str, help='Path to save the updated file')
    parser.add_argument('-od', '--output_dirs', nargs='+', help='Output dirs from which to take output files')
    parser.add_argument('-f', '--file_paths', type=str, help='Excel file/s to work with')
    parser.add_argument('-sf', '--second_file_path', type=str, help='Patient excel file to update')
    parser.add_argument('-sd', '--save_dir', type=str, help='Directory to save metrics plots')
    parser.add_argument('-c', '--compute_metrics', action='store_true', default=False, help='Whether to compute metrics')
    parser.add_argument('-pl', '--plot_labels', nargs='+', help='Curve labels to show in the metrics plot')
    parser.add_argument('-l', '--label_column', type=str, help='Name of the label column in the file')
    parser.add_argument('-s', '--score_columns', nargs='+', help='Name of the predicted score column')
    # parser.add_argument('-a', '--add_columns', nargs='+', help='Names of columns to add')
    parser.add_argument('-fr', '--from_columns', nargs='+', help='Names of columns to take values from')
    parser.add_argument('-m', '--mark_matches', action='store_true', default=False, help='Whether to mark existing blocks')
    parser.add_argument('-p', '--patient_df_update', action='store_true', default=False, help='Whether to update patient df')
    parser.add_argument('-pca', '--pca', action='store_true', help='Whether to compute and show embeddings pca')
    parser.add_argument('-fld', '--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='Folds number to filter by')
    parser.add_argument('-sld', '--slide_name', type=str, default=None, help='Slide name to filter by')
    parser.add_argument('-tsd', '--tile_score_dir', type=str, default=None, help='Dir from which to take tile scores')
    parser.add_argument('-ce', '--comp_embeds', action='store_true', default=False, help='Use compressed tile embeddings')
    parser.add_argument('-ccp', '--compute_ci_pval', action='store_true', default=False, help='Compute 95% CI and p-val for tile predictions')
    parser.add_argument('-pty', '--pred_tile_y', action='store_true', default=False, help='Use predicted tile y for computing CI and p-val')

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
            add_cols = [f'mpp1_{fr_col}' for fr_col in args.from_columns]
            add_col_to_patient_df(file_path=args.second_file_path, my_file_path=her2_csv_path,
                                  save_file=f'{args.second_file_path.split("_marked.xlsx")[0]}.xlsx',
                                  add_cols=add_cols, from_cols=args.from_columns)

    if args.compute_ci_pval:
        if len(args.output_dirs) >= 2:
            baseline_dfs = []
            local_dfs = []
            for i in range(1, 6):
                if args.pred_tile_y:
                    preds_dir = f"her2/{args.output_dirs[-1]}_infer_tumor_tile{i}/eval_pretrained_her2/inference_results/"
                    preds_file = f'tile_y_preds_mpp2_val{i}.csv'
                else:
                    preds_dir = f"her2/{args.output_dirs[-1]}_infer{i}/eval_pretrained_her2/inference_results/"
                    preds_file = 'slide_scores.csv'
                y_pred1_path = os.path.join(os.getcwd(), 'outputs/mpp2', args.output_dirs[0], 
                                            f"her2/{args.output_dirs[0]}_infer{i}/eval_pretrained_her2/inference_results/", preds_file)
                y_pred2_path = os.path.join(os.getcwd(), 'outputs/mpp2', args.output_dirs[1], preds_dir, preds_file)
                baseline_dfs.append(pd.read_csv(y_pred1_path))
                local_dfs.append(pd.read_csv(y_pred2_path))

            compute_ci_pval(baseline_dfs=baseline_dfs, local_dfs=local_dfs, pred_tile_y=args.pred_tile_y)
        else:
            print(f'Please provide exactly two output dirs to compute 95% CI and p-val.\n'
                f'output_dirs = {args.output_dirs}')


if __name__ == '__main__':
    main()
