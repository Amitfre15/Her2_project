import argparse
import re
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
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
        full_output_dir = os.path.join(os.getcwd(), 'outputs', output_dir)
        config = 'her2' if output_dir.endswith('score') else 'her2_status'
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
            slide_key = "SlideName" if key.startswith('IHC') else "Matched_HE_SlideName"
            her2_slidename = her2_row[slide_key]
            batch_matches = batch_df[batch_df['slide_name'].str.startswith(her2_slidename.split('.')[0])]

            if not batch_matches.empty:
                her2_df.at[i, key] = batch_matches['score'].values[0]

    her2_df.to_csv(save_file, index=False)

    print(f"Updated file '{save_file}' has been created.")


def compute_metrics(her2_file_path, score_cols, label_col, plot_labels, save_dir=None):
    read_columns = score_cols + ['patient barcode', label_col, 'fold']
    her2_df = pd.read_csv(her2_file_path)[read_columns]
    aucs_dict = {plot_label: [] for plot_label in plot_labels}
    for i in range(1, 6):
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
            if "score" in label_col:
                # Case 1: [0, 0.5, 1] vs [2, 3]
                binary_labels_case1 = np.isin(y_true, [2, 3]).astype(int)
                y_trues2_3.append(binary_labels_case1)
                # Case 2: [0, 0.5, 1, 2] vs [3]
                binary_labels_case2 = np.isin(y_true, [3]).astype(int)
                y_trues3.append(binary_labels_case2)
            else:
                y_true_binary = (y_true >= 0.5).astype(int)  # Convert to 0 or 1
                y_trues_bin.append(y_true_binary)
            y_score = per_pat_her2_df[score_col].values
            y_scores.append(y_score)
        patient_num = len(per_pat_her2_df)

        if len(y_trues3) > 0:
            aucs2_3 = show_roc_and_calc_auc(y_trues=y_trues2_3, y_scores=y_scores,
                                  score_title=f'[0, 0.5, 1] vs [2, 3] {score_cols[0]} ({patient_num} patients)',
                                  plot_labels=plot_labels, save_dir=save_dir)
            aucs3 = show_roc_and_calc_auc(y_trues=y_trues3, y_scores=y_scores,
                                  score_title=f'[0, 0.5, 1, 2] vs [3] {score_cols[0]} ({patient_num} patients)',
                                  plot_labels=plot_labels, save_dir=save_dir)
        else:
            aucs = show_roc_and_calc_auc(y_trues=y_trues_bin, y_scores=y_scores,
                                  score_title=f'Her2 positive vs negative fold {i} ({patient_num} patients)',
                                  plot_labels=plot_labels, save_dir=save_dir)
            for plot_label in plot_labels:
                aucs_dict[plot_label] += aucs[plot_label]

    for plot_label in plot_labels:
        print(f"Mean AUC for {plot_label}: {np.mean(np.array(aucs_dict[plot_label]))}")
        

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
        plt.savefig(os.path.join(save_dir, f'{score_title} ROC Curve.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return aucs


def show_pca_per_class(args):
    from sklearn.decomposition import PCA

    her2_df = pd.read_csv(args.file_paths) #['SlideName', 'label', 'fold']

    fold_df = her2_df[her2_df['fold'] == 1]
    slide_key = 'Matched_HE_SlideName' if 'HE_to_Her2' in args.output_dirs[0] else 'SlideName'

    ihc_embeddings = None
    pca = PCA(n_components=2)
    rows_to_drop = []
    for i, her2_row in fold_df.iterrows():
        slide_name = her2_row[slide_key].split('.')[0]
        full_file_path = os.path.join(args.output_dirs[0], f'{slide_name}.npy')
        if ihc_embeddings is None:
            ihc_embeddings = np.load(full_file_path)
        else:
            try:
                ihc_embeddings = np.hstack([ihc_embeddings, np.load(full_file_path)])
            except BaseException as e:
                # print(e)
                rows_to_drop.append(i)

    fold_df = fold_df.drop(rows_to_drop)
    # print(f'*******ihc_embeddings.shape = {ihc_embeddings.shape}, ihc_embeddings = {ihc_embeddings}')
    ihc_embeddings_2d = pca.fit_transform(ihc_embeddings.reshape(len(fold_df), -1))  # Reduce to 2D
    print(f'*******ihc_embeddings_2d.shape = {ihc_embeddings_2d.shape}')

    # ===== Plot the PCA Results =====
    plt.figure(figsize=(8, 6))
    ihc_labels = fold_df['label']
    colors = {0: 'red', 0.5: 'orange', 1: 'yellow', 2: 'blue', 3: 'purple'}

    for label in np.unique(ihc_labels):
        mask = ihc_labels == label
        # print(f'ihc_embeddings_2d.shape = {ihc_embeddings_2d.shape}')
        plt.scatter(ihc_embeddings_2d[mask, 0], ihc_embeddings_2d[mask, 1],
                    label=f'Her2 Class {label}', alpha=0.7, color=colors[label])

    if len(args.plot_labels) > 0:
        prefix = f'{args.plot_labels[0]}'
    else:
        prefix = ''

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"2D PCA Projection of {prefix} Embeddings")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, f'{prefix}_embeds_PCA.png'), dpi=300, bbox_inches='tight')


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Search and replace file names and content based on Excel mapping.")
    # parser.add_argument('-r', '--root', required=True, help='Root directory to search for files.')
    parser.add_argument('-u', '--update_file', action='store_true', help='Whether to update the file')
    parser.add_argument('-sp', '--save_path', type=str, help='Path to save the updated file')
    parser.add_argument('-od', '--output_dirs', nargs='+', help='Output dirs from which to take output files')
    parser.add_argument('-f', '--file_paths', required=True, type=str, help='Excel file/s to work with')
    parser.add_argument('-sf', '--second_file_path', type=str, help='Patient excel file to update')
    parser.add_argument('-sd', '--save_dir', type=str, help='Directory to save metrics plots')
    parser.add_argument('-c', '--compute_metrics', action='store_true', help='Whether to compute metrics')
    parser.add_argument('-pl', '--plot_labels', nargs='+', help='Curve labels to show in the metrics plot')
    parser.add_argument('-l', '--label_column', type=str, help='Name of the label column in the file')
    parser.add_argument('-s', '--score_columns', nargs='+', help='Name of the predicted score column')
    # parser.add_argument('-a', '--add_columns', nargs='+', help='Names of columns to add')
    parser.add_argument('-fr', '--from_columns', nargs='+', help='Names of columns to take values from')
    parser.add_argument('-m', '--mark_matches', action='store_true', help='Whether to mark existing blocks')
    parser.add_argument('-p', '--patient_df_update', action='store_true', help='Whether to update patient df')

    # example command: -f ./excel_files/Her2_slides_matched_HE_folds_infer.csv -sf ./excel_files/carmel_per_block_marked.xlsx -fr IHC_to_Her2_score IHC_to_Her2_status HE_to_Her2_score HE_to_Her2_status -p
    args = parser.parse_args()
    print(f'args = {args}')

    # her2_csv_path = os.path.join(os.getcwd(), 'workspace', 'WSI', 'metadata_csvs', 'Her2_slides_matched_HE_folds.csv')
    her2_csv_path = args.file_paths
    if args.update_file:
        update_excel_file(file_path=her2_csv_path, save_file=args.save_path, output_dirs=args.output_dirs)

    # her2_csv_path = os.path.join('excel_files', 'Her2_slides_matched_HE_folds_infer.csv')
    if args.compute_metrics:
        if args.label_column is not None and args.score_columns is not None:
            compute_metrics(her2_file_path=her2_csv_path, label_col=args.label_column, score_cols=args.score_columns,
                            plot_labels=args.plot_labels, save_dir=args.save_dir)
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


if __name__ == '__main__':
    main()
