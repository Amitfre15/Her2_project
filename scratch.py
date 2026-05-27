import numpy as np
import os
import pandas as pd
import torch
from visualizations import load_npy_file


def main():
    # y_map_dir = "/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/y_map_mpp2/21-1335_1_7_b/"
    # gt_dir = "/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/slide_segmentations_mpp2/21-1335_1_7_d/"
    # ex_gt_file = "/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/annotated_tiles_mpp2/21-1335_1_7_b/her2_tile_gt.npy"
    ti_file = "/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/tumor_indices_from_cancer_map_mpp0.5/21-1335_1_7_b/tumor_indices5.npy"
    # ex_gt_npy = load_npy_file(ex_gt_file)
    ti_npy = load_npy_file(ti_file)
    cp_file = "/SSDStorage/Breast/Carmel/Her2/gigapath_HE/cancer_probs_from_cancer_map_mpp0.5/21-1335_1_7_b/cancer_prob_val5.npy"
    cp_npy = load_npy_file(cp_file)
    tile_y_file = "/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/tile_y_from_y_map_mpp0.5/21-1335_1_7_b/tile_y_ad_val5.npy"
    tile_y_npy = load_npy_file(tile_y_file)
    local_tile_y_file = "/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/tile_y_mpp0.5/21-1335_1_7_b/tile_y_ad_val5.npy"
    local_tile_y_npy = load_npy_file(local_tile_y_file)

    local_tile_y_pred_file = "/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/tile_y_pred_mpp0.5/21-1335_1_7_b/local_binary_tile_y_pred_add_virchow2_ad_val5.npy"
    local_tile_y_pred_npy = load_npy_file(local_tile_y_pred_file)
    mt_file = "/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/matching_tiles_mpp0.5/21-1335_1_7_b/ihc_tiles.npy"
    mt_npy = load_npy_file(mt_file)
    tile_logits_file = "/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/tile_y_pred_mpp0.5/21-1335_1_7_b/local_tile_y_pred_add_virchow2_ad_val5.npy"
    tile_logits_npy = load_npy_file(tile_logits_file)
    tiles_dir = "/SSDStorage/Breast/Carmel/Her2/gigapath_HE/png_tiles_mpp0.5/21-1335_1_7_b"
    tiles = os.listdir(tiles_dir)

    

    # gt_name = next(filter(lambda x: x.endswith(".npy"), os.listdir(gt_dir)), None)
    # gt_file = os.path.join(gt_dir, gt_name)
    # gt_npy = load_npy_file(gt_file)
    # bl_name = next(filter(lambda x: "tumor from map SW baseline" in x and x.endswith(f".npy"), os.listdir(y_map_dir)), None)
    # bl_file = os.path.join(y_map_dir, bl_name)
    # baseline_npy = load_npy_file(bl_file)
    # pred_name = next(filter(lambda x: "tumor from map predictions" in x and x.endswith(f".npy"), os.listdir(y_map_dir)), None)
    # model_file = os.path.join(y_map_dir, pred_name)
    # model_npy = load_npy_file(model_file)

    # mask = (gt_npy != -1) & (baseline_npy != -1) & (model_npy != -1)
    # valid_indices = np.where(mask)
    # valid_gt, valid_bl, valid_pred = gt_npy[valid_indices], baseline_npy[valid_indices], model_npy[valid_indices]

    feat_dir = "/SSDStorage/Breast/Carmel/Her2/gigapath_cancer_classification/gigapath_features_mpp0.5"
    # dirs = os.listdir(feat_dir)
    # df = pd.read_csv("/home/amitf/workspace/WSI/metadata_csvs/prelim_cancer_classification_slides.csv")
    # slides = df['file'].tolist()
    # slides = [s.split('.')[0] for s in slides]
    # for s in slides:
    #     if s not in dirs:
    #         print(f"There is no dir for {s}")

def create_thufa_csv():
    num_folds = 5
    relevant_cols = ['id', 'file', 'MPP', 'patient barcode', 'fold', 'label']
    haemek_df = pd.read_excel(f"/SSDStorage/Breast/Haemek/Haemek_cancer_HE/Batch_1/HAEMEK1/slides_data_HAEMEK1.xlsx")
    # keep only rows where 'Her2 status' is 'Positive' or 'Negative'
    haemek_df = haemek_df[haemek_df['Her2 status'].isin(['Positive', 'Negative'])]
    # add a 'fold' column with values from 1 to 5 where rows with the same PatientIndex have the same fold value
    haemek_df['fold'] = (haemek_df.groupby('PatientIndex').ngroup() % num_folds) + 1
    # add a 'label' column which holds the 'Her2 status' as int
    haemek_df['label'] = haemek_df['Her2 status'].apply(lambda x: 1 if "Positive" in str(x) else 0)
    haemek_df['patient barcode'] = haemek_df['PatientIndex']  # For consistency with FinHer dfs
    haemek_df = haemek_df[relevant_cols]

    finher_dfs = []
    for i in range(1, 4):
        finher_df = pd.read_excel(f"/SSDStorage/Breast/FinHer/Batch{i}/FINHER_{i}/slides_data_FINHER_{i}.xlsx")
        finher_df = finher_df[finher_df['Her2 status'].isin(['Positive', 'Negative'])]
        finher_df['fold'] = (finher_df.groupby('patient barcode').ngroup() % num_folds) + 1
        finher_df['label'] = finher_df['Her2 status'].apply(lambda x: 1 if "Positive" in str(x) else 0)
        finher_df = finher_df[relevant_cols]
        finher_dfs.append(finher_df)
    
    ucmc_pp_df = pd.read_excel("/SSDStorage/Breast/UCMC/UCMC_metadata_per_patient.xlsx")
    ucmc_sld_df = pd.read_excel("/SSDStorage/Breast/UCMC/UCMC/slides_data_UCMC.xlsx")
    ucmc_pp_df = ucmc_pp_df[ucmc_pp_df['label_Her2'].isin([0, 1])]
    # merge the dataframes to add 'label_Her2' to ucmc_sld_df according to pp_df['PatientID'] == sld_df['patient barcode']
    ucmc_sld_df = ucmc_sld_df.merge(ucmc_pp_df[['PatientID', 'label_Her2']], left_on='patient barcode', right_on='PatientID', how='left')
    ucmc_sld_df = ucmc_sld_df[ucmc_sld_df['label_Her2'].isin([0, 1])]
    ucmc_sld_df = ucmc_sld_df.rename(columns={'label_Her2': 'label'})
    ucmc_sld_df['fold'] = (ucmc_sld_df.groupby('patient barcode').ngroup() % num_folds) + 1
    ucmc_sld_df = ucmc_sld_df[relevant_cols]

    abctb_df = pd.read_excel("/SSDStorage/Breast/ABCTB_TIF/slides_data_ABCTB.xlsx")
    abctb_df = abctb_df[abctb_df['Her2 status'].isin(['Positive', 'Negative'])]
    abctb_df['fold'] = (abctb_df.groupby('patient barcode').ngroup() % num_folds) + 1
    abctb_df['label'] = abctb_df['Her2 status'].apply(lambda x: 1 if "Positive" in str(x) else 0)
    abctb_df = abctb_df[relevant_cols]

    tcga_df = pd.read_excel("/SSDStorage/Breast/TCGA/slides_data.xlsx")
    tcga_df = tcga_df[tcga_df['Her2 status'].isin(['Positive', 'Negative'])]
    tcga_df['fold'] = (tcga_df.groupby('patient barcode').ngroup() % num_folds) + 1
    tcga_df['label'] = tcga_df['Her2 status'].apply(lambda x: 1 if "Positive" in str(x) else 0)
    tcga_df = tcga_df[relevant_cols]

    carmel_df = pd.read_csv("workspace/WSI/metadata_csvs/Her2_slides_matched_HE_folds_HE.csv")
    carmel_df['label'] = carmel_df['Her2_status']
    carmel_df = carmel_df[relevant_cols]

    df = pd.concat([haemek_df] + finher_dfs + [ucmc_sld_df] + [abctb_df] + [tcga_df] + [carmel_df], ignore_index=True)
    df.to_csv("/SSDStorage/Breast/THUFA/THUFAC_slides.csv", index=False)

def create_ev_as_test_csv():
    df = pd.read_csv("workspace/WSI/metadata_csvs/Her2_slides_matched_HE_folds.csv")
    # split the rows with fold 6 between folds 1-5, keeping the same patient in the same fold
    fold_6_df = df[df['fold'] == 6]
    other_df = df[df['fold'] != 6]
    # fold_6_df['fold'] = (fold_6_df.groupby('patient barcode').ngroup() % 5) + 1
    # new_df = pd.concat([other_df, fold_6_df], ignore_index=True)
    new_df = other_df.copy()
    # make the fold of rows with label 2 to be of current fold + 5 (i.e. 1->6, 2->7, 3->8, 4->9, 5->10)
    new_df['fold'] = new_df.apply(lambda row: row['fold'] + 5 if row['label'] == 2 else row['fold'], axis=1)
    # move all rows of patients in test folds to their fold
    patients_in_test_folds = new_df[new_df['fold'] >= 6]['patient barcode'].unique()
    # set fold of all rows with patient barcode in patients_in_test_folds to the highest fold of this patient rows
    for patient in patients_in_test_folds:
        max_fold = new_df[new_df['patient barcode'] == patient]['fold'].max()
        new_df.loc[new_df['patient barcode'] == patient, 'fold'] = max_fold
    new_df.to_csv("workspace/WSI/metadata_csvs/Her2_slides_matched_HE_folds_wo_orig_test_ev_as_test.csv", index=False)

def add_clinical_features():
    df = pd.read_csv("workspace/WSI/metadata_csvs/Her2_slides_matched_HE_folds_HE.csv")

    summary_df = pd.read_excel("workspace/WSI/metadata_csvs/Summary_data_per_slide.xlsx")
    summary_df = summary_df[summary_df['BatchID'].isin(['Carmel4', 'Carmel5', 'Carmel6', 'Carmel7', 'Carmel8', 'Carmel9', 'Carmel11'])]
    test_scores_df = pd.read_csv("/home/amitf/outputs/mpp0.5/ci_pvals/avg_test_ci_pval_per_slide_all_IHC_slide_her2_status_baseline_vs_paired_mw_to_Her2_status/score_differences.csv")
    cv_scores_df = pd.read_csv("/home/amitf/outputs/mpp0.5/ci_pvals/avg_ci_pval_per_slide_all_IHC_slide_her2_status_baseline_vs_paired_mw_to_Her2_status/score_differences.csv")
    
    clinical_features = ['TumorType', 'Age', 'Grade', 'label_ER', 'label_PR']
    df = df.merge(summary_df[['SlideName'] + clinical_features], left_on='Matched_HE_SlideName', right_on='SlideName', how='left')
    
    tumor_types = {"idc": "invasive ductal carcinoma", 
                   "ilc": "invasive lobular carcinoma", 
                   "muc": "mucinous carcinoma", 
                   "epc": "encapsulated papillary carcinoma", 
                   "ic": "invasive carcinoma", 
                   "spc": "solid papillary carcinoma", 
                   "pc": "papillary carcinoma", 
                   "necb": "necrotic breast tissue", 
                   "lcis": "lobular carcinoma in situ", 
                   "dcis": "ductal carcinoma in situ", 
                   "meta": "metaplastic carcinoma", 
                   "imc": "intermediate mixed carcinoma", 
                   "ipc": "inflammatory papillary carcinoma", 
                   "cpc": "cystic-pc"}
    for tt in tumor_types.keys():
        df[f'{tt}'] = df['TumorType'].apply(lambda x: 1 if (tt in str(x).lower() or tumor_types[tt] in str(x).lower()) else 0)

    # fill nan values 
    df[clinical_features] = df[clinical_features].fillna(0)

    df['block_id'] = df['file'].apply(lambda x: x.split('.')[0].rsplit("_", 1)[0] + '_')
    df = df.merge(cv_scores_df[['slide_name', 'score']], left_on='block_id', right_on='slide_name', how='left')
    df = df.merge(test_scores_df[['slide_name', 'score']], left_on='block_id', right_on='slide_name', how='left')
    # merge the score columns by taking the one which is not nan
    df['score'] = df['score_x'].fillna(df['score_y'])
    df.drop(columns=['slide_name_x', 'slide_name_y', 'score_x', 'score_y'], inplace=True)

    df.to_csv("workspace/WSI/metadata_csvs/Her2_slides_matched_HE_folds_HE_w_clinical.csv", index=False)

def check_cancer_tiles():
    he_slide_name = "19-14590_1_1_e"
    ihc_slide_name = "19-14590_1_1_n"
    mt_file = f"/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/matching_tiles_mpp0.5/{he_slide_name}/ihc_tiles.npy"
    he_tiles_path = f"/SSDStorage/Breast/Carmel/Her2/gigapath_HE/png_tiles_mpp0.5/{he_slide_name}/"
    ihc_tiles_path = f"/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/png_tiles_mpp0.5/{ihc_slide_name}/"
    ti_path = f"/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/tumor_indices_from_cancer_map_mpp0.5/{he_slide_name}/local_tumor_indices_ensemble.npy" # local_tumor_indices_ensemble
    matching_tiles = load_npy_file(mt_file)
    tumor_indices = load_npy_file(ti_path)
    he_tiles = np.array(os.listdir(he_tiles_path))
    ihc_tiles = np.array(os.listdir(ihc_tiles_path))

    valid_matching_tiles = (~np.isnan(matching_tiles)) & (matching_tiles < ihc_tiles.shape[0])
    valid_matching_indices = np.nonzero(valid_matching_tiles.flatten())[0]
    tumor_in_non_nan = np.isin(valid_matching_indices, tumor_indices)
    tumor_matching_indices = np.where(tumor_in_non_nan)[0]
    valid_tumor_matching_indices = valid_matching_indices[tumor_matching_indices].flatten()
    matching_tiles = matching_tiles.flatten()[valid_tumor_matching_indices].astype(int)
    he_tiles = np.array(he_tiles)[valid_tumor_matching_indices]
    ihc_tiles = np.array(ihc_tiles)[matching_tiles]
    print(f"Number of valid tumor matching tiles: {len(he_tiles)}")
    # images, img_coords = images[:, valid_tumor_matching_indices], img_coords[:, valid_tumor_matching_indices]
    # print(mt_npy)


if __name__ == "__main__":
    # main()
    # create_thufa_csv()
    # create_ev_as_test_csv()
    # add_clinical_features()
    check_cancer_tiles()

    # np_path = "/SSDStorage/Breast/Carmel/Her2/gigapath_HE/conch_features_mpp0.5/18-2394_1_9_a/tile_embeds_18-2394_1_9_a.npy"
    # embeds = load_npy_file(np_path)
    # print(embeds.shape)