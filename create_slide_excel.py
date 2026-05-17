import os
import pandas as pd


def get_batch_df_list(cohort: str, base_dir: str, prefix: str):
    batch_df_list = []
    full_base_dir = os.path.join('/data/Breast', cohort, base_dir)
    # Walk through the base directory and its subdirectories
    for batch_dir in os.listdir(full_base_dir):
        if not batch_dir.lower().startswith('batch'):
            continue
        batch_dir_path = os.path.join(full_base_dir, batch_dir)
        inner_dir = next(filter(lambda d: d.lower().startswith(prefix.lower()), os.listdir(batch_dir_path)), None)
        batch_dir_path = os.path.join(batch_dir_path, inner_dir)
        batch_num = batch_dir.split('_')[1]
        if batch_num == '10':
            continue
        slides_excel_file = next(filter(lambda f: f.startswith(f'slides_data_{prefix.upper()}{"_" if prefix == "Her2" else ""}{batch_num}.xlsx'), os.listdir(batch_dir_path)), None)
        batch_slides_df = pd.read_excel(os.path.join(batch_dir_path, slides_excel_file))
        batch_df_list.append(batch_slides_df)
    return batch_df_list

csvs_dir = '/home/amitf/workspace/WSI/metadata_csvs'
paired_csv_file = os.path.join(csvs_dir, 'Her2_slides_matched_HE_folds_HE.csv')
paired_df = pd.read_csv(paired_csv_file)
cancer_csv_file = os.path.join(csvs_dir, 'prelim_cancer_classification_slides.csv')
cancer_df = pd.read_csv(cancer_csv_file)
# Define the base directory
# base_dirs = ['Her2']
cohorts = ['Carmel', 'Carmel', 'Carmel', 'Haemek']
base_dirs = ['1-8', '9-11', 'Her2', 'Haemek_cancer_HE']
inner_prefixes = ['CARMEL', 'CARMEL', 'Her2', 'Haemek']

# Initialize lists to store the file names and paths
# slide_names = []
# paths = []
he_batch_df_list = []
ihc_batch_df_list = []
haemek_batch_df_list = []

for cohort, base_dir, inner_prefix in zip(cohorts, base_dirs, inner_prefixes):
    batch_df_list = get_batch_df_list(cohort, base_dir, inner_prefix)
    if inner_prefix == 'CARMEL':
        he_batch_df_list.extend(batch_df_list)
    elif inner_prefix == 'Her2':
        ihc_batch_df_list.extend(batch_df_list)
    else:
        haemek_batch_df_list.extend(batch_df_list)


# Concatenate all batch DataFrames into a single DataFrame
df = pd.concat(he_batch_df_list, ignore_index=True)
# choose only relevant columns
df = df[['file', 'patient barcode', 'id', 'Her2 status', 'Her2 score']]
# create a new column 'slide_wo_block' by removing the block identifier from 'file'
df['slide_wo_block'] = df['file'].apply(lambda x: '_'.join(x.split('_')[:-1]))

ihc_df = pd.concat(ihc_batch_df_list, ignore_index=True)
ihc_df['slide_wo_block'] = ihc_df['file'].apply(lambda x: '_'.join(x.split('_')[:-1]))
ihc_df = ihc_df.drop_duplicates(subset=['slide_wo_block'], keep='first')

# merge with ihc_df to get IHC slide info
df = pd.merge(df, ihc_df[['slide_wo_block', 'file', 'Her2 status', 'Her2 score']], on='slide_wo_block', how='left', suffixes=('_he', '_ihc'))
# drop rows with no Her2_status_ihc
df = df[(df['Her2 status_ihc'] != 'Missing Data') | (df['Her2 status_he'] != 'Missing Data')]
# pick the first slide with Her2 status not "Missing Data" for each slide_wo_block
df = df.sort_values(by=['slide_wo_block']).drop_duplicates(subset=['slide_wo_block'], keep='first')

#### Avoid contamination between train and test slides ####
paired_df['slide_wo_block'] = paired_df['file'].apply(lambda x: '_'.join(x.split('_')[:-1]))
paired_df = paired_df.drop_duplicates(subset=['slide_wo_block'], keep='first')
# keep only rows in df that don't have a match in paired_df
df = df[~df['slide_wo_block'].isin(paired_df['slide_wo_block'])]


# Hold the Her2 status from IHC if exists, otherwise from HE
df["Her2 status"] = df.apply(lambda row: row['Her2 status_he'] if row['Her2 status_he'] in ['Positive', 'Negative', 'Equivocal'] else row['Her2 status_ihc'], axis=1)
df["Her2 score"] = df.apply(lambda row: row['Her2 score_he'] if row['Her2 status_he'] in ['Positive', 'Negative', 'Equivocal'] else row['Her2 score_ihc'], axis=1)
# drop unneeded columns
df = df[['file_he', 'patient barcode', 'id', 'Her2 status', 'Her2 score']]
df = df.rename(columns={'file_he': 'file'})
# drop rows with NaN values in Her2 status
df = df.dropna(subset=['Her2 status'])


# HAEMEK
haemek_df = pd.concat(haemek_batch_df_list, ignore_index=True)
# keep only relevant columns
haemek_df = haemek_df[['file', 'patient barcode', 'id', 'Her2 status', 'Her2 score']]
# keep only rows with Her2 status in ['Positive', 'Negative']
haemek_df = haemek_df[haemek_df['Her2 status'].isin(['Positive', 'Negative'])]
# concatenate with df
df = pd.concat([df, haemek_df], ignore_index=True)

# Save the DataFrame to an Excel file
# output_file = 'Her2_slides_info.xlsx'
output_file = os.path.join(csvs_dir, 'HE_slides_wo_IHC.csv')
df.to_csv(output_file, index=False)

print(f"csv file '{output_file}' has been created with slide information.")


# Create a DataFrame with the collected data
# df = pd.DataFrame({
#     'SlideName': slide_names,
#     'Path': paths
# })