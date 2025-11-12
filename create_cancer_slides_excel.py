import pandas as pd
import os


def main():
    base_path = "/data/Breast/Carmel"
    matched_csv_path = os.path.join("workspace", "WSI", "metadata_csvs", "Her2_slides_matched_HE_folds_HE.csv")
    matched_df = pd.read_csv(matched_csv_path)
    matched_df["block"] = matched_df["file"].astype(str).apply(extract_block)
    # List of Excel files (adjust base directory if needed)
    excel_paths = [
        os.path.join(base_path, "Benign/slides_data_BENIGN_merged.xlsx"),
        os.path.join(base_path, "IS/Batch_1/CARMEL_1/slides_data_CARMEL_1.xlsx"),
    ]
    excel_paths += [os.path.join(base_path, f"1-8/Batch_{i}/CARMEL{i}/slides_data_CARMEL{i}.xlsx") for i in [1, 2, 3, 8]]
    excel_paths += [os.path.join(base_path, f"9-11/Batch_{i}/CARMEL{i}/slides_data_CARMEL{i}.xlsx") for i in [9, 11]]
        
    all_dfs = []
    cancer_col = "is_cancer status"
    num_folds = 5

    for path in excel_paths:
        df = pd.read_excel(path)
        
        # Add/handle is_cancer_status
        if cancer_col in df.columns:
            pass  # already there
        else:
            if "IS" in path.split(os.sep):  # if path contains "IS" directory
                df[cancer_col] = "Negative"
            elif "1-8" in path or "9-11" in path:
                df[cancer_col] = "Positive"
            else:
                df[cancer_col] = None  # or "Unknown", depending on your preference
        
        # Keep only relevant columns if they exist
        keep_cols = [col for col in ["file", "patient barcode", "MPP", cancer_col] if col in df.columns]
        df = df[keep_cols]

        # Add full path to the source Excel
        df["path"] = os.path.dirname(path)

        if "Benign" in path.split(os.sep) or "1-8" in path.split(os.sep):
            df = df.sort_values("file").drop_duplicates("patient barcode", keep="first")
        
        if "9-11" in path.split(os.sep):
            df["block"] = df["file"].astype(str).apply(extract_block)
            df = df.sort_values("file").drop_duplicates("block", keep="first")

        # if "Benign" in path.split(os.sep):
        #     df = df[:802]

        all_dfs.append(df)

    # Concatenate all
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df[final_df[cancer_col] != "Missing Data"]
    final_df[cancer_col] = final_df[cancer_col].replace({
        "Positive": 1,
        "Negative": 0
    })

    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    final_df["fold"] = (final_df.index % num_folds) + 1  # Assign folds
    final_df["id"] = "cancer_classification"
    final_df["label"] = final_df[cancer_col]

    merged_df = pd.merge(final_df, matched_df[["block", "fold"]], on="block", how="left", suffixes=("_replaced", ""))
    merged_df["fold"] = merged_df["fold"].combine_first(merged_df["fold_replaced"])

    # # Read annotations.csv and keep only needed columns
    # ann_df = pd.read_csv(os.path.join('workspace', 'WSI', 'metadata_csvs', "annotations.csv"))[["file", "MPP"]].drop_duplicates("file")

    # # Add the other required columns
    # ann_df["path"] = "/data/Breast/Carmel/9-11/Batch_11/CARMEL11"
    # ann_df["label"] = 1
    # ann_df["fold"] = 6
    # ann_df["id"] = "cancer_classification"

    # # Concat with the existing dataframe
    # final_df = pd.concat([final_df, ann_df], ignore_index=True)

    print(merged_df.head())

    # Optionally save
    save_path = os.path.join("workspace", "WSI", "metadata_csvs", "prelim_cancer_classification_slides.csv")
    merged_df.to_csv(save_path, index=False)

# Function to extract block substring
def extract_block(s):
    # remove file extension if present
    s = s.split(".")[0]
    # split by underscores
    parts = s.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return s  # fallback if unexpected format

def add_annotated_slides():
    csvs_path = os.path.join("workspace", "WSI", "metadata_csvs")
    # Load the files
    cancer_df = pd.read_csv(os.path.join(csvs_path, "cancer_classification_slides.csv"))
    her2_df = pd.read_csv(os.path.join(csvs_path, "Her2_slides_matched_HE_folds.csv"))

    # Create block columns
    cancer_df["block"] = cancer_df["file"].astype(str).apply(extract_block)
    her2_df["block"] = her2_df["Matched_HE_SlideName"].astype(str).apply(extract_block)

    # --- Step 2: Find biopsies present in Her2 file ---
    her2_blocks = set(her2_df["block"])

    # --- Step 3: Update fold values ---
    mask = (cancer_df["block"].isin(her2_blocks) == False) & (cancer_df["fold"] == 6)
    cancer_df.loc[mask, "fold"] = 7

    # Save updated cancer file
    cancer_df.to_csv(os.path.join(csvs_path, "cancer_classification_slides_ann.csv"), index=False)


if __name__ == "__main__":
    main()
    # add_annotated_slides()