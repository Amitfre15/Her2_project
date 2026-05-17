import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
import cv2
from tqdm import tqdm
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from workspace.Gigapath_GIP.finetune.utils import load_npy_file

NUM_H_TICKS = 48
NUM_W_TICKS = 21

def visualize_ihc_maps(maps, titles, ihc_thumb_path, slide_name, save_path, vmin=-1, vmax=3):
    num_axes = len(maps) + 1
    fig, axes = plt.subplots(1, num_axes, figsize=(18, 9), gridspec_kw={"width_ratios": [1 for _ in range(num_axes)]}, constrained_layout=True)

    # IHC thumbnail
    ihc_thumb = cv2.imread(ihc_thumb_path)  # shape (5115, 2237, 3)
    ihc_thumb = cv2.cvtColor(ihc_thumb, cv2.COLOR_BGR2RGB)

    # num_h_ticks = maps[-1].shape[0] // 2
    # num_w_ticks = maps[-1].shape[1] // 2
    x_tick_labels = np.round(np.linspace(0, maps[-1].shape[1], NUM_W_TICKS)).astype(int)
    y_tick_labels = np.round(np.linspace(0, maps[-1].shape[0], NUM_H_TICKS)).astype(int)

    axes[0].set_title("IHC slide", fontsize=18)
    axes[0].imshow(ihc_thumb)
    axes[0].grid(True, color="black", linestyle="--", linewidth=0.3, alpha=0.7)
    axes[0].set_xticks(np.linspace(0, ihc_thumb.shape[1], NUM_W_TICKS), x_tick_labels) 
    axes[0].set_yticks(np.linspace(0, ihc_thumb.shape[0], NUM_H_TICKS), y_tick_labels)
    axes[0].tick_params(labelsize=6)
    axes[0].tick_params(axis="x", labelrotation=45)

    for ax, m, title in zip(axes[1:], maps, titles):
        # shift a row down
        m = np.roll(m, 1, axis=0)

        if "Annotations".lower() in title.lower():            
            cmap = plt.get_cmap("jet")
            class_indices = [0, 64, 128, 192, 255]
            class_labels = ['Background', 'HER2 0', 'HER2 1+', 'HER2 2+', 'HER2 3+']
            
            legend_elements = [
                mpatches.Patch(color=cmap(i), label=label)
                for i, label in zip(class_indices, class_labels)
            ]
            
            plt.legend(handles=legend_elements, loc='upper right', title="Annotations")

        im = ax.imshow(
            m,
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
            # aspect="auto"
        )
        ax.set_title(title, fontsize=18)
        ax.grid(True, color="white", linestyle="--", linewidth=0.3, alpha=0.7)
        ax.set_xticks(np.linspace(0, m.shape[1], NUM_W_TICKS), x_tick_labels)
        ax.set_yticks(np.linspace(0, m.shape[0], NUM_H_TICKS), y_tick_labels)
        ax.tick_params(labelsize=6)
        ax.tick_params(axis="x", labelrotation=45)

    # Single shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label("Score")
    # set overall title
    fig.suptitle(f"Score maps for slide {slide_name}", fontsize=20)

    save_path = os.path.join(save_path, "score_maps.png")

    plt.show()
    plt.savefig(save_path)
    plt.close()


def visualize_he_maps(maps, titles, he_thumb_path, slide_name, save_path, vmin=-1, vmax=3):
    num_axes = len(maps) + 1
    fig, axes = plt.subplots(1, num_axes, figsize=(12, 9), gridspec_kw={"width_ratios": [1 for _ in range(num_axes)]}, constrained_layout=True)

    # HE thumbnail
    he_thumb = cv2.imread(he_thumb_path)  # shape (5115, 2237, 3)
    he_thumb = cv2.cvtColor(he_thumb, cv2.COLOR_BGR2RGB)

    num_h_ticks = maps[-1].shape[0] // 2
    num_w_ticks = maps[-1].shape[1] // 2
    x_tick_labels = np.round(np.linspace(0, maps[-1].shape[1], NUM_W_TICKS)).astype(int)
    y_tick_labels = np.round(np.linspace(0, maps[-1].shape[0], NUM_H_TICKS)).astype(int)

    axes[0].set_title("HE slide", fontsize=18)
    axes[0].imshow(he_thumb)
    axes[0].grid(True, color="black", linestyle="--", linewidth=0.3, alpha=0.7)
    axes[0].set_xticks(np.linspace(0, he_thumb.shape[1], NUM_W_TICKS), x_tick_labels) 
    axes[0].set_yticks(np.linspace(0, he_thumb.shape[0], NUM_H_TICKS), y_tick_labels)
    axes[0].tick_params(labelsize=6)
    axes[0].tick_params(axis="x", labelrotation=45)

    for ax, m, title in zip(axes[1:], maps, titles):
        # shift a row down
        m = np.roll(m, 1, axis=0)
        if "Tumor" in title:
            # swap 5 and 0 values for background and tumor
            m = np.where(m == 5, 0, np.where(m == 0, 5, m))

            cmap = plt.get_cmap("jet")
            class_indices = [0, 154, 255]
            class_labels = ['Background', 'Non-Tumor', 'Tumor']
            
            legend_elements = [
                mpatches.Patch(color=cmap(i), label=label)
                for i, label in zip(class_indices, class_labels)
            ]
            
            plt.legend(handles=legend_elements, loc='upper right', title="Annotations")
            vmin, vmax = 0, 5
            
        im = ax.imshow(
            m,
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=18)
        ax.grid(True, color="white", linestyle="--", linewidth=0.3, alpha=0.7)
        ax.set_xticks(np.linspace(0, m.shape[1], NUM_W_TICKS), x_tick_labels)
        ax.set_yticks(np.linspace(0, m.shape[0], NUM_H_TICKS), y_tick_labels)
        ax.tick_params(labelsize=6)
        ax.tick_params(axis="x", labelrotation=45)



    # Single shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label("Score")
    # set overall title
    fig.suptitle(f"Score maps for slide {slide_name}", fontsize=20)

    save_path = os.path.join(save_path, "score_maps.png")

    plt.show()
    plt.savefig(save_path)
    plt.close()


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Visualize score maps.")
    parser.add_argument('-imp', '--ihc_maps_path', type=str, help='Path to IHC saved maps')
    parser.add_argument('-hmp', '--he_maps_path', type=str, help='Path to HE saved maps')
    parser.add_argument('-cgt', '--cancer_gt_path', type=str, help='Path to cancer annotation maps')
    parser.add_argument('-her2_gt', '--her2_gt_path', type=str, help='Path to HER2 annotation maps')
    parser.add_argument('-e', '--excel_path', type=str, help='Slides excel path', required=True)
    parser.add_argument('-ihc', '--ihc_maps', action='store_true', help='Whether to visualize IHC maps')
    parser.add_argument('-he', '--he_maps', action='store_true', help='Whether to visualize HE maps')

    args = parser.parse_args()
    print(f'args = {args}')

    slides_df = pd.read_csv(args.excel_path)
    thumbs_dir = "/SSDStorage/Breast/Carmel/png_thumb_pairs_karin"

    # Process each slide with progress tracking and saving
    for _, row in tqdm(slides_df.iterrows(), total=len(slides_df)):
        slide_file, matching_he_slide, val_fold = row['SlideName'], row['Matched_HE_SlideName'].split('.')[0], row['fold']
        # if not slide_file.startswith("21-1335"):
        #     continue
        
        if args.ihc_maps:
            slide_dir = next(filter(lambda x: x.startswith(matching_he_slide[:-1]), os.listdir(args.ihc_maps_path)), None)
            if slide_dir is None or not os.path.exists(os.path.join(args.ihc_maps_path, slide_dir)):
                print(f"No ihc_maps_dir was found for slide {slide_file}, fold = {val_fold}. Skipping...")
                continue
            ihc_maps_dir = os.path.join(args.ihc_maps_path, slide_dir)
            

            slide_name = os.path.splitext(slide_file)[0]
            print(f"slide_name = {slide_name}")
            wo_block = slide_name.rsplit("_", 1)[0]
            slide_thumb_path = os.path.join(thumbs_dir, wo_block)

            # Load IHC thumb
            ihc_thumb_name = next(filter(lambda x: x.startswith("copy"), os.listdir(slide_thumb_path)), None)
            ihc_thumb_path = os.path.join(slide_thumb_path, ihc_thumb_name)

            # Load score maps
            y_map = next(filter(lambda x: "Tile HER2 contribution (y) from SW (MPP" in x and x.endswith(f"{val_fold}.npy"), os.listdir(ihc_maps_dir)), None)
            y_file = os.path.join(ihc_maps_dir, y_map)
            y_map_npy = load_npy_file(y_file)
            bl_name = next(filter(lambda x: "regional tumor tile HER2 contribution (y) baseline" in x and x.endswith(f".npy"), os.listdir(ihc_maps_dir)), None)
            if bl_name is None:
                print(f"No SW baseline was found for slide {slide_name}, probably no tumor indices were detected. Skipping...")
                continue

            bl_file = os.path.join(ihc_maps_dir, bl_name)
            baseline_map = load_npy_file(bl_file)
            pred_name = next(filter(lambda x: "regional tumor tile HER2 contribution (y) predictions" in x and x.endswith(f".npy"), os.listdir(ihc_maps_dir)), None)
            model_file = os.path.join(ihc_maps_dir, pred_name)
            model_map = load_npy_file(model_file)

            # Delete
            # ti_dir = args.ihc_maps_path.replace("y_map", "tumor_indices_from_cancer_map")
            # tumor_indices = load_npy_file(os.path.join(ti_dir, matching_he_slide, f"tumor_indices{val_fold}.npy"))

            her2_gt_dir = args.her2_gt_path
            slide_dir = next(filter(lambda x: x.startswith(matching_he_slide[:-1]), os.listdir(args.her2_gt_path)), None)            
            maps = [y_map_npy, baseline_map, model_map]
            titles = ["Y (Pseudo labels)", "Baseline", "Ours"]

            if slide_dir is None or not os.path.exists(os.path.join(args.her2_gt_path, slide_dir)):
                print(f"No her2_gt_dir was found for slide {slide_name}, fold = {val_fold}.")
            else:
                her2_gt_dir = os.path.join(args.her2_gt_path, slide_dir) 
                gt_name = next(filter(lambda x: x.endswith(f"Annotations.npy"), os.listdir(her2_gt_dir)), None)
                gt_file = os.path.join(her2_gt_dir, gt_name)
                gt_map = load_npy_file(gt_file)
                # gt_img = mpimg.imread(gt_file)
                maps.append(gt_map)
                titles.append("HER2 annotations")
            
            visualize_ihc_maps(maps, titles, ihc_thumb_path, slide_name, save_path=ihc_maps_dir, vmin=-1, vmax=3)
        
        if args.he_maps:
            slide_name = os.path.splitext(matching_he_slide)[0]
            print(f"slide_name = {slide_name}")
            wo_block = slide_name.rsplit("_", 1)[0]
            slide_thumb_path = os.path.join(thumbs_dir, wo_block)

            he_maps_dir = os.path.join(args.he_maps_path, matching_he_slide)
            cancer_gt_dir = os.path.join(args.cancer_gt_path, matching_he_slide)
            if not os.path.exists(he_maps_dir):
                print(f"he_maps_dir = {he_maps_dir}")
                print(f"No he_maps_dir was found for slide {slide_name}, fold = {val_fold}. Skipping...")
                continue

            # Load HE thumb
            he_thumb_name = next(filter(lambda x: x.endswith(f"{matching_he_slide}.png"), os.listdir(slide_thumb_path)), None)
            he_thumb_path = os.path.join(slide_thumb_path, he_thumb_name)

            # Load cancer maps
            pred_name = next(filter(lambda x: x.startswith("Tile cancer probability from SW") and x.endswith(f"{val_fold}.npy"), os.listdir(he_maps_dir)), None)
            model_file = os.path.join(he_maps_dir, pred_name)
            model_map = load_npy_file(model_file)
            
            maps = [model_map]
            titles = ["Ours"]
            if not os.path.exists(cancer_gt_dir):
                print(f"No cancer_gt_dir was found for slide {slide_name}, fold = {val_fold}.")
            else:
                gt_name = next(filter(lambda x: x.endswith(f"Annotations.npy"), os.listdir(cancer_gt_dir)), None)
                gt_file = os.path.join(cancer_gt_dir, gt_name)
                gt_map = load_npy_file(gt_file)
                maps.append(gt_map)
                titles.append("Tumor labels")
            
            visualize_he_maps(maps, titles, he_thumb_path, slide_name, save_path=he_maps_dir, vmin=-1, vmax=1)



if __name__ == "__main__":
    main()