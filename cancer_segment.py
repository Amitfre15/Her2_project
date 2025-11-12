import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import mode
import argparse
import torch
# from shapely.geometry import Point, Polygon
import matplotlib.patches as mpatches
from workspace.Gigapath_GIP.finetune.utils import parse_tile_name, correct_coords, slide_to_thumb_coord, thumb_to_slide_coord, SLIDE_HEIGHT, SLIDE_WIDTH

OPENSLIDE_PATH = r"C:\Program Files\openslide-bin-4.0.0.3-windows-x64\openslide-bin-4.0.0.3-windows-x64\bin"

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

SEG_THUMB_HEIGHT = 97
SEG_THUMB_WIDTH = 42


def filter_bg(arr: np.array) -> np.array:
    # img_rgb is your input image of shape (H, W, 3)
    img_rgb = arr.copy()  # Replace with your actual array

    # Extract RGB channels
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    # Create a boolean mask based on the condition
    mask = (R > 200) & (G > 100) & (G < 150) & (B > 100) & (B < 150)

    # Option 3: Create a new image where non-matching pixels are zeroed out
    filtered_image = np.zeros_like(img_rgb)
    filtered_image[mask] = img_rgb[mask]

    return filtered_image, mask

def segment_tiles_from_annotations(tumor_polygons: list[np.array], non_tumor_polygons: list[np.array], segment_map: np.array) -> np.array:
    """
    Segment tiles based on tumor polygons.
    Args:
        tumor_polygonss list(np.array): Tumor polygon coordinates.
        non_tumor_polygons list(np.array): Non-Tumor polygon coordinates.
    Returns:
        np.array: Segmented tile map.
    """
    t_polys = [Polygon(poly) for poly in tumor_polygons]
    nt_polys = [Polygon(poly) for poly in non_tumor_polygons]
    for i in range(SEG_THUMB_HEIGHT):
        for j in range(SEG_THUMB_WIDTH):
            slide_point = thumb_to_slide_coord(x_y_thumb_coords=np.array([[i + 0.5, j + 0.5]]),
                                                   thumb_size=(SEG_THUMB_WIDTH, SEG_THUMB_HEIGHT),
                                                   slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT))
            for poly in t_polys:
                if poly.contains(Point(slide_point)):
                    segment_map[i, j] = 0  # 0 for tumor area
            for poly in nt_polys:
                if poly.contains(Point(slide_point)):
                    segment_map[i, j] = 3  # 3 for non-tumor area

    return segment_map


def segment_tiles(segment_map: np.array, bg_mask: np.array, segment_dir: str) -> np.array:
    segment_map[bg_mask] = 5  # Set background pixels to 5

    # Target window size
    target_shape = (SEG_THUMB_HEIGHT, SEG_THUMB_WIDTH)
    window_size = round(segment_map.shape[0] / target_shape[0]) # 1242 / 97 = 13

    # Compute padding needed
    pad_h = window_size * target_shape[0] - segment_map.shape[0]  # 1261 - 1242 = 19
    pad_w = window_size * target_shape[1] - segment_map.shape[1]  # 546 - 543 = 3

    # Pad with zeros
    arr_padded = np.pad(segment_map, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=5)

    # Reshape into (97, 13, 42, 13) then swapaxes to (97, 42, 13, 13)
    reshaped = arr_padded.reshape(target_shape[0], window_size, target_shape[1], window_size).swapaxes(1, 2)

    # Compute mode in each 13x13 window (ignore zeros)
    windows = reshaped.reshape(target_shape[0], target_shape[1], -1)

    # init output with full fives
    output = np.full(target_shape, fill_value=5, dtype=int) 

    # Threshold for 25% zeros
    zero_thresh = 0.05 * (window_size ** 2)

    # Compute output
    for i in range(target_shape[0]):
        for j in range(target_shape[1]):
            window = windows[i, j]
            zero_count = np.count_nonzero(window == 0)
            if zero_count > zero_thresh:
                output[i, j] = 0
            else:
                output[i, j] = mode(window, keepdims=False).mode
    
    return output

def save_segmented_tiles(segment_map: np.array, segment_dir: str, save_name: str):
    """
    Save segmented tiles as an image and numpy array.
    Args:
        segment_map (np.array): Segmented tile map.
        segment_dir (str): Directory to save the segmented tiles.
    """
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir, exist_ok=True)
    
    save_path = os.path.join(segment_dir, f'{save_name}.png')
    np.save(os.path.join(segment_dir, f'{save_name}.npy'), segment_map)
    plt.figure(figsize=(10, 8))
    plt.title(save_name)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.imshow(segment_map, cmap='jet', interpolation='nearest')
    # plt.colorbar()
    # write label for color bar 
    
    if "resnet" in save_name:
        cbar = plt.colorbar()
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['Tumor', 'Stroma', 'Inflammation', 'Necrosis', 'Other', 'Background'])
    elif "Tumor Annotations" in save_name:
        # cbar.set_ticks([0, 3, 5])
        # cbar.set_ticklabels(['Tumor', 'Non-Tumor', 'Background'])
        cmap = plt.get_cmap("jet")   # 🔴 replace "tab10" with your actual cmap
        class_indices = [0, 154, 255]
        class_labels = ['Tumor', 'Non-Tumor', 'Background']
        
        legend_elements = [
            mpatches.Patch(color=cmap(i), label=label)
            for i, label in zip(class_indices, class_labels)
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', title="Annotations")
    else:
        cbar = plt.colorbar()
        # set label to cbar
        cbar.set_label("Cancer Probability", rotation=270, labelpad=15)
    plt.show()
    plt.savefig(save_path)
    plt.close()

def parse_coordinates(coord_str):
    """
    Parse coordinates from string format like:
    '30088.0x40076.0y | 27018.0x44149.0y | ...'
    into np.array([[x1, y1], [x2, y2], ...])
    """
    coords = []
    for part in coord_str.split(" | "):
        x_str, y_str = part.split("x")
        x = float(x_str)
        y = float(y_str.replace("y", ""))
        coords.append([x, y])
    return np.array(coords)

def is_inside_polygon(test_point, polygon_coords):
    """
    Check if test_point (np.array([x,y])) is inside polygon
    """
    polygon = Polygon(polygon_coords)
    point = Point(test_point)
    return polygon.contains(point)

def load_file_polygons(df: pd.DataFrame):
    """
    Group polygons by file.
    Returns a DataFrame with one row per file and a list of polygons.
    """
    # Parse coordinates into np.array polygons
    df["polygon"] = df["Coordinates"].apply(parse_coordinates)

    # Split tumor vs non-tumor
    df_tumor = df[df["Annotation"] == "Tumor"].groupby("file")["polygon"].apply(list)
    df_non_tumor = df[df["Annotation"] == "Non-tumor"].groupby("file")["polygon"].apply(list)

    # Combine into a single DataFrame
    grouped = pd.DataFrame({
        "tumor_polygons": df_tumor,
        "non_tumor_polygons": df_non_tumor
    }).reset_index()

    # Fill missing groups with empty list
    grouped["tumor_polygons"] = grouped["tumor_polygons"].apply(lambda x: x if isinstance(x, list) else [])
    grouped["non_tumor_polygons"] = grouped["non_tumor_polygons"].apply(lambda x: x if isinstance(x, list) else [])

    return grouped

def read_slide_file(excel_path: str, args):
    # Read slide metadata
    try:
        # slides_df = pd.read_excel(excel_path)
        slides_df = pd.read_csv(excel_path)
    except Exception as e:
        raise FileNotFoundError(f"Error reading the Excel file: {e}")

    # Check if 'mpp' column exists
    if 'MPP' not in slides_df.columns:
        raise KeyError(f"Column 'MPP' not found in the Excel file. Available columns: {slides_df.columns.tolist()}")

    # Drop rows with null or string 'MPP' values
    slides_df = slides_df.dropna(subset=['MPP'])
    slides_df = slides_df[slides_df['MPP'].apply(lambda x: isinstance(x, (int, float)))]

    if args.cancer_annotations:
        slides_df = load_file_polygons(slides_df)

    if args.registration_annotated:
        batches = ['Batch_1', 'Batch_2']
        slide_path_key = 'Path'
        slides_df = slides_df[slides_df[slide_path_key].str.contains('|'.join(batches), na=False)]
    
    return slides_df

def segment_from_model_or_annotations(args, row, slide_file, segment_dir, st):
    if args.registration_annotated:
        owc = os.path.join(segment_dir, 'overlay_with_colorbar.npy')
        pred = os.path.join(segment_dir, 'wsi_pred.npy')
        if not os.path.exists(st):
            if os.path.exists(owc):
                owc_arr = np.load(owc)
                bg_filter, bg_mask = filter_bg(owc_arr)
            else:
                print(f"Overlay with colorbar {owc} does not exist. Skipping...")
                return None
            if os.path.exists(pred):
                pred_arr = np.load(pred)
                tile_segment = segment_tiles(pred_arr, bg_mask, segment_dir)
            else:
                print(f"Prediction file {pred} does not exist. Skipping...")
                return None
        else:
            tile_segment = np.load(st)
    
    elif args.cancer_annotations:
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir, exist_ok=True)

        # Initialize a blank segment map
        tile_segment = np.full((SEG_THUMB_HEIGHT, SEG_THUMB_WIDTH), fill_value=5, dtype=int)  # 5 for background
        # tumor_polygons = row['polygon']
        tumor_polygons = row['tumor_polygons']
        non_tumor_polygons = row['non_tumor_polygons']
        
        if tumor_polygons is None:
            print(f"No tumor polygon for slide {slide_file}. Skipping...")
            return None
        slide = openslide.OpenSlide(slide_file)
        bbox_x = int(slide.properties['openslide.bounds-x'])
        bbox_y = int(slide.properties['openslide.bounds-y'])
        for tumor_polygon in tumor_polygons:
            tumor_polygon[:, 0] += bbox_y
            tumor_polygon[:, 1] += bbox_x
        for non_tumor_polygon in non_tumor_polygons:
            non_tumor_polygon[:, 0] += bbox_y
            non_tumor_polygon[:, 1] += bbox_x
        tile_segment = segment_tiles_from_annotations(tumor_polygons=tumor_polygons, non_tumor_polygons=non_tumor_polygons, segment_map=tile_segment)
    
    save_name = 'Tumor Annotations' if args.cancer_annotations else 'fcn_resnet50_unet-bcss predictions'
    save_segmented_tiles(segment_map=tile_segment, segment_dir=segment_dir, save_name=save_name)
    return tile_segment

def save_tumor_indices_and_cancer_probs(args, full_ti_dir, ti, nti, cp, segment_dir, slide_name, valid_tiles, valid_indices, tile_segment):
    tumor_indices = []
    non_tumor_indices = []
    if args.seg_from_cancer_predictions:
        if os.path.exists(cp):
            cancer_prob = np.load(cp)
            cancer_prob = cancer_prob[valid_indices]  # Filter to valid tiles
        else:
            print(f"cancer_prob file {cp} does not exist. Skipping...")
            return None
        tile_pred_segment = np.full((SEG_THUMB_HEIGHT, SEG_THUMB_WIDTH), fill_value=-1, dtype=float)
        
    for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
        x_start, x_end, y_start, y_end = parse_tile_name(tile_name=tile_name)
        he_window_center = ((x_end + x_start) // 2, (y_end + y_start) // 2)
        he_window_center = np.array(he_window_center).reshape(-1, 2)  # Ensure it's a 2D array for processing
        he_window_center = correct_coords(he_window_center)
        
        he_thumb_window_center = slide_to_thumb_coord(x_y_slide_coords=he_window_center, slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                    thumb_size=(SEG_THUMB_WIDTH, SEG_THUMB_HEIGHT))
        # clip to ensure coordinates are within bounds
        he_thumb_window_center[0][1] = np.clip(he_thumb_window_center[0][1], 0, SEG_THUMB_WIDTH - 1)
        he_thumb_window_center[0][0] = np.clip(he_thumb_window_center[0][0], 0, SEG_THUMB_HEIGHT - 1)

        if args.save_tumor_indices_from_seg_map:
            if tile_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] == 0:
                tumor_indices.append(idx)
            if tile_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] == 3:
                non_tumor_indices.append(idx)

        if args.seg_from_cancer_predictions:
            tile_pred_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] = cancer_prob[idx]
            if cancer_prob[idx] >= 0.5:
                tumor_indices.append(idx)
            
    
    if args.seg_from_cancer_predictions:
        save_segmented_tiles(segment_map=tile_pred_segment, segment_dir=segment_dir, save_name=f'Binary Cancer Probability (MPP=2 GigaPath features trained model) val_fold = {args.val_fold}')

    if args.save_tumor_indices_from_seg_map or args.seg_from_cancer_predictions:
        if len(tumor_indices) == 0:
            print(f"No tumor tiles found for slide {slide_name}.")
            # return None

        if not os.path.exists(full_ti_dir):
            os.makedirs(full_ti_dir, exist_ok=True)

        tumor_indices = np.array(tumor_indices)
        print(f"Saving tumor indices for {slide_name} to {ti}")
        np.save(ti, tumor_indices)

        if args.cancer_annotations:
            non_tumor_indices = np.array(non_tumor_indices)
            print(f"Saving non-tumor indices for {slide_name} to {nti}")
            np.save(nti, non_tumor_indices)
    
    return tumor_indices

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Search and replace file names and content based on Excel mapping.")
    parser.add_argument('-s', '--save_path', type=str, help='Path to save tiles and embeds', required=True)
    parser.add_argument('-e', '--excel_path', type=str, help='Slides excel path', required=True)
    parser.add_argument('-cpp', '--cancer_prob_path', type=str, help='Cancer probabilities path', required=True)
    parser.add_argument('-seg', '--segment_tiles', action='store_true', default=False, help='Segment image tiles')
    parser.add_argument('-y_std', '--y_std', action='store_true', default=False, help='Calculate standard deviation of tile_y')
    parser.add_argument('-reg_an', '--registration_annotated', action='store_true', default=False, help='filter annotated slides for registration')
    parser.add_argument('-cancer_an', '--cancer_annotations', action='store_true', default=False, help='process annotated slides for cancer classification')
    parser.add_argument('-save_tifsm', '--save_tumor_indices_from_seg_map', action='store_true', default=False, help='save tumor tile indices from segmentation map')
    parser.add_argument('-seg_from_cp', '--seg_from_cancer_predictions', action='store_true', default=False, help='save tile cancer prediction map')
    parser.add_argument('-vf', '--val_fold', type=str, help='validation fold used for cancer model', default='', choices=['1', '2', '3', '4', '5'])

    args = parser.parse_args()
    print(f'args = {args}')

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    target_mpp = 2 # 1 
    suffix = f'_mpp{target_mpp}' if target_mpp != 0.5 else ''
    
    # Paths
    # excel_path = os.path.join(base_path, "slides_data_HER2_2.xlsx")
    # excel_path = os.path.join(base_path, "slides_data_CARMEL9.xlsx")
    # excel_path = "/home/amitf/workspace/WSI/metadata_csvs/Her2_slides_matched_HE_folds_HE.csv"
    excel_path = args.excel_path
    save_path = args.save_path
    cancer_prob_path = args.cancer_prob_path
    tile_y_dir = os.path.join(save_path, f'tile_y{suffix}')
    tiles_dir = os.path.join(save_path.replace('IHC', 'HE'), f'png_tiles{suffix}')
    matching_tiles_dir = os.path.join(save_path, f'matching_tiles{suffix}')

    slides_df = read_slide_file(excel_path=excel_path, args=args)

    results_df = pd.DataFrame(columns=['slide_name', 'tumor_tile_y_std', 'num_tiles', 'num_tumor_tiles', 'label', 'fold'])
    all_tumor_y = np.array([], dtype=np.float32)

    # Process each slide with progress tracking and saving
    for idx, row in tqdm(slides_df.iterrows(), total=len(slides_df)):
        tile_segment = None

        if not args.cancer_annotations:
            slide_file, matching_he_slide = row['file'], row['Matched_HE_SlideName'].split('.')[0]
            slide_name = os.path.splitext(slide_file)[0]
        else:
            matching_he_slide = row['file'].split('.')[0]
            slide_name = matching_he_slide
            slides_dir = os.path.join("/data/Breast/Carmel/9-11/Batch_11/CARMEL11")
            slide_file = os.path.join(slides_dir, row['file'])
        
        ######################################################
        # if not slide_name.startswith('21-1518_'):
        #     continue
        
        full_tiles_dir = os.path.join(tiles_dir, matching_he_slide)
        full_tile_y_dir = os.path.join(tile_y_dir, matching_he_slide)
        full_mt_dir = os.path.join(matching_tiles_dir, matching_he_slide)
        full_ti_dir = full_mt_dir.replace('matching_tiles', 'tumor_indices')
        tile_pred_y_dir = os.path.join(full_tile_y_dir.replace('tile_y', 'tile_pred_y').replace('IHC', 'HE'))
        segment_dir = os.path.join(save_path.replace('IHC', 'HE'), f'slide_segmentations{suffix}', matching_he_slide)
        cancer_prob_dir = os.path.join(cancer_prob_path, "cancer_probs_mpp2", matching_he_slide)
        ti = os.path.join(full_ti_dir, f'tumor_indices{args.val_fold}.npy')
        if args.save_tumor_indices_from_seg_map:
            ti = ti.replace('tumor_indices.npy', 'tumor_ann_indices.npy')
        nti = os.path.join(full_ti_dir, 'non_tumor_indices.npy')
        cp = os.path.join(cancer_prob_dir, f'cancer_prob_val{args.val_fold}.npy')
        st = os.path.join(segment_dir, 'segmented_tiles.npy')

        tiles = os.listdir(full_tiles_dir)

        if args.y_std:
            if not os.path.exists(full_tile_y_dir):
                print(f"Directory {full_tile_y_dir} does not exist. Skipping...")
                continue

            tile_y = np.load(os.path.join(full_tile_y_dir, 'tile_y.npy'))
            # tile_pred_y = np.load(os.path.join(tile_pred_y_dir, 'tile_pred_y.npy'))

        if not args.cancer_annotations:
            matching_tiles = np.load(os.path.join(full_mt_dir, 'ihc_tiles.npy'))
            valid_tiles = [tile for idx, tile in enumerate(tiles) if not np.isnan(matching_tiles[idx])]
            valid_indices = [idx for idx, tile in enumerate(tiles) if not np.isnan(matching_tiles[idx])]
        else:
            valid_tiles = tiles
            valid_indices = list(range(len(tiles)))

        if args.segment_tiles:
            tile_segment = segment_from_model_or_annotations(args, row, slide_file, segment_dir, st)
            if tile_segment is None:
                continue
        elif args.save_tumor_indices_from_seg_map:
            if tile_segment is None and os.path.exists(st):
                tile_segment = np.load(st)
            else:
                print(f"Tile segmentation map not found for slide {slide_name}. Skipping...")
                continue
            
        if args.save_tumor_indices_from_seg_map or args.seg_from_cancer_predictions:
            tumor_indices = save_tumor_indices_and_cancer_probs(args, full_ti_dir, ti, nti, cp, segment_dir, slide_name, valid_tiles, valid_indices, tile_segment)
        else:
            tumor_indices = np.load(ti) if os.path.exists(ti) else None
            
        if args.y_std:
            if tumor_indices.size == 0:
                print(f"No tumor indices found for slide {slide_name}. Skipping...")
                continue
            tumor_tile_y = tile_y[tumor_indices]
            all_tumor_y = np.concatenate((all_tumor_y, tumor_tile_y.flatten()), axis=0)

            # Calculate standard deviation of tile_y
            tumor_tile_y_std = np.std(tumor_tile_y)
            num_tiles = tile_y.shape[0]
            num_tumor_tiles = tumor_tile_y.shape[0]
            label = row['label']
            fold = row['fold']
            results_df.loc[len(results_df)] = {
                'slide_name': slide_name,
                'tumor_tile_y_std': tumor_tile_y_std,
                'num_tiles': num_tiles,
                'num_tumor_tiles': num_tumor_tiles,
                'label': label,
                'fold': fold
            }
    
    if args.y_std:
        print(f"all_tumor_y.max() = {all_tumor_y.max()}")
        print(f"all_tumor_y.min() = {all_tumor_y.min()}")

        # Save results to CSV
        results_file = os.path.join(save_path, f'tumor_tile_y_std_results_w_my_cancer_model{suffix}.csv')
        results_df.to_csv(results_file, index=False)



if __name__ == '__main__':
    main()
    # prob_cc = np.load('/SSDStorage/Breast/Carmel/Her2/gigapath_cancer_classification/cancer_probs_mpp2/21-2110_1_1_b/cancer_prob.npy')
    # prob_he = np.load('/SSDStorage/Breast/Carmel/Her2/gigapath_HE/cancer_probs_mpp2/21-2110_1_1_b/cancer_prob.npy')
    # mt = np.load('/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/matching_tiles_mpp2/21-6922_1_1_e/ihc_tiles.npy')
    # tile_y = np.load('/SSDStorage/Breast/Carmel/Her2/gigapath_IHC/tile_y_mpp2/21-1518_1_5_a/tile_y.npy')
    # pass