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
from workspace.Gigapath_GIP.finetune.utils import (parse_tile_name, correct_coords, slide_to_thumb_coord, thumb_to_slide_coord, load_npy_file, \
                                                   SLIDE_HEIGHT, SLIDE_WIDTH, SLIDE_MPP, SLIDE_PATCH_SIZE)

OPENSLIDE_PATH = r"C:\Program Files\openslide-bin-4.0.0.3-windows-x64\openslide-bin-4.0.0.3-windows-x64\bin"

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


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

def get_tile_thumb_window_center(tile_name: str, target_mpp: float, seg_thumb_width: int, seg_thumb_height: int) -> np.array:
    x_start, x_end, y_start, y_end = parse_tile_name(tile_name=tile_name)
    window_center = ((x_end + x_start) // 2, (y_end + y_start) // 2)
    window_center = np.array(window_center).reshape(-1, 2)  # Ensure it's a 2D array for processing
    window_center = correct_coords(window_center, desired_mpp=target_mpp)
    thumb_window_center = slide_to_thumb_coord(x_y_slide_coords=window_center, slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                thumb_size=(seg_thumb_width, seg_thumb_height))
    # clip to ensure coordinates are within bounds
    thumb_window_center[0][1] = np.clip(thumb_window_center[0][1], 0, seg_thumb_width - 1)
    thumb_window_center[0][0] = np.clip(thumb_window_center[0][0], 0, seg_thumb_height - 1)
    return thumb_window_center

def segment_tiles_for_cancer_from_annotations(args, tumor_polygons: list[np.array], non_tumor_polygons: list[np.array], segment_map: np.array) -> np.array:
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
    for i in range(args.seg_thumb_height):
        for j in range(args.seg_thumb_width):
            slide_point = thumb_to_slide_coord(x_y_thumb_coords=np.array([[i + 0.5, j + 0.5]]),
                                                   thumb_size=(args.seg_thumb_width, args.seg_thumb_height),
                                                   slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT))
            for poly in t_polys:
                if poly.contains(Point(slide_point)):
                    segment_map[i, j] = 0  # 0 for tumor area
            for poly in nt_polys:
                if poly.contains(Point(slide_point)):
                    segment_map[i, j] = 3  # 3 for non-tumor area

    return segment_map

def segment_tiles_for_her2_from_annotations(args, her2_polygons: list[np.array], her2_labels: list[str], segment_map: np.array) -> np.array:
    """
    Segment tiles based on HER2 polygons.
    Args:
        her2_polygons list(np.array): HER2 polygon coordinates.
        her2_labels list(str): HER2 labels.
    Returns:
        np.array: Segmented tile map.
    """
    h_polys = [Polygon(poly) for poly in her2_polygons]
    for i in range(args.seg_thumb_height):
        for j in range(args.seg_thumb_width):
            slide_point = thumb_to_slide_coord(x_y_thumb_coords=np.array([[i + 0.5, j + 0.5]]),
                                                   thumb_size=(args.seg_thumb_width, args.seg_thumb_height),
                                                   slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT))
            for idx, poly in enumerate(h_polys):
                if not isinstance(her2_labels[idx], str):
                    continue
                
                try:
                    label = int(her2_labels[idx])  # set to corresponding HER2 label

                except BaseException as e:
                    print(f"Error converting HER2 label {her2_labels[idx]} to int: {e}")
                    continue
                
                if poly.contains(Point(slide_point)):
                    segment_map[i, j] = label

    return segment_map


def segment_tiles(args, segment_map: np.array, bg_mask: np.array, segment_dir: str) -> np.array:
    segment_map[bg_mask] = 5  # Set background pixels to 5

    # Target window size
    target_shape = (args.seg_thumb_height, args.seg_thumb_width)
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

def save_segmented_tiles(segment_map: np.array, segment_dir: str, save_name: str, value_label: str, vmax: int = 1):
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
    plt.imshow(segment_map, cmap='jet', interpolation='nearest', vmax=vmax)
    # plt.colorbar()
    # write label for color bar 
    
    if "resnet" in save_name:
        cbar = plt.colorbar()
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['Tumor', 'Stroma', 'Inflammation', 'Necrosis', 'Other', 'Background'])
    elif "Tumor Annotations" in save_name:
        cmap = plt.get_cmap("jet")
        class_indices = [0, 154, 255]
        class_labels = ['Tumor', 'Non-Tumor', 'Background']
        
        legend_elements = [
            mpatches.Patch(color=cmap(i), label=label)
            for i, label in zip(class_indices, class_labels)
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', title="Annotations")
    elif "HER2 Annotations" in save_name:
        cmap = plt.get_cmap("jet")
        class_indices = [0, 64, 128, 192, 255]
        class_labels = ['Background', 'HER2 0', 'HER2 1+', 'HER2 2+', 'HER2 3+']
        
        legend_elements = [
            mpatches.Patch(color=cmap(i), label=label)
            for i, label in zip(class_indices, class_labels)
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', title="HER2 Class")
    else:
        cbar = plt.colorbar()
        # set label to cbar
        cbar.set_label(value_label, rotation=270, labelpad=15)
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

def load_file_cancer_polygons(df: pd.DataFrame):
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

def load_file_her2_polygons(df: pd.DataFrame):
    """
    Group HER2 polygons by file.
    Returns a DataFrame with one row per file and a list of polygons and labels.
    """
    # Parse coordinates into np.array polygons
    df["polygon"] = df["Coordinates"].apply(parse_coordinates)

    # Group by file
    grouped = df.groupby("file").agg({
        "polygon": list,
        "Annotation": list
    }).reset_index()

    # Rename columns
    grouped = grouped.rename(columns={
        "polygon": "polygons",
        "Annotation": "Annotation"
    })

    return grouped

def read_slide_file(excel_path: str, args, only_read: bool = False) -> pd.DataFrame:
    # Read slide metadata
    try:
        # slides_df = pd.read_excel(excel_path)
        slides_df = pd.read_csv(excel_path)
    except Exception as e:
        raise FileNotFoundError(f"Error reading the Excel file: {e}")
    if only_read:
        return slides_df

    # Check if 'mpp' column exists
    if 'MPP' not in slides_df.columns:
        raise KeyError(f"Column 'MPP' not found in the Excel file. Available columns: {slides_df.columns.tolist()}")

    # Drop rows with null or string 'MPP' values
    slides_df = slides_df.dropna(subset=['MPP'])
    slides_df = slides_df[slides_df['MPP'].apply(lambda x: isinstance(x, (int, float)))]

    if args.cancer_annotations:
        slides_df = load_file_cancer_polygons(slides_df)
    elif args.her2_annotations or args.save_her2_annotations:
        slides_df = load_file_her2_polygons(slides_df)

    if args.registration_annotated:
        slide_path_key = 'Path'
        batches = ['Batch_1', 'Batch_2']
        slides_df = slides_df[slides_df[slide_path_key].str.contains('|'.join(batches), na=False)]

    if args.test_set:
        test_fold = 6
        slides_df = slides_df[slides_df['fold'] == test_fold]
    
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
                tile_segment = segment_tiles(args=args, pred_arr=pred_arr, bg_mask=bg_mask, segment_dir=segment_dir)
            else:
                print(f"Prediction file {pred} does not exist. Skipping...")
                return None
        else:
            tile_segment = np.load(st)
        save_name = 'fcn_resnet50_unet-bcss predictions'
    
    elif args.cancer_annotations:
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir, exist_ok=True)

        # Initialize a blank segment map
        tile_segment = np.full((args.seg_thumb_height, args.seg_thumb_width), fill_value=5, dtype=int)  # 5 for background
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
        tile_segment = segment_tiles_for_cancer_from_annotations(args=args,tumor_polygons=tumor_polygons, non_tumor_polygons=non_tumor_polygons, segment_map=tile_segment)
        save_name = 'Tumor Annotations'

    elif args.her2_annotations:
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir, exist_ok=True)

        # Initialize a blank segment map
        tile_segment = np.full((args.seg_thumb_height, args.seg_thumb_width), fill_value=-1, dtype=int)  # -1 for background
        her2_polygons = row['polygons']
        her2_labels = row['Annotation']
        
        if her2_polygons is None:
            print(f"No HER2 polygon for slide {slide_file}. Skipping...")
            return None
        slide = openslide.OpenSlide(slide_file)
        bbox_x = int(slide.properties['openslide.bounds-x'])
        bbox_y = int(slide.properties['openslide.bounds-y'])
        for tumor_polygon in her2_polygons:
            tumor_polygon[:, 0] += bbox_y
            tumor_polygon[:, 1] += bbox_x

        tile_segment = segment_tiles_for_her2_from_annotations(args=args, her2_polygons=her2_polygons, her2_labels=her2_labels, segment_map=tile_segment)
        save_name = f'HER2 Annotations'

    save_segmented_tiles(segment_map=tile_segment, segment_dir=segment_dir, save_name=save_name, value_label='HER2 Class', vmax=3)
    return tile_segment


def save_tumor_indices_and_cancer_probs(args, full_ti_dir, ti, nti, cp, segment_dir, slide_name, valid_tiles, valid_indices, tile_segment):
    tumor_indices = []
    non_tumor_indices = []
    if args.seg_from_my_cancer_predictions:
        if os.path.exists(cp):
            cancer_prob = np.load(cp)
            cancer_prob = cancer_prob[valid_indices]  # Filter to valid tiles
        else:
            print(f"cancer_prob file {cp} does not exist. Skipping...")
            return None
        tile_pred_segment = np.full((args.seg_thumb_height, args.seg_thumb_width), fill_value=-1, dtype=float)
        
    for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
        he_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

        if args.save_tumor_indices_from_seg_map:
            if tile_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] == 0:
                tumor_indices.append(idx)
            if tile_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] == 3:
                non_tumor_indices.append(idx)

        if args.seg_from_my_cancer_predictions:
            tile_pred_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] = cancer_prob[idx]
            if cancer_prob[idx] >= 0.5:
                tumor_indices.append(idx)
            
    
    if args.seg_from_my_cancer_predictions:
        save_segmented_tiles(segment_map=tile_pred_segment, segment_dir=segment_dir, save_name=f'Binary Local Cancer Probability (MPP={args.source_mpp} GigaPath features trained model) val_fold = {args.val_fold}',
                             value_label='Cancer Probability')

    if args.save_tumor_indices_from_seg_map or args.seg_from_my_cancer_predictions:
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

def save_ensemble_tumor_indices(args, full_ti_dir, ti, cp, segment_dir, slide_name, valid_tiles, valid_indices, tile_segment):
    tumor_indices = []
    if os.path.exists(os.path.dirname(cp)):
        cancer_probs = os.listdir(os.path.dirname(cp))
        cp_npys = []
        for cancer_prob_file in cancer_probs:
            if cancer_prob_file.endswith('.npy'):
                cp_path = os.path.join(os.path.dirname(cp), cancer_prob_file)
                if os.path.exists(cp_path):
                    cancer_prob = np.load(cp_path)
                    cancer_prob = cancer_prob[valid_indices]  # Filter to valid tiles
                    cp_npys.append(cancer_prob)
                else:
                    print(f"cancer_prob file {cp_path} does not exist. Skipping...")
        cp_npys = np.array(cp_npys)
        # tiles where all models predict cancer (prob >= 0.5) should hold the mean, otherwise set to 0
        all_models_predict_cancer = np.all(cp_npys >= 0.5, axis=0)

        # ensemble_cancer_prob = np.mean(cp_npys, axis=0)
        ensemble_cancer_prob = np.where(all_models_predict_cancer, np.mean(cp_npys, axis=0), 0)

        tile_pred_segment = np.full((args.seg_thumb_height, args.seg_thumb_width), fill_value=-1, dtype=float)
        for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
            he_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

            tile_pred_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] = ensemble_cancer_prob[idx]
            if ensemble_cancer_prob[idx] >= 0.5:
                tumor_indices.append(idx)
        
        save_segmented_tiles(segment_map=tile_pred_segment, segment_dir=segment_dir, save_name=f'Ensemble Binary Local Cancer Probability (MPP={args.source_mpp} GigaPath features trained model)',
                            value_label='Cancer Probability')

        if len(tumor_indices) == 0:
            print(f"No tumor tiles found for slide {slide_name}.")
            # return None

        tumor_indices = np.array(tumor_indices)
        print(f"Saving tumor indices for {slide_name} to {ti}")
        np.save(ti, tumor_indices)
        
    else:
        print(f"cancer_prob file {cp} does not exist. Skipping...")
        return None



def save_her2_annotations(args, full_ant_dir, slide_name, valid_tiles, tile_segment):
    annotated_indices = []
    tile_gt = []

    for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
        ihc_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

        if tile_segment[ihc_thumb_window_center[0][0], ihc_thumb_window_center[0][1]] != -1:
            annotated_indices.append(idx)
            tile_gt.append(tile_segment[ihc_thumb_window_center[0][0], ihc_thumb_window_center[0][1]])

    if not os.path.exists(full_ant_dir):
        os.makedirs(full_ant_dir, exist_ok=True)

    ani = os.path.join(full_ant_dir, f'her2_annotated_indices.npy')
    tgt = os.path.join(full_ant_dir, f'her2_tile_gt.npy')

    annotated_indices = np.array(annotated_indices)
    print(f"Saving annotated indices for {slide_name} to {ani}")
    np.save(ani, annotated_indices)

    tile_gt = np.array(tile_gt)
    print(f"Saving tile ground truth for {slide_name} to {tgt}")
    np.save(tgt, tile_gt)


def local_y_maps(args, tile_y: np.array, segment_dir: str, slide_name: str, valid_tiles: np.array, save_name: str):
    tile_pred_segment = np.full((args.seg_thumb_height, args.seg_thumb_width), fill_value=-1, dtype=float)
        
    for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
        ihc_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

        tile_pred_segment[ihc_thumb_window_center[0][0], ihc_thumb_window_center[0][1]] = tile_y[idx]
            
    save_segmented_tiles(segment_map=tile_pred_segment, segment_dir=segment_dir, save_name=save_name,
                         value_label='Tile HER2 contribution (y)', vmax=3)
    

def extract_tile_y_from_y_map(args, y_map: np.array, tile_y_dir: str, slide_name: str, valid_tiles: np.array, save_name: str):
    tile_y = []

    for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
        ihc_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

        tile_y.append(y_map[ihc_thumb_window_center[0][0], ihc_thumb_window_center[0][1]])
            
    tile_y = np.array(tile_y)
    save_path = os.path.join(tile_y_dir, save_name)
    np.save(save_path, tile_y)


def extract_tumor_indices_from_cancer_map(args, cancer_map: np.array, tumor_indices_dir: str, cancer_probs_dir: str, slide_name: str, valid_tiles: np.array, save_name: str):
    tumor_indices = []
    cancer_probs = []

    for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
        thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)
        cancer_prob = cancer_map[thumb_window_center[0][0], thumb_window_center[0][1]]
        if cancer_prob >= 0.5:
            tumor_indices.append(idx)
        cancer_probs.append(cancer_prob)

    tumor_indices = np.array(tumor_indices)
    print(f"tumor_indices.shape = {tumor_indices.shape}")
    save_path = os.path.join(tumor_indices_dir, save_name)
    np.save(save_path, tumor_indices)

    cp_save_name = save_name.replace('tumor_indices', 'cancer_prob_val')
    cancer_probs = np.array(cancer_probs)
    cp_save_path = os.path.join(cancer_probs_dir, cp_save_name)
    np.save(cp_save_path, cancer_probs)


def convert_he_y_map_to_ihc_y_map(args, he_y_map: np.array, segment_dir: str, slide_name: str, valid_tiles: np.array, save_name: str):
    tile_pred_segment = np.full((args.seg_thumb_height, args.seg_thumb_width), fill_value=-1, dtype=float)
        
    for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
        he_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

        tile_pred_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] = he_y_map[he_thumb_window_center[0][0], he_thumb_window_center[0][1]]
            
    save_segmented_tiles(segment_map=tile_pred_segment, segment_dir=segment_dir, save_name=save_name,
                         value_label='Tile HER2 contribution (y) from HE', vmax=3)
    

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Create and process score maps.")
    parser.add_argument('-s', '--save_path', type=str, help='Path to save tiles and embeds', required=True)
    parser.add_argument('-e', '--excel_path', type=str, help='Slides excel path', required=True)
    parser.add_argument('-blod', '--baseline_output_dir', type=str, help='Output dir for baseline predictions')
    parser.add_argument('-pod', '--predictions_output_dir', type=str, help='Output dir for model predictions')
    parser.add_argument('-cypc', '--create_y_sw_bl_csv', action='store_true', default=False, help='Create csvs for y predictions')
    parser.add_argument('-cpp', '--cancer_prob_path', type=str, help='Cancer probabilities path')
    parser.add_argument('-seg', '--segment_tiles', action='store_true', default=False, help='Segment image tiles')
    parser.add_argument('-y_std', '--y_std', action='store_true', default=False, help='Calculate standard deviation of tile_y')
    parser.add_argument('-reg_an', '--registration_annotated', action='store_true', default=False, help='filter annotated slides for registration')
    parser.add_argument('-ad', '--all_data', action='store_true', default=False, help='Use all the data (annotated)')
    parser.add_argument('-cancer_an', '--cancer_annotations', action='store_true', default=False, help='process annotated slides for cancer classification')
    parser.add_argument('-her2_an', '--her2_annotations', action='store_true', default=False, help='process annotated slides for HER2 classification')
    parser.add_argument('-save_her2_an', '--save_her2_annotations', action='store_true', default=False, help='save HER2 annotations from segmentation map')
    parser.add_argument('-save_tifsm', '--save_tumor_indices_from_seg_map', action='store_true', default=False, help='save tumor tile indices from segmentation map')
    parser.add_argument('-seg_from_cp', '--seg_from_my_cancer_predictions', action='store_true', default=False, help='save tile cancer prediction map')
    parser.add_argument('-save_ens_ti_seg', '--save_ensemble_tumor_indices', action='store_true', default=False, help='save ensemble tumor indices')
    parser.add_argument('-test_set', '--test_set', action='store_true', default=False, help='iterate only on test set')
    parser.add_argument('-vf', '--val_fold', type=str, help='validation fold used for cancer model', default='', choices=['1', '2', '3', '4', '5'])
    parser.add_argument('-tmpp', '--target_mpp', type=float, help='Target tiles MPP', default=1, choices=[0.5, 1, 2], required=True)
    parser.add_argument('-smpp', '--source_mpp', type=float, help='Source tiles MPP', choices=[0.5, 1, 2])
    parser.add_argument('-seg_y', '--seg_y_labels', action='store_true', default=False, help='save HER2 contribution (y) pseudo labels map')
    parser.add_argument('-seg_y_pred', '--seg_y_predictions', action='store_true', default=False, help='save HER2 contribution (y) prediction map')
    parser.add_argument('-seg_ot', '--seg_only_tumor', action='store_true', default=False, help='save only tumor tiles in HER2 contribution (y) prediction map')
    parser.add_argument('-efy_map', '--extract_tile_y_from_y_map', action='store_true', default=False, help='extract tile y from y map')
    parser.add_argument('-etfcm', '--extract_tumor_indices_from_cancer_map', action='store_true', default=False, help='extract tumor indices from cancer map')
    parser.add_argument('-y_map_fhe', '--y_map_from_he', action='store_true', default=False, help='extract tile y from y map constructed from HE slide')
    parser.add_argument('-y_type', '--y_type', type=str, help='Type of y map to use', default='regional', choices=['regional', 'local'])
    parser.add_argument('-y_pred_type', '--y_pred_type', type=str, help='Type of y_pred file to use', default='regional', choices=['regional', 'local'])
    parser.add_argument('-ti_type', '--tumor_indices_type', type=str, help='Type of tumor_indices file to use', default='local', choices=['regional', 'local'])
    parser.add_argument('-aht', '--all_he_tiles', action='store_true', default=False, help='extract tile y from y map constructed from HE slide and use all HE tiles')
    parser.add_argument('-exter', '--external_he_model', action='store_true', default=False, help='extract tile y from y map constructed by external HE model')

    args = parser.parse_args()
    print(f'args = {args}')

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.target_mpp = int(args.target_mpp) if args.target_mpp != 0.5 else args.target_mpp
    args.source_mpp = int(args.source_mpp) if args.source_mpp != 0.5 else args.source_mpp
    suffix = f'_mpp{args.target_mpp}'
    src_suffix = suffix if not args.source_mpp else f'_mpp{args.source_mpp}'
    
    # Paths
    # excel_path = os.path.join(base_path, "slides_data_HER2_2.xlsx")
    # excel_path = os.path.join(base_path, "slides_data_CARMEL9.xlsx")
    # excel_path = "/home/amitf/workspace/WSI/metadata_csvs/Her2_slides_matched_HE_folds_HE.csv"
    excel_path = args.excel_path
    save_path = args.save_path
    cancer_prob_path = args.cancer_prob_path
    tile_y_dir = os.path.join(save_path, f'tile_y{suffix}')
    he_tiles_dir = os.path.join(save_path.replace('IHC', 'HE'), f'png_tiles{suffix}')
    ihc_tiles_dir = os.path.join(save_path, f'png_tiles{suffix}')
    matching_tiles_dir = os.path.join(save_path, f'matching_tiles{suffix}')

    matched_csv = "/home/amitf/workspace/WSI/metadata_csvs/Her2_slides_matched_HE_folds.csv"
    matched_df = read_slide_file(excel_path=matched_csv, only_read=True, args=args)
    slides_df = read_slide_file(excel_path=excel_path, args=args)

    if args.create_y_sw_bl_csv:
        baseline_dfs = [pd.DataFrame(columns=['tile_y', 'tile_y_bin', 'tile_pred']) for fold in range(5)]
        map_baseline_dfs = [pd.DataFrame(columns=['tile_gt', 'tile_gt_bin', 'map_tile_pred']) for fold in range(5)]
        local_dfs = [pd.DataFrame(columns=['tile_y', 'tile_y_bin', 'tile_pred']) for fold in range(5)]
        map_local_dfs = [pd.DataFrame(columns=['tile_gt', 'tile_gt_bin', 'map_tile_pred']) for fold in range(5)]

    args.seg_thumb_height = int(SLIDE_HEIGHT * SLIDE_MPP // (SLIDE_PATCH_SIZE * args.source_mpp))
    args.seg_thumb_width = int(SLIDE_WIDTH * SLIDE_MPP // (SLIDE_PATCH_SIZE * args.source_mpp))

    # Process each slide with progress tracking and saving
    for _, row in tqdm(slides_df.iterrows(), total=len(slides_df)):
        tile_segment = None

        if args.cancer_annotations:
            matching_he_slide = row['file'].split('.')[0]
            slide_name = matching_he_slide
            slides_dir = os.path.join("/data/Breast/Carmel/9-11/Batch_11/CARMEL11")
            slide_file = os.path.join(slides_dir, row['file'])
        elif args.her2_annotations or args.save_her2_annotations:
            ihc_slide = row['file'].split('.')[0]
            slide_name = ihc_slide
            ihc_path = matched_df[matched_df['SlideName'].str.contains(ihc_slide)]['Path'].values[0]
            matching_he_slide = matched_df[matched_df['SlideName'].str.contains(ihc_slide)]['Matched_HE_SlideName'].values[0].split('.')[0]
            slide_file = os.path.join("/data/Breast/Carmel", ihc_path)
        else:
            slide_file, matching_he_slide = row['SlideName'], row['Matched_HE_SlideName'].split('.')[0]
            slide_name = os.path.splitext(slide_file)[0]
            
        
        ######################################################
        # if not slide_name.startswith('21-2989_1_1'):
        #     continue
        
        # Set paths
        full_he_tiles_dir = os.path.join(he_tiles_dir, matching_he_slide)
        block_dir = next(filter(lambda x: x.startswith(slide_name), os.listdir(ihc_tiles_dir)), None)
        full_ihc_tiles_dir = os.path.join(ihc_tiles_dir, block_dir)
        if not args.her2_annotations:
            full_tile_y_dir = os.path.join(tile_y_dir, matching_he_slide)
            full_mt_dir = os.path.join(matching_tiles_dir, matching_he_slide)
            full_ti_dir = full_mt_dir.replace('matching_tiles', 'tumor_indices_from_cancer_map')
            full_ant_dir = full_mt_dir.replace('matching_tiles', 'annotated_tiles')
            ti = os.path.join(full_ti_dir, f'{args.tumor_indices_type}_tumor_indices{args.val_fold}.npy') if not args.seg_y_predictions else os.path.join(full_ti_dir, f'{args.tumor_indices_type}_tumor_indices.npy')
            if args.save_tumor_indices_from_seg_map:
                ti = ti.replace('tumor_indices.npy', 'tumor_ann_indices.npy')
            if args.save_ensemble_tumor_indices:
                ti = ti.replace(f'tumor_indices{args.val_fold}.npy', 'tumor_indices_ensemble.npy')
            nti = os.path.join(full_ti_dir, 'non_tumor_indices.npy')
        
        if (args.seg_y_labels or args.seg_y_predictions or args.extract_tile_y_from_y_map):
            segment_dir = os.path.join(save_path, f'y_map{src_suffix}')
            if args.external_he_model:
                segment_dir = os.path.join(save_path.replace('IHC', 'HE'), f'HER2_status_map{src_suffix}')
            slide_dir = next(filter(lambda x: x.startswith(matching_he_slide[:-1]), os.listdir(segment_dir)), None)
            if slide_dir is None:
                print(f"y_map directory for slide {matching_he_slide} not found. Skipping...")
                continue
            segment_dir = os.path.join(segment_dir, slide_dir)
        elif (args.her2_annotations or args.save_her2_annotations):
            segment_dir = os.path.join(save_path, f'slide_segmentations{suffix}', slide_name)
        elif args.extract_tumor_indices_from_cancer_map or args.seg_from_my_cancer_predictions:
            segment_dir = os.path.join(save_path.replace('IHC', 'HE'), f'cancer_map{src_suffix}', matching_he_slide)
        else:
            segment_dir = os.path.join(save_path.replace('IHC', 'HE'), f'slide_segmentations{suffix}', matching_he_slide)


        st = os.path.join(segment_dir, 'segmented_tiles.npy')
        h2a = os.path.join(segment_dir, 'HER2 Annotations.npy')

        if cancer_prob_path is not None and not args.her2_annotations:
            cancer_prob_dir = os.path.join(cancer_prob_path, f"cancer_probs_mpp{args.target_mpp}", matching_he_slide)
            cp = os.path.join(cancer_prob_dir, f'cancer_prob_val{args.val_fold}.npy')

        if args.create_y_sw_bl_csv:
            gt_dir = segment_dir.replace("y_map", "slide_segmentations").replace(f"{matching_he_slide}", f"{slide_name}")
            if os.path.exists(gt_dir) and os.path.exists(segment_dir):
                gt_name = next(filter(lambda x: x.endswith(".npy"), os.listdir(gt_dir)), None)
                gt_file = os.path.join(gt_dir, gt_name)
                map_gt_npy = load_npy_file(gt_file)
                bl_name = next(filter(lambda x: "tumor from map SW baseline" in x and x.endswith(f".npy"), os.listdir(segment_dir)), None)
                bl_file = os.path.join(segment_dir, bl_name)
                map_baseline_npy = load_npy_file(bl_file)
                pred_name = next(filter(lambda x: "tumor from map predictions" in x and x.endswith(f".npy"), os.listdir(segment_dir)), None)
                model_file = os.path.join(segment_dir, pred_name)
                map_model_npy = load_npy_file(model_file)
            else:
                map_gt_npy = None
            

        ##### set valid tiles #####
        he_tiles = os.listdir(full_he_tiles_dir)
        ihc_tiles = os.listdir(full_ihc_tiles_dir)   
   
        if (args.save_her2_annotations or args.extract_tile_y_from_y_map):
            matching_tiles = np.load(os.path.join(full_mt_dir, 'ihc_tiles.npy'))
            valid_tiles = [ihc_tiles[matching_tiles[idx].astype(int)] for idx, _ in enumerate(he_tiles) if not np.isnan(matching_tiles[idx])]
            # valid_indices = [idx for idx, _ in enumerate(he_tiles) if not np.isnan(matching_tiles[idx])]
            if args.y_map_from_he:
                if args.all_he_tiles:
                    valid_tiles = he_tiles
                else:
                    valid_tiles = [tile for idx, tile in enumerate(he_tiles) if not np.isnan(matching_tiles[idx])]
        elif args.her2_annotations:
            valid_tiles = ihc_tiles
            valid_indices = list(range(len(ihc_tiles)))
        elif args.seg_y_labels or args.seg_y_predictions:
            matching_tiles = np.load(os.path.join(full_mt_dir, 'ihc_tiles.npy'))
            non_nan_matching_indices = np.array([idx for idx, _ in enumerate(he_tiles) if not np.isnan(matching_tiles[idx])])
        elif args.cancer_annotations or args.extract_tumor_indices_from_cancer_map or args.seg_from_my_cancer_predictions:
            valid_tiles = he_tiles
            valid_indices = list(range(len(he_tiles)))
        else: 
            matching_tiles = np.load(os.path.join(full_mt_dir, 'ihc_tiles.npy'))
            valid_tiles = [tile for idx, tile in enumerate(he_tiles) if not np.isnan(matching_tiles[idx])]
            valid_indices = [idx for idx, _ in enumerate(he_tiles) if not np.isnan(matching_tiles[idx])]
            

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
        
        # load tumor indices
        if args.save_tumor_indices_from_seg_map or (args.seg_from_my_cancer_predictions and not args.save_ensemble_tumor_indices):
            tumor_indices = save_tumor_indices_and_cancer_probs(args, full_ti_dir, ti, nti, cp, segment_dir, slide_name, valid_tiles, valid_indices, tile_segment)
        
        if args.save_ensemble_tumor_indices:
            save_ensemble_tumor_indices(args, full_ti_dir, ti, cp, segment_dir, slide_name, valid_tiles, valid_indices, tile_segment)

        elif args.seg_y_predictions:
            y_pred_folder=full_tile_y_dir.replace('tile_y', 'tile_y_pred')
            tile_y_folder=full_tile_y_dir.replace('tile_y', 'tile_y_from_y_map') if args.y_type == 'regional' else full_tile_y_dir
            
            if not os.path.exists(y_pred_folder):
                print(f"y_pred_folder {y_pred_folder} does not exist. Skipping...")
                continue
            y_pred_name = 'from_map_pred_ad_' if args.y_pred_type == 'regional' else 'local_tile_y_pred_add_virchow'
            y_pred_npy = next(filter(lambda f: y_pred_name in f and f.endswith('.npy'), os.listdir(y_pred_folder)), None)

            if y_pred_npy is None:
                print(f"No y prediction file found in {y_pred_folder}, probably no tumor indices were found. Skipping...")
                continue
            y_pred_file=os.path.join(y_pred_folder, y_pred_npy)
            fold = y_pred_npy.split('val')[-1].split('.')[0]  # extract fold from filename like tile_y_pred_val1.npy
            ti = ti.replace(f'tumor_indices.npy', f'tumor_indices{fold}.npy')
            tumor_indices = np.load(ti).astype(int) if os.path.exists(ti) else None
            tile_y_npy = next(filter(lambda f: f'val{fold}' in f, os.listdir(tile_y_folder)), None)
            y_file = os.path.join(tile_y_folder, tile_y_npy)
            tile_y = np.load(y_file).flatten() if os.path.exists(y_file) else None
        elif not args.her2_annotations:
            tumor_indices = np.load(ti).astype(int) if os.path.exists(ti) else None
        
        if args.save_her2_annotations:
            if tile_segment is None and os.path.exists(h2a):
                tile_segment = np.load(h2a)
            save_her2_annotations(args, full_ant_dir, slide_name, valid_tiles, tile_segment)

        # seg_y
        if args.seg_y_labels:
            y_npy = f'tile_y_ad_val{args.val_fold}.npy' if args.all_data else f'tile_y.npy'
            y_file=os.path.join(full_tile_y_dir, y_npy)

            if os.path.exists(y_file):
                tile_y = np.load(y_file).flatten()
            else:
                print(f"y_file file {y_file} does not exist. Skipping...")
                return None
            
            if args.seg_only_tumor:
                if tumor_indices is None:
                    continue
                print(f"tumor_indices.shape = {tumor_indices.shape}", "non_nan_matching_indices.shape =", non_nan_matching_indices.shape)
                valid_indices = np.intersect1d(tumor_indices, non_nan_matching_indices)
                tumor_in_non_nan = np.isin(non_nan_matching_indices, tumor_indices)
                tumor_matching_indices = np.where(tumor_in_non_nan)[0]
                valid_y = tile_y[tumor_matching_indices]
            else:
                valid_indices = non_nan_matching_indices
                valid_y = tile_y

            valid_matching_indices = matching_tiles[valid_indices].astype(int)
            valid_tiles = list(np.array(ihc_tiles)[valid_matching_indices])
            local_y_maps(args, tile_y=valid_y, segment_dir=segment_dir, slide_name=slide_name, valid_tiles=valid_tiles, 
                         save_name=f'Tile HER2 contribution (y) (MPP={args.source_mpp} GigaPath features trained model) val_fold = {args.val_fold}')
        elif args.seg_y_predictions:
            common_indices = np.intersect1d(tumor_indices, non_nan_matching_indices)
            tumor_in_non_nan = np.isin(non_nan_matching_indices, tumor_indices)
            tumor_matching_indices = np.where(tumor_in_non_nan)[0]
            tumor_matching_tiles = matching_tiles[common_indices].astype(int)
            tumor_tile_y = tile_y[tumor_matching_indices]
            valid_y = tumor_tile_y != -1
            valid_tile_y = tumor_tile_y[valid_y]
            tumor_matching_tiles = tumor_matching_tiles[valid_y].astype(int)

            y_bl_name = 'from_sw_bl_map' if args.y_pred_type == 'regional' else 'local_tile_y_pred_bl'            
            y_bl_npy = next(filter(lambda f: y_bl_name in f, os.listdir(y_pred_folder)), None)
            y_bl_file=os.path.join(y_pred_folder, y_bl_npy)

            tile_pred_y = load_npy_file(y_pred_file).flatten()
            tile_bl_y = load_npy_file(y_bl_file)
            # keep only paired tumor entries with valid y values
            tile_pred_y = tile_pred_y[common_indices][valid_y]
            tile_bl_y = tile_bl_y[common_indices][valid_y]
            
            valid_tiles = list(np.array(ihc_tiles)[tumor_matching_tiles])

            # if tile_bl_y.shape != tile_pred_y.shape:
            #     tile_bl_y = tile_bl_y[tumor_indices]

            if not args.create_y_sw_bl_csv:
                local_y_maps(args, tile_y=tile_pred_y, segment_dir=segment_dir, slide_name=slide_name, valid_tiles=valid_tiles,
                            save_name=f'{args.y_pred_type} tumor tile HER2 contribution (y) predictions (MPP={args.source_mpp} GigaPath features trained model)')
                local_y_maps(args, tile_y=tile_bl_y, segment_dir=segment_dir, slide_name=slide_name, valid_tiles=valid_tiles,
                            save_name=f'{args.y_pred_type} tumor tile HER2 contribution (y) baseline (MPP={args.source_mpp} GigaPath features trained model)')
            
            else: # create_y_sw_bl_csv
                # for pseudo-labels (y) comparison
                fold_idx = int(fold) - 1
                tile_y_bin = (valid_tile_y >= 2).astype(int)

                bl_df_new = pd.DataFrame({'tile_y': valid_tile_y, 'tile_y_bin': tile_y_bin, 'tile_pred': tile_bl_y})
                local_df_new = pd.DataFrame({'tile_y': valid_tile_y, 'tile_y_bin': tile_y_bin, 'tile_pred': tile_pred_y})
                baseline_dfs[fold_idx] = pd.concat([baseline_dfs[fold_idx], bl_df_new], ignore_index=True)
                local_dfs[fold_idx] = pd.concat([local_dfs[fold_idx], local_df_new], ignore_index=True)

                # for gt comparison
                if map_gt_npy is not None:
                    mask = (map_gt_npy != -1) & (map_baseline_npy != -1) & (map_model_npy != -1)
                    valid_indices = np.where(mask)
                    valid_gt, valid_bl, valid_pred = map_gt_npy[valid_indices], map_baseline_npy[valid_indices], map_model_npy[valid_indices]
                    valid_gt_bin = (valid_gt >= 2).astype(int)
                    
                    map_local_df_new = pd.DataFrame({'tile_gt': valid_gt, 'tile_gt_bin': valid_gt_bin, 'map_tile_pred': valid_pred})
                    map_bl_df_new = pd.DataFrame({'tile_gt': valid_gt, 'tile_gt_bin': valid_gt_bin, 'map_tile_pred': valid_bl})
                    map_baseline_dfs[fold_idx] = pd.concat([map_baseline_dfs[fold_idx], map_bl_df_new], ignore_index=True)
                    map_local_dfs[fold_idx] = pd.concat([map_local_dfs[fold_idx], map_local_df_new], ignore_index=True)


        if args.extract_tile_y_from_y_map:
            y_npy = f'tile_y_ad_val{args.val_fold}.npy' if args.all_data else f'tile_y.npy'
            tile_y_from_y_map_dir = full_tile_y_dir.replace('tile_y', 'tile_y_from_y_map')
            if not os.path.exists(tile_y_from_y_map_dir):
                os.makedirs(tile_y_from_y_map_dir, exist_ok=True)

            y_map_file = os.path.join(segment_dir, f'Tile HER2 contribution (y) from SW (MPP={args.source_mpp} GigaPath features trained model) val_fold = {args.val_fold}.npy')
            if args.y_map_from_he:
                y_npy = f'tile_y_from_sw_bl_map_ad_val{args.val_fold}.npy'
                tile_y_from_y_map_dir = full_tile_y_dir.replace('tile_y', 'tile_y_pred')
                y_map = next(filter(lambda f: f'SW baseline (MPP={args.source_mpp}' in f and f.endswith(f'val_fold = {args.val_fold}.npy'), os.listdir(segment_dir)), None)
                if y_map is None:
                    print(f"No y_map file found in {segment_dir} for fold {args.val_fold}. Skipping...")
                    continue
                y_map_file = os.path.join(segment_dir, y_map)
                # y_map_file = os.path.join(segment_dir, f'Tile HER2 contribution (y) SW baseline (MPP={args.source_mpp} GigaPath features trained model) val_fold = {args.val_fold}.npy')
                if not os.path.exists(tile_y_from_y_map_dir):
                    os.makedirs(tile_y_from_y_map_dir, exist_ok=True)
                if args.all_he_tiles:
                    y_npy = f'tile_y_from_sw_bl_map_all_he_tiles_val{args.val_fold}.npy'
                if args.external_he_model:
                    # y_map_file = y_map_file.replace('SW baseline', 'from THUFA SW')
                    y_map_file = os.path.join(segment_dir, "Tile HER2 status from SW (MPP=0.5 GigaPath features trained on THUFA model).npy")
                    y_npy = y_npy.replace('sw_bl', 'sw_thufa')

            tile_y_map = load_npy_file(y_map_file)
            if tile_y_map is None:
                print(f"y_map file {y_map_file} does not exist. Skipping...")
                continue
            
            extract_tile_y_from_y_map(args, y_map=tile_y_map, tile_y_dir=tile_y_from_y_map_dir, slide_name=slide_name, valid_tiles=valid_tiles, save_name=y_npy)

        if args.extract_tumor_indices_from_cancer_map:
            ti_npy = f'{args.tumor_indices_type}_tumor_indices{args.val_fold}.npy'
            tumor_indices_from_cancer_map_dir = full_tile_y_dir.replace('tile_y', 'tumor_indices_from_cancer_map')
            cancer_probs_from_cancer_map_dir = full_he_tiles_dir.replace('png_tiles', 'cancer_probs_from_cancer_map')
            if not os.path.exists(tumor_indices_from_cancer_map_dir):
                os.makedirs(tumor_indices_from_cancer_map_dir, exist_ok=True)
            if not os.path.exists(cancer_probs_from_cancer_map_dir):
                os.makedirs(cancer_probs_from_cancer_map_dir, exist_ok=True)

            cancer_map_file = os.path.join(segment_dir, f'Tile cancer probability from SW (MPP={args.source_mpp} GigaPath features trained model) val_fold = {args.val_fold}.npy')
            cancer_map = load_npy_file(cancer_map_file)
            if cancer_map is not None:
                extract_tumor_indices_from_cancer_map(args, cancer_map=cancer_map, tumor_indices_dir=tumor_indices_from_cancer_map_dir, 
                                                    cancer_probs_dir=cancer_probs_from_cancer_map_dir, slide_name=slide_name, 
                                                    valid_tiles=valid_tiles, save_name=ti_npy)
            
            
        if args.y_std:
            if not os.path.exists(full_tile_y_dir):
                print(f"Directory {full_tile_y_dir} does not exist. Skipping...")
                continue

            tile_y = np.load(os.path.join(full_tile_y_dir, 'tile_y.npy'))

            results_df = pd.DataFrame(columns=['slide_name', 'tumor_tile_y_std', 'num_tiles', 'num_tumor_tiles', 'label', 'fold'])
            all_tumor_y = np.array([], dtype=np.float32)

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

    if args.create_y_sw_bl_csv:
        if args.baseline_output_dir is not None and os.path.exists(args.baseline_output_dir):
            her2_subdir = os.path.join(args.baseline_output_dir, "her2")
            for i in range(1, 6):
                infer_fold_dir = next(filter(lambda x: 'infer' in x and x.endswith(f'{i}'), os.listdir(her2_subdir)))
                full_infer_dir = os.path.join(her2_subdir, infer_fold_dir, 'eval_pretrained_her2', 'inference_results')
                csv_name = f"tumor_tile_valid_{args.y_type}_y_preds_mpp{args.target_mpp}_val{i}.csv" 
                baseline_dfs[i-1].to_csv(os.path.join(full_infer_dir, csv_name), index=False)
                if map_gt_npy is not None:
                    map_baseline_dfs[i-1].to_csv(os.path.join(full_infer_dir, f"tile_preds_vs_gt_mpp{args.target_mpp}_val{i}.csv"), index=False)
        
        if args.predictions_output_dir is not None and os.path.exists(args.predictions_output_dir):
            her2_subdir = os.path.join(args.predictions_output_dir, "her2")
            for i in range(1, 6):
                infer_fold_dir = next(filter(lambda x: 'infer' in x and x.endswith(f'{i}'), os.listdir(her2_subdir)))
                full_infer_dir = os.path.join(her2_subdir, infer_fold_dir, 'eval_pretrained_her2', 'inference_results')
                csv_name = f"tumor_tile_valid_{args.y_type}_y_preds_mpp{args.target_mpp}_val{i}.csv"
                local_dfs[i-1].to_csv(os.path.join(full_infer_dir, csv_name), index=False)
                if map_gt_npy is not None:
                    map_local_dfs[i-1].to_csv(os.path.join(full_infer_dir, f"tile_preds_vs_gt_mpp{args.target_mpp}_val{i}.csv"), index=False)
    
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