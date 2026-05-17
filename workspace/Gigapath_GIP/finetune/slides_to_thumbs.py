import os
import math
import argparse
import re
import pandas as pd
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from datasets.slide_datatset import SlidingWindowDataset
from params import get_finetune_params
from task_configs.utils import load_task_config
from utils import get_test_loader


OPENSLIDE_PATH = r"C:\Program Files\openslide-bin-4.0.0.3-windows-x64\openslide-bin-4.0.0.3-windows-x64\bin"
# DESIRED_MPP = 0.5
# DESIRED_MPP = 1
# DESIRED_MPP = 2
SLIDE_PATCH_SIZE = 256
SLIDE_MPP = 0.242797397769517
SLIDE_HEIGHT = 204614
SLIDE_WIDTH = 89484
HE_THUMB_HEIGHT = 10230
HE_THUMB_WIDTH = 4473
IHC_THUMB_HEIGHT = 5115
IHC_THUMB_WIDTH = 2237
SEG_THUMB_HEIGHT = 97
SEG_THUMB_WIDTH = 42
IHC_SCALE_PATCH_SIZE = round(SLIDE_PATCH_SIZE * (IHC_THUMB_WIDTH / SLIDE_WIDTH) / 0.25)
SEG_SCALE_PATCH_SIZE = math.ceil(SLIDE_PATCH_SIZE * (SEG_THUMB_WIDTH / SLIDE_WIDTH) / 0.25)

# Function to round mpp values to the nearest 1/(2n) or 1
def round_mpp(mpp_value):
    try:
        if abs(mpp_value - 1) < 0.1:
            return 1
        else:
            n = int(round(1 / (2 * mpp_value)))
            return 1 / (2 * n)
    except:
        return None
    
ROUNDED_SLIDE_MPP = round_mpp(SLIDE_MPP)

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def slide_to_thumb_coord(slide_coord, slide_dimensions, thumb_size):
    """
    Converts a coordinate from the slide to the corresponding coordinate in the thumbnail.

    Parameters:
    - slide: The OpenSlide object.
    - thumb: The thumbnail image (PIL Image or similar).
    - slide_coord: A tuple (x, y) representing the coordinate on the slide.

    Returns:
    - A tuple (x, y) representing the corresponding coordinate on the thumbnail.
    """
    # Get dimensions of the thumbnail
    thumb_width, thumb_height = thumb_size

    # Get dimensions of the slide
    slide_width, slide_height = slide_dimensions

    # Calculate scale factors
    scale_x = thumb_width / slide_width
    scale_y = thumb_height / slide_height

    # Convert slide coordinates to thumbnail coordinates
    thumb_x = int(slide_coord[0] * scale_x)
    thumb_y = int(slide_coord[1] * scale_y)

    return thumb_x, thumb_y


def thumb_to_slide_coord(thumb_coord, slide_dimensions, thumb_size):
    """
    Converts a coordinate from the thumbnail to the corresponding coordinate in the slide.

    Parameters:
    - thumb: The thumbnail image (PIL Image or similar).
    - slide: The OpenSlide object.
    - thumb_coord: A tuple (x, y) representing the coordinate on the thumb.

    Returns:
    - A tuple (x, y) representing the corresponding coordinate on the slide.
    """
    # Get dimensions of the thumbnail
    thumb_width, thumb_height = thumb_size

    # Get dimensions of the slide
    slide_width, slide_height = slide_dimensions

    # Calculate scale factors
    scale_x = slide_width / thumb_width
    scale_y = slide_height / thumb_height

    # Convert slide coordinates to thumbnail coordinates
    slide_x = int(thumb_coord[0] * scale_x)
    slide_y = int(thumb_coord[1] * scale_y)

    return slide_x, slide_y


def rotate_point(x, y, cx, cy, angle_rad):
    # Translate point to the origin (relative to the center)
    translated_x = x - cx
    translated_y = -(y - cy)

    # Apply the rotation matrix
    rotated_x = translated_x * math.cos(angle_rad) - translated_y * math.sin(angle_rad)
    rotated_y = translated_x * math.sin(angle_rad) + translated_y * math.cos(angle_rad)
    rotated_y = -rotated_y

    # Translate the point back
    # new_x = int(rotated_x + cx)
    # new_y = int(rotated_y + cy)

    # return new_x, new_y
    return rotated_x, rotated_y


def rotate_coordinates(corners, angle, image_center=None):
    """
    Rotates the coordinates of the four corners of an image by a given angle.

    Parameters:
    - x1, y1: Coordinates of the top-left corner.
    - x2, y2: Coordinates of the top-right corner.
    - x3, y3: Coordinates of the bottom-right corner.
    - x4, y4: Coordinates of the bottom-left corner.
    - angle: Rotation angle in degrees (counterclockwise).
    - image_center: (cx, cy) The center of the image for rotation. If None, it is calculated as the center of the given coordinates.

    Returns:
    - A list of tuples representing the new coordinates of the corners after rotation.
    """
    # Convert the angle to radians
    angle_rad = math.radians(angle)

    # Calculate the center of the image if not provided
    if image_center is None:
        cx = (corners[0] + corners[2]) / 2
        cy = (corners[1] + corners[5]) / 2
    else:
        cx, cy = image_center

    # Rotate all four corners
    new_x1, new_y1 = rotate_point(corners[0], corners[1], cx, cy, angle_rad)
    new_x2, new_y2 = rotate_point(corners[2], corners[3], cx, cy, angle_rad)
    new_x3, new_y3 = rotate_point(corners[4], corners[5], cx, cy, angle_rad)
    new_x4, new_y4 = rotate_point(corners[6], corners[7], cx, cy, angle_rad)

    return [(new_x1, new_y1), (new_x2, new_y2), (new_x3, new_y3), (new_x4, new_y4)]


def get_padded_slide_coords(slide_coords: list, slide_width: int, slide_height: int):
    # Calculate the original patch width and height
    patch_width = slide_coords[2] - slide_coords[0]
    patch_height = slide_coords[3] - slide_coords[1]

    # Calculate the padding using 2 scaling factor
    scaling_factor = 2
    padding_width = int((scaling_factor - 1) * patch_width / 2)
    padding_height = int((scaling_factor - 1) * patch_height / 2)

    # Add padding to the coordinates
    padded_slide_coords = (
        max(0, slide_coords[0] - padding_width),
        max(0, slide_coords[1] - padding_height),
        min(slide_width, slide_coords[2] + padding_width),
        min(slide_height, slide_coords[3] + padding_height)
    )

    return padded_slide_coords, patch_width, patch_height


def extract_padded_slide_patch(padded_slide_coords: list, slide, patch_width: int, patch_height: int,
                               rotation_angle: float):
    # Extract padded slide patch
    padded_slide_width = padded_slide_coords[2] - padded_slide_coords[0]
    padded_slide_height = padded_slide_coords[3] - padded_slide_coords[1]
    slide_patch = slide.read_region((padded_slide_coords[0], padded_slide_coords[1]), 0,
                                    (padded_slide_width, padded_slide_height))
    # slide_patch.show(title="Slide Patch")
    angle_rad = math.radians(rotation_angle)
    actual_pad_width = (slide_patch.width - patch_width) // 2
    actual_pad_height = (slide_patch.height - patch_height) // 2
    corners = [actual_pad_width, actual_pad_height,
               slide_patch.width - actual_pad_width, actual_pad_height,
               actual_pad_width, slide_patch.height - actual_pad_height,
               slide_patch.width - actual_pad_width, slide_patch.height - actual_pad_height]
    slide_patch = slide_patch.rotate(rotation_angle, expand=True)
    slide_patch.show(title="Slide Patch")

    return slide_patch, corners


def crop_padded_patch(corners: list, rotation_angle: float, slide_patch):
    # Crop the padded patch
    rotated_corners = rotate_coordinates(corners=corners, angle=rotation_angle)
    new_center = slide_patch.width // 2, slide_patch.height // 2
    xs = [rc[0] + new_center[0] for rc in rotated_corners]
    ys = [rc[1] + new_center[1] for rc in rotated_corners]
    small_x, big_x = min(xs), max(xs)
    small_y, big_y = min(ys), max(ys)
    crop_box_slide = (small_x, small_y, big_x, big_y)
    slide_patch = slide_patch.crop(crop_box_slide)

    return slide_patch


def get_mpp_origin(slide):
    x_origin = int(slide.properties.get('openslide.bounds-x'))
    y_origin = int(slide.properties.get('openslide.bounds-y'))
    mpp_x = float(slide.properties.get('openslide.mpp-x'))
    mpp_y = float(slide.properties.get('openslide.mpp-y'))

    return mpp_x, mpp_y, x_origin, y_origin


def extract_and_show_patch(slide, thumb, rotation_mat=None, rotation_angle=None, slide_coords=None, thumb_coords=None,
                           prefix=''):
    """
    Extracts patches from the slide and the thumbnail based on the given slide coordinates and shows them.

    Parameters:
    - slide: The OpenSlide object.
    - thumb: The thumbnail image (PIL Image or similar).
    - slide_coords: A tuple (x1, y1, x2, y2) representing the top-left and bottom-right coordinates on the slide.
    - rotation: A float representing the rotation angle.

    Returns:
    - None, but displays the extracted patches.
    """
    if slide_coords:
        # padded_slide_coords, patch_width, patch_height = get_padded_slide_coords(slide_coords=slide_coords,
        #                                                                          slide_width=slide.dimensions[0],
        #                                                                          slide_height=slide.dimensions[1])

        # Map the slide coordinates to thumbnail coordinates
        thumb_coord1 = slide_to_thumb_coord((slide_coords[0], slide_coords[1]),
                                            slide_dimensions=slide.dimensions, thumb_size=thumb.size)
        thumb_coord2 = slide_to_thumb_coord((slide_coords[2], slide_coords[3]),
                                            slide_dimensions=slide.dimensions, thumb_size=thumb.size)
        thumb_coords = list(thumb_coord1) + list(thumb_coord2)
        
        slide_patch = slide.read_region((slide_coords[0], slide_coords[1]), 0,
                                        (slide_coords[2] - slide_coords[0], slide_coords[3] - slide_coords[1]))
    else:
        # Map the thumbnail coordinates to slide coordinates
        slide_coord1 = thumb_to_slide_coord(slide_dimensions=slide.dimensions, thumb_size=thumb.size,
                                            thumb_coord=(thumb_coords[0], thumb_coords[1]))
        slide_coord2 = thumb_to_slide_coord(slide_dimensions=slide.dimensions, thumb_size=thumb.size,
                                            thumb_coord=(thumb_coords[2], thumb_coords[3]))
        slide_coords = list(slide_coord1) + list(slide_coord2)

        padded_slide_coords, patch_width, patch_height = get_padded_slide_coords(slide_coords=slide_coords,
                                                                                 slide_width=slide.dimensions[0],
                                                                                 slide_height=slide.dimensions[1])
        slide_patch, corners = extract_padded_slide_patch(padded_slide_coords=padded_slide_coords, slide=slide,
                                                          patch_width=patch_width, patch_height=patch_height,
                                                          rotation_angle=rotation_angle)
        slide_patch = crop_padded_patch(corners=corners, rotation_angle=rotation_angle, slide_patch=slide_patch)

    # Extract thumb patches
    thumb_patch = thumb.crop((thumb_coords[0], thumb_coords[1], thumb_coords[2], thumb_coords[3]))

    # Convert slide patch to RGB (it might be RGBA, depending on the format)
    slide_patch_rgb = slide_patch.convert("RGB").resize((SLIDE_PATCH_SIZE, SLIDE_PATCH_SIZE))
    # slide_patch_rgb.show(title="Slide Patch")
    # thumb_patch = thumb_patch.rotate(rotation_angle, expand=True)

    # if prefix == 'ihc_':
    # Display both patches
    thumb_patch.show(title="Thumbnail Patch")
    slide_patch_rgb.show(title="Slide Patch")
    slide_patch_rgb.save(fp=os.path.join('slides_to_thumbs_output', f'{prefix}slide_patch.png'), format="PNG")
    thumb_patch.save(fp=os.path.join('slides_to_thumbs_output', f'{prefix}thumb_patch.png'), format="PNG")

    return thumb_coords


def extract_matching_slide_thumb_patches(he_slide_path: str, he_thumb_path, ihc_slide_path: str, ihc_thumb_path: str,
                                         rotation_mat_path: str, qupath_he_coords: tuple, args=None):
    h_e_slide = openslide.OpenSlide(he_slide_path)
    thumb = Image.open(he_thumb_path)

    mpp_x, mpp_y, x_origin, y_origin = get_mpp_origin(h_e_slide)

    # qupath_location = (16200, 42400)  # 21-3263 slides
    # openslide_location = (round(qupath_location[0] / mpp_x) + x_origin, round(qupath_location[1] / mpp_y) + y_origin)
    openslide_location = (round(qupath_he_coords[0] / mpp_x) + x_origin, round(qupath_he_coords[1] / mpp_y) + y_origin)
    mpp_scale_factor = args.target_mpp / mpp_x
    scaled_patch_size = int(SLIDE_PATCH_SIZE * mpp_scale_factor)

    slide_coords = tuple(list(openslide_location) + [cor + scaled_patch_size for cor in openslide_location])
    rotation_img = cv2.imread(rotation_mat_path, cv2.IMREAD_UNCHANGED)
    rotation_img = cv2.cvtColor(rotation_img, cv2.COLOR_BGR2RGB)
    rotation_mat = np.array(rotation_img).transpose(1, 0, 2)

    thumb_coords = extract_and_show_patch(h_e_slide, thumb, rotation_mat=rotation_mat, slide_coords=slide_coords)
    # minus 1 since the matrix holds rotation angles from HE to IHC, but we rotate IHC to HE
    rotation_angle = -1 * (rotation_mat[int(thumb_coords[0]), int(thumb_coords[1])][-1] / 100)
    thumb_center = [(thumb_coords[0] + thumb_coords[2]) // 2, (thumb_coords[1] + thumb_coords[3]) // 2]
    thumb_width, thumb_height = thumb_coords[2] - thumb_coords[0], thumb_coords[3] - thumb_coords[1]
    ihc_center = list(rotation_mat[thumb_center[0], thumb_center[1]][:2])

    # IHC
    ihc_slide = openslide.OpenSlide(ihc_slide_path)
    ihc_thumb = Image.open(ihc_thumb_path)
    x_ratio = ihc_thumb.width / thumb.width
    y_ratio = ihc_thumb.height / thumb.height
    corresp_ihc_thumb_coords = [max(0, ihc_center[0] - (thumb_width // 2) * x_ratio),
                                max(0, ihc_center[1] - (thumb_height // 2) * y_ratio),
                                min(ihc_thumb.width, ihc_center[0] + (thumb_width // 2) * x_ratio),
                                min(ihc_thumb.height,
                                    ihc_center[1] + (thumb_height // 2) * y_ratio)]

    extract_and_show_patch(ihc_slide, ihc_thumb, rotation_angle=rotation_angle, thumb_coords=corresp_ihc_thumb_coords,
                           prefix='ihc_')


def parse_window_name(window_name):
    # Example format: "21-1518_1_5_d_32256_33024x_10240_12288y"
    match = re.match(r"(.*)_(\d+)_(\d+)x_(\d+)_(\d+)y", window_name)
    if match:
        slide_name = match.group(1)
        x_start, x_end, y_start, y_end = map(int, match.groups()[1:])
        x_end, y_end = x_end + SLIDE_PATCH_SIZE, y_end + SLIDE_PATCH_SIZE
        return slide_name, x_start, x_end, y_start, y_end
    else:
        raise ValueError(f"Invalid window_name format: {window_name}")


def parse_tile_name(tile_name):
    # Example format: "33024x_12288y.png"
    match = re.match(r"(\d+)x_(\d+)y.png", tile_name)
    if match:
        x_start, y_start = map(int, match.groups())
        x_end, y_end = x_start + SLIDE_PATCH_SIZE, y_start + SLIDE_PATCH_SIZE
        return x_start, x_end, y_start, y_end
    else:
        raise ValueError(f"Invalid tile_name format: {tile_name}")

def correct_coords(window_center: list, inverse: bool = False, target_mpp: float = None): 
    mpp_correction_factor = SLIDE_MPP / ROUNDED_SLIDE_MPP
    if not inverse:
        window_center = tuple(map(lambda x: x * (mpp_correction_factor * target_mpp) / SLIDE_MPP, window_center))
    else:  # inverse correction
        window_center = tuple(map(lambda x: int(x * SLIDE_MPP / (mpp_correction_factor * target_mpp)), window_center))
    return window_center

def find_matching_score(score_mat: np.array, map_mat: np.array, tile_name: str = None, window_name: str = None, find_tile: bool = False, 
                        tiles: list = None, tile_df_row: dict = None, ihc_tile: bool = None, target_mpp=None):
    if tile_name is not None:
        x_start, x_end, y_start, y_end = parse_tile_name(tile_name=tile_name)
    elif window_name is not None:
        _, x_start, x_end, y_start, y_end = parse_window_name(window_name=window_name)
    else:
        print("No tile/window name was given, can not find matching score")
        return

    he_window_center = ((x_end + x_start) // 2, (y_end + y_start) // 2)
    he_window_center = correct_coords(he_window_center, target_mpp=target_mpp)
    
    if not ihc_tile:
        he_thumb_window_center = slide_to_thumb_coord(slide_coord=he_window_center,
                                                    slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                    thumb_size=(HE_THUMB_WIDTH, HE_THUMB_HEIGHT))
    else: # IHC tile
        he_thumb_window_center = slide_to_thumb_coord(slide_coord=he_window_center,
                                                    slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                    thumb_size=(IHC_THUMB_WIDTH, IHC_THUMB_HEIGHT))
    
    he_thumb_window_center = min(he_thumb_window_center[0], map_mat.shape[0] - 1), min(he_thumb_window_center[1], map_mat.shape[1] - 1)
    if window_name is not None:
        matching_coords = map_mat[he_thumb_window_center[0] - 150: he_thumb_window_center[0] + 150,
                                he_thumb_window_center[1] - 150: he_thumb_window_center[1] + 150][:, :, :2]
        mask = ~(np.all(matching_coords == [0, 0], axis=-1))
        ihc_thumb_window_center = matching_coords[mask]
        if ihc_thumb_window_center.size == 0:
            return None
        ihc_thumb_window_center = np.mean(ihc_thumb_window_center, axis=0).astype(int)
        matching_scores = score_mat[ihc_thumb_window_center[1] - 50: ihc_thumb_window_center[1] + 50,
                      ihc_thumb_window_center[0] - 50: ihc_thumb_window_center[0] + 50].flatten()
        score = matching_scores[matching_scores != 0]
        score = np.mean(score)
    else:
        if not ihc_tile:
            ihc_thumb_window_center = map_mat[he_thumb_window_center[0], he_thumb_window_center[1]][:2]
            if np.all(ihc_thumb_window_center == [0, 0]):
                return None
            if not find_tile:    
                score = score_mat[ihc_thumb_window_center[1], ihc_thumb_window_center[0]]
        else: # IHC tile
            ihc_thumb_window_center = min(he_thumb_window_center[0], score_mat.shape[0] - 1), min(he_thumb_window_center[1], score_mat.shape[1] - 1)
            if not find_tile:
                score = score_mat[ihc_thumb_window_center[0], ihc_thumb_window_center[1]]
        if not find_tile and score == 0:
            return None
        if find_tile:
            ihc_window_center = thumb_to_slide_coord(thumb_coord=ihc_thumb_window_center,
                                                  slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                  thumb_size=(IHC_THUMB_WIDTH, IHC_THUMB_HEIGHT))
            ihc_window_center = correct_coords(ihc_window_center, inverse=True, target_mpp=target_mpp)
            ret_val = find_tile_index(ref_x=ihc_window_center[1], ref_y=ihc_window_center[0], tiles=tiles) # check
            if ret_val is None:
                return None
            else:
                tile_index, ihc_tile = ret_val
            if tile_df_row:
                tile_df_row['IHC_tile'], tile_df_row['HE_tile_thumb_coords'], tile_df_row['IHC_tile_thumb_coords'] = ihc_tile, he_thumb_window_center, np.flip(ihc_thumb_window_center)
                tile_df_row['tile_label'] = score
            return tile_index        

    return score


def find_tile_index(ref_x, ref_y, tiles):
    for idx, tile_name in enumerate(tiles):
        x_start, x_end, y_start, y_end = parse_tile_name(tile_name)
            
        # Check if reference coordinates fall within this tile
        if x_start <= ref_x <= x_end and y_start <= ref_y <= y_end:
            print(f"Reference coordinates ({ref_x}, {ref_y}) belong to tile {tile_name}")
            return idx, tile_name  # Return the index in the list
    
    # print(f"No matching tile found for coordinates ({ref_x}, {ref_y}).")
    return None  # No matching tile found


def show_score_matrix(score_matrix, slide_name, save_dir=None, vmin=-1, vmax=4):
    plt.figure(figsize=(10, 8))
    plt.title(f"Score Matrix for Slide: {slide_name}")
    plt.imshow(score_matrix, cmap='jet', interpolation='none', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Score')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    if save_dir:
        full_save_dir = save_dir.replace(".npy", ".png")
        plt.savefig(full_save_dir, dpi=300)
    plt.show()


def gaussian_2d_kernel(shape, sigma=1, window_amp: bool = False, patch_num: int = None) -> torch.tensor:
    """
    Generate a 2D Gaussian kernel for a non-square shape.

    Parameters:
        shape (tuple): The shape of the kernel (height, width).
        sigma (float): Standard deviation of the Gaussian along the x-axis.

    Returns:
        torch.tensor: A 2D Gaussian kernel.
    """
    height, width = shape
    gaussian_x = gaussian(width, std=sigma)  # 1D Gaussian along the x-axis
    gaussian_y = gaussian(height, std=sigma)  # 1D Gaussian along the y-axis

    kernel = np.outer(gaussian_y, gaussian_x)  # Create 2D Gaussian from 1D arrays
    kernel /= kernel.sum()  # Normalize to ensure the kernel sums to 1

    if window_amp and patch_num is not None:
        ker_center = kernel[height // 2, width // 2]
        desired_center_val = 1 / patch_num
        center_factor = desired_center_val / ker_center
        kernel *= center_factor
    return torch.tensor(kernel)


def create_weighted_score_matrices(slide_scores_csv: str):
    mpp_correction_factor = SLIDE_MPP / ROUNDED_SLIDE_MPP

    scores_df = pd.read_csv(slide_scores_csv)

    # Parse window_name and create columns for easy processing
    scores_df[['slide_name', 'x_start', 'x_end', 'y_start', 'y_end']] = scores_df['window_name'].apply(
        lambda x: pd.Series(parse_window_name(x))
    )

    # Group by slide
    for slide_name, group in scores_df.groupby('slide_name'):
        # Initialize score and weight matrices
        weighted_score_matrix = np.zeros((IHC_THUMB_HEIGHT, IHC_THUMB_WIDTH), dtype=np.float64)
        weight_matrix = np.zeros((IHC_THUMB_HEIGHT, IHC_THUMB_WIDTH), dtype=np.float64)

        # Fill the score matrix with scores from each window
        for _, row in group.iterrows():
            score = row['score']
            x_start, x_end, y_start, y_end = row['x_start'], row['x_end'], row['y_start'], row['y_end']
            x_start, x_end, y_start, y_end = tuple(map(lambda x: x * mpp_correction_factor / SLIDE_MPP, (x_start, x_end, y_start, y_end)))
            x_start, y_start = slide_to_thumb_coord(slide_coord=(x_start, y_start),
                                                    slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                    thumb_size=(IHC_THUMB_WIDTH, IHC_THUMB_HEIGHT))
            x_end, y_end = slide_to_thumb_coord(slide_coord=(x_end, y_end),
                                                slide_dimensions=(SLIDE_WIDTH, SLIDE_HEIGHT),
                                                thumb_size=(IHC_THUMB_WIDTH, IHC_THUMB_HEIGHT))

            # Create a Gaussian window for the current region
            window_size = (x_end - x_start, y_end - y_start)
            gaussian_window = gaussian_2d_kernel(shape=window_size)

            # Apply the Gaussian window to the score
            weighted_region = gaussian_window * score

            try:
                # Add to the cumulative matrices
                weighted_score_matrix[x_start:x_end, y_start:y_end] += weighted_region
                weight_matrix[x_start:x_end, y_start:y_end] += gaussian_window
            except BaseException as e:
                print(e)

        # Normalize the weighted score matrix by the weight matrix
        final_score_matrix = np.divide(
            weighted_score_matrix,
            weight_matrix,
            out=np.zeros_like(weighted_score_matrix),
            where=weight_matrix > 0  # Avoid division by zero
        )

        show_score_matrix(score_matrix=final_score_matrix, slide_name=slide_name)

        # Save the score matrix as a .npz file (NumPy compressed format)
        compressed_score_matrix = final_score_matrix.astype(np.float16)
        npz_output_path = os.path.join(output_dir, f"{slide_name}_score_matrix.npz")
        np.savez_compressed(npz_output_path, compressed_score_matrix)
        # loaded_matrix = np.load(npz_output_path)['arr_0']
        # show_score_matrix(score_matrix=loaded_matrix, slide_name=slide_name)

        print(f"Saved score matrix for slide: {slide_name} at {npz_output_path}")


def gaussian_patches_matrix(weight_matrix: torch.tensor, gaus_window: torch.tensor, img_coords: torch.tensor):
    # Get the dimensions of the weight matrix
    w_height, w_width = weight_matrix.size()

    # Get the dimensions of the gaus window
    k_height, k_width = gaus_window.size()

    # Flatten img_coords for easier handling
    start_x = img_coords[:, 0].long().flatten()  # x_start
    start_y = img_coords[:, 1].long().flatten()  # y_start

    # pad_pixels = int(IHC_SCALE_PATCH_SIZE * 0.25)
    pad_pixels = int(SEG_SCALE_PATCH_SIZE * 0.25)
    # Create a grid for the small_matrix offsets
    x_range = torch.arange(-pad_pixels, k_height - pad_pixels).repeat(k_height).view(k_height,
                                                           k_height).t().flatten().to(device=start_x.device)  # Row indices for the small matrix
    y_range = torch.arange(-pad_pixels, k_height - pad_pixels).repeat(k_width).view(-1, 1).to(device=start_y.device)  # Column indices for the small matrix

    # Compute all region coordinates by adding offsets
    x_offsets = start_x.view(-1, 1, 1) + x_range  # Broadcast offsets for x
    y_offsets = start_y.view(-1, 1, 1) + y_range  # Broadcast offsets for y

    # Flatten offsets for direct indexing
    x_offsets = x_offsets.flatten()  # Shape: (n_regions * k_height * k_width)
    y_offsets = y_offsets.flatten()  # Shape: (n_regions * k_height * k_width)

    # Tile the small_matrix to match all regions
    gaus_window_tiled = gaus_window.repeat(len(start_x), 1,
                                             1).flatten().to(device=start_x.device)  # Shape: (n_regions * k_height * k_width)

    # Filter out-of-bound indices
    valid_mask = (x_offsets >= 0) & (x_offsets < w_height) & (y_offsets >= 0) & (y_offsets < w_width)

    # Apply the mask
    x_offsets = x_offsets[valid_mask].to(device=start_x.device)
    y_offsets = y_offsets[valid_mask].to(device=start_x.device)
    gaus_window_tiled = gaus_window_tiled[valid_mask].to(device=start_x.device)

    # Add the small_matrix values to the weight_matrix
    weight_matrix.index_put_((x_offsets, y_offsets), gaus_window_tiled, accumulate=True)
    return weight_matrix


def patch_weighted_score_matrices(weighted_score_matrix: torch.tensor, weight_matrix: torch.tensor, img_coords: torch.tensor, window_score: float, 
                                  target_mpp: float, seg_thumb_width: int):
    # img_coords = img_coords * mpp_correction_factor / SLIDE_MPP
    img_coords = correct_coords(img_coords, target_mpp=target_mpp)[0]

    # Calculate scale factors
    # scale = IHC_THUMB_WIDTH / SLIDE_WIDTH
    scale = seg_thumb_width / SLIDE_WIDTH

    # Convert slide coordinates to thumbnail coordinates
    img_thumb_coords = (img_coords * scale).to(torch.int32)

    # pad_pixels = int(IHC_SCALE_PATCH_SIZE * 0.25)
    # gaus_win_side = IHC_SCALE_PATCH_SIZE + 2 * pad_pixels
    seg_scale_patch_size = math.ceil(SLIDE_PATCH_SIZE * (seg_thumb_width / SLIDE_WIDTH) / ROUNDED_SLIDE_MPP)
    pad_pixels = int(seg_scale_patch_size * ROUNDED_SLIDE_MPP)
    gaus_win_side = seg_scale_patch_size + 2 * pad_pixels
    sq_gaus_window_size = (gaus_win_side, gaus_win_side)
    gaussian_window = gaussian_2d_kernel(shape=sq_gaus_window_size, window_amp=True, patch_num=img_coords.size(1))

    curr_weight_matrix = torch.zeros_like(weight_matrix).to(device=weight_matrix.device)
    curr_weight_matrix = gaussian_patches_matrix(weight_matrix=curr_weight_matrix, gaus_window=gaussian_window, img_coords=img_thumb_coords)
    weighted_score_matrix += curr_weight_matrix * window_score
    weight_matrix += curr_weight_matrix


def create_tile_scores_matching_tiles(window: bool = False, matching_tile: bool = False, create_samples_csv: bool = False, ihc_tile: bool = False, args=None):
    if window:
        window_args = get_finetune_params()
        print(window_args)
        window_args.task_config = load_task_config(window_args.task_cfg_path)
    
    slides_excel_path = os.path.join("workspace", "WSI", "metadata_csvs", "Her2_slides_matched_HE_folds.csv")
    tile_df_save_path = os.path.join("workspace", "WSI", "metadata_csvs", "Tile_samples.csv")
    df = pd.read_csv(slides_excel_path)
    if args.only_first_annotated:
        df = df[(df["Path"].str.contains("Batch_1", case=False, na=False)) | (df["Path"].str.contains("Batch_2", case=False, na=False))]
    if args.folds:
        df = df[df["fold"] == int(args.folds)]
    df = df[["SlideName", "Matched_HE_SlideName", "fold", "label", "patient barcode", "label"]]
    df["SlideName"] = df["SlideName"].apply(lambda x: x.split('.')[0])
    df["Matched_HE_SlideName"] = df["Matched_HE_SlideName"].apply(lambda x: x.split('.')[0])
    df["block"] = df["Matched_HE_SlideName"].str.extract(r'(^[\d-]+_\d+_\d+)')
    df["patient_id"] = df["patient barcode"]

    map_matrix_dir = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "png_thumb_pairs_karin")
    score_matrix_dir = os.path.join(os.sep, "home", "amitf", "outputs", "SW_IHC_to_Her2_score", "her2")
    # tiles_path = os.path.join(os.sep, "SSDStorage", "Breast", "gigapath_CAT_features", "png_tiles_mpp1")
    tiles_path = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_HE", f"png_tiles{args.suffix}")
    # labels_path = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", "tile_labels_mpp1")
    labels_path = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", f"tile_labels{args.suffix}")
    # ihc_tiles_path = labels_path.replace("tile_labels", "png_tiles")
    ihc_tiles_path = tiles_path.replace("HE", "IHC")
    if ihc_tile:
        labels_path = labels_path.replace("tile", "ihc_tile")
    matching_tiles_path = labels_path.replace("tile_labels", "matching_tiles")
    if window:
        labels_path = labels_path.replace("tile", "window")
    labels_hist_path = os.path.join(labels_path, "local_labels_hist.npy")
    if create_samples_csv:
        df_columns = ['HE_slide', 'IHC_slide', 'HE_tile', 'IHC_tile', 'HE_tile_thumb_coords', 'IHC_tile_thumb_coords', 'Slide_label', 'tile_label',
                      'fold', 'patient_id', 'HE_features_path', 'HE_features_tile_index', 'HE_tile_path', 'IHC_features_path', 'IHC_features_tile_index', 'IHC_tile_path']
        tile_df = pd.DataFrame(columns=df_columns)
    all_local_labels = None

    if not os.path.exists(labels_path):
        os.makedirs(labels_path, exist_ok=True)
    
    if matching_tile and not os.path.exists(matching_tiles_path):
        os.makedirs(matching_tiles_path, exist_ok=True)

    for row in df.itertuples(index=True, name="Row"):
        ihc_slide, dir, fold, block, patient_id, slide_label = row.SlideName, row.Matched_HE_SlideName, row.fold, row.block, row.patient_id, row.label
        if not dir.startswith('19-14590'): 
            continue
        if ihc_tile:
            dir = ihc_slide
        dir_labels_path = os.path.join(labels_path, dir)
        dir_mt_path = os.path.join(matching_tiles_path, dir)
        save_path = os.path.join(dir_labels_path, "tile_scores.npy") if not matching_tile else os.path.join(dir_mt_path, "ihc_tiles.npy")
        if window:
            save_path = save_path.replace("tile", "window")

        if os.path.exists(save_path):
            if not matching_tile:
                local_labels = np.load(save_path).flatten()
                valid_local_labels = local_labels[~np.isnan(local_labels.astype(float))]
                if valid_local_labels.size < 10000:
                    print(f"ihc_slide = {ihc_slide}, valid_local_labels.size = {valid_local_labels.size}") # Delete
                if all_local_labels is None:
                    all_local_labels = valid_local_labels
                else:
                    all_local_labels = np.concatenate([all_local_labels, valid_local_labels])
            else:
                matching_tiles = np.load(save_path).flatten()
                valid_matching_tiles = matching_tiles[~np.isnan(matching_tiles.astype(float))]
                if valid_matching_tiles.size < 100:
                    print(f"ihc_slide = {ihc_slide}, valid_matching_tiles.size = {valid_matching_tiles.size}") # Delete
            # continue

        if not os.path.exists(dir_labels_path):
            os.makedirs(dir_labels_path, exist_ok=True)

        if matching_tile and not os.path.exists(dir_mt_path):
            os.makedirs(dir_mt_path, exist_ok=True)
        
        score_mat_fold_dir = os.path.join(score_matrix_dir, f"SW_stride1_sigma1_0.3win_IHC_to_Her2_score_infer{fold}", "eval_pretrained_her2", "inference_results")
        score_mat_full_path = os.path.join(score_mat_fold_dir, f"{ihc_slide}_score_matrix.npz")
        if os.path.exists(score_mat_full_path) and not matching_tile:  # for matching tiles, we can skip if score matrix not found
            score_mat = np.load(score_mat_full_path)['arr_0']  # saved as .npz file
            show_score_matrix(score_matrix=score_mat, slide_name=ihc_slide, save_dir=dir_labels_path)
            print(f"score matrix was not found at {score_mat_full_path}")
            continue

        block_map_matrix_dir = os.path.join(map_matrix_dir, block)
        try:
            map_matrix_file = next(filter(lambda x: x.startswith('map') and ihc_slide in x, os.listdir(block_map_matrix_dir)))
        except BaseException as e:
            print(f"map matrix was not found at {block_map_matrix_dir}\n{e}")
            continue

        map_matrix_path = os.path.join(block_map_matrix_dir, map_matrix_file)
        map_img = cv2.imread(map_matrix_path, cv2.IMREAD_UNCHANGED)
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
        map_mat = np.array(map_img)  # .transpose(1, 0, 2)

        full_tiles_dir = os.path.join(tiles_path, dir) if not ihc_tile else os.path.join(ihc_tiles_path, dir)

        if not matching_tile:
            scores = []
            if not window:
                for tile in os.listdir(full_tiles_dir):
                    scores.append(find_matching_score(tile_name=tile, score_mat=score_mat, map_mat=map_mat, ihc_tile=ihc_tile, 
                                                      target_mpp=args.target_mpp))
            else:
                sliding_window_ds = SlidingWindowDataset(data_df=df, root_path=window_args.root_path, task_config=window_args.task_config, slide_key='Matched_HE_SlideName', label=window_args.label, \
                                    dataset_name=window_args.test_dataset, folds=window_args.test_fold,
                                    use_clinical_features=window_args.clinical_features, \
                                    test_on_all=window_args.test_on_all, get_single_slide=dir,
                                    window_size=window_args.window_size, stride=window_args.window_size)
                window_loader = get_test_loader(sliding_window_ds, for_heatmap=True, **vars(window_args))

                for batch in window_loader:
                    # load the batch and transform this batch
                    img_coords = batch['coords']
                    # Separate x and y coordinates
                    x_coords = img_coords[:, :, 0]  # Extract all x-coordinates
                    y_coords = img_coords[:, :, 1]  # Extract all y-coordinates
                    x_min, y_min = int(x_coords.min().item()), int(y_coords.min().item())
                    x_max, y_max = int(x_coords.max().item()), int(y_coords.max().item())
                    window_name = f'{dir}_{x_min}_{x_max}x_{y_min}_{y_max}y'
                    scores.append(find_matching_score(window_name=window_name, score_mat=score_mat, map_mat=map_mat, target_mpp=args.target_mpp))
            scores = np.array(scores, dtype=np.float16)
            print(f'ihc_slide = {ihc_slide}, scores = {scores}, num_not_nan = {np.sum(~np.isnan(scores.astype(float)))}, slide_label = {slide_label}') # Delete
            
            np.save(save_path, scores)
        else: # matching_tile
            # if not os.path.exists(save_path):
                slide_dir = next(filter(lambda x: x.startswith(ihc_slide), os.listdir(ihc_tiles_path)))
                ihc_tiles_dir = os.path.join(ihc_tiles_path, slide_dir)
                tiles = os.listdir(ihc_tiles_dir)
                matching_indices = []
                if not window:
                    for he_idx, tile in enumerate(os.listdir(full_tiles_dir)):
                        tile_df_row = {'HE_slide': row.Matched_HE_SlideName, 'IHC_slide': ihc_slide, 'HE_tile': tile, 'Slide_label': slide_label, 
                                    'fold': fold, 'patient_id': patient_id, 'HE_tile_path': os.path.join(full_tiles_dir, tile),
                                    'HE_features_path': os.path.join(full_tiles_dir.replace('png_tiles', 'gigapath_features'), f'tile_embeds_{dir}.npy'), 'HE_features_tile_index': he_idx} if create_samples_csv else None                    
                        ihc_tile_index = find_matching_score(tile_name=tile, score_mat=None, map_mat=map_mat, find_tile=matching_tile, tiles=tiles, tile_df_row=tile_df_row, 
                                                             target_mpp=args.target_mpp)
                        matching_indices.append(ihc_tile_index)
                        if ihc_tile_index is not None and tile_df_row: # documenting matching indices to a file
                            tile_df_row['IHC_features_path'], tile_df_row['IHC_features_tile_index'] = os.path.join(labels_path.replace('tile_labels', 'gigapath_features'), slide_dir, f'tile_embeds_{slide_dir}.npy'), ihc_tile_index
                            tile_df_row['IHC_tile_path'] = os.path.join(ihc_tiles_dir, tile_df_row['IHC_tile'])
                            tile_df.loc[len(tile_df)] = tile_df_row
                            if len(tile_df) % 500 == 0:
                                break
                
                matching_indices = np.array(matching_indices, dtype=np.float64)
                print(f'ihc_slide = {ihc_slide}, matching_indices = {matching_indices}')
        
                np.save(save_path, matching_indices)
            # else:
            #     print(f'{save_path} already exists, skipping matching tiles for {ihc_slide}')

    if create_samples_csv:
        tile_df.to_csv(tile_df_save_path)
        print("Saved tile_samples csv")
    
    if not matching_tile:
        print(f"len(all_local_labels) = {len(all_local_labels)}, min(all_local_labels) = {min(all_local_labels)}, max(all_local_labels) = {max(all_local_labels)}")
        print(f"all_local_labels.mean() = {all_local_labels.mean()}") # Delete
        print(f"all_local_labels.std() = {all_local_labels.astype(float).std()}")
        # bin_size = 1
        # bins = len(np.arange(min(all_local_labels), max(all_local_labels) + bin_size, bin_size))
        # hist_values, bins_array = np.histogram(all_local_labels, bins)
        # print(f"hist_values = {hist_values}, bins_array = {bins_array}")
        # np.save(labels_hist_path, hist_values)
        # np.save(labels_hist_path.replace("hist", "bins"), bins_array)

def create_tile_x_min_max():
    slides_excel_path = os.path.join("workspace", "WSI", "metadata_csvs", "Her2_slides_matched_HE_folds_infer_15_05_25.csv")
    df = pd.read_csv(slides_excel_path)
    df = df[(df["Path"].str.contains("Batch_1", case=False, na=False)) | (df["Path"].str.contains("Batch_2", case=False, na=False))][["SlideName", "Matched_HE_SlideName", "fold", "label", "patient barcode", "label"]]
    df["SlideName"] = df["SlideName"].apply(lambda x: x.split('.')[0])

    tile_x_path = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", "tile_x_mpp1")
    all_tile_x = []

    for row in df.itertuples(index=True, name="Row"):
        ihc_slide = row.SlideName
        save_path = os.path.join(tile_x_path, ihc_slide, "tile_x.npy")
        if os.path.exists(save_path):
            tile_x = np.load(save_path)
            tile_x = torch.from_numpy(tile_x)
            all_tile_x.append(tile_x.squeeze(0))

    all_tile_x = torch.cat(all_tile_x, dim=0)
    tile_x_max = all_tile_x.max(dim=-2)[0]
    tile_x_min = all_tile_x.min(dim=-2)[0]

    np.save(os.path.join(tile_x_path, "tile_x_max.npy"), tile_x_max)
    np.save(os.path.join(tile_x_path, "tile_x_min.npy"), tile_x_min)


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Process slides through thumbs and mapping matrices.")
    # parser.add_argument('-s', '--save_path', type=str, help='Path to save tiles and embeds', required=True)
    # parser.add_argument('-e', '--excel_path', type=str, help='Slides excel path', required=True)
    parser.add_argument('-tmpp', '--target_mpp', type=float, help='Target tiles MPP', default=1, choices=[0.5, 1, 2], required=True)
    parser.add_argument('-folds', '--folds', type=str, help='Fold to filter', default='', choices=['1', '2', '3', '4', '5', '6'])
    parser.add_argument('-ofa', '--only_first_annotated', action='store_true', help='Only use the first annotated slides')
    

    args = parser.parse_args()
    print(f'args = {args}')

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.target_mpp = int(args.target_mpp) if args.target_mpp != 0.5 else args.target_mpp
    args.suffix = f'_mpp{args.target_mpp}'

    create_tile_scores_matching_tiles(args=args, matching_tile=True) # create_samples_csv=True
    # create_tile_scores(window=True)
    # create_tile_scores_matching_tiles(ihc_tile=True)
    # create_tile_scores_matching_tiles()

    # create_tile_x_min_max()
        

    # first_slide_score_mat_path = os.path.join('score_matrices', '21-1518_1_5_d_score_matrix.npz')
    # sigma = 'sigma01'
    # weighted_slide_score_mat_paths = [
    #     os.path.join('weighted_score_matrices', sigma, '21-1518_1_5_d_score_matrix.npz'),
    #     os.path.join('weighted_score_matrices', sigma, '21-178_1_5_d_score_matrix.npz'),
    #     os.path.join('weighted_score_matrices', sigma, '21-178_1_7_d_score_matrix.npz'),
    #     os.path.join('weighted_score_matrices', sigma, '21-1829_1_1_m_score_matrix.npz'),
    #     os.path.join('weighted_score_matrices', sigma, '21-2025_2_1_d_score_matrix.npz')
    #                                   ]
    # for w_path in weighted_slide_score_mat_paths:
    #     loaded_matrix = np.load(w_path)['arr_0']
    #     # show_score_matrix(loaded_matrix[3990: 4100, 480:590], slide_name='')
    #     show_score_matrix(loaded_matrix, slide_name=w_path.split('\\')[-1].split('_score')[0])

    # slide_scores_csv_path = os.path.join('excel_files', 'sw_stride1_slide_scores.csv')
    # create_weighted_score_matrices(slide_scores_csv=slide_scores_csv_path)

    # rotation_img = cv2.imread(os.path.join('slides_to_amit', 'map_HE_20-10015_1_1_e_to_Her2_20-10015_1_1_m.png'), cv2.IMREAD_UNCHANGED)
    # rotation_img = cv2.imread(os.path.join('slides_to_amit2', '17-8750_2_10', 'map_HE_17-8750_2_10_a_to_Her2_17-8750_2_10_d.png'), cv2.IMREAD_UNCHANGED)
    # rotation_img = cv2.imread(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', 'map_HE_17-8750_2_10_a_labeled_to_Her2_17-8750_2_10_d_labeled.png'), cv2.IMREAD_UNCHANGED)
    # rotation_img = cv2.imread(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', 'map_HE_17-8750_2_10_a_to_Her2_17-8750_2_10_d.png'), cv2.IMREAD_UNCHANGED)
    # rotation_img = cv2.imread(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', 'map_HE_21-163_3_9_d_to_Her2_21-163_3_9_f.png'), cv2.IMREAD_UNCHANGED)
    # rotation_img = cv2.imread(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', 'map_HE_21-2644_1_1_e_to_Her2_21-2644_1_1_m.png'), cv2.IMREAD_UNCHANGED)

    # ihc_slide = openslide.OpenSlide(os.path.join('slides_to_amit', '20-10015_1_1_m.mrxs'))
    # ihc_slide = openslide.OpenSlide(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '17-8750_2_10_d.mrxs'))
    # ihc_slide = openslide.OpenSlide(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '21-163_3_9_f.mrxs'))
    # ihc_slide = openslide.OpenSlide(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '21-2644_1_1_m.mrxs'))

    # ihc_thumb = Image.open(os.path.join('slides_to_amit', '0004_0_thumb_20-10015_1_1_m.jpg'))
    # ihc_thumb = Image.open(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '0000_0_thumb_17-8750_2_10_d.jpg'))
    # ihc_thumb = Image.open(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '0028_0_thumb_21-163_3_9_f.jpg'))
    # ihc_thumb = Image.open(os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', '0093_0_thumb_21-2644_1_1_m.png'))


if __name__ == "__main__":
    main()