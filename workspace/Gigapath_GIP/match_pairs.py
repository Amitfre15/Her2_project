import math
import itertools
import json
import os
import random
import time
import traceback
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import tqdm
import open3d as o3d

from skimage.morphology import disk
from skimage.morphology import binary_opening
from skimage.measure import regionprops, label
from scipy.spatial import Delaunay, ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field

HER2_PATCH_SIZE = 26
MIN_OUTER_POINTS = '5'
DESIRED_MPP = 1
SLIDE_PATCH_SIZE = 256
EXPECTED_THUMBS = 3 # H&E, Her2, and Her2 copy for orientation checking
MIN_LANDMARKS = 6 # At least 4 for global transformation and 2 for evaluation
NUM_INIT_LANDMARKS = 4 # At least 4 for global transformation
LANDMARK_COLOR = [0.0, 0.470588, 0.843137] # (0, 120, 215) - Photos default
INIT_ICP_MATCH_THRESHOLD = 500
MIN_ICP_MATCH_THRESHOLD = 50
ICP_MATCH_THRESHOLD_STEP = 50
UPPER_PAIR_DIST_THRESHOLD = 80
FITNESS_THRESHOLD = 0.9

# =========================================================
# Dataclasses
# =========================================================

@dataclass
class SlideContext:
    dir: str
    folder_path: str

    im_HE: np.ndarray
    im_Her2: np.ndarray
    im_Her2_copy: np.ndarray
    output_mapping_file: str

    S_land_HE: np.ndarray
    S_land_Her2: np.ndarray
    S_land_HE_3d: np.ndarray
    S_land_Her2_3d: np.ndarray

    extra: Dict[str, Any] = field(default_factory=dict)

# =========================================================
# Argument parsing
# =========================================================

def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser(description="Match pairs of thumbnails through a mapping matrix.")
    parser.add_argument('-r',  '--root', default=os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', 'Marked', 'png_thumb_pairs_karin'), help='Root directory to search for thumbnail pairs.')
    parser.add_argument('-dn', '--dict_name', type=str, default='rigid_one_out_LS', help='Name for the distance dictionary saved.')
    parser.add_argument('-d',  '--display', action='store_true', default=False, help='Whether to display images and plots during processing.')
    parser.add_argument('-ut', '--use_triangulation', action='store_true', default=False, help='Whether to match using Delunay triangulation.')

    return parser.parse_args()

def get_folder_path(args, dir):
    folder = os.path.join(args.root, dir)
    if not os.path.isdir(folder):
        print(f'{dir} is not a directory')
        raise NotADirectoryError(f'{dir} is not a directory')
    
    return folder

def get_thumb_names(args, folder):
    png_images = list(filter(lambda x: x.endswith('.png') and not x.startswith('map') and not x.startswith('Transformed'), os.listdir(folder)))
    if len(png_images) < EXPECTED_THUMBS:
        # tri_dist[dir] = 'Directory does not contain enough images'
        args.global_dist_dict[dir] = 'Directory does not contain enough images'
        print('Directory does not contain enough images')
        raise ValueError('Directory does not contain enough images')

    thumb_HE, thumb_Her2, thumb_Her2_copy = parse_thumb_names(png_images=png_images)
    return thumb_HE, thumb_Her2, thumb_Her2_copy

def load_thumbs(args, folder, thumb_HE, thumb_Her2, thumb_Her2_copy):
    im_HE, im_Her2, im_Her2_copy, output_mapping_file = load_and_display_thumbs(folder_path=folder, HE_thumb_name=thumb_HE, Her2_thumb_name=thumb_Her2,
                                                                                Her2_thumb_copy_name=thumb_Her2_copy, display=args.display)

    im_Her2 = rotate_back_annotation_img(im_Her2=im_Her2, im_Her2_copy=im_Her2_copy, display=args.display)
    if type(im_Her2) is int:
        args.global_dist_dict[dir] = 'Could not rotate Her2 image back to original orientation'
        raise ValueError('Could not rotate Her2 image back to original orientation')

    return im_HE, im_Her2, im_Her2_copy, output_mapping_file

def detect_landmarks(args, im_HE, im_Her2):
    # Find landmarks
    landmark_color = np.array(LANDMARK_COLOR).astype(np.float32)

    try:
        if dir in args.global_dist_dict:
            args.global_dist_dict.pop(dir)
        S_land_HE, S_land_Her2 = landmark_detection(im_HE=im_HE, im_Her2=im_Her2, landmark_color=landmark_color, display=args.display)
        if len(S_land_HE) == 0:  # try other landmark color (some cases were annotated in black)
            S_land_HE, S_land_Her2 = landmark_detection(im_HE=im_HE, im_Her2=im_Her2, landmark_color=np.array([0.0, 0.0, 0.0]))
    except ValueError as e:
        args.global_dist_dict[dir] = str(e)
        print(e)
        raise e

    if len(S_land_HE) < MIN_LANDMARKS:
        args.global_dist_dict[dir] = f'Insufficient landmark count: {len(S_land_HE)}'
        print(f'Insufficient landmark count: {len(S_land_HE)}')
        raise ValueError(f'Insufficient landmark count: {len(S_land_HE)}')

    return S_land_HE, S_land_Her2

def verify_no_duplicate_mappings(args, slide_ctx, HE_Her2_land_mapping):
    # Check for duplicate elements
    _, counts = np.unique(HE_Her2_land_mapping, return_counts=True)
    has_duplicates = np.any(counts > 1)
    if has_duplicates:
        args.global_dist_dict[slide_ctx.dir] = f"HE_Her2_land_mapping has correspondence overlap: {HE_Her2_land_mapping}"
        print(f"HE_Her2_land_mapping has correspondence overlap: {HE_Her2_land_mapping}")
        raise ValueError(f"HE_Her2_land_mapping has correspondence overlap: {HE_Her2_land_mapping}")

def evaluate_registration_w_inner_points(args, slide_ctx, HE_Her2_land_mapping, inner_points_mask):
    for i in np.arange(slide_ctx.PT_HE.shape[0]):
        inner_points_indices = np.where(inner_points_mask)[0]
        curr_tri_dist, curr_global_dist = evaluate_w_sub_group(slide_ctx, HE_Her2_land_mapping=HE_Her2_land_mapping, inner_points_indices=inner_points_indices, metric_point_index=i)
        if slide_ctx.dir not in args.global_dist_dict:
            # tri_dist[slide_ctx.dir] = curr_tri_dist
            args.global_dist_dict[slide_ctx.dir] = curr_global_dist
        else:
            for k in args.global_dist_dict[slide_ctx.dir].keys():
                args.global_dist_dict[slide_ctx.dir][k] = np.hstack((args.global_dist_dict[slide_ctx.dir][k], curr_global_dist[k]))

    for k in args.global_dist_dict[slide_ctx.dir].keys():
        args.global_dist_dict[slide_ctx.dir][k] = np.mean(args.global_dist_dict[slide_ctx.dir][k])

def create_save_final_map(args, slide_ctx):
    im_map = create_global_map(args, slide_ctx)
    if args.use_triangulation:
        im_map = triangulate_and_create_map(PT_HE=slide_ctx.PT_HE, PT_Her2=slide_ctx.PT_Her2, im_HE=slide_ctx.im_HE, im_Her2=slide_ctx.im_Her2)

    img_16bit = im_map.astype(np.uint16)
    # Now we can use cv2.cvtColor
    bgr_im_map = cv2.cvtColor(img_16bit, cv2.COLOR_RGB2BGR)

    cv2.imwrite(slide_ctx.output_mapping_file, bgr_im_map)
    print(f'saved mapping file to {slide_ctx.output_mapping_file}')

    # Verify the saved file
    im_map_uint16_rec = cv2.imread(slide_ctx.output_mapping_file, cv2.IMREAD_UNCHANGED)
    if not np.array_equal(bgr_im_map, im_map_uint16_rec):
        print("Warning: The saved mapping file does not match the original.")

def prepare_slide_context(args, dir):
    folder_path = get_folder_path(args, dir)
    thumb_HE, thumb_Her2, thumb_Her2_copy = get_thumb_names(args, folder_path)
    im_HE, im_Her2, im_Her2_copy, output_mapping_file = load_thumbs(args, folder_path, thumb_HE, thumb_Her2, thumb_Her2_copy)
    S_land_HE, S_land_Her2 = detect_landmarks(args, im_HE, im_Her2)
    pts_moving = S_land_HE / 2  # HE thumbs were created twice the size of the Her2 thumbs for some reason
    pts_fixed = S_land_Her2

    S_land_HE_3d = np.hstack([pts_moving, np.zeros((len(S_land_HE), 1))])  # For point cloud
    S_land_Her2_3d = np.hstack([pts_fixed, np.zeros((len(S_land_Her2), 1))])

    if args.display:
        plot_landmarks(S_land_HE_3d[:, [1, 0, 2]], S_land_Her2_3d[:, [1, 0, 2]], f'Initial H&E Landmarks - {dir}', 'Her2 Landmarks')

    return SlideContext(
        dir = dir,
        folder_path=folder_path,

        im_HE=im_HE,
        im_Her2=im_Her2,
        im_Her2_copy=im_Her2_copy,
        output_mapping_file=output_mapping_file,

        S_land_HE=S_land_HE,
        S_land_Her2=S_land_Her2,
        S_land_HE_3d=S_land_HE_3d,
        S_land_Her2_3d=S_land_Her2_3d,
    )

def process_slide(args, slide_ctx):
    HE_Her2_land_mapping, pairs_dist, best_trnsfrm, trans_init_rotated, translation = match_landmarks(args, slide_ctx)

    # Mapping for H&E and Her2
    slide_ctx.PT_HE = slide_ctx.S_land_HE[HE_Her2_land_mapping]
    slide_ctx.PT_Her2 = slide_ctx.S_land_Her2

    inner_points_mask = get_inner_hull_point_mask(points=slide_ctx.PT_HE)

    evaluate_registration_w_inner_points(args, slide_ctx, HE_Her2_land_mapping, inner_points_mask)
    create_save_final_map(args, slide_ctx)
    


# Function to load and normalize image
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def display_image(img, title="Image", save_path: str = None):
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.show()
    plt.close()
    if save_path:
        plt.imsave(f"{os.path.join(save_path, f'{title}.png')}", img)

def load_and_display_thumbs(folder_path: str, HE_thumb_name: str, Her2_thumb_name: str, Her2_thumb_copy_name,
                            display: bool = True):
    output_mapping_file = os.path.join(folder_path, f'map_HE_{HE_thumb_name[13:-4]}_to_Her2_{Her2_thumb_name[13:-4]}.png')
    dir_name = folder_path.split('\\')[-1]

    # Load images
    im_HE = load_image(os.path.join(folder_path, HE_thumb_name))
    im_Her2 = load_image(os.path.join(folder_path, Her2_thumb_name))
    im_Her2_copy = load_image(os.path.join(folder_path, Her2_thumb_copy_name))

    if im_Her2.shape != im_Her2_copy.shape and im_Her2.shape[0] != im_Her2_copy.shape[1]:
        im_HE, im_Her2 = im_Her2, im_HE

    if display:
        display_image(im_HE, f"H&E Image - {dir_name}")
        display_image(im_Her2, "Her2 Image")
        display_image(im_Her2_copy, "Her2 Image original orientation")

    return im_HE, im_Her2, im_Her2_copy, output_mapping_file

def landmark_detection(im_HE: np.array, im_Her2: np.array, landmark_color: np.array, color_threshold: float = 0.001,
                       display: bool = False):
    # For each pixel, measure its distance from landmark_color
    dist_HE = np.sqrt(np.sum((im_HE - landmark_color) ** 2, axis=2))

    dist_HE_b = dist_HE < color_threshold  # % distance < color_threshold should be true for the pixels of the landmarks
    dist_HE_b = binary_opening(dist_HE_b, disk(2))  # remove small noise pixels

    if display:
        display_image(dist_HE_b.astype(np.float64), "Landmarks H&E")

    # Extract centroid of landmarks
    labeled_HE = label(dist_HE_b)  # get all connected components
    regions_HE = regionprops(labeled_HE)
    S_land_HE = np.array([region.centroid for region in regions_HE])

    # Repeat for Her2
    dist_Her2 = np.sqrt(np.sum((im_Her2 - landmark_color) ** 2, axis=2))
    dist_Her2_b = dist_Her2 < color_threshold
    dist_Her2_b = binary_opening(dist_Her2_b, disk(2))

    if display:
        display_image(dist_Her2_b.astype(np.float64), "Landmarks Her2")

    labeled_Her2 = label(dist_Her2_b)
    regions_Her2 = regionprops(labeled_Her2)
    S_land_Her2 = np.array([region.centroid for region in regions_Her2])

    # Ensure the number of landmarks is the same
    if len(S_land_HE) != len(S_land_Her2):
        raise ValueError("Number of landmarks does not match.")

    return S_land_HE, S_land_Her2


# Visualize the results
def plot_landmarks(landmarks_fixed, landmarks_moving, title_fixed, title_moving, unchosen_fixed: np.array = None,
                   unchosen_moving: np.array = None, color_unchosen: str = 'g'):
    import matplotlib.pyplot as plt
    plt.scatter(landmarks_fixed[:, 0], landmarks_fixed[:, 1], c='b', label=title_fixed)
    plt.scatter(landmarks_moving[:, 0], landmarks_moving[:, 1], c='r', label=title_moving)
    # plot unchosen landmarks in evaluation stage
    if unchosen_fixed is not None:
        plt.scatter(unchosen_fixed[:, 0], unchosen_fixed[:, 1], c='b')
        plt.scatter(unchosen_moving[:, 0], unchosen_moving[:, 1], c=color_unchosen, label=f'{title_moving} unchosen')
    # Invert the y-axis
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()


def get_rotation_matrix_45_degrees(axis, angle_deg):
    """Returns a rotation matrix for 45-degree increments along a given axis."""
    angle_rad = np.radians(angle_deg)
    if axis == 'x':
        rotation_vector = [angle_rad, 0, 0]
    elif axis == 'y':
        rotation_vector = [0, angle_rad, 0]
    elif axis == 'z':
        rotation_vector = [0, 0, angle_rad]
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    return o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)


# Perform Iterative Closest Point (ICP) to find the correspondence between the landmarks
def run_icp(source, target, trans_init, threshold):
    """Runs ICP and returns the result."""
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return reg_p2p.transformation, reg_p2p.fitness


def find_and_apply_best_trnsfrm(S_land_HE_3d: np.array, S_land_Her2_3d: np.array, left_out_points: np.array = None,
                                display: bool = True, icp_threhold: int = 150, relevant_mask: np.array = None,
                                left_out_true_pairs: np.array = None):
    pcd_moving = o3d.geometry.PointCloud()
    pcd_fixed = o3d.geometry.PointCloud()
    # pcd_moving.points = o3d.utility.Vector3dVector(S_land_Her2_3d)
    pcd_moving.points = o3d.utility.Vector3dVector(S_land_HE_3d)
    # pcd_fixed.points = o3d.utility.Vector3dVector(S_land_HE_3d)
    pcd_fixed.points = o3d.utility.Vector3dVector(S_land_Her2_3d)
    fixed_center = pcd_fixed.get_center()[:2]

    trans_init = np.eye(4)

    # List to store results for different rotations
    results = []
    rotated_lo, trans_rotated_lo = None, None

    # Perform ICP for 90-degree rotations (90°, 180°, 270°)
    for i in range(0, 8):
        # Generate a 90-degree rotation matrix around the z-axis (change axis if needed)
        rotation_matrix = get_rotation_matrix_45_degrees('z', 45 * i)

        # Construct a 4x4 transformation matrix
        trans_rot = np.eye(4)
        trans_rot[:3, :3] = rotation_matrix

        # Multiply the rotation matrix with the initial transformation matrix
        trans_init_rotated = np.dot(trans_rot, trans_init)

        pcd_moving_tmp = o3d.geometry.PointCloud()
        # rotated_points = np.dot(S_land_Her2_3d, trans_init_rotated[:3, :3].T)
        rotated_points = np.dot(S_land_HE_3d, trans_init_rotated[:3, :3].T)
        if left_out_points is not None:
            rotated_lo = np.dot(left_out_points, trans_init_rotated[:3, :3].T)
        pcd_moving_tmp.points = o3d.utility.Vector3dVector(rotated_points)
        # plot_landmarks(S_land_HE_3d, rotated_points, 'HE Landmarks', 'Transformed HE Landmarks before translation')

        translation = fixed_center - pcd_moving_tmp.get_center()[:2]
        trans_rot_points = rotated_points + np.hstack([translation, np.zeros(1)])
        if rotated_lo is not None:
            trans_rotated_lo = rotated_lo + np.hstack([translation, np.zeros(1)])
        pcd_moving_tmp.points = o3d.utility.Vector3dVector(trans_rot_points)
        # plot_landmarks(S_land_HE_3d, trans_rot_points, 'HE Landmarks', 'Transformed HE Landmarks after translation')

        # Run ICP with the rotated transformation
        transformation_rotated, fitness_rotated = run_icp(pcd_moving_tmp, pcd_fixed, trans_init, icp_threhold)

        # Store result
        results.append((transformation_rotated, fitness_rotated, trans_rot_points, trans_rotated_lo, trans_init_rotated, translation))

    max_fitness = max([r[1] for r in results])
    # Select the best transformation based on fitness
    filtered_results = list(filter(lambda x: x[1] == max_fitness, results))

    best_mean_global_dist = math.inf
    best_fitness = 0
    if left_out_points is not None:
        for transformation, fitness, rot_points, rot_lo, trans_init_rotated, translation in filtered_results:
            rot_lo = np.dot(rot_lo, transformation[:3, :3].T) + transformation[:3, 3]
            if relevant_mask is not None:
                mean_global_dist = np.mean(np.sqrt(np.sum((left_out_true_pairs - rot_lo)[relevant_mask] ** 2, axis=1)))
            else:
                mean_global_dist = np.sqrt(np.sum((left_out_true_pairs - rot_lo) ** 2))
            if mean_global_dist < best_mean_global_dist:
                best_transformation, best_fitness, best_rot_points, best_rot_lo, best_trans_init_rotated, best_translation = (
                    transformation, fitness, rot_points, rot_lo, trans_init_rotated, translation)
                best_mean_global_dist = mean_global_dist
    else:
        best_transformation, best_fitness, best_rot_points, best_rot_lo, best_trans_init_rotated, best_translation = max(results, key=lambda x: x[1])


    # Output the best transformation
    print(f"Best transformation fitness is: {best_fitness}")

    # Perform transformation
    # S_land_Her2_transformed = np.dot(best_rot_points, best_transformation[:3, :3].T) + best_transformation[:3, 3]
    S_land_HE_transformed = np.dot(best_rot_points, best_transformation[:3, :3].T) + best_transformation[:3, 3]

    if display:
        time.sleep(2)
        if left_out_true_pairs is not None:
            plot_landmarks(landmarks_fixed=S_land_Her2_3d[:, [1, 0, 2]], landmarks_moving=S_land_HE_transformed[:, [1, 0, 2]],
                           title_fixed='Her2 Landmarks', title_moving='Transformed unchosen HE Landmarks',
                           unchosen_fixed=left_out_true_pairs[:, [1, 0, 2]], unchosen_moving=best_rot_lo[:, [1, 0, 2]],
                           color_unchosen='g')
        else:
            plot_landmarks(landmarks_fixed=S_land_Her2_3d[:, [1, 0, 2]],
                           landmarks_moving=S_land_HE_transformed[:, [1, 0, 2]],
                           title_fixed='Her2 Landmarks', title_moving='Transformed HE Landmarks')

    # best_transformation = np.dot()
    return S_land_HE_transformed, best_fitness, best_rot_lo, best_transformation, best_trans_init_rotated, best_translation


def create_global_map(args, slide_ctx):
    # Define bounding box and pixels in H&E
    min_y, min_x = np.floor(np.min(slide_ctx.PT_HE, axis=0)).astype(int)
    max_y, max_x = np.ceil(np.max(slide_ctx.PT_HE, axis=0)).astype(int)
    y, x = np.meshgrid(np.arange(min_y, max_y), np.arange(min_x, max_x))
    pixels = np.vstack([y.ravel(), x.ravel()]).T
    # pixels_to_trnsfrm = pixels // 2
    # pixels_hom = np.hstack([pixels_to_trnsfrm, np.ones((len(pixels_to_trnsfrm), 1))])
    pixels_hom = np.hstack([pixels, np.ones((len(pixels), 1))])

    # Initialize transformed image and mapping image
    im_transformed = np.copy(slide_ctx.im_HE)
    im_map = np.zeros_like(slide_ctx.im_HE)

    # Create the (x, y) grid for the Her2 image
    im_XY = np.dstack(np.meshgrid(np.arange(slide_ctx.im_Her2.shape[1]), np.arange(slide_ctx.im_Her2.shape[0])))

    # pixels_transformed_Her2 = np.dot(pixels_hom, trans_init_rotated[:3, :3].T)
    # pixels_transformed_Her2 = pixels_transformed_Her2 + np.hstack((translation, 0))
    # pixels_transformed_Her2 = np.dot(pixels_transformed_Her2, best_trnsfrm[:3, :3].T) + best_trnsfrm[:3, 3]

    # Compute the affine transformation matrix between the two triangles
    A = np.linalg.lstsq(np.hstack([slide_ctx.PT_HE, np.ones((len(slide_ctx.PT_HE), 1))]), slide_ctx.PT_Her2, rcond=None)[0]

    # Apply the transformation to the pixels of the current triangle
    pixels_transformed_Her2 = np.dot(pixels_hom, A)

    # Round and clip the coordinates for image indexing
    pixels_transformed_Her2 = np.round(pixels_transformed_Her2).astype(int)[:, :2]
    pixels_transformed_Her2 = np.clip(pixels_transformed_Her2, 0, np.array(slide_ctx.im_Her2.shape[:2]) - 1)

    # Convert pixel coordinates to integer after rounding
    y_transformed = np.round(pixels_transformed_Her2[:, 0]).astype(int)
    x_transformed = np.round(pixels_transformed_Her2[:, 1]).astype(int)

    y_curr = np.round(pixels[:, 0]).astype(int)
    x_curr = np.round(pixels[:, 1]).astype(int)

    # Filter out any out-of-bound indices
    valid_mask = (x_transformed >= 0) & (x_transformed < slide_ctx.im_Her2.shape[1]) & \
                 (y_transformed >= 0) & (y_transformed < slide_ctx.im_Her2.shape[0]) & \
                 (x_curr >= 0) & (x_curr < slide_ctx.im_HE.shape[1]) & \
                 (y_curr >= 0) & (y_curr < slide_ctx.im_HE.shape[0])

    x_transformed = x_transformed[valid_mask]
    y_transformed = y_transformed[valid_mask]
    x_curr = x_curr[valid_mask]
    y_curr = y_curr[valid_mask]

    # Update the transformed image for each channel (R, G, B)
    im_transformed[y_curr, x_curr, 0] = slide_ctx.im_Her2[y_transformed, x_transformed, 0]  # Red channel
    im_transformed[y_curr, x_curr, 1] = slide_ctx.im_Her2[y_transformed, x_transformed, 1]  # Green channel
    im_transformed[y_curr, x_curr, 2] = slide_ctx.im_Her2[y_transformed, x_transformed, 2]  # Blue channel

    # Set the mapping matrix values
    im_map[y_curr, x_curr, 0] = im_XY[y_transformed, x_transformed, 0]
    im_map[y_curr, x_curr, 1] = im_XY[y_transformed, x_transformed, 1]

    # Compute the rotation (theta) and store in the blue channel
    # svd_U, _, svd_V = np.linalg.svd(best_trnsfrm[:2, :2])
    svd_U, _, svd_V = np.linalg.svd(A[:2, :2])
    R_mat = np.dot(svd_U, svd_V)
    theta = np.arctan2(R_mat[1, 0], R_mat[0, 0])

    # init_svd_U, _, init_svd_V = np.linalg.svd(trans_init_rotated[:2, :2])
    # init_R_mat = np.dot(init_svd_U, init_svd_V)
    # init_theta = np.arctan2(init_R_mat[1, 0], init_R_mat[0, 0])
    #
    # theta += init_theta

    if theta < 0:
        theta += 2 * np.pi

    theta = (theta / (2 * np.pi)) * 360 * 100  # Convert to degrees and scale
    # im_map.ravel()[ind_curr_b] = theta
    im_map[y_curr, x_curr, 2] = theta

    # Display the transformed image
    if args.display:
        im_downsampled = cv2.resize(im_transformed[max(min_y - 700, 0):min(max_y + 700, im_transformed.shape[0]), :], (0, 0), fx=0.25, fy=0.25)  # 25% size
        display_image(im_downsampled, f"Transformed Her2 on H&E - {slide_ctx.dir}", save_path=slide_ctx.folder_path)

    return im_map



def triangulate_and_create_map(PT_HE: np.array, PT_Her2: np.array, im_HE: np.array, im_Her2: np.array,
                               display: bool = True):
    tri = Delaunay(PT_HE)

    # Define bounding box and pixels in H&E
    min_y, min_x = np.floor(np.min(PT_HE, axis=0)).astype(int)
    max_y, max_x = np.ceil(np.max(PT_HE, axis=0)).astype(int)
    y, x = np.meshgrid(np.arange(min_y, max_y), np.arange(min_x, max_x))
    pixels = np.vstack([y.ravel(), x.ravel()]).T

    # Triangle indices for H&E image
    triangle_indices = tri.find_simplex(pixels)

    # Initialize transformed image and mapping image
    im_transformed = np.copy(im_HE)
    im_map = np.zeros_like(im_HE)

    # Create the (x, y) grid for the Her2 image
    im_XY = np.dstack(np.meshgrid(np.arange(im_Her2.shape[1]), np.arange(im_Her2.shape[0])))

    # Iterate over each triangle and compute the transformation
    for i in range(len(tri.simplices)):
        # Get the pixels inside the current triangle in H&E
        tri_curr_pixels_HE = pixels[triangle_indices == i]

        if len(tri_curr_pixels_HE) == 0:
            continue

        # Get the vertices of the triangle in H&E and Her2
        tri_curr_PT_HE = PT_HE[tri.simplices[i]]
        tri_curr_PT_Her2 = PT_Her2[tri.simplices[i]]

        # Compute the affine transformation matrix between the two triangles
        A = np.linalg.lstsq(np.hstack([tri_curr_PT_HE, np.ones((3, 1))]), tri_curr_PT_Her2, rcond=None)[0]

        # Apply the transformation to the pixels of the current triangle
        tri_curr_pixels_HE_hom = np.hstack([tri_curr_pixels_HE, np.ones((len(tri_curr_pixels_HE), 1))])
        pixels_transformed_Her2 = np.dot(tri_curr_pixels_HE_hom, A)

        # Round and clip the coordinates for image indexing
        pixels_transformed_Her2 = np.round(pixels_transformed_Her2).astype(int)
        pixels_transformed_Her2 = np.clip(pixels_transformed_Her2, 0, np.array(im_Her2.shape[:2]) - 1)

        # Convert pixel coordinates to integer after rounding
        y_transformed = np.round(pixels_transformed_Her2[:, 0]).astype(int)
        x_transformed = np.round(pixels_transformed_Her2[:, 1]).astype(int)

        y_curr = np.round(tri_curr_pixels_HE[:, 0]).astype(int)
        x_curr = np.round(tri_curr_pixels_HE[:, 1]).astype(int)

        # Filter out any out-of-bound indices
        valid_mask = (x_transformed >= 0) & (x_transformed < im_Her2.shape[1]) & \
                     (y_transformed >= 0) & (y_transformed < im_Her2.shape[0]) & \
                     (x_curr >= 0) & (x_curr < im_HE.shape[1]) & \
                     (y_curr >= 0) & (y_curr < im_HE.shape[0])

        x_transformed = x_transformed[valid_mask]
        y_transformed = y_transformed[valid_mask]
        x_curr = x_curr[valid_mask]
        y_curr = y_curr[valid_mask]

        # Update the transformed image for each channel (R, G, B)
        im_transformed[y_curr, x_curr, 0] = im_Her2[y_transformed, x_transformed, 0]  # Red channel
        im_transformed[y_curr, x_curr, 1] = im_Her2[y_transformed, x_transformed, 1]  # Green channel
        im_transformed[y_curr, x_curr, 2] = im_Her2[y_transformed, x_transformed, 2]  # Blue channel

        # Set the mapping matrix values
        im_map[y_curr, x_curr, 0] = im_XY[y_transformed, x_transformed, 0]
        im_map[y_curr, x_curr, 1] = im_XY[y_transformed, x_transformed, 1]

        # Compute the rotation (theta) and store in the blue channel
        svd_U, _, svd_V = np.linalg.svd(A[:2, :2])
        R_mat = np.dot(svd_U, svd_V)
        theta = np.arctan2(R_mat[1, 0], R_mat[0, 0])

        if theta < 0:
            theta += 2 * np.pi

        theta = (theta / (2 * np.pi)) * 360 * 100  # Convert to degrees and scale
        # im_map.ravel()[ind_curr_b] = theta
        im_map[y_curr, x_curr, 2] = theta

    # Display the transformed image
    if display:
        display_image(im_transformed[max(min_y - 700, 0):min(max_y + 700, im_transformed.shape[0]), :], "Transformed Her2 on H&E")

    return im_map


def choose_top_index(PT_HE_chosen: np.array, PT_HE_unchosen: np.array, PT_Her2_chosen: np.array,
                     PT_Her2_unchosen: np.array, HE_3d_chosen: np.array, HE_3d_unchosen: np.array,
                     Her2_3d_chosen: np.array, Her2_3d_unchosen: np.array, top_index: np.array):
    # Move the selected samples from unchosen to chosen
    PT_HE_chosen = np.vstack((PT_HE_chosen, PT_HE_unchosen[top_index]))
    PT_HE_unchosen = np.delete(PT_HE_unchosen, top_index, axis=0)
    PT_Her2_chosen = np.vstack((PT_Her2_chosen, PT_Her2_unchosen[top_index]))
    PT_Her2_unchosen = np.delete(PT_Her2_unchosen, top_index, axis=0)
    HE_3d_chosen = np.vstack((HE_3d_chosen, HE_3d_unchosen[top_index]))
    HE_3d_unchosen = np.delete(HE_3d_unchosen, top_index, axis=0)
    Her2_3d_chosen = np.vstack((Her2_3d_chosen, Her2_3d_unchosen[top_index]))
    Her2_3d_unchosen = np.delete(Her2_3d_unchosen, top_index, axis=0)

    return (PT_HE_chosen, PT_HE_unchosen, PT_Her2_chosen, PT_Her2_unchosen, HE_3d_chosen, HE_3d_unchosen, Her2_3d_chosen
            , Her2_3d_unchosen)


def transform_sub_group(PT_HE_chosen: np.array, PT_HE_unchosen: np.array, PT_Her2_chosen: np.array, PT_Her2_unchosen: np.array,
                        im_HE: np.array, im_Her2: np.array, S_land_HE_3d: np.array, HE_3d_chosen: np.array,
                        HE_3d_unchosen: np.array, Her2_3d_chosen: np.array, Her2_3d_unchosen: np.array, metric_point: np.array,
                        metric_point_3d: np.array, Her2_metric_point_3d: np.array, icp_threshold: int = 500):
    # global evaluation
    trnsfrm_fitness = 0
    icp_trials = 0
    curr_thresh = icp_threshold
    mean_global_dist = math.inf

    mean_dists = []
    while (mean_global_dist > 100 or trnsfrm_fitness < 0.9) and curr_thresh > 0:
        best_results = find_and_apply_best_trnsfrm(S_land_HE_3d=HE_3d_chosen, S_land_Her2_3d=Her2_3d_chosen, left_out_points=metric_point_3d,
                                                   left_out_true_pairs=Her2_metric_point_3d, icp_threhold=curr_thresh, display=False)
        trnsfrm_fitness, transformed_lo = best_results[1], best_results[2]

        mean_global_dist = np.sqrt(np.sum((Her2_metric_point_3d - transformed_lo) ** 2))
        mean_dists.append(mean_global_dist)
        icp_trials += 1
        curr_thresh = icp_threshold - icp_trials * 50
    mean_global_dist = min(mean_dists)

    # return mean_tri_dist, mean_global_dist
    return None, mean_global_dist


def evaluate_w_sub_group(slide_ctx, HE_Her2_land_mapping: np.array, inner_points_indices: np.array, metric_point_index: np.array = None):
    tri_dist, global_dist = {}, {}
    S_land_HE_3d = slide_ctx.S_land_HE_3d[HE_Her2_land_mapping]

    metric_point, metric_point_3d, Her2_metric_point_3d = np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1))

    # init metric point for both triangulation and global transform
    metric_point, Her2_metric_point = slide_ctx.PT_HE[metric_point_index].reshape(1, -1), slide_ctx.PT_Her2[metric_point_index].reshape(1, -1)
    curr_PT_HE, curr_PT_Her2 = np.delete(slide_ctx.PT_HE, metric_point_index, axis=0), np.delete(slide_ctx.PT_Her2, metric_point_index, axis=0)

    metric_point_3d, Her2_metric_point_3d = slide_ctx.S_land_HE_3d[metric_point_index].reshape(1, -1), slide_ctx.S_land_Her2_3d[metric_point_index].reshape(1, -1)
    S_land_HE_3d, S_land_Her2_3d_mapped = np.delete(slide_ctx.S_land_HE_3d, metric_point_index, axis=0), np.delete(slide_ctx.S_land_Her2_3d, metric_point_index, axis=0)

    # init other points
    PT_HE_chosen, PT_HE_unchosen = curr_PT_HE[0].reshape(1, -1), curr_PT_HE[1:]
    PT_Her2_chosen, PT_Her2_unchosen = curr_PT_Her2[0].reshape(1, -1), curr_PT_Her2[1:]
    HE_3d_chosen, HE_3d_unchosen = S_land_HE_3d[0].reshape(1, -1), S_land_HE_3d[1:]
    Her2_3d_chosen, Her2_3d_unchosen = S_land_Her2_3d_mapped[0].reshape(1, -1), S_land_Her2_3d_mapped[1:]

    # max_area = calculate_area(points=PT_HE)

    while PT_HE_chosen.shape[0] < NUM_INIT_LANDMARKS:
        top_index, chosen_area = diversity_sampling(chosen_points=PT_HE_chosen, unchosen_points=PT_HE_unchosen)
        arrays_tuple = choose_top_index(PT_HE_chosen=PT_HE_chosen, PT_HE_unchosen=PT_HE_unchosen, PT_Her2_chosen=PT_Her2_chosen,
                                        PT_Her2_unchosen=PT_Her2_unchosen, HE_3d_chosen=HE_3d_chosen, HE_3d_unchosen=HE_3d_unchosen,
                                        Her2_3d_chosen=Her2_3d_chosen, Her2_3d_unchosen=Her2_3d_unchosen, top_index=top_index)
        PT_HE_chosen, PT_HE_unchosen, PT_Her2_chosen, PT_Her2_unchosen, HE_3d_chosen, HE_3d_unchosen, Her2_3d_chosen, Her2_3d_unchosen = arrays_tuple

    unchosen_indices = np.arange(PT_HE_unchosen.shape[0])
    for j in range(1, PT_HE_unchosen.shape[0] + 1):
        tmp_len = PT_HE_chosen.shape[0] + j
        tri_dist[str(tmp_len)], global_dist[str(tmp_len)] = [], []
        possible_ind_groups = list(itertools.combinations(unchosen_indices, j))
        index_groups = random.sample(possible_ind_groups, min(len(possible_ind_groups), 10))
        for ind_gr in index_groups:
            arrays_tuple = choose_top_index(PT_HE_chosen=PT_HE_chosen, PT_HE_unchosen=PT_HE_unchosen, PT_Her2_chosen=PT_Her2_chosen,
                                            PT_Her2_unchosen=PT_Her2_unchosen, HE_3d_chosen=HE_3d_chosen, HE_3d_unchosen=HE_3d_unchosen,
                                            Her2_3d_chosen=Her2_3d_chosen, Her2_3d_unchosen=Her2_3d_unchosen, top_index=np.array(ind_gr))
            PT_HE_chosen_tmp, PT_HE_unchosen_tmp, PT_Her2_chosen_tmp, PT_Her2_unchosen_tmp, HE_3d_chosen_tmp, HE_3d_unchosen_tmp, Her2_3d_chosen_tmp, Her2_3d_unchosen_tmp = arrays_tuple

            # mean_tri_dist, mean_global_dist = transform_sub_group(PT_HE_chosen=PT_HE_chosen_tmp, PT_HE_unchosen=PT_HE_unchosen_tmp,
            #                                            PT_Her2_chosen=PT_Her2_chosen_tmp,
            #                                            PT_Her2_unchosen=PT_Her2_unchosen_tmp, im_HE=im_HE,
            #                                            im_Her2=im_Her2, S_land_HE_3d=S_land_HE_3d,
            #                                            HE_3d_chosen=HE_3d_chosen_tmp, HE_3d_unchosen=HE_3d_unchosen_tmp,
            #                                            Her2_3d_chosen=Her2_3d_chosen_tmp,
            #                                            Her2_3d_unchosen=Her2_3d_unchosen_tmp, metric_point=metric_point,
            #                                            metric_point_3d=metric_point_3d,
            #                                            Her2_metric_point_3d=Her2_metric_point_3d)

            # Compute the affine transformation matrix between the two triangles
            A = np.linalg.lstsq(np.hstack([PT_HE_chosen_tmp, np.ones((len(PT_HE_chosen_tmp), 1))]),
                                np.hstack([PT_Her2_chosen_tmp, np.ones((len(PT_Her2_chosen_tmp), 1))]), rcond=None)[0]

            # Apply the transformation to the pixels of the current triangle
            metric_transformed_to_Her2 = np.dot(np.hstack([metric_point, np.ones((len(metric_point), 1))]), A)

            mean_global_dist = np.sqrt(np.sum((metric_transformed_to_Her2[:, :2] - Her2_metric_point) ** 2))

            if mean_global_dist is not None:
                # tri_dist[str(tmp_len)].append(mean_tri_dist)
                global_dist[str(tmp_len)].append(mean_global_dist)
    # for k in tri_dist.keys():
    for k in global_dist.keys():
        # tri_dist[k] = np.mean(tri_dist.get(k))
        global_dist[k] = np.mean(global_dist.get(k))

    return tri_dist, global_dist


def show_dist_plot(tri_dist: list, global_dist: list):
    tri_dist, global_dist = np.array(tri_dist), np.array(global_dist)
    global_minus_tri = global_dist - tri_dist
    print(f'tri_dist = {tri_dist}\ntri_dist max = {max(tri_dist)}\ntri_dist avg = {np.mean(tri_dist)}')
    print(f'global_dist = {global_dist}\nglobal_dist max = {max(global_dist)}\nglobal_dist avg = {np.mean(global_dist)}')

    # Compute mean and max for each list
    tri_mean, tri_max = np.mean(tri_dist), np.max(tri_dist)
    global_mean, global_max = np.mean(global_dist), np.max(global_dist)
    diff_mean = np.mean(global_minus_tri)

    # Create histograms
    plt.figure(figsize=(10, 6))
    # plt.plot(tri_dist, marker='o', linestyle='-', label='Triangulation distances', color='blue')
    # plt.plot(global_dist, marker='o', linestyle='-', label='Global Transformation distances', color='orange')
    plt.plot(global_minus_tri, marker='o', linestyle='-', label='Global minus Triangulation', color='orange')

    # Plot mean and max lines for tri_dist
    # plt.axhline(tri_mean, color='blue', linestyle='dashed', linewidth=2, label='Mean (tri_dist)')
    # plt.axhline(tri_max, color='blue', linestyle='solid', linewidth=2, label='Max (tri_dist)')

    # Plot mean and max lines for global_dist
    # plt.axhline(global_mean, color='orange', linestyle='dashed', linewidth=2, label='Mean (global_dist)')
    # plt.axhline(global_max, color='orange', linestyle='solid', linewidth=2, label='Max (global_dist)')
    plt.axhline(diff_mean, color='orange', linestyle='solid', linewidth=2, label='Mean (global_dist - tri dist)')

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel(f'Distance diff as a percentage out of patch size ({HER2_PATCH_SIZE}px)')
    plt.title('Line plots with Mean line')
    plt.legend()

    # Show plot
    plt.show()


def show_dist_hist(global_data_dict: dict):
    final_tri_dists, final_global_dists = {}, {}
    for k in global_data_dict.keys():
        # final_tri_dists[k] = list(tri_dist[k].values())[-1]
        if any(np.array(list(global_data_dict[k].values())) > 100):
            print(f'k = {k}, v.values() = {global_data_dict[k].values()}')
        final_global_dists[k] = list(global_data_dict[k].values())[-1] if len(global_data_dict[k].values()) > 0 else list(global_data_dict[k].values())


    # np_final_tri = np.array(list(final_tri_dists.values())) / HER2_PATCH_SIZE
    np_final_global = np.array(list(final_global_dists.values())) / HER2_PATCH_SIZE

    # Compute mean and max for each list
    # tri_mean, tri_max = np.mean(np_final_tri), np.max(np_final_tri)
    global_mean, global_max = np.mean(np_final_global), np.max(np_final_global)
    print(f'global_mean = {global_mean}')

    # Double font sizes globally
    plt.rcParams.update({'font.size': 16})

    # Create histograms
    plt.figure(figsize=(10, 6))

    # plt.hist(np_final_tri, bins=10, alpha=0.5, label='Triangulation distances', color='blue')
    plt.hist(np_final_global, bins=10, alpha=0.5, label='Global transformation distances', color='orange')

    # Plot mean and max lines for tri_dist
    # plt.axvline(tri_mean, color='blue', linestyle='dashed', linewidth=2, label='Mean (tri_dist)')
    # plt.axvline(tri_max, color='blue', linestyle='solid', linewidth=2, label='Max (tri_dist)')

    # Plot mean and max lines for global_dist
    plt.axvline(global_mean, color='orange', linestyle='dashed', linewidth=2, label='Mean line (global distances)')
    # plt.axvline(global_max, color='orange', linestyle='solid', linewidth=2, label='Max (global_dist)')

    # Add labels and legend
    plt.xlabel(f'Distance (proportion out of MPP={DESIRED_MPP} {SLIDE_PATCH_SIZE}px patch)')
    plt.ylabel('Frequency')
    plt.title('Matched landmarks distance histogram with mean line')
    plt.legend()

    # Show plot
    plt.show()


def show_mean_dist_area_prop(global_data_dict: dict, avg_dirs: bool = False, show_area: bool = False):
    if not avg_dirs:
        # Calculate rows needed for two columns
        num_dirs = len(global_data_dict)
        ncols = 2
        nrows = math.ceil(num_dirs / ncols)

        # Setup the figure with two columns of subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows), sharex=True)
        fig.suptitle('Distance by Number of Chosen Points', fontsize=18)

        # Flatten axes array for easy indexing, in case there's only one row
        axes = axes.flatten()

        # Plot each directory's data
        # for i, (directory, values) in enumerate(tri_data_dict.items()):
        for i, (directory, values) in enumerate(global_data_dict.items()):
            global_values = global_data_dict[directory]
            ax = axes[i]

            # Extract data for plotting
            num_points = list(values.keys())
            tri_distances = [values[n][0] / HER2_PATCH_SIZE for n in num_points]  # proportion out of 13px patch
            area_proportions = [values[n][1] for n in num_points]
            global_distances = [global_values[n] / HER2_PATCH_SIZE for n in num_points]  # proportion out of 13px patch

            # Plot distance on the left y-axis
            ax.plot(num_points, tri_distances, color='blue', marker='o', label='Tri Distance')
            ax.plot(num_points, global_distances, color='green', marker='o', label='Global Distance')

            # Create secondary y-axis for area proportion
            ax2 = ax.twinx()
            ax2.plot(num_points, area_proportions, color='red', marker='x', linestyle='--', label='Area Proportion')

            # Set title for each subplot
            ax.set_title(f"Directory: {directory}")

        # Adjust layout to fit everything
        plt.subplots_adjust(top=0.92, hspace=0.8, wspace=0.3)

        # Add shared axis labels
        fig.text(0.5, 0.04, 'Number of Chosen Points', ha='center', fontsize=14)
        fig.text(0.03, 0.5, f'Triangulation Distance (proportion out of {HER2_PATCH_SIZE}px patch)', va='center', ha='center', color='blue', rotation='vertical', fontsize=14)
        fig.text(0.06, 0.5, f'Global Distance (proportion out of {HER2_PATCH_SIZE}px patch)', va='center', ha='center', color='green',
                 rotation='vertical', fontsize=14)
        fig.text(0.96, 0.5, 'Area Proportion', va='center', ha='center', color='red', rotation='vertical', fontsize=14)

        plt.show()

    else:
        global_distances, tri_distances = {}, {}
        if show_area:
            area_proportions = {}
        # Setup the figure with two columns of subplots
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Distance by Number of Chosen Points', fontsize=18)

        # Plot each directory's data
        # for i, (directory, values) in enumerate(tri_data_dict.items()):
        for i, (directory, values) in enumerate(global_data_dict.items()):
            # global_values = global_data_dict[directory]

            # Extract data for plotting
            num_points = list(values.keys())
            for n in num_points:
                # if n not in tri_distances:
                #     tri_distances[n] = []
                # if show_area:
                #     # proportion out of IHC thumb scale patch size
                #     tri_distances[n].append(values[n][0] / HER2_PATCH_SIZE)
                #     if n not in area_proportions:
                #         area_proportions[n] = []
                #     area_proportions[n].append(values[n][1])
                # else:
                #     # tri_distances[n].append(values[n] / HER2_PATCH_SIZE) if n == MIN_OUTER_POINTS \
                #     #     else tri_distances[n].append((values[n] - values[MIN_OUTER_POINTS]) / HER2_PATCH_SIZE)
                #     pass

                if n not in global_distances:
                    global_distances[n] = []
                global_distances[n].append(values[n] / HER2_PATCH_SIZE) if n == MIN_OUTER_POINTS \
                        else global_distances[n].append((values[n] - values[MIN_OUTER_POINTS]) / HER2_PATCH_SIZE)

        # num_points = list(tri_distances.keys())
        num_points = list(global_distances.keys())
        # for k in tri_distances.keys():
        for k in global_distances.keys():
            # tri_distances[k] = np.mean(tri_distances.get(k))
            global_distances[k] = np.mean(global_distances.get(k))
            if show_area:
                area_proportions[k] = np.mean(area_proportions.get(k))

        # Adjust layout to fit everything
        plt.subplots_adjust(top=0.92, hspace=0.8, wspace=0.3)

        # Plot distance on the left y-axis
        # ax.plot(num_points, [tri_distances[n] if n == MIN_OUTER_POINTS else tri_distances[MIN_OUTER_POINTS] + tri_distances[n] for n in tri_distances], color='blue', marker='o', label='Tri Distance')
        ax.plot(num_points, [global_distances[n] if n == MIN_OUTER_POINTS else global_distances[MIN_OUTER_POINTS] + global_distances[n] for n in global_distances], color='green', marker='o', label='Global Distance')
        plt.legend()

        if show_area:
            # Create secondary y-axis for area proportion
            ax2 = ax.twinx()
            ax2.plot(num_points, [area_proportions[n] for n in area_proportions], color='red', marker='x', linestyle='--', label='Area Proportion')

        # Add shared axis labels
        fig.text(0.5, 0.04, 'Number of Chosen Points', ha='center', fontsize=14)
        # fig.text(0.03, 0.5, f'Triangulation Distance (proportion out of {HER2_PATCH_SIZE}px patch)', va='center',
        #          ha='center', color='blue', rotation='vertical', fontsize=14)
        fig.text(0.06, 0.5, f'Global Distance (proportion out of {HER2_PATCH_SIZE}px patch)', va='center', ha='center',
                 color='green',
                 rotation='vertical', fontsize=14)
        if show_area:
            fig.text(0.96, 0.5, 'Area Proportion', va='center', ha='center', color='red', rotation='vertical', fontsize=14)

        plt.legend()
        plt.show()

def calculate_area(points: np.array):
    """Calculate the area of the convex hull formed by a set of points."""
    if len(points) < 3:
        return 0  # No area can be formed with less than 3 points
    hull = ConvexHull(points)
    return hull.volume  # For 2D, hull.volume is the area of the convex hull


def get_inner_hull_point_mask(points: np.array):
    # Calculate the convex hull
    hull = ConvexHull(points)

    # Find indices of points on the hull
    hull_indices = hull.vertices

    # Create a mask for points that are *not* on the convex hull
    inside_points_mask = np.ones(len(points), dtype=bool)
    inside_points_mask[hull_indices] = False

    return inside_points_mask


def top_k(arr, k):
    """Given a list of distances, return k largest distances """
    return np.argpartition(arr, -k)[-k:]


def diversity_sampling(chosen_points, unchosen_points, n_select=1):
    """ Perform Diversity sampling - choose samples which are furthest from any labeled sample"""
    chosen_area = None

    # Can't calculate area
    if chosen_points.shape[0] < 2:
        # Calculate distances between unlabeled and labeled samples
        distances = pairwise_distances(chosen_points, unchosen_points)

        # Find the minimum distance for each unlabeled sample
        min_distances = distances.min(axis=0)

        # Select the top n_select samples with the largest minimum distances
        top_indices = top_k(min_distances, min(n_select, len(unchosen_points)))

    else:
        areas = np.array([calculate_area(np.vstack((chosen_points, unchosen_points[i]))) for i in range(unchosen_points.shape[0])])
        top_indices = top_k(areas, min(n_select, len(unchosen_points)))
        chosen_area = areas[top_indices]

    return top_indices, chosen_area


def rotate_back_annotation_img(im_Her2: np.array, im_Her2_copy: np.array, display: bool = False):
    rotation_imgs_diffs = []

    for _ in range(4):
        if im_Her2.shape == im_Her2_copy.shape:
            rotation_imgs_diffs.append((im_Her2, np.sum((im_Her2 - im_Her2_copy) ** 2)))

        im_Her2 = cv2.rotate(im_Her2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if len(rotation_imgs_diffs) == 0:
        return -1
    correct_rotation_index = np.argmin([diff for img, diff in rotation_imgs_diffs])
    im_Her2 = rotation_imgs_diffs[correct_rotation_index][0]
    if display:
        display_image(im_Her2, "Her2 Image rotated back")

    return im_Her2


def match_landmarks(args, slide_ctx):
    # Find global transformation using ICP
    trnsfrm_fitness = 0
    max_dist = math.inf
    curr_icp_thresh = INIT_ICP_MATCH_THRESHOLD
    th_pixels = UPPER_PAIR_DIST_THRESHOLD
    best_trnsfrm = None
    has_duplicates = True

    mappings, max_dists, best_trnsfrms, tirs, trnsltns = [], [], [], [], []
    while (trnsfrm_fitness < FITNESS_THRESHOLD or max_dist > th_pixels or has_duplicates) and curr_icp_thresh > MIN_ICP_MATCH_THRESHOLD:
        curr_icp_thresh -= ICP_MATCH_THRESHOLD_STEP
        best_results = find_and_apply_best_trnsfrm(S_land_HE_3d=slide_ctx.S_land_HE_3d, S_land_Her2_3d=slide_ctx.S_land_Her2_3d,
                                                   icp_threhold=curr_icp_thresh, display=args.display)

        S_land_HE_transformed, trnsfrm_fitness, best_trnsfrm = best_results[0], best_results[1], best_results[3]
        trans_init_rotated, translation = best_results[4], best_results[5]

        # KNN search to map Her2 landmarks to H&E landmarks
        knn = NearestNeighbors(n_neighbors=1)
        # knn.fit(S_land_Her2_transformed[:, :2])
        knn.fit(S_land_HE_transformed[:, :2])
        # HE_Her2_land_mapping = knn.kneighbors(S_land_HE_3d[:, :2], return_distance=False).flatten()
        HE_Her2_land_mapping = knn.kneighbors(slide_ctx.S_land_Her2_3d[:, :2], return_distance=False).flatten()
        mappings.append(HE_Her2_land_mapping)
        best_trnsfrms.append(best_trnsfrm)
        tirs.append(trans_init_rotated)
        trnsltns.append(translation)

        _, counts = np.unique(HE_Her2_land_mapping, return_counts=True)
        has_duplicates = np.any(counts > 1)

        # Calculate the Euclidean distance between corresponding landmarks
        # pairs_dist = np.sqrt(np.sum((S_land_Her2_transformed[HE_Her2_land_mapping] - S_land_HE_3d) ** 2, axis=1))
        pairs_dist = np.sqrt(np.sum((S_land_HE_transformed[HE_Her2_land_mapping] - slide_ctx.S_land_Her2_3d) ** 2, axis=1))
        max_dist = np.max(pairs_dist)
        max_dists.append(max_dist)
        print(f'max pair dist: {max_dist:.3f}')

        # Check if any landmark pair exceeds the threshold
        if max_dist > th_pixels:
            print(f'There is at least one landmark pair with more than {th_pixels} distance')

    min_max_dist_ind = np.argmin(np.array(max_dists))
    HE_Her2_land_mapping, pairs_dist = mappings[min_max_dist_ind], max_dists[min_max_dist_ind]
    best_trnsfrm, trans_init_rotated, translation = best_trnsfrms[min_max_dist_ind], tirs[min_max_dist_ind], trnsltns[min_max_dist_ind]

    return HE_Her2_land_mapping, pairs_dist, best_trnsfrm, trans_init_rotated, translation


def init_dist_dict(args):
    args.global_dist_path = os.path.join(args.root, f'{args.dict_name}.json')

    if os.path.exists(args.global_dist_path):
        with open(args.global_dist_path, 'r') as f:
            global_dist_dict = json.load(f)
    else:
        global_dist_dict = {}

    if args.use_triangulation:
        args.tri_dist_path = os.path.join(args.root, 'tri_one_out_LS.json')
        if os.path.exists(args.tri_dist_path):
            with open(args.tri_dist_path, 'r') as f:
                tri_dist_dict = json.load(f)
        else:
            tri_dist_dict = {}

    args.global_dist_dict = global_dist_dict
    args.tri_dist_dict = tri_dist_dict

def parse_thumb_names(png_images: list):
    # extract the 4-digit prefix (after removing optional "copy_")
    def get_prefix(name):
        clean = name.replace("copy_", "")
        return clean.split("_")[0]

    groups = {}
    for f in png_images:
        prefix = get_prefix(f)
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(f)

    # find the prefix that occurs twice and the one that occurs once
    ihc_prefix = [p for p, items in groups.items() if len(items) == 2][0]
    he_prefix = [p for p, items in groups.items() if len(items) == 1][0]

    ihc_thumbs = groups[ihc_prefix]
    copy_index = 0 if "copy_" in ihc_thumbs[0] else 1
    thumb_Her2_copy = ihc_thumbs[copy_index]
    thumb_Her2 = ihc_thumbs[1 - copy_index]
    thumb_HE = groups[he_prefix][0]

    return thumb_HE, thumb_Her2, thumb_Her2_copy


def main():
    args = parse_args()
    print(f'args = {args}')
        
    init_dist_dict(args)

    dirs_to_iterate = os.listdir(args.root)

    try:
        for _, dir in tqdm(dirs_to_iterate, total=len(dirs_to_iterate)):
            print(f'dir = {dir}')
            # if dir in tri_dist:
            # if dir in args.global_dist_dict and dir != '19-9595_1_1':  #
            #     print(f'Directory already done')
            #     continue
            # if dir != '19-9595_1_1':
            #     continue
            
            try:
                slide_ctx = prepare_slide_context(args=args, dir=dir)
                process_slide(args=args, slide_ctx=slide_ctx)

            except Exception as e:
                print(f"Failed processing " f"{row['SlideName']}:\n{e}")

                continue

        dist_to_plot = {key: value for key, value in args.global_dist_dict.items() if not isinstance(value, str)}
        issue_dirs = {key: value for key, value in args.global_dist_dict.items() if isinstance(value, str)}
        print(f"Number of dirs: {len(args.global_dist_dict.keys())}")
        print(issue_dirs)
        # show_dist_plot(tri_dist=tri_dist, global_dist=global_dist)
        show_mean_dist_area_prop(global_data_dict=dist_to_plot, avg_dirs=True)
        show_dist_hist(global_data_dict=dist_to_plot)

    except BaseException as e:
        traceback.print_exc()
    finally:
        # with open(tri_dist_path, "w") as f:
        #     json.dump(tri_dist, f, indent=4)
        with open(args.global_dist_path, "w") as f:
            json.dump(args.global_dist_dict, f, indent=4)


if __name__ == "__main__":
    main()
