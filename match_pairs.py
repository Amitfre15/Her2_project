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
from skimage.morphology import disk
from skimage.morphology import binary_opening
from skimage.measure import regionprops, label
import open3d as o3d
from scipy.spatial import Delaunay, ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

HER2_PATCH_SIZE = 26
MIN_OUTER_POINTS = '5'
DESIRED_MPP = 1
SLIDE_PATCH_SIZE = 256

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

def landmark_detection(im_HE: np.array, im_Her2: np.array, color_landmark: np.array, color_threshold: float = 0.001,
                       display: bool = False):
    # For each pixel, measure its distance from color_landmark
    dist_HE = np.sqrt(np.sum((im_HE - color_landmark) ** 2, axis=2))

    dist_HE_b = dist_HE < color_threshold  # % distance < color_threshold should be true for the pixels of the landmarks
    dist_HE_b = binary_opening(dist_HE_b, disk(2))  # remove small noise pixels

    if display:
        display_image(dist_HE_b.astype(np.float64), "Landmarks H&E")

    # Extract centroid of landmarks
    labeled_HE = label(dist_HE_b)  # get all connected components
    regions_HE = regionprops(labeled_HE)
    S_land_HE = np.array([region.centroid for region in regions_HE])

    # Repeat for Her2
    dist_Her2 = np.sqrt(np.sum((im_Her2 - color_landmark) ** 2, axis=2))
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

    best_mean_rigid_dist = math.inf
    best_fitness = 0
    if left_out_points is not None:
        for transformation, fitness, rot_points, rot_lo, trans_init_rotated, translation in filtered_results:
            rot_lo = np.dot(rot_lo, transformation[:3, :3].T) + transformation[:3, 3]
            if relevant_mask is not None:
                mean_rigid_dist = np.mean(np.sqrt(np.sum((left_out_true_pairs - rot_lo)[relevant_mask] ** 2, axis=1)))
            else:
                mean_rigid_dist = np.sqrt(np.sum((left_out_true_pairs - rot_lo) ** 2))
            if mean_rigid_dist < best_mean_rigid_dist:
                best_transformation, best_fitness, best_rot_points, best_rot_lo, best_trans_init_rotated, best_translation = (
                    transformation, fitness, rot_points, rot_lo, trans_init_rotated, translation)
                best_mean_rigid_dist = mean_rigid_dist
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


def create_rigid_map(PT_HE: np.array, PT_Her2: np.array, im_HE: np.array, im_Her2: np.array, dir_name: str, base_path: str = None,
                     display: bool = True):
    # Define bounding box and pixels in H&E
    min_y, min_x = np.floor(np.min(PT_HE, axis=0)).astype(int)
    max_y, max_x = np.ceil(np.max(PT_HE, axis=0)).astype(int)
    y, x = np.meshgrid(np.arange(min_y, max_y), np.arange(min_x, max_x))
    pixels = np.vstack([y.ravel(), x.ravel()]).T
    # pixels_to_trnsfrm = pixels // 2
    # pixels_hom = np.hstack([pixels_to_trnsfrm, np.ones((len(pixels_to_trnsfrm), 1))])
    pixels_hom = np.hstack([pixels, np.ones((len(pixels), 1))])

    # Initialize transformed image and mapping image
    im_transformed = np.copy(im_HE)
    im_map = np.zeros_like(im_HE)

    # Create the (x, y) grid for the Her2 image
    im_XY = np.dstack(np.meshgrid(np.arange(im_Her2.shape[1]), np.arange(im_Her2.shape[0])))

    # pixels_transformed_Her2 = np.dot(pixels_hom, trans_init_rotated[:3, :3].T)
    # pixels_transformed_Her2 = pixels_transformed_Her2 + np.hstack((translation, 0))
    # pixels_transformed_Her2 = np.dot(pixels_transformed_Her2, best_trnsfrm[:3, :3].T) + best_trnsfrm[:3, 3]

    # Compute the affine transformation matrix between the two triangles
    A = np.linalg.lstsq(np.hstack([PT_HE, np.ones((len(PT_HE), 1))]), PT_Her2, rcond=None)[0]

    # Apply the transformation to the pixels of the current triangle
    pixels_transformed_Her2 = np.dot(pixels_hom, A)

    # Round and clip the coordinates for image indexing
    pixels_transformed_Her2 = np.round(pixels_transformed_Her2).astype(int)[:, :2]
    pixels_transformed_Her2 = np.clip(pixels_transformed_Her2, 0, np.array(im_Her2.shape[:2]) - 1)

    # Convert pixel coordinates to integer after rounding
    y_transformed = np.round(pixels_transformed_Her2[:, 0]).astype(int)
    x_transformed = np.round(pixels_transformed_Her2[:, 1]).astype(int)

    y_curr = np.round(pixels[:, 0]).astype(int)
    x_curr = np.round(pixels[:, 1]).astype(int)

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
    if display:
        save_path = os.path.join(base_path, dir_name) if base_path else None
        im_downsampled = cv2.resize(im_transformed[max(min_y - 700, 0):min(max_y + 700, im_transformed.shape[0]), :], (0, 0), fx=0.25, fy=0.25)  # 25% size
        display_image(im_downsampled,
                      f"Transformed Her2 on H&E - {dir_name}", save_path=save_path)

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
    # tri_dist, rigid_dist = {}, {}

    # im_map_tmp = triangulate_and_create_map(PT_HE=PT_HE_chosen, PT_Her2=PT_Her2_chosen, im_HE=im_HE,
    #                                         im_Her2=im_Her2, display=False)

    # mapped_metric_point = im_map_tmp[(metric_point[0]).astype(int), (metric_point[1]).astype(int)][:2][
    #                       ::-1]  # reverse the coordinate order
    # if not np.allclose(mapped_metric_point, np.zeros_like(mapped_metric_point.shape), atol=1e-8):
    #     metric_tri_dist = np.sqrt(np.sum((Her2_metric_point - mapped_metric_point) ** 2))
    #     tri_dist[PT_HE_chosen.shape[0]] = metric_tri_dist, chosen_area / max_area

    # mapped_pairs = im_map_tmp[(PT_HE_unchosen[:, 0]).astype(int), (PT_HE_unchosen[:, 1]).astype(int)][:, :2][:, ::-1]  # reverse the coordinate order
    # # Keep only rows different from [0.0, 0.0]
    # relevant_mask = ~np.all(mapped_pairs == [0.0, 0.0], axis=1)
    # if not any(relevant_mask):
    #     return None, None
    # mapped_pairs = mapped_pairs[relevant_mask]
    #
    # mean_tri_dist = np.mean(np.sqrt(np.sum((PT_Her2_unchosen[relevant_mask] - mapped_pairs) ** 2, axis=1)))
    # tri_dist[PT_HE_chosen.shape[0]] = mean_tri_dist  # , chosen_area / max_area

    # Rigid evaluation
    trnsfrm_fitness = 0
    icp_trials = 0
    curr_thresh = icp_threshold
    mean_rigid_dist = math.inf

    mean_dists = []
    while (mean_rigid_dist > 100 or trnsfrm_fitness < 0.9) and curr_thresh > 0:
        best_results = find_and_apply_best_trnsfrm(S_land_HE_3d=HE_3d_chosen, S_land_Her2_3d=Her2_3d_chosen,
                                                                              # left_out_points=Her2_3d_unchosen,
                                                                              # left_out_points=Her2_metric_point_3d,
                                                                              left_out_points=metric_point_3d,
                                                                              # left_out_true_pairs=HE_3d_unchosen,
                                                                              # left_out_true_pairs=metric_point_3d,
                                                                              left_out_true_pairs=Her2_metric_point_3d,
                                                                              icp_threhold=curr_thresh,
                                                                              display=False,
                                                                              # relevant_mask=relevant_mask
                                                                              )
        trnsfrm_fitness, transformed_lo = best_results[1], best_results[2]

        # mean_rigid_dist = np.mean(np.sqrt(np.sum((HE_3d_unchosen - transformed_lo)[relevant_mask] ** 2, axis=1)))
        # mean_rigid_dist = np.sqrt(np.sum((metric_point_3d - transformed_lo) ** 2))
        mean_rigid_dist = np.sqrt(np.sum((Her2_metric_point_3d - transformed_lo) ** 2))
        mean_dists.append(mean_rigid_dist)
        icp_trials += 1
        curr_thresh = icp_threshold - icp_trials * 50
    mean_rigid_dist = min(mean_dists)

    # return mean_tri_dist, mean_rigid_dist
    return None, mean_rigid_dist


def evaluate_w_sub_group(PT_HE: np.array, PT_Her2: np.array, im_HE: np.array, im_Her2: np.array, S_land_HE_3d: np.array,
                         S_land_Her2_3d: np.array, HE_Her2_land_mapping: np.array,
                         inner_points_indices: np.array, metric_point_index: np.array = None):
    tri_dist, rigid_dist = {}, {}
    # S_land_Her2_3d_mapped = S_land_Her2_3d[HE_Her2_land_mapping]
    S_land_HE_3d = S_land_HE_3d[HE_Her2_land_mapping]

    metric_point, metric_point_3d, Her2_metric_point_3d = np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1))
    # init metric point for both triangulation and rigid transform
    metric_point, Her2_metric_point = PT_HE[metric_point_index].reshape(1, -1), PT_Her2[metric_point_index].reshape(1, -1)
    PT_HE, PT_Her2 = np.delete(PT_HE, metric_point_index, axis=0), np.delete(PT_Her2, metric_point_index, axis=0)
    # metric_point_3d, Her2_metric_point_3d = S_land_HE_3d[metric_point_index].reshape(1, -1), S_land_Her2_3d_mapped[metric_point_index].reshape(1, -1)
    # S_land_HE_3d, S_land_Her2_3d_mapped = np.delete(S_land_HE_3d, metric_point_index, axis=0), np.delete(S_land_Her2_3d_mapped, metric_point_index, axis=0)
    metric_point_3d, Her2_metric_point_3d = S_land_HE_3d[metric_point_index].reshape(1, -1), S_land_Her2_3d[
        metric_point_index].reshape(1, -1)
    S_land_HE_3d, S_land_Her2_3d_mapped = np.delete(S_land_HE_3d, metric_point_index, axis=0), np.delete(
        S_land_Her2_3d, metric_point_index, axis=0)
    
    # init other points
    PT_HE_chosen, PT_HE_unchosen = PT_HE[0].reshape(1, -1), PT_HE[1:]
    PT_Her2_chosen, PT_Her2_unchosen = PT_Her2[0].reshape(1, -1), PT_Her2[1:]
    HE_3d_chosen, HE_3d_unchosen = S_land_HE_3d[0].reshape(1, -1), S_land_HE_3d[1:]
    Her2_3d_chosen, Her2_3d_unchosen = S_land_Her2_3d_mapped[0].reshape(1, -1), S_land_Her2_3d_mapped[1:]

    # max_area = calculate_area(points=PT_HE)

    # start with 4 points
    while PT_HE_chosen.shape[0] < 4:
        top_index, chosen_area = diversity_sampling(chosen_points=PT_HE_chosen, unchosen_points=PT_HE_unchosen)
        # for ind, point in enumerate(PT_HE):
            # if np.allclose(point, PT_HE_unchosen[top_index], atol=1e-8) and ind in inner_points_indices:
            #     inner_points_indices.pop(top_index)
        arrays_tuple = choose_top_index(PT_HE_chosen=PT_HE_chosen, PT_HE_unchosen=PT_HE_unchosen,
                                        PT_Her2_chosen=PT_Her2_chosen,
                                        PT_Her2_unchosen=PT_Her2_unchosen, HE_3d_chosen=HE_3d_chosen,
                                        HE_3d_unchosen=HE_3d_unchosen,
                                        Her2_3d_chosen=Her2_3d_chosen, Her2_3d_unchosen=Her2_3d_unchosen,
                                        top_index=top_index)
        PT_HE_chosen, PT_HE_unchosen, PT_Her2_chosen, PT_Her2_unchosen, HE_3d_chosen, HE_3d_unchosen, Her2_3d_chosen, Her2_3d_unchosen = arrays_tuple

    # inner_points = PT_HE[inner_points_indices]
    # # Create a mask for elements in PT_HE_unchosen that are NOT in inner_points
    # outer_unchosen_mask = np.all(~np.isin(PT_HE_unchosen, inner_points), axis=1)
    #
    # # Use the mask to filter out elements
    # outer_unchosen_indices = np.where(outer_unchosen_mask)[0]
    # if len(outer_unchosen_indices) == 0:
    #     print('No more outer landmarks to choose')
    #     tri_dist = 'No more outer landmarks to choose'
    # else:
    # for j in range(1, outer_unchosen_indices.shape[0] + 1):
    unchosen_indices = np.arange(PT_HE_unchosen.shape[0])
    for j in range(1, PT_HE_unchosen.shape[0] + 1):
        tmp_len = PT_HE_chosen.shape[0] + j
        tri_dist[str(tmp_len)], rigid_dist[str(tmp_len)] = [], []
        # Generate all possible index groups of size j
        # index_groups = list(itertools.combinations(outer_unchosen_indices, j))[:20]
        possible_ind_groups = list(itertools.combinations(unchosen_indices, j))
        index_groups = random.sample(possible_ind_groups, min(len(possible_ind_groups), 10))
        for ind_gr in index_groups:
            arrays_tuple = choose_top_index(PT_HE_chosen=PT_HE_chosen, PT_HE_unchosen=PT_HE_unchosen,
                                            PT_Her2_chosen=PT_Her2_chosen,
                                            PT_Her2_unchosen=PT_Her2_unchosen, HE_3d_chosen=HE_3d_chosen,
                                            HE_3d_unchosen=HE_3d_unchosen,
                                            Her2_3d_chosen=Her2_3d_chosen, Her2_3d_unchosen=Her2_3d_unchosen,
                                            top_index=np.array(ind_gr))
            PT_HE_chosen_tmp, PT_HE_unchosen_tmp, PT_Her2_chosen_tmp, PT_Her2_unchosen_tmp, HE_3d_chosen_tmp, HE_3d_unchosen_tmp, Her2_3d_chosen_tmp, Her2_3d_unchosen_tmp = arrays_tuple

            # mean_tri_dist, mean_rigid_dist = transform_sub_group(PT_HE_chosen=PT_HE_chosen_tmp, PT_HE_unchosen=PT_HE_unchosen_tmp,
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

            mean_rigid_dist = np.sqrt(np.sum((metric_transformed_to_Her2[:, :2] - Her2_metric_point) ** 2))

            if mean_rigid_dist is not None:
                # tri_dist[str(tmp_len)].append(mean_tri_dist)
                rigid_dist[str(tmp_len)].append(mean_rigid_dist)
    # for k in tri_dist.keys():
    for k in rigid_dist.keys():
        # tri_dist[k] = np.mean(tri_dist.get(k))
        rigid_dist[k] = np.mean(rigid_dist.get(k))

    return tri_dist, rigid_dist


def show_dist_plot(tri_dist: list, rigid_dist: list):
    tri_dist, rigid_dist = np.array(tri_dist), np.array(rigid_dist)
    rigid_minus_tri = rigid_dist - tri_dist
    print(f'tri_dist = {tri_dist}\ntri_dist max = {max(tri_dist)}\ntri_dist avg = {np.mean(tri_dist)}')
    print(f'rigid_dist = {rigid_dist}\nrigid_dist max = {max(rigid_dist)}\nrigid_dist avg = {np.mean(rigid_dist)}')

    # Compute mean and max for each list
    tri_mean, tri_max = np.mean(tri_dist), np.max(tri_dist)
    rigid_mean, rigid_max = np.mean(rigid_dist), np.max(rigid_dist)
    diff_mean = np.mean(rigid_minus_tri)

    # Create histograms
    plt.figure(figsize=(10, 6))
    # plt.plot(tri_dist, marker='o', linestyle='-', label='Triangulation distances', color='blue')
    # plt.plot(rigid_dist, marker='o', linestyle='-', label='Rigid Transformation distances', color='orange')
    plt.plot(rigid_minus_tri, marker='o', linestyle='-', label='Rigid minus Triangulation', color='orange')

    # Plot mean and max lines for tri_dist
    # plt.axhline(tri_mean, color='blue', linestyle='dashed', linewidth=2, label='Mean (tri_dist)')
    # plt.axhline(tri_max, color='blue', linestyle='solid', linewidth=2, label='Max (tri_dist)')

    # Plot mean and max lines for rigid_dist
    # plt.axhline(rigid_mean, color='orange', linestyle='dashed', linewidth=2, label='Mean (rigid_dist)')
    # plt.axhline(rigid_max, color='orange', linestyle='solid', linewidth=2, label='Max (rigid_dist)')
    plt.axhline(diff_mean, color='orange', linestyle='solid', linewidth=2, label='Mean (rigid_dist - tri dist)')

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel(f'Distance diff as a percentage out of patch size ({HER2_PATCH_SIZE}px)')
    plt.title('Line plots with Mean line')
    plt.legend()

    # Show plot
    plt.show()


def show_dist_hist(rigid_dist: dict):
    final_tri_dists, final_rigid_dists = {}, {}
    for k in rigid_dist.keys():
        # final_tri_dists[k] = list(tri_dist[k].values())[-1]
        if any(np.array(list(rigid_dist[k].values())) > 100):
            print(f'k = {k}, v.values() = {rigid_dist[k].values()}')
        final_rigid_dists[k] = list(rigid_dist[k].values())[-1] if len(rigid_dist[k].values()) > 0 else list(rigid_dist[k].values())


    # np_final_tri = np.array(list(final_tri_dists.values())) / HER2_PATCH_SIZE
    np_final_rigid = np.array(list(final_rigid_dists.values())) / HER2_PATCH_SIZE

    # tri_dist, rigid_dist = np.array(tri_dist), np.array(rigid_dist)
    # print(f'tri_dist = {tri_dist}\ntri_dist max = {max(tri_dist)}\ntri_dist avg = {np.mean(tri_dist)}')
    # print(f'rigid_dist = {rigid_dist}\nrigid_dist max = {max(rigid_dist)}\nrigid_dist avg = {np.mean(rigid_dist)}')

    # Compute mean and max for each list
    # tri_mean, tri_max = np.mean(np_final_tri), np.max(np_final_tri)
    rigid_mean, rigid_max = np.mean(np_final_rigid), np.max(np_final_rigid)
    print(f'rigid_mean = {rigid_mean}')

    # Double font sizes globally
    plt.rcParams.update({'font.size': 16})

    # Create histograms
    plt.figure(figsize=(10, 6))

    # plt.hist(np_final_tri, bins=10, alpha=0.5, label='Triangulation distances', color='blue')
    plt.hist(np_final_rigid, bins=10, alpha=0.5, label='Rigid transformation distances', color='orange')

    # Plot mean and max lines for tri_dist
    # plt.axvline(tri_mean, color='blue', linestyle='dashed', linewidth=2, label='Mean (tri_dist)')
    # plt.axvline(tri_max, color='blue', linestyle='solid', linewidth=2, label='Max (tri_dist)')

    # Plot mean and max lines for rigid_dist
    plt.axvline(rigid_mean, color='orange', linestyle='dashed', linewidth=2, label='Mean line (rigid distances)')
    # plt.axvline(rigid_max, color='orange', linestyle='solid', linewidth=2, label='Max (rigid_dist)')

    # Add labels and legend
    plt.xlabel(f'Distance (proportion out of MPP={DESIRED_MPP} {SLIDE_PATCH_SIZE}px patch)')
    plt.ylabel('Frequency')
    plt.title('Matched landmarks distance histogram with mean line')
    plt.legend()

    # Show plot
    plt.show()


def show_mean_dist_area_prop(rigid_data_dict: dict, avg_dirs: bool = False, show_area: bool = False):
    if not avg_dirs:
        # Calculate rows needed for two columns
        num_dirs = len(rigid_data_dict)
        ncols = 2
        nrows = math.ceil(num_dirs / ncols)

        # Setup the figure with two columns of subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows), sharex=True)
        fig.suptitle('Distance by Number of Chosen Points', fontsize=18)

        # Flatten axes array for easy indexing, in case there's only one row
        axes = axes.flatten()

        # Plot each directory's data
        # for i, (directory, values) in enumerate(tri_data_dict.items()):
        for i, (directory, values) in enumerate(rigid_data_dict.items()):
            rigid_values = rigid_data_dict[directory]
            ax = axes[i]

            # Extract data for plotting
            num_points = list(values.keys())
            tri_distances = [values[n][0] / HER2_PATCH_SIZE for n in num_points]  # proportion out of 13px patch
            area_proportions = [values[n][1] for n in num_points]
            rigid_distances = [rigid_values[n] / HER2_PATCH_SIZE for n in num_points]  # proportion out of 13px patch

            # Plot distance on the left y-axis
            ax.plot(num_points, tri_distances, color='blue', marker='o', label='Tri Distance')
            ax.plot(num_points, rigid_distances, color='green', marker='o', label='Rigid Distance')

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
        fig.text(0.06, 0.5, f'Rigid Distance (proportion out of {HER2_PATCH_SIZE}px patch)', va='center', ha='center', color='green',
                 rotation='vertical', fontsize=14)
        fig.text(0.96, 0.5, 'Area Proportion', va='center', ha='center', color='red', rotation='vertical', fontsize=14)

        plt.show()

    else:
        rigid_distances, tri_distances = {}, {}
        if show_area:
            area_proportions = {}
        # Setup the figure with two columns of subplots
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Distance by Number of Chosen Points', fontsize=18)

        # Plot each directory's data
        # for i, (directory, values) in enumerate(tri_data_dict.items()):
        for i, (directory, values) in enumerate(rigid_data_dict.items()):
            # rigid_values = rigid_data_dict[directory]

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

                if n not in rigid_distances:
                    rigid_distances[n] = []
                rigid_distances[n].append(values[n] / HER2_PATCH_SIZE) if n == MIN_OUTER_POINTS \
                        else rigid_distances[n].append((values[n] - values[MIN_OUTER_POINTS]) / HER2_PATCH_SIZE)

        # num_points = list(tri_distances.keys())
        num_points = list(rigid_distances.keys())
        # for k in tri_distances.keys():
        for k in rigid_distances.keys():
            # tri_distances[k] = np.mean(tri_distances.get(k))
            rigid_distances[k] = np.mean(rigid_distances.get(k))
            if show_area:
                area_proportions[k] = np.mean(area_proportions.get(k))

        # Adjust layout to fit everything
        plt.subplots_adjust(top=0.92, hspace=0.8, wspace=0.3)

        # Plot distance on the left y-axis
        # ax.plot(num_points, [tri_distances[n] if n == MIN_OUTER_POINTS else tri_distances[MIN_OUTER_POINTS] + tri_distances[n] for n in tri_distances], color='blue', marker='o', label='Tri Distance')
        ax.plot(num_points, [rigid_distances[n] if n == MIN_OUTER_POINTS else rigid_distances[MIN_OUTER_POINTS] + rigid_distances[n] for n in rigid_distances], color='green', marker='o', label='Rigid Distance')
        plt.legend()

        if show_area:
            # Create secondary y-axis for area proportion
            ax2 = ax.twinx()
            ax2.plot(num_points, [area_proportions[n] for n in area_proportions], color='red', marker='x', linestyle='--', label='Area Proportion')

        # Add shared axis labels
        fig.text(0.5, 0.04, 'Number of Chosen Points', ha='center', fontsize=14)
        # fig.text(0.03, 0.5, f'Triangulation Distance (proportion out of {HER2_PATCH_SIZE}px patch)', va='center',
        #          ha='center', color='blue', rotation='vertical', fontsize=14)
        fig.text(0.06, 0.5, f'Rigid Distance (proportion out of {HER2_PATCH_SIZE}px patch)', va='center', ha='center',
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

    for i in range(4):
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


def match_landmarks(S_land_HE_3d: np.array, S_land_Her2_3d: np.array, display: bool = False):
    # Find rigid transformation using ICP
    trnsfrm_fitness = 0
    max_dist = math.inf
    curr_icp_thresh = 500
    th_pixels = 80
    best_trnsfrm = None
    has_duplicates = True

    mappings, max_dists, best_trnsfrms, tirs, trnsltns = [], [], [], [], []
    while (trnsfrm_fitness < 0.9 or max_dist > th_pixels or has_duplicates) and curr_icp_thresh > 50:
        curr_icp_thresh -= 50
        best_results = find_and_apply_best_trnsfrm(S_land_HE_3d=S_land_HE_3d, S_land_Her2_3d=S_land_Her2_3d,
                                                   icp_threhold=curr_icp_thresh, display=display)

        S_land_HE_transformed, trnsfrm_fitness, best_trnsfrm = best_results[0], best_results[1], best_results[3]
        trans_init_rotated, translation = best_results[4], best_results[5]

        # KNN search to map Her2 landmarks to H&E landmarks
        knn = NearestNeighbors(n_neighbors=1)
        # knn.fit(S_land_Her2_transformed[:, :2])
        knn.fit(S_land_HE_transformed[:, :2])
        # HE_Her2_land_mapping = knn.kneighbors(S_land_HE_3d[:, :2], return_distance=False).flatten()
        HE_Her2_land_mapping = knn.kneighbors(S_land_Her2_3d[:, :2], return_distance=False).flatten()
        mappings.append(HE_Her2_land_mapping)
        best_trnsfrms.append(best_trnsfrm)
        tirs.append(trans_init_rotated)
        trnsltns.append(translation)

        _, counts = np.unique(HE_Her2_land_mapping, return_counts=True)
        has_duplicates = np.any(counts > 1)

        # Calculate the Euclidean distance between corresponding landmarks
        # pairs_dist = np.sqrt(np.sum((S_land_Her2_transformed[HE_Her2_land_mapping] - S_land_HE_3d) ** 2, axis=1))
        pairs_dist = np.sqrt(np.sum((S_land_HE_transformed[HE_Her2_land_mapping] - S_land_Her2_3d) ** 2, axis=1))
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


def init_dist_dict(dict_json_path: str):
    if os.path.exists(dict_json_path):
        with open(dict_json_path, 'r') as f:
            dist_dict = json.load(f)
    else:
        dist_dict = {}

    return dist_dict


def main():
    # Argument parser
    # parser = argparse.ArgumentParser(description="Search and replace file names and content based on Excel mapping.")
    # parser.add_argument('-r', '--root', required=True, help='Root directory to search for files.')
    # parser.add_argument('-m', '--mapping_file', required=True,
    #                     help='Excel file with current_name and changed_name columns.')

    # args = parser.parse_args()
    # print(f'args = {args}')

    display = False

    # marked_folder_path = os.path.join(os.sep, 'SSDStorage', 'Breast', 'Carmel', 'png_thumb_pairs_karin')
    marked_folder_path = os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', 'Marked', 'png_thumb_pairs_karin')
    # marked_folder_path = os.path.join('slides_to_amit2', 'match_thumbs_HE_Her2', 'Marked', 'OneDrive_1_20-01-2025')
    # tri_dist_path = os.path.join(marked_folder_path, 'tri_one_out_LS.json')
    rigid_dist_path = os.path.join(marked_folder_path, 'rigid_one_out_LS.json')


    # tri_dist, rigid_dist = [], []
    # tri_dist, rigid_dist = init_dist_dict(tri_dist_path), init_dist_dict(rigid_dist_path)
    rigid_dist = init_dist_dict(rigid_dist_path)

    try:
        for ind, dir in enumerate(os.listdir(marked_folder_path)):
            print(f'dir = {dir}')
            # if dir in tri_dist:
            if dir in rigid_dist and dir != '21-6755_1_1':  #
                print(f'Directory already done')
                continue
            if dir != '21-6755_1_1':
                continue

            # Initialize paths
            folder = os.path.join(marked_folder_path, dir)
            if not os.path.isdir(folder):
                print(f'{dir} is not a directory')
                continue

            png_images = list(filter(lambda x: x.endswith('.png') and not x.startswith('map') and not x.startswith('Transformed'), os.listdir(folder)))
            if len(png_images) < 2:
                # tri_dist[dir] = 'Directory does not contain enough images'
                rigid_dist[dir] = 'Directory does not contain enough images'
                print('Directory does not contain enough images')
                continue

            # order the thumbs by the last letter in their name
            ordered_thumbs = sorted(png_images, key=lambda x: x.split('_')[-1][0])
            if len(ordered_thumbs) == 3:
                thumb_HE, thumb_Her2, thumb_Her2_copy = ordered_thumbs[0], ordered_thumbs[1], ordered_thumbs[2]
            else:
                thumb_HE, thumb_Her2, thumb_Her2_copy = ordered_thumbs[0], ordered_thumbs[1], ordered_thumbs[1]

            im_HE, im_Her2, im_Her2_copy, output_mapping_file = load_and_display_thumbs(folder_path=folder,
                                                                                        HE_thumb_name=thumb_HE,
                                                                                        Her2_thumb_name=thumb_Her2,
                                                                                        Her2_thumb_copy_name=thumb_Her2_copy,
                                                                                        display=False)

            im_Her2 = rotate_back_annotation_img(im_Her2=im_Her2, im_Her2_copy=im_Her2_copy, display=False)
            if type(im_Her2) is int:
                continue

            # Find landmarks in H&E thumb
            # color_landmark = np.array([0.0156863, 0.2, 1])  # (4, 51, 255)
            color_landmark = np.array([0.0, 0.470588, 0.843137]).astype(np.float32)  # (0, 120, 215) - Photos default
            # color_landmark = np.array([0.0, 0.0, 0.0]) if ind > 17 else np.array([0.0156863, 0.2, 1])  # (4, 51, 255)
            try:
                if dir in rigid_dist:
                    rigid_dist.pop(dir)
                S_land_HE, S_land_Her2 = landmark_detection(im_HE=im_HE, im_Her2=im_Her2, color_landmark=color_landmark, display=False)
                if len(S_land_HE) == 0:  # try other landmark color (black)
                    S_land_HE, S_land_Her2 = landmark_detection(im_HE=im_HE, im_Her2=im_Her2, color_landmark=np.array([0.0, 0.0, 0.0]))
            except ValueError as e:
                rigid_dist[dir] = str(e)
                print(e)
                continue

            if len(S_land_HE) < 6:
                rigid_dist[dir] = f'Insufficient landmark count: {len(S_land_HE)}'
                print(f'Insufficient landmark count: {len(S_land_HE)}')
                continue

            # pts_moving = S_land_Her2
            pts_moving = S_land_HE / 2
            # pts_fixed = S_land_HE / 2
            pts_fixed = S_land_Her2

            # S_land_HE_3d = np.hstack([pts_fixed, np.zeros((len(S_land_HE), 1))])  # For point cloud
            S_land_HE_3d = np.hstack([pts_moving, np.zeros((len(S_land_HE), 1))])  # For point cloud
            # S_land_Her2_3d = np.hstack([pts_moving, np.zeros((len(S_land_Her2), 1))])
            S_land_Her2_3d = np.hstack([pts_fixed, np.zeros((len(S_land_Her2), 1))])
            if display:
                plot_landmarks(S_land_HE_3d[:, [1, 0, 2]], S_land_Her2_3d[:, [1, 0, 2]],
                               f'Initial H&E Landmarks - {dir}', 'Her2 Landmarks')

            HE_Her2_land_mapping, pairs_dist, best_trnsfrm, trans_init_rotated, translation = match_landmarks(S_land_HE_3d=S_land_HE_3d,
                                                                                                              S_land_Her2_3d=S_land_Her2_3d,
                                                                                                              display=True)
            # Check for duplicate elements
            _, counts = np.unique(HE_Her2_land_mapping, return_counts=True)
            has_duplicates = np.any(counts > 1)
            if has_duplicates:
                rigid_dist[dir] = f"HE_Her2_land_mapping has correspondence overlap: {HE_Her2_land_mapping}"
                print(f"HE_Her2_land_mapping has correspondence overlap: {HE_Her2_land_mapping}")
                continue

            # Mapping for H&E and Her2
            PT_HE = S_land_HE[HE_Her2_land_mapping]
            # PT_HE = pts_moving
            # PT_Her2 = S_land_Her2[HE_Her2_land_mapping]
            PT_Her2 = S_land_Her2

            inner_points_mask = get_inner_hull_point_mask(points=PT_HE)

            for i in np.arange(PT_HE.shape[0]):
                metric_point_index = i
                inner_points_indices = np.where(inner_points_mask)[0]
                curr_tri_dist, curr_rigid_dist = evaluate_w_sub_group(PT_HE=PT_HE,  # PT_HE=pts_moving,
                                                                      PT_Her2=PT_Her2, im_HE=im_HE,
                                                                      im_Her2=im_Her2, S_land_HE_3d=S_land_HE_3d,
                                                                      S_land_Her2_3d=S_land_Her2_3d,
                                                                      HE_Her2_land_mapping=HE_Her2_land_mapping,
                                                                      inner_points_indices=inner_points_indices,
                                                                      metric_point_index=metric_point_index)
                if dir not in rigid_dist:
                    # tri_dist[dir] = curr_tri_dist
                    rigid_dist[dir] = curr_rigid_dist
                else:
                    for k in rigid_dist[dir].keys():
                        rigid_dist[dir][k] = np.hstack((rigid_dist[dir][k], curr_rigid_dist[k]))

            for k in rigid_dist[dir].keys():
                rigid_dist[dir][k] = np.mean(rigid_dist[dir][k])

            im_map = create_rigid_map(PT_HE=PT_HE, PT_Her2=PT_Her2, im_HE=im_HE, im_Her2=im_Her2, dir_name=dir, base_path=marked_folder_path)
            # im_map = triangulate_and_create_map(PT_HE=PT_HE, PT_Her2=PT_Her2, im_HE=im_HE, im_Her2=im_Her2)
            #
            # Convert the image to 8-bit unsigned integers
            img_16bit = im_map.astype(np.uint16)

            # Now we can use cv2.cvtColor
            bgr_im_map = cv2.cvtColor(img_16bit, cv2.COLOR_RGB2BGR)
            # display_image(img=bgr_im_map)
            cv2.imwrite(output_mapping_file, bgr_im_map)
            print(f'saved mapping file to {output_mapping_file}')

            # Verify the saved file
            im_map_uint16_rec = cv2.imread(output_mapping_file, cv2.IMREAD_UNCHANGED)
            if not np.array_equal(bgr_im_map, im_map_uint16_rec):
                print("Warning: The saved mapping file does not match the original.")

        dist_to_plot = {key: value for key, value in rigid_dist.items() if not isinstance(value, str)}
        issue_dirs = {key: value for key, value in rigid_dist.items() if isinstance(value, str)}
        print(f"Number of dirs: {len(rigid_dist.keys())}")
        print(issue_dirs)
        # show_dist_plot(tri_dist=tri_dist, rigid_dist=rigid_dist)
        show_mean_dist_area_prop(rigid_data_dict=dist_to_plot, avg_dirs=True)
        show_dist_hist(rigid_dist=dist_to_plot)

    except BaseException as e:
        traceback.print_exc()
    finally:
        # with open(tri_dist_path, "w") as f:
        #     json.dump(tri_dist, f, indent=4)
        with open(rigid_dist_path, "w") as f:
            json.dump(rigid_dist, f, indent=4)


if __name__ == "__main__":
    main()

    # # Given data as numpy arrays
    # tri_dist = np.array([
    #     10.03883224, 45.20179753, 37.6084809, 24.81791358, 16.70417869, 27.34309731,
    #     26.1130023, 22.81568661, 29.1585703, 15.60626655, 24.4994331, 28.05748068,
    #     23.94054558, 4.21718431, 5.37400052, 10.86893968, 13.54935883, 14.73978681,
    #     11.10065857, 12.83426889, 11.51144108, 23.17967749, 16.35209386, 23.94573381,
    #     16.66999967, 3.34995854, 9.06764701, 10.749677, 36.31005027, 12.26075957,
    #     35.77189377, 36.06582165, 35.20498507, 46.13121303, 25.24654072, 6.89885261,
    #     61.21746723, 76.0274003, 66.0767269, 88.45189102, 7.85632922, 23.3896222,
    #     53.50689765, 11.0053438, 6.90133202, 9.29622238, 56.94048043, 12.80269343,
    #     3.26342862, 36.70829267, 21.67470572, 42.94899763, 56.56314762, 65.22215975,
    #     29.52979096, 42.79278236, 77.84314713, 31.52600338, 6.74948558, 33.74742789,
    #     18.30110767, 7.89469603, 5.85087174, 4.24801716, 25.67314546, 23.42126289,
    #     12.97005097, 2.4267033, 21.73067468, 29.53340858, 22.78067705, 14.3097439,
    #     12.02077074, 25.44163334, 29.38184099, 35.55644574, 11.23375903, 13.94173393,
    #     9.06977585, 24.40935045, 10.08332943, 56.6665009, 25.83667171, 97.73413248,
    #     40.24661269, 29.61484329, 10.59381142, 29.08479183, 16.85033268, 15.55955978,
    #     18.68360185, 14.67712343, 9.96021676, 15.2432864, 48.05856981, 16.45547676,
    #     35.2754763, 39.29417038, 36.50505464, 77.76177104, 29.43803088, 40.17408484,
    #     13.55901075
    # ])
    #
    # rigid_dist = np.array([
    #     3.78385135, 50.56527605, 22.40750206, 23.04012448, 14.28585017,
    #     19.5278628, 4.69962848, 17.98866511, 10.94150182, 13.56472502,
    #     15.67137755, 9.61621343, 26.57650013, 12.55367715, 5.5109596,
    #     23.3924256, 59.04020882, 44.545065, 17.77278119, 54.94558654,
    #     5.64261532, 13.92565432, 13.31798422, 18.37661341, 22.0625028,
    #     14.39670045, 9.5020172, 1.04151958, 22.91021481, 25.97685311,
    #     33.91587594, 38.85748956, 26.03558135, 51.02857928, 39.03474551,
    #     31.98495344, 39.36492668, 97.49393235, 83.43399, 50.59011517,
    #     6.15042138, 15.47073101, 7.36677328, 45.76023971, 51.60343171,
    #     19.82823302, 25.18934756, 7.13439341, 6.12962674, 96.97989263,
    #     95.2590565, 64.40454616, 15.11687392, 29.18388706, 36.38702459,
    #     22.934182, 28.88748333, 48.93935444, 60.98116922, 49.1220076,
    #     41.25754815, 19.45755143, 18.70913121, 7.78272333, 33.46507857,
    #     10.52183465, 2.95256404, 6.79606654, 43.7163289, 37.68818898,
    #     72.91862621, 56.13078184, 8.76027704, 27.91538706, 25.5510089,
    #     44.38074149, 2.73488157, 9.9496242, 12.16916047, 22.01246078,
    #     41.04676196, 53.54936436, 13.94182651, 101.36108441, 21.45753803,
    #     28.18494631, 48.04952857, 13.36783054, 5.45086462, 8.70348268,
    #     26.09070381, 31.46460621, 24.90688994, 22.66567023, 54.6071548,
    #     14.62536511, 18.70278915, 32.58019979, 43.68864578, 94.50784221,
    #     16.48687765, 17.3821217, 5.88521505
    # ])
    # dist_pairs = list(zip(tri_dist / 26, rigid_dist / 26))
    # sorted_dists = sorted(dist_pairs, key=lambda x: x[1] - x[0])
    # sorted_tri_dist, sorted_rigid_dist = [d[0] for d in sorted_dists], [d[1] for d in sorted_dists]
    # show_dist_plot(tri_dist=sorted_tri_dist, rigid_dist=sorted_rigid_dist)
    # # show_dist_hist(tri_dist=tri_dist, rigid_dist=rigid_dist)

    # ******** area trial ********
    # data_dict = {'17-8750_2_10': {4: (22.02441207279937, np.array([0.90462975])), 5: (19.739386002997833, np.array([0.9548438])),
    #                   6: (19.54238162387375, np.array([0.98076832])), 7: (19.54238162387375, np.array([1.])),
    #                   8: (25.859610693054993, np.array([1.])), 9: (26.50624762672625, np.array([1.])),
    #                   10: (10.03883223969227, np.array([1.]))},
    #  '17-9612_1_8': {4: (29.011326231390722, np.array([0.69941711])), 5: (31.27536426162592, np.array([0.81000723])),
    #                  6: (31.27536426162592, np.array([0.91439682])), 7: (22.744527822896345, np.array([0.96499244])),
    #                  8: (22.744527822896345, np.array([0.97828497])), 9: (22.744527822896345, np.array([0.98658425])),
    #                  10: (22.744527822896345, np.array([0.99274814])), 11: (21.76150063275847, np.array([0.9976953])),
    #                  12: (21.76150063275847, np.array([1.])), 13: (16.70417869273491, np.array([1.]))},
    #  '20-10017_1_1': {4: (24.801164375537617, np.array([0.82872286])), 5: (20.795540657013355, np.array([0.9016647])),
    #                   6: (21.875917836996578, np.array([0.96378935])), 7: (21.875917836996578, np.array([1.])),
    #                   8: (20.48031856272022, np.array([1.])), 9: (20.960199468469046, np.array([1.])),
    #                   10: (21.746349504723725, np.array([1.])), 11: (26.748550746816555, np.array([1.])),
    #                   12: (26.11300229557859, np.array([1.]))},
    #  '20-10023_2_10': {4: (24.430693476786313, np.array([0.66445652])), 5: (24.430693476786313, np.array([0.82277207])),
    #                    6: (22.362438055858533, np.array([0.91131877])), 7: (20.851903236541464, np.array([0.99602629])),
    #                    8: (21.151967323679216, np.array([1.])), 9: (24.09275250386426, np.array([1.])),
    #                    10: (17.093469621166506, np.array([1.])), 11: (23.940545583843754, np.array([1.]))},
    #  '20-10023_2_5': {4: (24.32094198517525, np.array([0.6873434])), 5: (28.924201613022166, np.array([0.88718867])),
    #                   6: (28.87439478246895, np.array([0.95869799])), 7: (28.87439478246895, np.array([0.99210916])),
    #                   8: (15.49999400035329, np.array([0.99809361])), 9: (20.49484135390564, np.array([1.])),
    #                   10: (27.776759905173893, np.array([1.])), 11: (21.5476007070303, np.array([1.])),
    #                   12: (13.549358830459804, np.array([1.]))},
    #  '20-10040_1_1': {4: (11.281519491137912, np.array([0.72963072])), 5: (10.444358937689682, np.array([0.8750629])),
    #                   6: (13.961020843774618, np.array([0.93940333])), 7: (13.961020843774618, np.array([0.97152163])),
    #                   8: (19.887824130595416, np.array([0.99281344])), 9: (19.285320533884846, np.array([1.])),
    #                   10: (18.950423284362408, np.array([1.])), 11: (17.819026439767924, np.array([1.])),
    #                   12: (11.511441078786273, np.array([1.]))},
    #  '20-10043_1_1': {4: (15.089850913733002, np.array([0.7894286])), 5: (15.208940791376083, np.array([0.93355799])),
    #                   6: (15.208940791376083, np.array([0.95902296])), 7: (17.46342096438474, np.array([0.98114772])),
    #                   8: (10.78508884035425, np.array([0.99663517])), 9: (10.78508884035425, np.array([1.])),
    #                   10: (8.368549218584466, np.array([1.])), 11: (8.238432649704013, np.array([1.])),
    #                   12: (16.669999666733165, np.array([1.]))},
    #  '20-10103_1_1': {4: (30.87764616215523, np.array([0.82172546])), 5: (25.892620589648484, np.array([0.99756754])),
    #                   6: (29.97411099477561, np.array([1.])), 7: (28.03206580454283, np.array([1.])),
    #                   8: (32.74157720836558, np.array([1.])), 9: (37.18594159792452, np.array([1.])),
    #                   10: (21.793891164906995, np.array([1.])), 11: (36.31005026890629, np.array([1.]))},
    #  '20-10105_1_1': {4: (65.00167902628029, np.array([0.91526374])), 5: (65.00167902628029, np.array([0.97216685])),
    #                   6: (65.00167902628029, np.array([1.])), 7: (75.55127916418328, np.array([1.])),
    #                   8: (60.201983217579794, np.array([1.])), 9: (38.424044815287154, np.array([1.])),
    #                   10: (17.989132444529055, np.array([1.])), 11: (25.246540715317455, np.array([1.]))},
    #  '20-10147_1_1': {4: (30.313248344480474, np.array([1.])), 5: (35.68544774266485, np.array([1.])),
    #                   6: (40.03598318311568, np.array([1.])), 7: (15.195670534354095, np.array([1.])),
    #                   8: (7.856329219418876, np.array([1.]))},
    #  '20-10148_1_1': {4: (11.620557727265588, np.array([0.73330697])), 5: (10.531509992891625, np.array([0.80274007])),
    #                   6: (10.607741140916936, np.array([0.87158508])), 7: (22.16247302467939, np.array([0.90813354])),
    #                   8: (22.16247302467939, np.array([0.94178661])), 9: (33.476464462978655, np.array([0.97196006])),
    #                   10: (32.53456384988357, np.array([0.9955288])), 11: (32.53456384988357, np.array([1.])),
    #                   12: (56.00575583423509, np.array([1.])), 13: (33.11835140514921, np.array([1.])),
    #                   14: (9.296222379575177, np.array([1.]))},
    #  '20-10169_1_1': {4: (43.30017850235388, np.array([0.85697408])), 5: (35.06157214060757, np.array([0.99156116])),
    #                   6: (34.9659104089476, np.array([1.])), 7: (37.2318121865967, np.array([1.])),
    #                   8: (31.725107624282174, np.array([1.])), 9: (23.844910748365006, np.array([1.])),
    #                   10: (29.19149919410155, np.array([1.])), 11: (36.70829266637958, np.array([1.]))},
    #  '20-10170_1_1': {4: (69.62141103002352, np.array([0.71225354])), 5: (70.05624234527711, np.array([0.85379493])),
    #                   6: (51.46601809751449, np.array([0.9471753])), 7: (51.46601809751449, np.array([0.98277889])),
    #                   8: (51.46601809751449, np.array([1.])), 9: (46.83892502764163, np.array([1.])),
    #                   10: (30.61801253780543, np.array([1.])), 11: (48.635924212078756, np.array([1.])),
    #                   12: (42.79278236130744, np.array([1.]))},
    #  '20-10177_1_6': {4: (27.714907244241832, np.array([0.69944925])), 5: (27.11078493385507, np.array([0.95623382])),
    #                   6: (20.27783888298085, np.array([0.9974134])), 7: (20.102456721165613, np.array([1.])),
    #                   8: (12.9715343758245, np.array([1.])), 9: (10.874001018951896, np.array([1.])),
    #                   10: (13.385565656168763, np.array([1.])), 11: (18.301107668592437, np.array([1.]))},
    #  '20-10179_1_9': {4: (40.635841044770395, np.array([0.76253275])), 5: (33.71885841782669, np.array([0.87187045])),
    #                   6: (29.905304002574827, np.array([0.93550289])), 7: (29.905304002574827, np.array([0.97060773])),
    #                   8: (17.712446256502336, np.array([1.])), 9: (16.408775776055922, np.array([1.])),
    #                   10: (13.801596223609982, np.array([1.])), 11: (17.090317666070014, np.array([1.])),
    #                   12: (23.421262894121732, np.array([1.]))},
    #  '20-10208_1_1': {4: (32.8261577021278, np.array([0.77401521])), 5: (32.8261577021278, np.array([0.88811016])),
    #                   6: (30.017232888181343, np.array([0.95620637])), 7: (29.718873244854002, np.array([1.])),
    #                   8: (27.041735804577357, np.array([1.])), 9: (24.22816739972193, np.array([1.])),
    #                   10: (32.7839406124707, np.array([1.])), 11: (22.7806770457251, np.array([1.]))},
    #  '20-10287_1_1': {4: (26.060669846838703, np.array([0.79438625])), 5: (26.060669846838703, np.array([0.89946043])),
    #                   6: (17.7046774250957, np.array([1.])), 7: (23.786757846434462, np.array([1.])),
    #                   8: (18.83586999763443, np.array([1.])), 9: (20.03089808123318, np.array([1.])),
    #                   10: (24.290857887228835, np.array([1.])), 11: (35.556445744241785, np.array([1.]))},
    #  '20-10300_1_7': {4: (59.04037343000895, np.array([0.76009719])), 5: (48.90381265651536, np.array([0.8820201])),
    #                   6: (40.41024205752488, np.array([0.98246825])), 7: (40.83532824577922, np.array([1.])),
    #                   8: (36.0526250103764, np.array([1.])), 9: (45.98084542622942, np.array([1.])),
    #                   10: (41.25158630558168, np.array([1.])), 11: (56.6665008983149, np.array([1.]))},
    #  '20-10303_1_8': {4: (27.658781180164173, np.array([0.79501732])), 5: (14.746698384038085, np.array([0.95960901])),
    #                   6: (13.031926136308561, np.array([1.])), 7: (12.211346271997146, np.array([1.])),
    #                   8: (13.59489738909305, np.array([1.])), 9: (13.942841150863677, np.array([1.])),
    #                   10: (14.870767344526005, np.array([1.])), 11: (17.068955300489037, np.array([1.])),
    #                   12: (19.44741386963799, np.array([1.])), 13: (10.593811423926242, np.array([1.]))},
    #  '20-10313_1_5': {4: (31.98050315005507, np.array([1.])), 5: (37.00078064807614, np.array([1.])),
    #                   6: (27.10710758663553, np.array([1.])), 7: (48.05856980693352, np.array([1.]))},
    #  '20-10313_3_2': {4: (29.760199896221707, np.array([0.74114443])), 5: (30.963666400345453, np.array([0.91128135])),
    #                   6: (35.520238583189766, np.array([0.99006779])), 7: (40.54331733152044, np.array([1.])),
    #                   8: (32.62084525470095, np.array([1.])), 9: (38.902980907438796, np.array([1.])),
    #                   10: (57.13341284292623, np.array([1.])), 11: (36.505054642476125, np.array([1.]))}}

    # ******* best point out trial ********
    # tri_data_dict = {'17-8750_2_10': {4: (8.482973832588268, np.array([0.90462975])), 5: (7.509050875069197, np.array([0.9548438])), 6: (8.482973832588268, np.array([0.98076832])), 7: (8.482973832588268, np.array([1.])), 8: (8.482973832588268, np.array([1.])), 9: (7.810697721108619, np.array([1.]))}, '17-9612_1_8': {4: (23.549268509600665, np.array([0.69941711])), 5: (28.077344570071062, np.array([0.81000723])), 6: (28.077344570071062, np.array([0.91439682])), 7: (16.324555867170137, np.array([0.96499244])), 8: (16.324555867170137, np.array([0.97828497])), 9: (16.324555867170137, np.array([0.98658425])), 10: (16.324555867170137, np.array([0.99274814])), 11: (7.127328852245127, np.array([0.9976953])), 12: (7.127328852245127, np.array([1.]))}, '20-10017_1_1': {4: (25.157283018817722, np.array([0.82872286])), 5: (27.33536577809462, np.array([0.9016647])), 6: (27.33536577809462, np.array([0.96378935])), 7: (27.33536577809462, np.array([1.])), 8: (27.33536577809462, np.array([1.])), 9: (27.33536577809462, np.array([1.])), 10: (26.11300229557859, np.array([1.])), 11: (26.11300229557859, np.array([1.]))}, '20-10023_2_10': {4: (32.53872724110683, np.array([0.66445652])), 5: (32.53872724110683, np.array([0.82277207])), 6: (32.53872724110683, np.array([0.91131877])), 7: (17.990929169331796, np.array([0.99602629])), 8: (19.082291524768028, np.array([1.])), 9: (22.55273627142163, np.array([1.])), 10: (5.374000519376838, np.array([1.]))}, '20-10023_2_5': {5: (41.01576016326157, np.array([0.88718867])), 6: (41.01576016326157, np.array([0.95869799])), 7: (41.01576016326157, np.array([0.99210916])), 8: (8.88826865223327, np.array([0.99809361])), 9: (18.896790687178846, np.array([1.])), 10: (33.04779356386894, np.array([1.])), 11: (11.100658566280925, np.array([1.]))}, '20-10040_1_1': {5: (7.932877277344993, np.array([0.8750629])), 6: (7.932877277344993, np.array([0.93940333])), 7: (7.932877277344993, np.array([0.97152163])), 8: (9.48430939136987, np.array([0.99281344])), 9: (7.074295004527593, np.array([1.])), 10: (7.074295004527593, np.array([1.])), 11: (12.4583753931637, np.array([1.]))}, '20-10043_1_1': {4: (9.067647005823506, np.array([0.7894286])), 5: (8.82546819658235, np.array([0.93355799])), 6: (8.82546819658235, np.array([0.95902296])), 7: (15.680844648451952, np.array([0.98114772])), 8: (5.374838498865606, np.array([0.99663517])), 9: (5.374838498865606, np.array([1.])), 10: (9.672412085697784, np.array([1.])), 11: (10.749676997731239, np.array([1.]))}, '20-10103_1_1': {4: (28.243079667268066, np.array([0.82172546])), 5: (30.929060782542642, np.array([0.99756754])), 6: (21.879221331438597, np.array([1.])), 7: (22.87789692687456, np.array([1.])), 8: (20.512032847519006, np.array([1.])), 9: (23.36454075985639, np.array([1.])), 10: (37.549329216794504, np.array([1.]))}, '20-10105_1_1': {4: (57.46999666350188, np.array([0.91526374])), 5: (57.46999666350188, np.array([0.97216685])), 6: (57.46999666350188, np.array([1.])), 7: (64.01728840645804, np.array([1.])), 8: (34.43314952693386, np.array([1.])), 9: (23.024291016670677, np.array([1.])), 10: (15.738655546381645, np.array([1.]))}, '20-10147_1_1': {4: (37.00858789346944, np.array([1.])), 5: (44.518716888610165, np.array([1.])), 6: (40.97712516150624, np.array([1.])), 7: (7.001718865912546, np.array([1.]))}, '20-10148_1_1': {4: (4.028982523019562, np.array([0.73330697])), 5: (4.028982523019562, np.array([0.80274007])), 6: (4.269976600731851, np.array([0.87158508])), 7: (4.269976600731851, np.array([0.90813354])), 8: (4.269976600731851, np.array([0.94178661])), 9: (9.901800331812879, np.array([0.97196006])), 10: (6.134197879432555, np.array([0.9955288])), 11: (6.134197879432555, np.array([1.])), 12: (3.2634286200023594, np.array([1.])), 13: (3.2634286200023594, np.array([1.]))}, '20-10169_1_1': {4: (55.45894723948463, np.array([0.85697408])), 5: (36.33920701070467, np.array([0.99156116])), 6: (12.531087831680306, np.array([1.])), 7: (33.86954102588526, np.array([1.])), 8: (19.920644663612496, np.array([1.])), 9: (29.529790964416655, np.array([1.])), 10: (29.529790964416655, np.array([1.]))}, '20-10170_1_1': {4: (41.39914116124727, np.array([0.71225354])), 5: (41.39914116124727, np.array([0.85379493])), 6: (41.39914116124727, np.array([0.9471753])), 7: (41.39914116124727, np.array([0.98277889])), 8: (41.39914116124727, np.array([1.])), 9: (41.39914116124727, np.array([1.])), 10: (44.12734098291245, np.array([1.])), 11: (32.93765153876173, np.array([1.]))}, '20-10177_1_6': {4: (25.31760062900212, np.array([0.69944925])), 5: (25.31760062900212, np.array([0.95623382])), 6: (15.855313732376539, np.array([0.9974134])), 7: (15.855313732376539, np.array([1.])), 8: (9.927559678893346, np.array([1.])), 9: (9.927559678893346, np.array([1.])), 10: (4.248017162287008, np.array([1.]))}, '20-10179_1_9': {4: (38.477987935383865, np.array([0.76253275])), 5: (13.186693629901685, np.array([0.87187045])), 6: (12.982894729433587, np.array([0.93550289])), 7: (12.982894729433587, np.array([0.97060773])), 8: (12.982894729433587, np.array([1.])), 9: (15.347819244295035, np.array([1.])), 10: (15.670212364724204, np.array([1.])), 11: (12.97005097222915, np.array([1.]))}, '20-10208_1_1': {4: (3.4513225890965478, np.array([0.77401521])), 5: (3.4513225890965478, np.array([0.88811016])), 6: (9.462065061764152, np.array([0.95620637])), 7: (9.870939020063712, np.array([1.])), 8: (10.255801718239724, np.array([1.])), 9: (12.020770740735019, np.array([1.])), 10: (12.020770740735019, np.array([1.]))}, '20-10287_1_1': {4: (8.174129281995103, np.array([0.79438625])), 5: (8.174129281995103, np.array([0.89946043])), 6: (21.943807218280423, np.array([1.])), 7: (13.96391752673201, np.array([1.])), 8: (20.202322882189346, np.array([1.])), 9: (17.976807165566893, np.array([1.])), 10: (13.025270030215886, np.array([1.]))}, '20-10300_1_7': {6: (22.757028796525763, np.array([0.98246825])), 7: (8.829331387029535, np.array([1.])), 8: (23.105894478377213, np.array([1.])), 9: (25.836671712848467, np.array([1.])), 10: (25.836671712848467, np.array([1.]))}, '20-10303_1_8': {4: (29.55660226301128, np.array([0.79501732])), 5: (10.898708596971339, np.array([0.95960901])), 6: (12.932585322425483, np.array([1.])), 7: (12.932585322425483, np.array([1.])), 8: (18.264521273054164, np.array([1.])), 9: (20.224730622133364, np.array([1.])), 10: (18.957339026084217, np.array([1.])), 11: (18.957339026084217, np.array([1.])), 12: (16.850332683537292, np.array([1.]))}, '20-10313_1_5': {4: (2.2160391234793853, np.array([1.])), 5: (15.929665588671398, np.array([1.])), 6: (13.467364409188802, np.array([1.]))}, '20-10313_3_2': {4: (7.273983048976076, np.array([0.74114443])), 5: (7.273983048976076, np.array([0.91128135])), 6: (22.964148373836466, np.array([0.99006779])), 7: (22.964148373836466, np.array([1.])), 8: (13.559010747738515, np.array([1.])), 9: (13.559010747738515, np.array([1.])), 10: (13.559010747738515, np.array([1.]))}}
    #
    # rigid_data_dict = {'17-8750_2_10': {4: 12.721734269721777, 5: 11.070827834238749, 6: 10.076476579162401, 7: 10.408592224724059, 8: 11.513341884619084, 9: 9.193073651728852}, '17-9612_1_8': {4: 17.11144805718279, 5: 13.501362339487828, 6: 23.383900409197768, 7: 20.61034829060198, 8: 21.528195609933285, 9: 16.0264229815798, 10: 17.642114918806755, 11: 16.801384010438895, 12: 14.43092913686377}, '20-10017_1_1': {4: 15.239691665891675, 5: 17.964763358517107, 6: 16.78377150733384, 7: 5.494377489371948, 8: 4.969125537579648, 9: 4.497727732713275, 10: 5.66238619814203, 11: 6.554935846730751}, '20-10023_2_10': {4: 20.25830308043308, 5: 21.715811570249553, 6: 16.606622101063724, 7: 11.003452055537968, 8: 6.640399097580669, 9: 8.26007508903743, 10: 6.747764617031562}, '20-10023_2_5': {5: 23.003522954358246, 6: 23.88127837241009, 7: 20.013951366290456, 8: 18.476635787585572, 9: 19.714563118152153, 10: 20.8369426860813, 11: 18.212075013144357}, '20-10040_1_1': {5: 10.297986486689622, 6: 8.79831107281684, 7: 7.429270232616335, 8: 7.2789239604464315, 9: 5.380825877844231, 10: 6.418956705014951, 11: 6.517144875101049}, '20-10043_1_1': {4: 6.703102124518803, 5: 6.558888581327318, 6: 7.251147338127791, 7: 9.761792350878913, 8: 6.873970522529702, 9: 3.997391504772334, 10: 3.771181743405897, 11: 2.3880126559216106}, '20-10103_1_1': {4: 22.281884440057798, 5: 24.548443645141013, 6: 19.1856355628361, 7: 21.157461863436122, 8: 18.50657601932264, 9: 17.760964601630914, 10: 21.300527128274616}, '20-10105_1_1': {4: 51.457531876586785, 5: 99.07829521872729, 6: 100.7370300443009, 7: 102.44408640887245, 8: 92.7145259160316, 9: 29.60457988645289, 10: 29.955589859687098}, '20-10147_1_1': {4: 25.355374182461638, 5: 11.935200482896843, 6: 6.392162637737383, 7: 5.062410954439418}, '20-10148_1_1': {4: 5.394834961781895, 5: 4.952937865534796, 6: 3.3965368544303653, 7: 9.115580076828355, 8: 9.9584474381366, 9: 8.528366059505952, 10: 8.291674593159716, 11: 7.900787719257342, 12: 7.535921519686023, 13: 6.323185868659116}, '20-10169_1_1': {4: 34.399348182236295, 5: 47.96830891938331, 6: 33.11521563991246, 7: 29.605828024404534, 8: 30.439463789465847, 9: 36.63637867258834, 10: 44.37991069000126}, '20-10170_1_1': {4: 26.983227044373802, 5: 13.922206541675, 6: 18.788129915198173, 7: 13.608345165753242, 8: 12.697823411345716, 9: 13.371485075863541, 10: 19.20899870834208, 11: 20.708188127712226}, '20-10177_1_6': {4: 22.538028269188587, 5: 12.2640619448283, 6: 1.6812153775815408, 7: 6.230120084817792, 8: 2.391594924010874, 9: 3.9829509023858005, 10: 5.857778265038015}, '20-10179_1_9': {4: 18.885945126743522, 5: 13.379121758801736, 6: 11.013175946782871, 7: 15.502393752472349, 8: 8.821091514919317, 9: 5.134094557559792, 10: 2.068119480087959, 11: 2.3699545054585607}, '20-10208_1_1': {4: 12.670199949831705, 5: 7.4711332858967845, 6: 16.417873239402187, 7: 15.710689668897917, 8: 17.0677549577134, 9: 18.08339544150134, 10: 12.859737608365718}, '20-10287_1_1': {4: 10.531530894069274, 5: 3.837758895352743, 6: 10.594994743163932, 7: 7.05650266753589, 8: 5.9984858618350145, 9: 5.108982403573714, 10: 4.538389360072342}, '20-10300_1_7': {6: 7.240644946827002, 7: 7.005826234178222, 8: 10.009617687791895, 9: 8.074011492808076, 10: 13.915000132572102}, '20-10303_1_8': {4: 21.808649184184482, 5: 14.585461912334958, 6: 10.670441415474075, 7: 7.045947133694094, 8: 7.255369927297619, 9: 10.078206475999968, 10: 8.634003215842384, 11: 7.72424729069351, 12: 6.267740053787436}, '20-10313_1_5': {4: 7.211252939303034, 5: 5.445190384539975, 6: 9.527360363591814}, '20-10313_3_2': {4: 11.065788334492328, 5: 4.732540090503264, 6: 13.427283135547587, 7: 11.982582515885529, 8: 10.151396904312762, 9: 10.339237587545693, 10: 7.65672567152606}}

    # ******* different outer groups trial ********
    # tri_data_dict_new = {'17-8750_2_10': {5: 21.207000366158177, 6: 20.37965688318297, 7: 19.54238162387375}, '17-9612_1_8': {5: 27.868081405993003, 6: 25.969427280490653, 7: 24.80287881058371, 8: 24.89004727958407, 9: 25.150740762227905, 10: 23.908999722653203, 11: 22.901155785853696, 12: 21.76150063275847}, '20-10017_1_1': {5: 21.641835118515093, 6: 20.666752939001416, 7: 21.875917836996578}, '20-10023_2_10': {5: 24.043952626039843, 6: 23.279615775515623, 7: 22.271136237349985, 8: 21.151967323679216}, '20-10023_2_5': {5: 20.990631382068763, 6: 20.94074224948207, 7: 20.643225061643154, 8: 20.138567649748527, 9: 20.49484135390564}, '20-10040_1_1': {5: 12.93068444324545, 6: 14.104935693365878, 7: 15.300098992603091, 8: 16.8444605086517, 9: 19.285320533884846}, '20-10043_1_1': {5: 13.12813536056378, 6: 11.815496180874188, 7: 11.06396503764992, 8: 10.764120659439493, 9: 10.78508884035425}, '20-10103_1_1': {5: 27.933365792212047, 6: 29.97411099477561}, '20-10105_1_1': {5: 65.00167902628029, 6: 65.00167902628029}, '20-10148_1_1': {5: 13.922309110307959, 6: 16.52946566203595, 7: 17.540496995587425, 8: 18.173562081730857, 9: 22.06302570027848, 10: 25.69398406319484, 11: 32.53456384988357}, '20-10169_1_1': {5: 37.156304500156494, 6: 34.9659104089476}, '20-10170_1_1': {5: 61.58589393055763, 6: 55.88148940865083, 7: 52.50819746430312, 8: 51.46601809751449}, '20-10177_1_6': {5: 22.154022887831033, 6: 20.052934041465946, 7: 20.102456721165613}, '20-10179_1_9': {5: 41.520015241989455, 6: 37.99417417635113, 7: 30.058317847855427, 8: 17.712446256502336}, '20-10208_1_1': {5: 31.753433155923243, 6: 30.52796785657144, 7: 29.718873244854002}, '20-10287_1_1': {5: 21.8826736359672, 6: 17.7046774250957}, '20-10300_1_7': {5: 49.86352068206122, 6: 45.18980627794539, 7: 40.83532824577922}, '20-10303_1_8': {5: 16.168498882040257, 6: 13.031926136308561}, '20-10313_3_2': {5: 33.350064446720786, 6: 36.56472257658334, 7: 40.54331733152044}}
    # tri_data_dict = {'17-8750_2_10': {5: (21.207000366158177, 0.9548438), 6: (20.37965688318297, 0.98076832), 7: (19.54238162387375, 1.0)}, '17-9612_1_8': {5: (27.868081405993003, 0.81000723), 6: (25.969427280490653, 0.91439682), 7: (24.80287881058371, 0.96499244), 8: (24.89004727958407, 0.97828497), 9: (25.150740762227905, 0.98658425), 10: (23.908999722653203, 0.99274814), 11: (22.901155785853696, 0.9976953), 12: (21.76150063275847, 1.0)}, '20-10017_1_1': {5: (21.641835118515093, 0.9016647), 6: (20.666752939001416, 0.96378935), 7: (21.875917836996578, 1.0)}, '20-10023_2_10': {5: (24.043952626039843, 0.82277207), 6: (23.279615775515623, 0.91131877), 7: (22.271136237349985, 0.99602629), 8: (21.151967323679216, 1.0)}, '20-10023_2_5': {5: (20.990631382068763, 0.88718867), 6: (20.94074224948207, 0.95869799), 7: (20.643225061643154, 0.99210916), 8: (20.138567649748527, 0.99809361), 9: (20.49484135390564, 1.0)}, '20-10040_1_1': {5: (12.93068444324545, 0.8750629), 6: (14.104935693365878, 0.93940333), 7: (15.300098992603091, 0.97152163), 8: (16.8444605086517, 0.99281344), 9: (19.285320533884846, 1.0)}, '20-10043_1_1': {5: (13.12813536056378, 0.93355799), 6: (11.815496180874188, 0.95902296), 7: (11.06396503764992, 0.98114772), 8: (10.764120659439493, 0.99663517), 9: (10.78508884035425, 1.0)}, '20-10103_1_1': {5: (27.933365792212047, 0.99756754), 6: (29.97411099477561, 1.0)}, '20-10105_1_1': {5: (65.00167902628029, 0.97216685), 6: (65.00167902628029, 1.0)}, '20-10148_1_1': {5: (13.922309110307959, 0.80274007), 6: (16.52946566203595, 0.87158508), 7: (17.540496995587425, 0.90813354), 8: (18.173562081730857, 0.94178661), 9: (22.06302570027848, 0.97196006), 10: (25.69398406319484, 0.9955288), 11: (32.53456384988357, 1.0)}, '20-10169_1_1': {5: (37.156304500156494, 0.99156116), 6: (34.9659104089476, 1.0)}, '20-10170_1_1': {5: (61.58589393055763, 0.85379493), 6: (55.88148940865083, 0.9471753), 7: (52.50819746430312, 0.98277889), 8: (51.46601809751449, 1.0)}, '20-10177_1_6': {5: (22.154022887831033, 0.95623382), 6: (20.052934041465946, 0.9974134), 7: (20.102456721165613, 1.0)}, '20-10179_1_9': {5: (41.520015241989455, 0.87187045), 6: (37.99417417635113, 0.93550289), 7: (30.058317847855427, 0.97060773), 8: (17.712446256502336, 1.0)}, '20-10208_1_1': {5: (31.753433155923243, 0.88811016), 6: (30.52796785657144, 0.95620637), 7: (29.718873244854002, 1.0)}, '20-10287_1_1': {5: (21.8826736359672, 0.89946043), 6: (17.7046774250957, 1.0)}, '20-10300_1_7': {5: (49.86352068206122, 0.95), 6: (45.18980627794539, 0.98246825), 7: (40.83532824577922, 1.0)}, '20-10303_1_8': {5: (16.168498882040257, 0.95960901), 6: (13.031926136308561, 1.0)}, '20-10313_3_2': {5: (33.350064446720786, 0.91128135), 6: (36.56472257658334, 0.99006779), 7: (40.54331733152044, 1.0)}}
    # rigid_data_dict = {'17-8750_2_10': {5: 27.944598888869212, 6: 27.221445072157437, 7: 26.46438384859622}, '17-9612_1_8': {5: 17.50470602002357, 6: 15.757951038526972, 7: 14.611194802040174, 8: 13.960459203001742, 9: 15.059469772734806, 10: 16.134665605566877, 11: 17.289049820199484, 12: 17.039849918955024}, '20-10017_1_1': {5: 15.422209055256403, 6: 13.610639170713108, 7: 11.592838879730648}, '20-10023_2_10': {5: 21.202007728132582, 6: 19.59782939229137, 7: 18.477242148293612, 8: 17.559525538902413}, '20-10023_2_5': {5: 48.25727889810376, 6: 45.11642479592547, 7: 43.7200047094129, 8: 43.28354929487479, 9: 42.9693969322507}, '20-10040_1_1': {5: 14.37438278571795, 6: 13.515307350578258, 7: 12.799167024451936, 8: 12.304489395428533, 9: 12.092245075713397}, '20-10043_1_1': {5: 39.19022412165589, 6: 36.530705354509635, 7: 14.454591121567608, 8: 14.015912336593384, 9: 13.588530862339397}, '20-10103_1_1': {5: 34.34523999630902, 6: 34.32045185795319}, '20-10105_1_1': {5: 85.53286244177494, 6: 67.61958612238651}, '20-10148_1_1': {5: 14.389147733763597, 6: 15.963829309217278, 7: 16.781802566594905, 8: 14.837597244105439, 9: 18.00236686887211, 10: 17.603195879373764, 11: 17.669513132258146}, '20-10169_1_1': {5: 53.152247439320846, 6: 49.942480628475614}, '20-10170_1_1': {5: 58.6337115686799, 6: 53.94725891400199, 7: 51.00744854176924, 8: 49.15133090859747}, '20-10177_1_6': {5: 23.921652468487675, 6: 22.56798790376132, 7: 22.593519738877806}, '20-10179_1_9': {5: 53.679224256534454, 6: 41.84324577749246, 7: 26.85863884159587, 8: 25.231495194070458}, '20-10208_1_1': {5: 43.45719432406665, 6: 41.09200015995527, 7: 39.79646888058684}, '20-10287_1_1': {5: 23.65391491408167, 6: 24.291307970986736}, '20-10300_1_7': {5: 46.82824150093679, 6: 43.62059996845187, 7: 40.66867517475093}, '20-10303_1_8': {5: 25.411226273121947, 6: 23.718134979636613}, '20-10313_3_2': {5: 37.57502277832326, 6: 38.52272815565134, 7: 38.07910013916895}}
    #
    # show_mean_dist_area_prop(tri_data_dict=tri_data_dict, rigid_data_dict=rigid_data_dict, avg_dirs=True)
