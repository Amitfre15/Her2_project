# Standard library imports
import argparse
import os
import sys

# Third-party imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import mode
# from shapely.geometry import Point, Polygon
from tqdm import tqdm
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field

# Local application imports
from workspace.Gigapath_GIP.finetune.utils import (
    parse_tile_name, correct_coords, slide_to_thumb_coord,
    thumb_to_slide_coord, load_npy_file, SLIDE_HEIGHT, SLIDE_WIDTH,
    SLIDE_MPP, SLIDE_PATCH_SIZE
)

# Constants
# Background filtering thresholds (RGB values for identifying background pixels)
BG_R_MIN = 200
BG_G_MIN = 100
BG_G_MAX = 150
BG_B_MIN = 100
BG_B_MAX = 150

# Segment labels
SEGMENT_TUMOR = 0
SEGMENT_NON_TUMOR = 3
SEGMENT_BACKGROUND = 5
SEGMENT_UNKNOWN = -1

# OpenSlide configuration
OPENSLIDE_PATH = r"C:\Program Files\openslide-bin-4.0.0.3-windows-x64\openslide-bin-4.0.0.3-windows-x64\bin"

# OpenSlide library loading
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class Paths:
    save_path: str
    tile_y_dir: str
    he_tiles_dir: str
    ihc_tiles_dir: str
    matching_tiles_dir: str


@dataclass
class SlideContext:
    row: pd.Series

    slide_name: str
    slide_file: str
    matching_he_slide: str

    paths: Paths

    valid_tiles: list
    valid_indices: list
    he_tiles: list
    ihc_tiles: list
    tile_segment: Optional[np.ndarray] = None
    tumor_indices: Optional[np.ndarray] = None
    matching_tiles: Optional[np.ndarray] = None
    non_nan_matching_indices: Optional[np.ndarray] = None
    tumor_indices: Optional[np.ndarray] = None
    tile_y: Optional[np.ndarray] = None
    map_gt_npy: Optional[np.ndarray] = None
    map_baseline_npy: Optional[np.ndarray] = None
    map_model_npy: Optional[np.ndarray] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Collectors:
    baseline_dfs: Optional[list] = None
    local_dfs: Optional[list] = None
    map_baseline_dfs: Optional[list] = None
    map_local_dfs: Optional[list] = None

    results_df: Optional[pd.DataFrame] = None
    all_tumor_y: Optional[np.ndarray] = None


# =========================================================
# Argument parsing
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Create and process score maps.")

    # Required
    parser.add_argument("-s", "--save_path", required=True, type=str, help='Path to save tiles and embeds')
    parser.add_argument("-e", "--excel_path", required=True, type=str, help='Path to Excel file')
    parser.add_argument("-tmpp", "--target_mpp", required=True, type=float, choices=[0.5, 1, 2], help='Target tiles MPP')

    # Optional
    parser.add_argument("-mcsv", "--matched_csv_path", default="/home/amitf/workspace/WSI/metadata_csvs/Her2_slides_matched_HE_folds.csv", type=str, help='Path to matched CSV file')
    parser.add_argument('-smpp', '--source_mpp', type=float, help='Source tiles MPP', choices=[0.5, 1, 2])
    parser.add_argument('-vf', '--val_fold', type=str, help='validation fold used for cancer model', default='', choices=['1', '2', '3', '4', '5'])
    parser.add_argument('-y_type', '--y_type', type=str, help='Type of y map to use', default='regional', choices=['regional', 'local'])
    parser.add_argument('-y_pred_type', '--y_pred_type', type=str, help='Type of y_pred file to use', default='regional', choices=['regional', 'local'])
    parser.add_argument('-ti_type', '--tumor_indices_type', type=str, help='Type of tumor_indices file to use', default='local', choices=['regional', 'local'])

    parser.add_argument('-blod', '--baseline_output_dir', type=str, help='Output dir for baseline predictions')
    parser.add_argument('-pod', '--predictions_output_dir', type=str, help='Output dir for model predictions')
    parser.add_argument('-cpp', '--cancer_prob_path', type=str, help='Cancer probabilities path')

    # Flags
    parser.add_argument('-seg', '--segment_tiles', action='store_true', default=False, help='Segment image tiles')
    parser.add_argument('-seg_y', '--seg_y_labels', action='store_true', default=False, help='save HER2 contribution (y) pseudo labels map')
    parser.add_argument('-seg_y_pred', '--seg_y_predictions', action='store_true', default=False, help='save HER2 contribution (y) prediction map')
    parser.add_argument('-seg_ot', '--seg_only_tumor', action='store_true', default=False, help='save only tumor tiles in HER2 contribution (y) prediction map')
    parser.add_argument('-amc', '--all_models_cancer', action='store_true', default=False, help='Only tiles classified as tumor by all cancer models will be considered tumor')
    parser.add_argument('-efy_map', '--extract_tile_y_from_y_map', action='store_true', default=False, help='extract tile y from y map')
    parser.add_argument('-etfcm', '--extract_tumor_indices_from_cancer_map', action='store_true', default=False, help='extract tumor indices from cancer map')
    parser.add_argument('-save_tifsm', '--save_tumor_indices_from_seg_map', action='store_true', default=False, help='save tumor tile indices from segmentation map')
    parser.add_argument('-save_ens_ti_seg', '--save_ensemble_tumor_indices', action='store_true', default=False, help='save ensemble tumor indices')
    parser.add_argument('-cypc', '--create_y_sw_bl_csv', action='store_true', default=False, help='Create csvs for y predictions')
    parser.add_argument('-y_std', '--y_std', action='store_true', default=False, help='Calculate standard deviation of tile_y')
    parser.add_argument('-reg_an', '--registration_annotated', action='store_true', default=False, help='filter annotated slides for registration')
    parser.add_argument('-ad', '--all_data', action='store_true', default=False, help='Use all the data (annotated)')
    parser.add_argument('-cancer_an', '--cancer_annotations', action='store_true', default=False, help='process annotated slides for cancer classification')
    parser.add_argument('-her2_an', '--her2_annotations', action='store_true', default=False, help='process annotated slides for HER2 classification')
    parser.add_argument('-save_her2_an', '--save_her2_annotations', action='store_true', default=False, help='save HER2 annotations from segmentation map')
    parser.add_argument('-seg_from_cp', '--seg_from_my_cancer_predictions', action='store_true', default=False, help='save tile cancer prediction map')
    parser.add_argument('-test_set', '--test_set', action='store_true', default=False, help='iterate only on test set')
    parser.add_argument('-y_map_fhe', '--y_map_from_he', action='store_true', default=False, help='extract tile y from y map constructed from HE slide')
    parser.add_argument('-aht', '--all_he_tiles', action='store_true', default=False, help='extract tile y from y map constructed from HE slide and use all HE tiles')
    parser.add_argument('-exter', '--external_he_model', action='store_true', default=False, help='extract tile y from y map constructed by external HE model')

    return parser.parse_args()


# =========================================================
# Setup
# =========================================================

def cast_mpp(mpp):
    if mpp is None:
        return None

    return int(mpp) if mpp != 0.5 else mpp


def setup_args(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.target_mpp = cast_mpp(args.target_mpp)
    args.source_mpp = cast_mpp(args.source_mpp)

    args.suffix = f"_mpp{args.target_mpp}"

    args.src_suffix = (
        args.suffix
        if args.source_mpp is None
        else f"_mpp{args.source_mpp}"
    )

    args.seg_thumb_height = int(SLIDE_HEIGHT * SLIDE_MPP // (SLIDE_PATCH_SIZE * args.source_mpp))
    args.seg_thumb_width = int(SLIDE_WIDTH * SLIDE_MPP // (SLIDE_PATCH_SIZE * args.source_mpp))

    return args


# =========================================================
# Paths
# =========================================================

def build_paths(args):

    return Paths(
        save_path=args.save_path,
        tile_y_dir=os.path.join(args.save_path, f"tile_y{args.suffix}"),
        he_tiles_dir=os.path.join(args.save_path.replace("IHC", "HE"), f"png_tiles{args.suffix}"),
        ihc_tiles_dir=os.path.join(args.save_path, f"png_tiles{args.suffix}"),
        matching_tiles_dir=os.path.join(args.save_path, f"matching_tiles{args.suffix}")
    )


# =========================================================
# Metadata
# =========================================================

def load_metadata(args):
    matched_df = read_slide_file(excel_path=args.matched_csv_path, only_read=True, args=args)

    slides_df = read_slide_file(excel_path=args.excel_path, args=args)

    return {"matched_df": matched_df, "slides_df": slides_df}


# =========================================================
# Collectors
# =========================================================

def initialize_collectors(args):

    collectors = Collectors()

    if args.create_y_sw_bl_csv:
        collectors.baseline_dfs = [pd.DataFrame(columns=['tile_y', 'tile_y_bin', 'tile_pred']) for fold in range(5)]
        collectors.map_baseline_dfs = [pd.DataFrame(columns=['tile_gt', 'tile_gt_bin', 'map_tile_pred']) for fold in range(5)]
        collectors.local_dfs = [pd.DataFrame(columns=['tile_y', 'tile_y_bin', 'tile_pred']) for fold in range(5)]
        collectors.map_local_dfs = [pd.DataFrame(columns=['tile_gt', 'tile_gt_bin', 'map_tile_pred']) for fold in range(5)]

    if args.y_std:
        collectors.results_df = pd.DataFrame(columns=["slide_name", "tumor_tile_y_std", "num_tiles", "num_tumor_tiles", "label", "fold"])
        collectors.all_tumor_y = np.array([], dtype=np.float32)

    return collectors


# =========================================================
# Slide preparation
# =========================================================

def resolve_slide_info(args, row, metadata):
    matched_df = metadata["matched_df"]

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

    return slide_name, slide_file, matching_he_slide


def build_slide_dirs(args, paths, slide_name, matching_he_slide):
    paths.full_he_tiles_dir = os.path.join(paths.he_tiles_dir, matching_he_slide)
    block_dir = next(filter(lambda x: x.startswith(slide_name), os.listdir(paths.ihc_tiles_dir)), None)
    paths.full_ihc_tiles_dir = os.path.join(paths.ihc_tiles_dir, block_dir)
    paths.full_tile_y_dir = os.path.join(paths.tile_y_dir, matching_he_slide)
    paths.full_mt_dir = os.path.join(paths.matching_tiles_dir, matching_he_slide)
    paths.full_ti_dir = paths.full_mt_dir.replace('matching_tiles', 'tumor_indices_from_cancer_map')
    paths.full_ant_dir = paths.full_mt_dir.replace('matching_tiles', 'annotated_tiles')
    paths.ti = os.path.join(paths.full_ti_dir, f'{args.tumor_indices_type}_tumor_indices{args.val_fold}.npy') if not args.seg_y_predictions else os.path.join(paths.full_ti_dir, f'{args.tumor_indices_type}_tumor_indices.npy')
    if args.save_tumor_indices_from_seg_map:
        paths.ti = paths.ti.replace('tumor_indices.npy', 'tumor_ann_indices.npy')
    if args.save_ensemble_tumor_indices:
        paths.ti = paths.ti.replace(f'tumor_indices{args.val_fold}.npy', 'tumor_indices_ensemble.npy')
    paths.nti = os.path.join(paths.full_ti_dir, 'non_tumor_indices.npy')

    if args.cancer_prob_path is not None and not args.her2_annotations:
        cancer_prob_dir = os.path.join(args.cancer_prob_path, f"cancer_probs_mpp{args.target_mpp}", matching_he_slide)
        paths.cp = os.path.join(cancer_prob_dir, f'cancer_prob_val{args.val_fold}.npy')


def resolve_segment_dir(args, paths, slide_name, matching_he_slide):
    if (args.seg_y_labels or args.seg_y_predictions or args.extract_tile_y_from_y_map):
        paths.segment_dir = os.path.join(paths.save_path, f'y_map{args.src_suffix}')
        if args.external_he_model:
            paths.segment_dir = os.path.join(paths.save_path.replace('IHC', 'HE'), f'HER2_status_map{args.src_suffix}')
        slide_dir = next(filter(lambda x: x.startswith(matching_he_slide[:-1]), os.listdir(paths.segment_dir)), None)
        if slide_dir is None:
            print(f"y_map directory for slide {matching_he_slide} not found. Skipping...")
            # continue
        paths.segment_dir = os.path.join(paths.segment_dir, slide_dir)
    elif (args.her2_annotations or args.save_her2_annotations):
        paths.segment_dir = os.path.join(paths.save_path, f'slide_segmentations{args.suffix}', slide_name)
    elif args.extract_tumor_indices_from_cancer_map or args.seg_from_my_cancer_predictions:
        paths.segment_dir = os.path.join(paths.save_path.replace('IHC', 'HE'), f'cancer_map{args.src_suffix}', matching_he_slide)
    else:
        paths.segment_dir = os.path.join(paths.save_path.replace('IHC', 'HE'), f'slide_segmentations{args.suffix}', matching_he_slide)

    paths.st = os.path.join(paths.segment_dir, 'segmented_tiles.npy')
    paths.h2a = os.path.join(paths.segment_dir, 'HER2 Annotations.npy')


def get_valid_tiles(args, paths, he_tiles, ihc_tiles):
    valid_tiles, valid_indices, matching_tiles, non_nan_matching_indices = None, None, None, None

    if (args.save_her2_annotations or args.extract_tile_y_from_y_map):
        matching_tiles = np.load(os.path.join(paths.full_mt_dir, 'ihc_tiles.npy'))
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
        matching_tiles = np.load(os.path.join(paths.full_mt_dir, 'ihc_tiles.npy'))
        non_nan_matching_indices = np.array([idx for idx, _ in enumerate(he_tiles) if not np.isnan(matching_tiles[idx])])
    elif args.cancer_annotations or args.extract_tumor_indices_from_cancer_map or args.seg_from_my_cancer_predictions:
        valid_tiles = he_tiles
        valid_indices = list(range(len(he_tiles)))
    else: 
        matching_tiles = np.load(os.path.join(paths.full_mt_dir, 'ihc_tiles.npy'))
        valid_tiles = [tile for idx, tile in enumerate(he_tiles) if not np.isnan(matching_tiles[idx])]
        valid_indices = [idx for idx, _ in enumerate(he_tiles) if not np.isnan(matching_tiles[idx])]

    return valid_tiles, valid_indices, matching_tiles, non_nan_matching_indices


def get_y_sw(args, paths, slide_name, matching_he_slide):
    map_gt_npy, map_baseline_npy, map_model_npy = None, None, None

    if args.create_y_sw_bl_csv:
        gt_dir = paths.segment_dir.replace("y_map", "slide_segmentations").replace(f"{matching_he_slide}", f"{slide_name}")
        if os.path.exists(gt_dir) and os.path.exists(paths.segment_dir):
            gt_name = next(filter(lambda x: x.endswith(".npy"), os.listdir(gt_dir)), None)
            gt_file = os.path.join(gt_dir, gt_name)
            map_gt_npy = load_npy_file(gt_file)
            bl_name = next(filter(lambda x: "tumor from map SW baseline" in x and x.endswith(f".npy"), os.listdir(paths.segment_dir)), None)
            bl_file = os.path.join(paths.segment_dir, bl_name)
            map_baseline_npy = load_npy_file(bl_file)
            pred_name = next(filter(lambda x: "tumor from map predictions" in x and x.endswith(f".npy"), os.listdir(paths.segment_dir)), None)
            model_file = os.path.join(paths.segment_dir, pred_name)
            map_model_npy = load_npy_file(model_file)
        
    return map_gt_npy, map_baseline_npy, map_model_npy


def prepare_slide_context(args, row, paths, metadata):
    slide_name, slide_file, matching_he_slide = resolve_slide_info(args, row, metadata)
    build_slide_dirs(args, paths, slide_name, matching_he_slide)
    resolve_segment_dir(args, paths, slide_name, matching_he_slide)
    he_tiles = os.listdir(paths.full_he_tiles_dir)
    ihc_tiles = os.listdir(paths.full_ihc_tiles_dir)
    valid_tiles, valid_indices, matching_tiles, non_nan_matching_indices = get_valid_tiles(args, paths, he_tiles, ihc_tiles)
    map_gt_npy, map_baseline_npy, map_model_npy = get_y_sw(args, paths, slide_name, matching_he_slide)

    return SlideContext(
        row=row,

        slide_name=slide_name,
        slide_file=slide_file,
        matching_he_slide=matching_he_slide,

        valid_tiles=valid_tiles,
        valid_indices=valid_indices,
        he_tiles=he_tiles,
        ihc_tiles=ihc_tiles,
        tile_segment=None,
        tumor_indices=None,
        matching_tiles=matching_tiles,
        non_nan_matching_indices=non_nan_matching_indices,
        tile_y=None,
        map_gt_npy=map_gt_npy,
        map_baseline_npy=map_baseline_npy,
        map_model_npy=map_model_npy,

        paths=paths
    )


# =========================================================
# Processing stages
# =========================================================   

def get_tile_segment(args, slide_ctx):
    if args.segment_tiles:
        slide_ctx.tile_segment = segment_from_model_or_annotations(args=args, row=slide_ctx.row, slide_file=slide_ctx.slide_file, segment_dir=slide_ctx.segment_dir, st=slide_ctx.paths.st)
        if slide_ctx.tile_segment is None:
            raise ValueError(f"tile_segment is None for slide {slide_ctx.slide_name}. Cannot proceed with segmentation-based processing.")

    elif args.save_tumor_indices_from_seg_map:
        if slide_ctx.tile_segment is None and os.path.exists(slide_ctx.paths.segment_dir):
            slide_ctx.tile_segment = np.load(slide_ctx.paths.segment_dir)
        else:
            raise ValueError(f"Tile segmentation map not found for slide {slide_ctx.slide_name}. Skipping...")
            
    elif args.save_her2_annotations:
        if slide_ctx.tile_segment is None and os.path.exists(slide_ctx.paths.h2a):
            slide_ctx.tile_segment = np.load(slide_ctx.paths.h2a)
        save_her2_annotations(args, slide_ctx.paths.full_ant_dir, slide_ctx.slide_name, slide_ctx.valid_tiles, slide_ctx.tile_segment)
    

def load_tumor_indices(args, slide_ctx):
    if args.save_tumor_indices_from_seg_map or (args.seg_from_my_cancer_predictions and not args.save_ensemble_tumor_indices):
        slide_ctx.tumor_indices = save_tumor_indices_and_cancer_probs(args, slide_ctx)

    elif args.seg_y_predictions:
        slide_ctx.paths.y_pred_folder=slide_ctx.paths.full_tile_y_dir.replace('tile_y', 'tile_y_pred')
        slide_ctx.paths.tile_y_folder=slide_ctx.paths.full_tile_y_dir.replace('tile_y', 'tile_y_from_y_map') if args.y_type == 'regional' else slide_ctx.paths.full_tile_y_dir

        if not os.path.exists(slide_ctx.paths.y_pred_folder):
            raise ValueError(f"y_pred_folder {slide_ctx.paths.y_pred_folder} does not exist. Skipping...")
            
        y_pred_name = 'from_map_pred_ad_' if args.y_pred_type == 'regional' else 'local_tile_y_pred_add_virchow'
        y_pred_npy = next(filter(lambda f: y_pred_name in f and f.endswith('.npy'), os.listdir(slide_ctx.paths.y_pred_folder)), None)

        if y_pred_npy is None:
            raise ValueError(f"No y prediction file found in {slide_ctx.paths.y_pred_folder}, probably no tumor indices were found. Skipping...")
        slide_ctx.paths.y_pred_file=os.path.join(slide_ctx.paths.y_pred_folder, y_pred_npy)
        slide_ctx.y_pred_fold = y_pred_npy.split('val')[-1].split('.')[0]  # extract fold from filename like tile_y_pred_val1.npy
        ti = slide_ctx.paths.ti.replace(f'tumor_indices.npy', f'tumor_indices{slide_ctx.y_pred_fold}.npy')
        slide_ctx.tumor_indices = np.load(ti).astype(int) if os.path.exists(ti) else None
        tile_y_npy = next(filter(lambda f: f'val{slide_ctx.y_pred_fold}' in f, os.listdir(slide_ctx.paths.tile_y_folder)), None)
        y_file = os.path.join(slide_ctx.paths.tile_y_folder, tile_y_npy)
        slide_ctx.tile_y = np.load(y_file).flatten() if os.path.exists(y_file) else None
    elif not args.her2_annotations:
        slide_ctx.tumor_indices = np.load(slide_ctx.paths.ti).astype(int) if os.path.exists(slide_ctx.paths.ti) else None
    
    if args.save_ensemble_tumor_indices:
        save_ensemble_tumor_indices(args, slide_ctx)


def process_y_labels(args, slide_ctx):
    if args.seg_y_labels:
        y_npy = f'tile_y_ad_val{args.val_fold}.npy' if args.all_data else f'tile_y.npy'
        y_file=os.path.join(slide_ctx.paths.full_tile_y_dir, y_npy)

        if os.path.exists(y_file):
            slide_ctx.tile_y = np.load(y_file).flatten()
        else:
            raise ValueError(f"y_file file {y_file} does not exist. Skipping...")
            # return None
        
        if args.seg_only_tumor:
            if slide_ctx.tumor_indices is None:
                raise ValueError(f"Tumor indices not found for slide {slide_ctx.slide_name}. Cannot filter to tumor tiles.")
            print(f"tumor_indices.shape = {slide_ctx.tumor_indices.shape}", "non_nan_matching_indices.shape =", slide_ctx.non_nan_matching_indices.shape)
            valid_indices = np.intersect1d(slide_ctx.tumor_indices, slide_ctx.non_nan_matching_indices)
            tumor_in_non_nan = np.isin(slide_ctx.non_nan_matching_indices, slide_ctx.tumor_indices)
            tumor_matching_indices = np.where(tumor_in_non_nan)[0]
            valid_y = slide_ctx.tile_y[tumor_matching_indices]
        else:
            valid_indices = slide_ctx.non_nan_matching_indices
            valid_y = slide_ctx.tile_y

        valid_matching_indices = slide_ctx.matching_tiles[valid_indices].astype(int)
        slide_ctx.valid_tiles = list(np.array(slide_ctx.ihc_tiles)[valid_matching_indices])
        local_y_maps(args, tile_y=valid_y, segment_dir=slide_ctx.paths.segment_dir, slide_name=slide_ctx.slide_name, valid_tiles=slide_ctx.valid_tiles, 
                        save_name=f'Tile HER2 contribution (y) (MPP={args.source_mpp} GigaPath features trained model) val_fold = {args.val_fold}')


def process_y_predictions(args, slide_ctx, collectors):
    if args.seg_y_predictions:
        common_indices = np.intersect1d(slide_ctx.tumor_indices, slide_ctx.non_nan_matching_indices)
        tumor_in_non_nan = np.isin(slide_ctx.non_nan_matching_indices, slide_ctx.tumor_indices)
        tumor_matching_indices = np.where(tumor_in_non_nan)[0]
        tumor_matching_tiles = slide_ctx.matching_tiles[common_indices].astype(int)
        tumor_tile_y = slide_ctx.tile_y[tumor_matching_indices]
        valid_y = tumor_tile_y != -1
        valid_tile_y = tumor_tile_y[valid_y]
        tumor_matching_tiles = tumor_matching_tiles[valid_y].astype(int)

        y_bl_name = 'from_sw_bl_map' if args.y_pred_type == 'regional' else 'local_tile_y_pred_bl'            
        y_bl_npy = next(filter(lambda f: y_bl_name in f, os.listdir(slide_ctx.paths.y_pred_folder)), None)
        y_bl_file=os.path.join(slide_ctx.paths.y_pred_folder, y_bl_npy)

        tile_pred_y = load_npy_file(slide_ctx.paths.y_pred_file).flatten()
        tile_bl_y = load_npy_file(y_bl_file)
        # keep only paired tumor entries with valid y values
        tile_pred_y = tile_pred_y[common_indices][valid_y]
        tile_bl_y = tile_bl_y[common_indices][valid_y]
        
        slide_ctx.valid_tiles = list(np.array(slide_ctx.ihc_tiles)[tumor_matching_tiles])

        if not args.create_y_sw_bl_csv:
            local_y_maps(args, tile_y=tile_pred_y, segment_dir=slide_ctx.paths.segment_dir, slide_name=slide_ctx.slide_name, valid_tiles=slide_ctx.valid_tiles,
                        save_name=f'{args.y_pred_type} tumor tile HER2 contribution (y) baseline (MPP={args.source_mpp} GigaPath features trained model)')
        
        else: # create_y_sw_bl_csv
            # for pseudo-labels (y) comparison
            fold_idx = int(slide_ctx.y_pred_fold) - 1
            tile_y_bin = (valid_tile_y >= 2).astype(int)

            bl_df_new = pd.DataFrame({'tile_y': valid_tile_y, 'tile_y_bin': tile_y_bin, 'tile_pred': tile_bl_y})
            local_df_new = pd.DataFrame({'tile_y': valid_tile_y, 'tile_y_bin': tile_y_bin, 'tile_pred': tile_pred_y})
            collectors.baseline_dfs[fold_idx] = pd.concat([collectors.baseline_dfs[fold_idx], bl_df_new], ignore_index=True)
            collectors.local_dfs[fold_idx] = pd.concat([collectors.local_dfs[fold_idx], local_df_new], ignore_index=True)

            # for gt comparison
            if slide_ctx.map_gt_npy is not None:
                mask = (slide_ctx.map_gt_npy != -1) & (slide_ctx.map_baseline_npy != -1) & (slide_ctx.map_model_npy != -1)
                valid_indices = np.where(mask)
                valid_gt, valid_bl, valid_pred = slide_ctx.map_gt_npy[valid_indices], slide_ctx.map_baseline_npy[valid_indices], slide_ctx.map_model_npy[valid_indices]
                valid_gt_bin = (valid_gt >= 2).astype(int)
                
                map_local_df_new = pd.DataFrame({'tile_gt': valid_gt, 'tile_gt_bin': valid_gt_bin, 'map_tile_pred': valid_pred})
                map_bl_df_new = pd.DataFrame({'tile_gt': valid_gt, 'tile_gt_bin': valid_gt_bin, 'map_tile_pred': valid_bl})
                collectors.map_baseline_dfs[fold_idx] = pd.concat([collectors.map_baseline_dfs[fold_idx], map_bl_df_new], ignore_index=True)
                collectors.map_local_dfs[fold_idx] = pd.concat([collectors.map_local_dfs[fold_idx], map_local_df_new], ignore_index=True)


def extract_tile_y(args, slide_ctx):
    if args.extract_tile_y_from_y_map:
        y_npy = f'tile_y_ad_val{args.val_fold}.npy' if args.all_data else f'tile_y.npy'
        tile_y_from_y_map_dir = slide_ctx.paths.full_tile_y_dir.replace('tile_y', 'tile_y_from_y_map')
        if not os.path.exists(tile_y_from_y_map_dir):
            os.makedirs(tile_y_from_y_map_dir, exist_ok=True)

        y_map_file = os.path.join(slide_ctx.paths.segment_dir, f'Tile HER2 contribution (y) from SW (MPP={args.source_mpp} GigaPath features trained model) val_fold = {args.val_fold}.npy')
        if args.y_map_from_he:
            y_npy = f'tile_y_from_sw_bl_map_ad_val{args.val_fold}.npy'
            tile_y_from_y_map_dir = slide_ctx.paths.full_tile_y_dir.replace('tile_y', 'tile_y_pred')
            y_map = next(filter(lambda f: f'SW baseline (MPP={args.source_mpp}' in f and f.endswith(f'val_fold = {args.val_fold}.npy'), os.listdir(slide_ctx.paths.segment_dir)), None)
            if y_map is None:
                raise ValueError(f"No y_map file found in {slide_ctx.paths.segment_dir} for fold {args.val_fold}. Skipping...")
            y_map_file = os.path.join(slide_ctx.paths.segment_dir, y_map)
            # y_map_file = os.path.join(slide_ctx.paths.segment_dir, f'Tile HER2 contribution (y) SW baseline (MPP={args.source_mpp} GigaPath features trained model) val_fold = {args.val_fold}.npy')
            if not os.path.exists(tile_y_from_y_map_dir):
                os.makedirs(tile_y_from_y_map_dir, exist_ok=True)
            if args.all_he_tiles:
                y_npy = f'tile_y_from_sw_bl_map_all_he_tiles_val{args.val_fold}.npy'
            if args.external_he_model:
                # y_map_file = y_map_file.replace('SW baseline', 'from THUFA SW')
                y_map_file = os.path.join(slide_ctx.paths.segment_dir, "Tile HER2 status from SW (MPP=0.5 GigaPath features trained on THUFA model).npy")
                y_npy = y_npy.replace('sw_bl', 'sw_thufa')

        tile_y_map = load_npy_file(y_map_file)
        if tile_y_map is None:
            raise ValueError(f"y_map file {y_map_file} does not exist. Skipping...")
        
        extract_tile_y_from_y_map(args, y_map=tile_y_map, tile_y_dir=tile_y_from_y_map_dir, slide_name=slide_ctx.slide_name, valid_tiles=slide_ctx.valid_tiles, save_name=y_npy)


def send_extract_tumor_indices_from_cancer_map(args, slide_ctx):
    if args.extract_tumor_indices_from_cancer_map:
        ti_npy = f'{args.tumor_indices_type}_tumor_indices{args.val_fold}.npy'
        tumor_indices_from_cancer_map_dir = slide_ctx.paths.full_tile_y_dir.replace('tile_y', 'tumor_indices_from_cancer_map')
        cancer_probs_from_cancer_map_dir = slide_ctx.paths.full_he_tiles_dir.replace('png_tiles', 'cancer_probs_from_cancer_map')
        if not os.path.exists(tumor_indices_from_cancer_map_dir):
            os.makedirs(tumor_indices_from_cancer_map_dir, exist_ok=True)
        if not os.path.exists(cancer_probs_from_cancer_map_dir):
            os.makedirs(cancer_probs_from_cancer_map_dir, exist_ok=True)

        cancer_map_file = os.path.join(slide_ctx.paths.segment_dir, f'Tile cancer probability from SW (MPP={args.source_mpp} GigaPath features trained model) val_fold = {args.val_fold}.npy')
        cancer_map = load_npy_file(cancer_map_file)
        if cancer_map is not None:
            extract_tumor_indices_from_cancer_map(args, cancer_map=cancer_map, tumor_indices_dir=tumor_indices_from_cancer_map_dir, 
                                                cancer_probs_dir=cancer_probs_from_cancer_map_dir, slide_name=slide_ctx.slide_name, 
                                                valid_tiles=slide_ctx.valid_tiles, save_name=ti_npy)


def compute_y_std(
    args,
    slide_ctx,
    collectors
):
    if args.y_std:
        if not os.path.exists(slide_ctx.paths.full_tile_y_dir):
            raise ValueError(f"Directory {slide_ctx.paths.full_tile_y_dir} does not exist. Skipping...")

        tile_y = np.load(os.path.join(slide_ctx.paths.full_tile_y_dir, 'tile_y.npy'))

        if slide_ctx.tumor_indices.size == 0:
            raise ValueError(f"No tumor indices found for slide {slide_ctx.slide_name}. Skipping...")
        tumor_tile_y = tile_y[slide_ctx.tumor_indices]
        collectors.all_tumor_y = np.concatenate((collectors.all_tumor_y, tumor_tile_y.flatten()), axis=0)

        # Calculate standard deviation of tile_y
        tumor_tile_y_std = np.std(tumor_tile_y)
        num_tiles = tile_y.shape[0]
        num_tumor_tiles = tumor_tile_y.shape[0]
        label = slide_ctx.row['label']
        fold = slide_ctx.row['fold']
        collectors.results_df.loc[len(collectors.results_df)] = {
            'slide_name': slide_ctx.slide_name,
            'tumor_tile_y_std': tumor_tile_y_std,
            'num_tiles': num_tiles,
            'num_tumor_tiles': num_tumor_tiles,
            'label': label,
            'fold': fold
        }


def process_slide(args, slide_ctx, collectors):
    get_tile_segment(args, slide_ctx)
    load_tumor_indices(args, slide_ctx)
    process_y_labels(args, slide_ctx)
    process_y_predictions(args, slide_ctx, collectors)
    extract_tile_y(args, slide_ctx)
    send_extract_tumor_indices_from_cancer_map(args, slide_ctx)
    compute_y_std(args, slide_ctx, collectors)


# =========================================================
# Finalization
# =========================================================

def save_prediction_csvs(args, collectors):
    if args.baseline_output_dir is not None and os.path.exists(args.baseline_output_dir):
        her2_subdir = os.path.join(args.baseline_output_dir, "her2")
        for i in range(1, 6):
            infer_fold_dir = next(filter(lambda x: 'infer' in x and x.endswith(f'{i}'), os.listdir(her2_subdir)))
            full_infer_dir = os.path.join(her2_subdir, infer_fold_dir, 'eval_pretrained_her2', 'inference_results')
            csv_name = f"tumor_tile_valid_{args.y_type}_y_preds_mpp{args.target_mpp}_val{i}.csv" 
            collectors.baseline_dfs[i-1].to_csv(os.path.join(full_infer_dir, csv_name), index=False)
            if len(collectors.map_baseline_dfs[i-1]) > 0:
                collectors.map_baseline_dfs[i-1].to_csv(os.path.join(full_infer_dir, f"tile_preds_vs_gt_mpp{args.target_mpp}_val{i}.csv"), index=False)
    
    if args.predictions_output_dir is not None and os.path.exists(args.predictions_output_dir):
        her2_subdir = os.path.join(args.predictions_output_dir, "her2")
        for i in range(1, 6):
            infer_fold_dir = next(filter(lambda x: 'infer' in x and x.endswith(f'{i}'), os.listdir(her2_subdir)))
            full_infer_dir = os.path.join(her2_subdir, infer_fold_dir, 'eval_pretrained_her2', 'inference_results')
            csv_name = f"tumor_tile_valid_{args.y_type}_y_preds_mpp{args.target_mpp}_val{i}.csv"
            collectors.local_dfs[i-1].to_csv(os.path.join(full_infer_dir, csv_name), index=False)
            if len(collectors.map_local_dfs[i-1]) > 0:
                collectors.map_local_dfs[i-1].to_csv(os.path.join(full_infer_dir, f"tile_preds_vs_gt_mpp{args.target_mpp}_val{i}.csv"), index=False)


def save_y_std_results(args, collectors, paths):
    print(f"all_tumor_y.max() = {collectors.all_tumor_y.max()}")
    print(f"all_tumor_y.min() = {collectors.all_tumor_y.min()}")

    # Save results to CSV
    results_file = os.path.join(paths.save_path, f'tumor_tile_y_std_results_w_my_cancer_model{args.suffix}.csv')
    collectors.results_df.to_csv(results_file, index=False)


def finalize_outputs(args, collectors, paths):
    if args.create_y_sw_bl_csv:
        save_prediction_csvs(args, collectors)

    if args.y_std:
        save_y_std_results(args, collectors, paths)


def filter_background(image_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Filters out background pixels from an RGB image based on specific color thresholds.

    Args:
        image_array (np.ndarray): The input RGB image array of shape (H, W, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - filtered_image (np.ndarray): A new image array where non-matching
              background pixels are zeroed out.
            - mask (np.ndarray): A boolean mask indicating the identified background pixels.
    """
    R = image_array[:, :, 0]
    G = image_array[:, :, 1]
    B = image_array[:, :, 2]

    # Create a boolean mask based on background color thresholds
    mask = (R > BG_R_MIN) & (G > BG_G_MIN) & (G < BG_G_MAX) & (B > BG_B_MIN) & (B < BG_B_MAX)

    filtered_image = np.zeros_like(image_array)
    filtered_image[mask] = image_array[mask]

    return filtered_image, mask

def get_tile_thumb_window_center(tile_name: str, target_mpp: float, seg_thumb_width: int, seg_thumb_height: int) -> np.ndarray:
    """
    Calculates the center of a tile's window in thumbnail coordinates.
    
    Args:
        tile_name (str): The name of the tile.
        target_mpp (float): The target microns per pixel (MPP) for the slide.
        seg_thumb_width (int): The width of the segmentation thumbnail.
        seg_thumb_height (int): The height of the segmentation thumbnail.
    
    Returns:
        np.ndarray: A 2D NumPy array representing the thumbnail coordinates
                    of the tile's window center.
    """
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

def segment_tiles_for_cancer_from_annotations(args: argparse.Namespace, tumor_polygons: List[np.ndarray], non_tumor_polygons: List[np.ndarray], segment_map: np.ndarray) -> np.ndarray:
    """
    Populate a segmentation map from tumor and non-tumor annotation polygons.

    Args:
        args (argparse.Namespace): Parsed CLI arguments (expects `seg_thumb_height` and `seg_thumb_width`).
        tumor_polygons (List[np.ndarray]): List of polygons for tumor regions (Nx2 arrays).
        non_tumor_polygons (List[np.ndarray]): List of polygons for non-tumor regions (Nx2 arrays).
        segment_map (np.ndarray): Pre-allocated segmentation map to populate.

    Returns:
        np.ndarray: The updated segmentation map.
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
                    segment_map[i, j] = SEGMENT_TUMOR
            for poly in nt_polys:
                if poly.contains(Point(slide_point)):
                    segment_map[i, j] = SEGMENT_NON_TUMOR

    return segment_map

def segment_tiles_for_her2_from_annotations(args: argparse.Namespace, her2_polygons: List[np.ndarray], her2_labels: List[Any], segment_map: np.ndarray) -> np.ndarray:
    """
    Populate a segmentation map using HER2 annotation polygons and labels.

    Args:
        args (argparse.Namespace): Parsed CLI arguments (expects `seg_thumb_height` and `seg_thumb_width`).
        her2_polygons (List[np.ndarray]): List of polygons for HER2-labeled regions.
        her2_labels (List[Any]): Corresponding labels (convertible to int when possible).
        segment_map (np.ndarray): Pre-allocated segmentation map to populate.

    Returns:
        np.ndarray: The updated segmentation map with HER2 labels.
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
                    label = int(her2_labels[idx])
                except BaseException as e:
                    print(f"Error converting HER2 label {her2_labels[idx]} to int: {e}")
                    continue

                if poly.contains(Point(slide_point)):
                    segment_map[i, j] = label

    return segment_map


def segment_tiles(args: argparse.Namespace, segment_map: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    """
    Downsample a high-resolution segmentation map into thumbnail tiles by computing
    the modal label over non-overlapping windows.

    Args:
        args (argparse.Namespace): Parsed CLI arguments (expects `seg_thumb_height` and `seg_thumb_width`).
        segment_map (np.ndarray): High-resolution segmentation map.
        bg_mask (np.ndarray): Boolean mask indicating background pixels.

    Returns:
        np.ndarray: Downsampled segmentation map of shape `(seg_thumb_height, seg_thumb_width)`.
    """
    segment_map[bg_mask] = SEGMENT_BACKGROUND

    # Target window size
    target_shape = (args.seg_thumb_height, args.seg_thumb_width)
    window_size = round(segment_map.shape[0] / target_shape[0])

    # Compute padding needed
    pad_h = window_size * target_shape[0] - segment_map.shape[0]
    pad_w = window_size * target_shape[1] - segment_map.shape[1]

    # Pad with background value
    arr_padded = np.pad(segment_map, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=SEGMENT_BACKGROUND)

    # Reshape into (H_tiles, window_h, W_tiles, window_w) then swap axes to group windows
    reshaped = arr_padded.reshape(target_shape[0], window_size, target_shape[1], window_size).swapaxes(1, 2)

    # Flatten each window to compute the mode
    windows = reshaped.reshape(target_shape[0], target_shape[1], -1)

    output = np.full(target_shape, fill_value=SEGMENT_BACKGROUND, dtype=int)

    # Threshold for proportion of zeros (background)
    zero_thresh = 0.05 * (window_size ** 2)

    for i in range(target_shape[0]):
        for j in range(target_shape[1]):
            window = windows[i, j]
            zero_count = np.count_nonzero(window == 0)
            if zero_count > zero_thresh:
                output[i, j] = 0
            else:
                output[i, j] = mode(window, keepdims=False).mode

    return output

def save_segmented_tiles(segment_map: np.ndarray, segment_dir: str, save_name: str, value_label: str, vmax: int = 1) -> None:
    """
    Save a segmentation map both as a numpy array and as an image with a colorbar/legend.

    Args:
        segment_map (np.ndarray): Segmented tile map to save.
        segment_dir (str): Directory where outputs will be written.
        save_name (str): Basename for saved files (without extension).
        value_label (str): Label for the colorbar axis.
        vmax (int, optional): Maximum value for color scaling. Defaults to 1.
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

def parse_coordinates(coord_str: str) -> np.ndarray:
    """
    Parse coordinates from a compact string into an Nx2 numpy array.

    Expected format: '30088.0x40076.0y | 27018.0x44149.0y | ...'

    Args:
        coord_str (str): Coordinate string to parse.

    Returns:
        np.ndarray: Array of shape (N, 2) with float coordinates [[x1, y1], ...].
    """
    coords: List[List[float]] = []
    for part in coord_str.split(" | "):
        x_str, y_str = part.split("x")
        x = float(x_str)
        y = float(y_str.replace("y", ""))
        coords.append([x, y])
    return np.array(coords)

def is_inside_polygon(test_point: np.ndarray, polygon_coords: np.ndarray) -> bool:
    """
    Return True if a 2D point is inside the polygon defined by polygon_coords.

    Args:
        test_point (np.ndarray): Array-like point [x, y].
        polygon_coords (np.ndarray): Polygon vertex coordinates of shape (N, 2).

    Returns:
        bool: True if the point is strictly inside the polygon.
    """
    polygon = Polygon(polygon_coords)
    point = Point(test_point)
    return polygon.contains(point)

def load_file_cancer_polygons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert an annotations DataFrame into grouped tumor/non-tumor polygon lists per file.

    Args:
        df (pd.DataFrame): DataFrame with columns `file`, `Coordinates`, and `Annotation`.

    Returns:
        pd.DataFrame: DataFrame with columns `file`, `tumor_polygons`, `non_tumor_polygons`.
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

def load_file_her2_polygons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group HER2 annotation polygons and labels by file.

    Args:
        df (pd.DataFrame): DataFrame with columns `file`, `Coordinates`, and `Annotation`.

    Returns:
        pd.DataFrame: DataFrame with columns `file`, `polygons`, and `Annotation` (list of labels).
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

def read_slide_file(excel_path: str, args: argparse.Namespace, only_read: bool = False) -> pd.DataFrame:
    """
    Read slide metadata from a CSV (or Excel) and optionally group annotations.

    Args:
        excel_path (str): Path to the CSV/Excel file containing slide metadata.
        args (argparse.Namespace): Parsed CLI arguments controlling annotation grouping.
        only_read (bool): If True, just return the read DataFrame without grouping.

    Returns:
        pd.DataFrame: Slides metadata, possibly grouped for annotations.
    """
    # Read slide metadata
    try:
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

def segment_from_model_or_annotations(args: argparse.Namespace, row: pd.Series, slide_file: str, segment_dir: str, st: str) -> Optional[np.ndarray]:
    """
    Build a tile-level segmentation map either from model outputs or from annotation polygons.

    Args:
        args (argparse.Namespace): Parsed CLI arguments selecting behavior.
        row (pd.Series): Row from slides dataframe with annotation information.
        slide_file (str): Path to the slide file.
        segment_dir (str): Directory where segmentation assets live.
        st (str): Path to precomputed segmented tiles file.

    Returns:
        Optional[np.ndarray]: Tile-level segmentation map, or None on failure.
    """
    if args.registration_annotated:
        owc = os.path.join(segment_dir, 'overlay_with_colorbar.npy')
        pred = os.path.join(segment_dir, 'wsi_pred.npy')
        if not os.path.exists(st):
            if os.path.exists(owc):
                owc_arr = np.load(owc)
                bg_filter, bg_mask = filter_background(owc_arr)
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

    elif args.cancer_annotations or args.her2_annotations:
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir, exist_ok=True)
        
        # Initialize a blank segment map
        tile_segment = np.full((args.seg_thumb_height, args.seg_thumb_width), fill_value=SEGMENT_BACKGROUND, dtype=int)
        slide = openslide.OpenSlide(slide_file)
        bbox_x = int(slide.properties['openslide.bounds-x'])
        bbox_y = int(slide.properties['openslide.bounds-y'])
    
        if args.cancer_annotations:
            tumor_polygons = row['tumor_polygons']
            non_tumor_polygons = row['non_tumor_polygons']
            
            if tumor_polygons is None:
                print(f"No tumor polygon for slide {slide_file}. Skipping...")
                return None
            
            for tumor_polygon in tumor_polygons:
                tumor_polygon[:, 0] += bbox_y
                tumor_polygon[:, 1] += bbox_x
            for non_tumor_polygon in non_tumor_polygons:
                non_tumor_polygon[:, 0] += bbox_y
                non_tumor_polygon[:, 1] += bbox_x
            tile_segment = segment_tiles_for_cancer_from_annotations(args=args,tumor_polygons=tumor_polygons, non_tumor_polygons=non_tumor_polygons, segment_map=tile_segment)
            save_name = 'Tumor Annotations'

        elif args.her2_annotations:
            her2_polygons = row['polygons']
            her2_labels = row['Annotation']
            
            if her2_polygons is None:
                print(f"No HER2 polygon for slide {slide_file}. Skipping...")
                return None
            
            for tumor_polygon in her2_polygons:
                tumor_polygon[:, 0] += bbox_y
                tumor_polygon[:, 1] += bbox_x

            tile_segment = segment_tiles_for_her2_from_annotations(args=args, her2_polygons=her2_polygons, her2_labels=her2_labels, segment_map=tile_segment)
            save_name = f'HER2 Annotations'

    save_segmented_tiles(segment_map=tile_segment, segment_dir=segment_dir, save_name=save_name, value_label='HER2 Class', vmax=3)
    return tile_segment


# def save_tumor_indices_and_cancer_probs(args: argparse.Namespace, full_ti_dir: str, ti: str, nti: str, cp: Optional[str], segment_dir: str, slide_name: str, valid_tiles: List[str], valid_indices: List[int], tile_segment: np.ndarray) -> Optional[np.ndarray]:
def save_tumor_indices_and_cancer_probs(args: argparse.Namespace, slide_ctx: SlideContext) -> Optional[np.ndarray]:
    """
    Save tumor / non-tumor tile indices derived from a segmentation map or cancer probabilities.

    Args:
        args (argparse.Namespace): Parsed CLI arguments controlling behavior.
        slide_ctx (SlideContext): Context object containing paths, tile info, and segmentation map.

    Returns:
        Optional[np.ndarray]: Array of tumor indices written to `ti`, or None on failure.
    """
    tumor_indices = []
    non_tumor_indices = []
    if args.seg_from_my_cancer_predictions:
        if os.path.exists(slide_ctx.paths.cp):
            cancer_prob = np.load(slide_ctx.paths.cp)
            cancer_prob = cancer_prob[slide_ctx.valid_indices]  # Filter to valid tiles
        else:
            print(f"cancer_prob file {slide_ctx.paths.cp} does not exist. Skipping...")
            return None
        tile_pred_segment = np.full((args.seg_thumb_height, args.seg_thumb_width), fill_value=SEGMENT_UNKNOWN, dtype=float)
        
    for idx, tile_name in tqdm(enumerate(slide_ctx.valid_tiles), desc=f"Processing tiles for {slide_ctx.slide_name}"):
        he_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

        if args.save_tumor_indices_from_seg_map:
            if slide_ctx.tile_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] == SEGMENT_TUMOR:
                tumor_indices.append(idx)
            if slide_ctx.tile_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] == SEGMENT_NON_TUMOR:
                non_tumor_indices.append(idx)

        if args.seg_from_my_cancer_predictions:
            tile_pred_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] = cancer_prob[idx]
            if cancer_prob[idx] >= 0.5:
                tumor_indices.append(idx)

    # Save segmentation visualization if using model predictions                
    if args.seg_from_my_cancer_predictions:
        save_segmented_tiles(segment_map=tile_pred_segment, segment_dir=slide_ctx.paths.segment_dir, save_name=f'Binary Local Cancer Probability (MPP={args.source_mpp} GigaPath features trained model) val_fold = {args.val_fold}',
                             value_label='Cancer Probability')

    # Save tumor/non-tumor indices if requested (either from seg map or from cancer predictions)
    if args.save_tumor_indices_from_seg_map or args.seg_from_my_cancer_predictions:
        if len(tumor_indices) == 0:
            print(f"No tumor tiles found for slide {slide_ctx.slide_name}.")

        if not os.path.exists(slide_ctx.paths.full_ti_dir):
            os.makedirs(slide_ctx.paths.full_ti_dir, exist_ok=True)

        tumor_indices = np.array(tumor_indices)
        print(f"Saving tumor indices for {slide_ctx.slide_name} to {slide_ctx.paths.ti}")
        np.save(slide_ctx.paths.ti, tumor_indices)

        if args.cancer_annotations:
            non_tumor_indices = np.array(non_tumor_indices)
            print(f"Saving non-tumor indices for {slide_ctx.slide_name} to {slide_ctx.paths.nti}")
            np.save(slide_ctx.paths.nti, non_tumor_indices)
    
    return tumor_indices

def save_ensemble_tumor_indices(args: argparse.Namespace, slide_ctx: SlideContext) -> Optional[np.ndarray]:
    """
    Create and save tumor indices using an ensemble of cancer probability files (all models must agree).

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        slide_ctx (SlideContext): Context object containing paths, tile info, and segmentation map.

    Returns:
        Optional[np.ndarray]: None on failure.
    """
    tumor_indices = []
    if os.path.exists(os.path.dirname(slide_ctx.paths.cp)):
        cancer_probs = os.listdir(os.path.dirname(slide_ctx.paths.cp))
        cp_npys = []
        for cancer_prob_file in cancer_probs:
            if cancer_prob_file.endswith('.npy'):
                cp_path = os.path.join(os.path.dirname(slide_ctx.paths.cp), cancer_prob_file)
                if os.path.exists(cp_path):
                    cancer_prob = np.load(cp_path)
                    cancer_prob = cancer_prob[slide_ctx.valid_indices]  # Filter to valid tiles
                    cp_npys.append(cancer_prob)
                else:
                    print(f"cancer_prob file {cp_path} does not exist. Skipping...")
        cp_npys = np.array(cp_npys)

        ensemble_str = ''
        if args.all_models_cancer:
            # tiles where all models predict cancer (prob >= 0.5) should hold the mean, otherwise set to 0
            all_models_predict_cancer = np.all(cp_npys >= 0.5, axis=0)
            ensemble_cancer_prob = np.where(all_models_predict_cancer, np.mean(cp_npys, axis=0), 0)
        else:
            ensemble_cancer_prob = np.mean(cp_npys, axis=0)
            ensemble_str = 'Mean '

        tile_pred_segment = np.full((args.seg_thumb_height, args.seg_thumb_width), fill_value=SEGMENT_UNKNOWN, dtype=float)
        for idx, tile_name in tqdm(enumerate(slide_ctx.valid_tiles), desc=f"Processing tiles for {slide_ctx.slide_name}"):
            he_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

            tile_pred_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] = ensemble_cancer_prob[idx]
            if ensemble_cancer_prob[idx] >= 0.5:
                tumor_indices.append(idx)
        
        save_segmented_tiles(segment_map=tile_pred_segment, segment_dir=slide_ctx.paths.segment_dir, save_name=f'Ensemble {ensemble_str}Binary Local Cancer Probability (MPP={args.source_mpp} GigaPath features trained model)',
                            value_label='Cancer Probability')

        if len(tumor_indices) == 0:
            print(f"No tumor tiles found for slide {slide_ctx.slide_name}.")
            # return None

        tumor_indices = np.array(tumor_indices)
        print(f"Saving tumor indices for {slide_ctx.slide_name} to {slide_ctx.paths.ti}")
        np.save(slide_ctx.paths.ti, tumor_indices)
        
    else:
        print(f"cancer_prob file {slide_ctx.paths.cp} does not exist. Skipping...")
        return None



def save_her2_annotations(args: argparse.Namespace, full_ant_dir: str, slide_name: str, valid_tiles: List[str], tile_segment: np.ndarray) -> None:
    """
    Extract annotated tile indices and ground-truth HER2 labels from a segmentation map and save them.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        full_ant_dir (str): Directory where annotation outputs will be stored.
        slide_name (str): Slide identifier.
        valid_tiles (List[str]): List of valid tile filenames.
        tile_segment (np.ndarray): Segmentation map with HER2 labels.
    """
    annotated_indices = []
    tile_gt = []

    for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
        ihc_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

        if tile_segment[ihc_thumb_window_center[0][0], ihc_thumb_window_center[0][1]] != SEGMENT_UNKNOWN:
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


def local_y_maps(args: argparse.Namespace, tile_y: np.ndarray, segment_dir: str, slide_name: str, valid_tiles: List[str], save_name: str) -> None:
    """
    Create and save a tile-level map visualizing per-tile HER2 contribution `y`.

    Args:
        args (argparse.Namespace): Parsed CLI arguments (expects `seg_thumb_height`/`seg_thumb_width`).
        tile_y (np.ndarray): Array of per-tile y values aligned with `valid_tiles`.
        segment_dir (str): Directory to save the y map.
        slide_name (str): Slide identifier.
        valid_tiles (List[str]): Filenames of tiles corresponding to `tile_y`.
        save_name (str): Name for the saved visualization.
    """
    tile_pred_segment = np.full((args.seg_thumb_height, args.seg_thumb_width), fill_value=-1, dtype=float)

    for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
        ihc_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

        tile_pred_segment[ihc_thumb_window_center[0][0], ihc_thumb_window_center[0][1]] = tile_y[idx]

    save_segmented_tiles(segment_map=tile_pred_segment, segment_dir=segment_dir, save_name=save_name, value_label='Tile HER2 contribution (y)', vmax=3)
    

def extract_tile_y_from_y_map(args: argparse.Namespace, y_map: np.ndarray, tile_y_dir: str, slide_name: str, valid_tiles: List[str], save_name: str) -> None:
    """
    Extract per-tile y values from a y_map and save as a 1-D numpy array aligned with `valid_tiles`.

    Args:
        args (argparse.Namespace): Parsed CLI arguments (expects `seg_thumb_height`/`seg_thumb_width`).
        y_map (np.ndarray): 2D map of y values.
        tile_y_dir (str): Directory where the extracted tile_y will be saved.
        slide_name (str): Slide identifier.
        valid_tiles (List[str]): Filenames of tiles to extract values for.
        save_name (str): Filename for the saved numpy array (including extension).
    """
    tile_y = []

    for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
        ihc_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

        tile_y.append(y_map[ihc_thumb_window_center[0][0], ihc_thumb_window_center[0][1]])

    tile_y = np.array(tile_y)
    save_path = os.path.join(tile_y_dir, save_name)
    np.save(save_path, tile_y)


def extract_tumor_indices_from_cancer_map(args: argparse.Namespace, cancer_map: np.ndarray, tumor_indices_dir: str, cancer_probs_dir: str, slide_name: str, valid_tiles: List[str], save_name: str) -> None:
    """
    From a per-pixel cancer probability map, extract tumor tile indices and save both
    tumor indices and the per-tile cancer probabilities.

    Args:
        args (argparse.Namespace): Parsed CLI arguments (expects `seg_thumb_height`/`seg_thumb_width`).
        cancer_map (np.ndarray): 2D array of cancer probabilities.
        tumor_indices_dir (str): Directory where tumor indices will be saved.
        cancer_probs_dir (str): Directory where per-tile cancer probs will be saved.
        slide_name (str): Slide identifier.
        valid_tiles (List[str]): Filenames of tiles to consider.
        save_name (str): Filename for the saved tumor indices npy.
    """
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


def convert_he_y_map_to_ihc_y_map(args, he_y_map: np.array, segment_dir: str, slide_name: str, valid_tiles: np.array, save_name: str) -> None:
    """
    Convert H&E y-map to IHC y-map. This is done by taking the y value from the H&E y-map at the center of each tile and creating a new map with those values at the corresponding tile locations.

    Args:
    args (argparse.Namespace): Parsed CLI arguments (expects `seg_thumb_height`/`seg_thumb_width`).
        he_y_map (np.ndarray): 2D array of y values from the H&E map.
        segment_dir (str): Directory to save the converted y map.
        slide_name (str): Slide identifier.
        valid_tiles (List[str]): Filenames of tiles to consider for conversion.
        save_name (str): Filename for the saved converted y map (including extension).
    """
    tile_pred_segment = np.full((args.seg_thumb_height, args.seg_thumb_width), fill_value=-1, dtype=float)
        
    for idx, tile_name in tqdm(enumerate(valid_tiles), desc=f"Processing tiles for {slide_name}"):
        he_thumb_window_center = get_tile_thumb_window_center(tile_name, args.target_mpp, seg_thumb_width=args.seg_thumb_width, seg_thumb_height=args.seg_thumb_height)

        tile_pred_segment[he_thumb_window_center[0][0], he_thumb_window_center[0][1]] = he_y_map[he_thumb_window_center[0][0], he_thumb_window_center[0][1]]
            
    save_segmented_tiles(segment_map=tile_pred_segment, segment_dir=segment_dir, save_name=save_name,
                         value_label='Tile HER2 contribution (y) from HE', vmax=3)
    

def main():
    args = parse_args()
    print(f'args = {args}')

    setup_args(args)
    paths = build_paths(args)
    metadata = load_metadata(args)
    collectors = initialize_collectors(args)

    slides_df = metadata["slides_df"]

    for _, row in tqdm(slides_df.iterrows(), total=len(slides_df)):
        # if '19-14590' not in row['SlideName']:
        #     continue
        try:
            slide_ctx = prepare_slide_context(args=args, row=row, paths=paths, metadata=metadata)
            process_slide(args=args, slide_ctx=slide_ctx, collectors=collectors)

        except Exception as e:
            print(f"Failed processing " f"{row['SlideName']}:\n{e}")

            continue

    finalize_outputs(args=args, collectors=collectors, paths=paths)


if __name__ == '__main__':
    main()

    # pass