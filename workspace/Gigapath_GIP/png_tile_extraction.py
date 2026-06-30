import os
import shutil
import ast
import numpy as np
import pandas as pd
# import openslide
import pickle
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
import cv2
import math
import argparse
import torch

import timm
import gigapath.slide_encoder as slide_encoder
from finetune.utils import map_colors
from gigapath.pipeline import load_tile_slide_encoder, run_inference_with_tile_encoder, run_inference_with_slide_encoder, \
load_virchow2_tile_encoder, run_inference_with_virchow2_tile_encoder, load_uni2_tile_encoder, run_inference_with_uni2_tile_encoder, \
load_conch_tile_encoder, run_inference_with_conch_tile_encoder
from finetune.cycle_gan import ResNetGenerator
from torchvision import transforms
from torchvision.utils import save_image

# Set the hf token
# with open("/home/amitf/workspace/Gigapath_GIP/finetune/hf_token.txt", "r") as file:
# with open("/home/amitf/workspace/Gigapath_GIP/finetune/hf_read_token_virchow2.txt", "r") as file:
#     os.environ["HUGGINGFACE_HUB_TOKEN"] = file.read()
#     os.environ["HF_TOKEN"] = file.read()

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_colormap = False


# Function to encode a slide and save its embeddings in the tile and slide level
def encode_slide(slide_patches_path, slide_name, slide_encoder_model, tile_encoder, output_dir, preprocess=None, only_tile=False, external_tile_encoder='virchow2'):
    # tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)
    # outputs a dict with keys 'tile_embeds' and 'coords'. 'tile_embeds' is a tensor of shape (N, 1536) where N is the number of tiles
    # 'coords' is a tensor of shape (N, 2) containing the coordinates of the tiles. The i-th row of 'coords' corresponds to the i-th row of 'tile_embeds'
    # slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)
    # outputs a dict with keys 'layer_<1, 2, ..., 12, last>_embed' where each value is a tensor containing the embeddings of the slide at the corresponding layer
    # 'final' is the final layer of the slide encoder.
    # We want to lave for each slide all of its tile embeddings and the 6-th and final layer slide embeddings.
    # We will save the embeddings in seperate files, in the same folder, with the same name as the slide.
    # The embeddings will be saved in the following format:
    # tile_embeds_<slide_name>.npy
    # layer_6_embed_<slide_name>.npy
    # final_embed_<slide_name>.npy
    # The embeddings will be saved in the output_dir
    # The slide patches are saved in the slide_patches_path
    # The slide name is the name of the slide
    # The slide encoder model is the slide encoder model
    # The tile encoder is the tile encoder
    # The output_dir is the directory where the embeddings will be saved
    # The function does not return anything

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the slide patches
    image_paths = [os.path.join(slide_patches_path, img) for img in os.listdir(slide_patches_path) if img.endswith('.png')]
    if only_tile:
        if external_tile_encoder == 'virchow2':
            # Run inference with the Virchow2 tile encoder
            tile_encoder_outputs = run_inference_with_virchow2_tile_encoder(image_paths, tile_encoder)
        elif external_tile_encoder == 'uni2':
            # Run inference with the Uni2 tile encoder
            tile_encoder_outputs = run_inference_with_uni2_tile_encoder(image_paths, tile_encoder)
        elif external_tile_encoder == 'conch':
            # Run inference with the Conch tile encoder
            tile_encoder_outputs = run_inference_with_conch_tile_encoder(image_paths, tile_encoder, preprocess)
    else: 
        # Run inference with the tile encoder
        tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)
    # Save the tile embeddings
    np.save(os.path.join(output_dir, f"tile_embeds_{slide_name}.npy"), tile_encoder_outputs['tile_embeds'])

    if not only_tile:
        # Run inference with the slide encoder
        slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)
        # Save the 6-th layer embeddings
        np.save(os.path.join(output_dir, f"layer_6_embed_{slide_name}.npy"), slide_embeds['layer_6_embed'])
        # Save the final layer embeddings
        np.save(os.path.join(output_dir, f"final_embed_{slide_name}.npy"), slide_embeds['last_layer_embed'])


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

def tiles_overlap(new_coord, existing_coord, tile_size):
    x_new, y_new = new_coord
    x_exis, y_exis = existing_coord
    return ((x_exis <= x_new < x_exis + tile_size) and (y_exis <= y_new < y_exis + tile_size or y_new <= y_exis < y_new + tile_size))

def remove_overlapping_tiles(coords, tile_size):
    kept = []
    for coord in coords:
        if all(not tiles_overlap(coord, existing, tile_size) for existing in kept):
            kept.append(coord)
    return kept

# Function to extract and save patches
def extract_patches(slide_path, segmentation_path, output_dir, mpp, target_mpp = 0.5):
    slide = openslide.open_slide(slide_path)
    mpp_seg = 1
    output_tile_size = 256

    # Calculate the closest level for target MPP
    downsample_factors = [float(level_downsample) for level_downsample in slide.level_downsamples]
    closest_level = find_best_level(mpp, target_mpp, downsample_factors)
    scale_factor_image = target_mpp / (mpp * downsample_factors[closest_level])
    best_level_output_tile_size = int(round(output_tile_size * scale_factor_image))
    scale_factor_level_0_to_half_mpp = target_mpp / mpp
    scale_factor_seg = mpp_seg / mpp
    level_0_seg_tile_size = output_tile_size * scale_factor_seg
    #level_0_output_tile_size = int(round(level_0_seg_tile_size // 2.))
    level_0_output_tile_size = int(round(level_0_seg_tile_size // (mpp_seg / target_mpp)))

    # Load segmentation data
    with open(segmentation_path, 'rb') as f:
        tiles_coords = pickle.load(f)
        
    if len(tiles_coords) == 0:
        return False
    elif target_mpp > 1:
        tiles_coords = remove_overlapping_tiles(tiles_coords, tile_size=level_0_output_tile_size)
    
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    for (x, y) in tiles_coords:
        for i in range(min(int(math.ceil(mpp_seg / target_mpp)),2)):
            for j in range(min(int(math.ceil(mpp_seg / target_mpp)),2)):
                coord_x = x + i * level_0_output_tile_size
                coord_y = y + j * level_0_output_tile_size
                try:
                    patch = slide.read_region((coord_y, coord_x), closest_level, (best_level_output_tile_size, best_level_output_tile_size)).convert('RGB')
                except:
                    print(f"couldn't read the coords {coord_y}, {coord_x} at level {closest_level} with size {best_level_output_tile_size}")
                    raise
                
                # Convert patch to numpy array for cv2.resize
                patch_np = np.array(patch)
                if to_colormap:
                    patch_np = map_colors(patch_np.transpose((2,0,1)).transpose((1,2,0)))
                resized_patch = cv2.resize(patch_np, (output_tile_size, output_tile_size))
                
                # Convert back to PIL Image
                resized_patch_img = Image.fromarray(resized_patch)
                patch_name = f"{int(round(coord_x / scale_factor_level_0_to_half_mpp))}x_{int(round(coord_y / scale_factor_level_0_to_half_mpp))}y.png"
                resized_patch_img.save(os.path.join(output_dir, patch_name))
    return True

def find_best_level(mpp, target_mpp, downsample_factors):
    best_level = 0
    while mpp * downsample_factors[best_level] <= target_mpp:
        if len(downsample_factors)==best_level+1 or mpp * downsample_factors[best_level+1] > target_mpp:
            return best_level
        else:
            best_level += 1
    print(best_level)
    print(len(downsample_factors))
    print(mpp * downsample_factors[best_level])
    print(mpp * downsample_factors[best_level+1])
    raise "slide level 0 mpp is smaller than target"

def generate_synth_ihc(slide_name, output_dir_patches, output_dir_features, g_HE_IHC, tile_encoder, args):
    patches_names = [img for img in os.listdir(output_dir_patches) if img.endswith('.png')]
    image_paths = [os.path.join(output_dir_patches, img) for img in patches_names]
    output_dir_syn_patches = output_dir_patches.replace('png_tiles', 'synth_ihc_tiles')
    output_dir_syn_features = output_dir_features.replace('gigapath_features', 'synth_ihc_gigapath_features')
    os.makedirs(output_dir_syn_patches, exist_ok=True)
    os.makedirs(output_dir_syn_features, exist_ok=True)

    tile_bsz = 16
    transform = transforms.Compose([transforms.ToTensor()])
    syn_img_tensor_list = []
    # Generate synthetic IHC images from HE images
    tiles = torch.stack([transform(Image.open(img_path).convert('RGB')) for img_path in image_paths]).to(args.device)
    with torch.no_grad():
        for i in range(0, len(tiles), tile_bsz):
            curr_tiles = tiles[i:i+tile_bsz]
            syn_img_tensor = g_HE_IHC(curr_tiles)
            syn_img_tensor_list.append(syn_img_tensor.cpu())
    syn_img_tensor = torch.cat(syn_img_tensor_list, dim=0)
    for i, patch_name in zip(range(syn_img_tensor.shape[0]), patches_names):
        save_image(syn_img_tensor[i] * 0.5 + 0.5, os.path.join(output_dir_syn_patches, f"{patch_name}"))
    
    # Run inference with the tile encoder on the synthetic IHC images
    syn_image_paths = [os.path.join(output_dir_syn_patches, img) for img in os.listdir(output_dir_syn_patches) if img.endswith('.png')]
    tile_encoder_outputs = run_inference_with_tile_encoder(syn_image_paths, tile_encoder)
    # Save the tile embeddings
    np.save(os.path.join(output_dir_syn_features, f"tile_embeds_{slide_name}.npy"), tile_encoder_outputs['tile_embeds'])


def main():
    def parse_list(input_string):
        return ast.literal_eval(input_string)
    
    # Argument parser
    parser = argparse.ArgumentParser(description="Extract Gigapath tiles and their features.")
    parser.add_argument('-b', '--base_path', type=str, help='Slide files path', required=True)
    parser.add_argument('-s', '--save_path', type=str, help='Path to save tiles and embeds', required=True)
    parser.add_argument('-e', '--excel_path', type=str, help='Slides excel path', required=True)
    parser.add_argument('-src', '--src_path', type=str, help='Path for files to copy')
    parser.add_argument('-c', '--copy_batches', action='store_true', default=False, help='Copy tiles and features of specific batches')
    parser.add_argument('-bch', '--batch', type=parse_list, default=[8,9,11], help='Batches to copy')
    parser.add_argument('-her2', '--her2_available', action='store_true', default=False, help='Filter only slides with Her2 annotation available')
    parser.add_argument('-g', '--generate_synth_ihc', action='store_true', default=False, help='Use cycle GAN to generate synthetic IHC images from HE images')
    parser.add_argument('-sf', '--synth_fold', type=int, help='Fold to generate synthetic IHC images from HE images', choices=[1, 2, 3, 4, 5, 6], default=-1)
    parser.add_argument('-gckpt', '--gan_ckpt', type=str, help='Checkpoint for the GAN model')
    parser.add_argument('-mpp', '--mpp', type=float, default=0.5, help='Target MPP to extract patches with')
    parser.add_argument('-ot', '--only_tile_encode', action='store_true', default=False, help='Only run tile encoding without slide encoding')
    parser.add_argument('-ete', '--external_tile_encoder', type=str, choices=['virchow2', 'conch', 'uni2'], default='virchow2', help='Use external tile encoder instead of Gigapath')
    parser.add_argument('-hf', '--hf_token', type=str, default=os.environ.get("HF_TOKEN"), help='Hugging Face token for model access')

    args = parser.parse_args()
    print(f'args = {args}')

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Set the environment variable for the Hugging Face token
    os.environ["HF_TOKEN"] = args.hf_token
    
    # Paths
    base_path = args.base_path
    excel_path = args.excel_path
    save_path = args.save_path
    target_mpp = args.mpp if args.mpp % 1 != 0 else int(args.mpp) # 2 # 1 
    print(f"Extracting patches at {target_mpp} mpp")
    segmentation_dir = os.path.join(base_path, "Grids_10")
    process_ind = "5"
    progress_file = os.path.join(base_path, f"progress{process_ind}.txt")


    # load the tile encoder
    # tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    # # load the slide encoder
    # slide_encoder_model = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
    preprocess = None
    if args.only_tile_encode:
        if args.external_tile_encoder == 'virchow2':
            tile_encoder = load_virchow2_tile_encoder()
        elif args.external_tile_encoder == 'uni2':
            tile_encoder = load_uni2_tile_encoder()
        elif args.external_tile_encoder == 'conch':
            tile_encoder, preprocess = load_conch_tile_encoder()
        else:
            raise ValueError(f"Unknown external tile encoder: {args.external_tile_encoder}")
        slide_encoder_model = None
    else:
        tile_encoder, slide_encoder_model = load_tile_slide_encoder()
    

    # Read slide metadata
    try:
        if excel_path.endswith('.xlsx'):
            slides_df = pd.read_excel(excel_path)
        else:
            slides_df = pd.read_csv(excel_path)
    except Exception as e:
        raise FileNotFoundError(f"Error reading the Excel file: {e}")

    # Check if 'mpp' column exists
    if 'MPP' not in slides_df.columns:
        raise KeyError(f"Column 'MPP' not found in the Excel file. Available columns: {slides_df.columns.tolist()}")

    # Drop rows with null or string 'MPP' values
    slides_df = slides_df.dropna(subset=['MPP'])
    slides_df = slides_df[slides_df['MPP'].apply(lambda x: isinstance(x, (int, float)))]

    # Round the 'mpp' column values
    slides_df['MPP'] = slides_df['MPP'].apply(round_mpp)

    slides_df = slides_df.dropna(subset=['MPP'])
    slides_df = slides_df[slides_df['MPP'].apply(lambda x: isinstance(x, (int, float)))]

    if args.her2_available:
        slides_df = slides_df[slides_df['Her2 status'].isin(['Positive', 'Negative', 'Equivocal'])]

    if args.copy_batches and args.batch is not None:
        # batches_str = '|'.join([f"Batch_{b}" for b in args.batch])
        batches_str = f"Batch_{args.batch}"
        mask = slides_df['path'].str.contains(batches_str, na=False)
        slides_df = slides_df[mask].reset_index(drop=True)

    # Load progress if it exists
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            start_index = int(f.read().strip())
    else:
        start_index = 0
    end_file = os.path.join(base_path, f"end{process_ind}.txt")
    if os.path.exists(end_file):
        with open(end_file, 'r') as f:
            end_index = int(f.read().strip())
    else:
        end_index = None

    slides_df = slides_df[start_index:end_index]

    if args.synth_fold != -1:
        # Filter slides from the specified fold
        slides_df = slides_df[slides_df['fold'] == args.synth_fold]
        print(f"Using fold {args.synth_fold} with {len(slides_df)} slides.")

    if args.generate_synth_ihc:
        g_HE_IHC = ResNetGenerator().to(args.device)
        g_HE_IHC.load_state_dict(torch.load(args.gan_ckpt), strict=False)

    # Process each slide with progress tracking and saving
    for idx, row in tqdm(slides_df.iterrows(), total=len(slides_df), initial=start_index):
        slide_file = row['file']
        print(f"slide = {slide_file}")
        slide_name = os.path.splitext(slide_file)[0]
        slide_path = os.path.join(base_path, slide_file)
        suffix = '_colormapped' if to_colormap else ''
        suffix += f'_mpp{target_mpp}' # if target_mpp != 0.5 else ''
        
        segmentation_path = os.path.join(segmentation_dir, f"{slide_name}--tlsz256.data")
        output_dir_patches = os.path.join(save_path, f'png_tiles{suffix}', slide_name)
        output_dir_features = os.path.join(save_path, f'gigapath_features{suffix}', slide_name)
        if args.only_tile_encode:
            output_dir_features = os.path.join(save_path, f'{args.external_tile_encoder}_features{suffix}', slide_name)

        # if not slide_file.startswith('17-5500'): 
        #     continue

        if args.copy_batches:
            src_dir_patches = os.path.join(args.src_path, f'png_tiles{suffix}', slide_name)
            src_dir_features = os.path.join(args.src_path, f'gigapath_features{suffix}', slide_name)
            if os.path.exists(src_dir_patches):
                shutil.copytree(src_dir_patches, output_dir_patches, dirs_exist_ok=True)
                shutil.copytree(src_dir_features, output_dir_features, dirs_exist_ok=True)
                continue
        
        if not os.path.exists(slide_path):
            print(f"slide: {slide_path} missing, skipping")
            with open(progress_file, 'w') as f:
                f.write(str(idx + 1))
            continue
        
        if args.generate_synth_ihc:
            generate_synth_ihc(slide_name, output_dir_patches, output_dir_features, g_HE_IHC, tile_encoder, args)
        else:
            if os.path.exists(output_dir_features):
                print(f"{output_dir_features} already exists, skipping")
                continue
            else:
                if not args.only_tile_encode:
                    try:
                        if not extract_patches(slide_path, segmentation_path, output_dir_patches, row['MPP'], target_mpp):
                            print(f"slide: {slide_path} has no legitimate tiles, skipping")
                            with open(progress_file, 'w') as f:
                                f.write(str(idx + 1))
                            continue
                    except:
                        print(f"couldn't extract patches from slide: {slide_path}")
                        raise
                encode_slide(output_dir_patches, slide_name, slide_encoder_model, tile_encoder, output_dir_features, preprocess=preprocess, only_tile=args.only_tile_encode, external_tile_encoder=args.external_tile_encoder)
        
        # Save progress
        with open(progress_file, 'w') as f:
            f.write(str(idx + 1))

    # Delete the progress file after completion
    if os.path.exists(progress_file):
        os.remove(progress_file)


if __name__ == '__main__':
    main()