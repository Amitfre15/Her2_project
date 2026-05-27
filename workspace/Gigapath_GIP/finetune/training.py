import os
import sys
from pathlib import Path

# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))

import time
import wandb
import torch
import numpy as np
import torch.utils.tensorboard as tensorboard
import pandas as pd
import gc
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

from gigapath.classification_head import get_model, TileClassificationHead
from metrics import calculate_metrics_with_task_cfg
from utils import (get_optimizer, get_loss_function, \
                   Monitor_Score, get_records_array,
                   log_writer, adjust_learning_rate,
                   get_regression_losses, LogCoshLoss, get_test_loader, 
                   Contrastive_Loss, DifferentiableHistogramKL, CLIPLoss, MidPenalizedLoss, margin_loss, Assymmetric_MSE_Loss, SoftSpearmanLoss)
from slides_to_thumbs import patch_weighted_score_matrices, show_score_matrix, SLIDE_HEIGHT, SLIDE_WIDTH, SLIDE_MPP, SLIDE_PATCH_SIZE
from datasets.slide_datatset import SlidingWindowDataset
from finetune.cycle_gan import ResNetGenerator, PatchDiscriminator, style_loss

VIRCHOW2_FEATURE_DIM = 2560
UNI2_FEATURE_DIM = 1536
CONCH_FEATURE_DIM = 512

def print_free_gpu_memory(device):
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

def slice_tiles(indices, batch):
    batch['imgs'] = batch['imgs'][:, indices, :]
    batch['coords'] = batch['coords'][:, indices, :]
    # if batch['ihc_imgs'] is not None:
    #     batch['ihc_imgs'] = batch['ihc_imgs'][:, indices, :]
    #     batch['ihc_coords'] = batch['ihc_coords'][:, indices, :]
    if batch['matching_tiles'] is not None:
        batch['matching_tiles'] = batch['matching_tiles'][:, indices]
    if batch['tile_logits'] is not None:
        batch['tile_logits'] = batch['tile_logits'][:, indices, :]
    if batch['cancer_prob'] is not None:
        batch['cancer_prob'] = batch['cancer_prob'][:, indices]
    if batch['tile_y'] is not None:
        print(f"Before slicing, batch['tile_y'].shape = {batch['tile_y'].shape}")
        batch['tile_y'] = batch['tile_y'][:, indices]    

def run_window_inference(data_df, dataset_class, args):
    data_df = data_df[data_df['id'].isin(args.test_dataset)]
    folds_list = data_df['fold']
    folds_list = pd.to_numeric(folds_list, errors='coerce').fillna(folds_list)
    data_df['fold'] = folds_list
    data_df = data_df[data_df['fold'].isin(args.test_fold)]
    data_df[args.slide_key] = data_df[args.slide_key].str.removesuffix('.svs')
    data_df[args.slide_key] = data_df[args.slide_key].str.removesuffix('.mrxs')
    data_df[args.slide_key] = data_df[args.slide_key].str.removesuffix('.tiff')
    data_df[args.slide_key] = data_df[args.slide_key].str.removesuffix('.tif')

    task_setting = args.task_config.get('setting', 'multi_class')
    if task_setting == 'continuous':
        args.n_classes = 1
    else:
        label_dict = args.task_config.get('label_dict', {})
        args.n_classes = len(label_dict)

    model = get_model(**vars(args))
    model = model.to(args.device)
    # set up the fp16 scaler
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 training')
    model.load_state_dict(torch.load(args.model_ckpt), strict=False)
    model.eval()
    results_df = pd.DataFrame({'window_name': [], 'score': [], 'label': []})

    seg_thumb_height = int(SLIDE_HEIGHT * SLIDE_MPP // (SLIDE_PATCH_SIZE * args.target_mpp))
    seg_thumb_width = int(SLIDE_WIDTH * SLIDE_MPP // (SLIDE_PATCH_SIZE * args.target_mpp))
    for slide in data_df[args.slide_key]:
        # if not slide.startswith("18-7057_2_1"):
        #     continue
        args.get_single_slide = slide
        print(f'slide = {slide}.mrxs')

        val_fold_value = args.val_fold[0] if isinstance(args.val_fold, list) else args.val_fold
        if args.create_y_map:
            npy_output_path = os.path.join(args.root_path.replace("gigapath_features", "y_map"))
            if args.pred_y_baseline or args.window_training:
                npy_output_path = npy_output_path.replace("HE", "IHC")
            block = slide.rsplit("_", 1)[0] + "_"
            try:
                matching_folder = next(filter(lambda x: block in x, os.listdir(npy_output_path)))
                npy_output_path = os.path.join(npy_output_path, matching_folder)
            except StopIteration:
                print(f"Making folder for block {block} in {npy_output_path}")
                npy_output_path = os.path.join(npy_output_path, block)
                os.makedirs(npy_output_path, exist_ok=True)
                
            npy_file = os.path.join(npy_output_path, f"Tile HER2 contribution (y) from SW (MPP={args.target_mpp} GigaPath features trained model) val_fold = {val_fold_value}.npy")
            if args.pred_y_baseline:
                npy_file = npy_file.replace("(y) from SW", "(y) from SW pred SW baseline")
            elif args.window_training:
                npy_file = npy_file.replace("(y) from SW", "(y) from SW pred SW training baseline")
        elif args.create_cancer_map:
            npy_output_path = os.path.join(args.root_path.replace("gigapath_features", "cancer_map"), slide)
            npy_file = os.path.join(npy_output_path, f"Tile cancer probability from SW (MPP={args.target_mpp} GigaPath features trained model) val_fold = {val_fold_value}.npy")
        elif args.create_HER2_status_map:
            npy_output_path = os.path.join(args.root_path.replace("gigapath_features", "HER2_status_map"), slide)
            npy_file = os.path.join(npy_output_path, f"Tile HER2 status from SW (MPP={args.target_mpp} GigaPath features trained on THUFA model).npy")

        test_data = dataset_class(data_df, args.root_path, args.task_config, slide_key=args.slide_key, label=args.label, \
                                  dataset_name=args.test_dataset, folds=args.test_fold,
                                  use_clinical_features=args.clinical_features, \
                                  test_on_all=args.test_on_all, get_single_slide=args.get_single_slide,
                                  window_size=args.window_size, stride=args.stride, val_fold=args.val_fold)
        test_loader = get_test_loader(test_data, for_heatmap=True, **vars(args))

        # Initialize score and weight matrices
        print(f'seg_thumb_height = {seg_thumb_height}, seg_thumb_width = {seg_thumb_width}')
        weighted_score_matrix = torch.zeros(size=(seg_thumb_height, seg_thumb_width), dtype=torch.float64).to(args.device)
        weight_matrix = torch.zeros(size=(seg_thumb_height, seg_thumb_width), dtype=torch.float64).to(args.device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # load the batch and transform this batch
                images, img_coords = batch['imgs'], batch['coords']
                # Separate x and y coordinates
                x_coords = img_coords[:, :, 0]  # Extract all x-coordinates
                y_coords = img_coords[:, :, 1]  # Extract all y-coordinates
                x_min, y_min = int(x_coords.min().item()), int(y_coords.min().item())
                x_max, y_max = int(x_coords.max().item()), int(y_coords.max().item())
                window_name = f'{slide}_{x_min}_{x_max}x_{y_min}_{y_max}y'
                images = images.to(args.device, non_blocking=True)
                img_coords = img_coords.to(args.device, non_blocking=True)

                with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                    # get the logits
                    logits = model(images, img_coords)
                    if not args.task_config.get('setting', 'multi_class') == 'continuous':
                        score = logits.item() if args.task_config.get('setting', 'multi_class') != 'binary' else torch.softmax(logits, dim=1)[0,1].item()
                        results = {'window_name': window_name, 'score': score,
                                   'label': test_data.label.item()}
                    else:
                        if not args.survival:
                            # un-normalize
                            score = logits # * model.std.item() + model.mean.item()
                            results = {'window_name': window_name, 'score': score.item(),
                                       'label': test_data.label.item()}

                    results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)

                patch_weighted_score_matrices(weighted_score_matrix=weighted_score_matrix, weight_matrix=weight_matrix, img_coords=img_coords,
                                              window_score=score, target_mpp=args.target_mpp, seg_thumb_width=seg_thumb_width)

        # Normalize the weighted score matrix by the weight matrix
        final_score_matrix = np.divide(
            weighted_score_matrix.cpu().numpy(),
            weight_matrix.cpu().numpy(),
            out=np.zeros_like(weighted_score_matrix.cpu().numpy()),
            where=weight_matrix.cpu().numpy() > 0  # Avoid division by zero
        )
        # replace zeros with minus one as background
        final_score_matrix[weight_matrix.cpu().numpy() == 0] = -1

        # Save the score matrix as a .npz file (NumPy compressed format)
        compressed_score_matrix = final_score_matrix.astype(np.float16)
        if not os.path.exists(npy_output_path):
            os.makedirs(npy_output_path, exist_ok=True)

        np.save(npy_file, compressed_score_matrix)
        print(f"Saved score matrix for slide: {slide} at {npy_file}")

        vmax = 3 if args.create_y_map else 1
        show_score_matrix(final_score_matrix, slide, save_dir=npy_file, vmin=-1, vmax=vmax)

        # npz_output_path = os.path.join(args.save_dir, f"{slide}_score_matrix.npz")
        # np.savez_compressed(npz_output_path, compressed_score_matrix)

        # print(f"Saved score matrix for slide: {slide} at {npz_output_path}")

    results_df.to_csv(os.path.join(args.save_dir, 'slide_scores.csv'), index=False)
    print(f'Saved slide_scores.csv')

    return


def save_plot(x, y, xlabel: str, ylabel: str, title: str, hist: bool = False, output_dir: str = ''):
    if hist:
        y_np = y.detach().cpu().numpy()
        bin_size = 0.1
        bins = np.arange(min(y_np), max(y_np) + bin_size, bin_size)
        plt.hist(y_np, bins=bins, density=True, edgecolor='black', alpha=0.75)
    else:
        plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    plt.title(f"{title}")

    # Save the figure to a file instead of showing it
    save_path = os.path.join(output_dir, f'{title}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Close the figure to free memory
    plt.close()


def generate_heatmap(heatmap_loader, args):
    # set up the model
    model = get_model(**vars(args))
    model = model.to(args.device)
    # set up the fp16 scaler
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 training')
    '''        
    task_setting = args.task_config.get('setting', 'multi_class')
    model.load_state_dict(torch.load(args.model_ckpt), strict = False)
    test_loader.dataset.update_clinical_features_params_and_handle_data(model)
    print('Testing on {} windows'.format(len(test_loader.dataset)))
    x_min, y_min = test_loader.dataset.img_coords.int().min(dim=0).values
    x_max, y_max = test_loader.dataset.img_coords.int().max(dim=0).values
    model.eval()
    scores_for_coords = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # load the batch and transform this batch
            images, img_coords = batch['imgs'], batch['coords']
            images = images.to(args.device, non_blocking=True)
            img_coords = img_coords.to(args.device, non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                # get the logits
                logits = model(images, img_coords*256)
            
            if task_setting == 'multi_label':
                Y_prob = torch.sigmoid(logits).numpy()
                scores = Y_prob.cpu().numpy()[1]
            elif task_setting == 'multi_class' or task_setting == 'binary':
                Y_prob = torch.softmax(logits, dim=1).cpu().numpy()
                scores = Y_prob.numpy()[0]                
            elif task_setting == 'continuous':
                scores = ((logits*model.std.item())+model.mean.item()).cpu().numpy()[0]
                
            #if not args.task_config.get('setting', 'multi_class') == 'continuous':
            #    results = {'slide_name': slide_id, 'score': test_records['prob'][idx][1], 'label': 0 if test_records['label'][idx][0] == 1 else 1}
            #else:
            #    results = {'slide_name': slide_id, 'score': test_records['prob'][idx][0], 'label': test_records['label'][idx][0]}
            
            img_coords = img_coords.cpu().int().numpy()
            for coord, score in zip(img_coords, scores):
                scores_for_coords[coord] = scores_for_coords.get(coord, []) + [score]
        heatmap = np.zeros([x_max-x_min,y_max-y_min])
        for coord in scores_for_coords:
            aligned_coord = [coord[0]-x_min, coord[1]-y_min]
            #scores_for_coords[aligned_coord] = np.array((scores_for_coords[aligned_coord])).mean()
            score = np.array((scores_for_coords[aligned_coord])).mean()
            heatmap[aligned_coord] = score
        '''

    model.load_state_dict(torch.load(args.model_ckpt), strict=False)
    heatmap_loader.dataset.update_clinical_features_params_and_handle_data(model)
    model.eval()

    for batch_idx, batch in enumerate(heatmap_loader):
        # load the batch and transform this batch
        images, img_coords, label = batch['imgs'], batch['coords'], batch['labels']
        # x_min, y_min = (img_coords.squeeze()/256).int().min(dim=0).values
        # x_max, y_max = (img_coords.squeeze()/256).int().max(dim=0).values
        x_max, y_max = (img_coords.squeeze()).max(dim=0).values
        # img_coords[..., 0] = x_max - img_coords[..., 0]
        images = images.to(args.device, non_blocking=True)
        images.requires_grad = True
        img_coords = img_coords.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True).long()
        # print(img_coords.int().min(dim=0).values)
        # print(img_coords.int().min(dim=0).values.shape)
        # print(img_coords.int().shape)
        for i in range(2):
            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                # get the logits
                logits = model(images, img_coords + torch.tensor([256, 0], device=img_coords.device) * i * 50)
                scores = ((logits * model.std.item()) + model.mean.item())  # .cpu().detach().numpy()[0]
            grads = torch.abs(
                torch.autograd.grad(outputs=scores, inputs=images, grad_outputs=torch.ones_like(scores))[0].squeeze())
            # scores = grads.norm(2, dim=1)
            # scores = scores.cpu().detach().numpy()
            # print(scores.max())
            # scores = scores/scores.max()*256
            # print(scores.max())
            # aligned_image_coords = (img_coords[0]/256).int().cpu().detach().numpy()
            # aligned_image_coords[:,0] -= x_min.cpu().detach().numpy()
            # aligned_image_coords[:,1] -= y_min.cpu().detach().numpy()
            # x_max -= x_min.cpu().detach().numpy()
            # y_max -= y_min.cpu().detach().numpy()
            # heatmap = np.zeros([y_max+1,x_max+1])
            # for coord, score in zip(aligned_image_coords, scores):
            #    aligned_coord = [coord[1], coord[0]]
            #    old = heatmap[coord[1], coord[0]]
            #    if old != 0:
            #        a=sslsl
            #    heatmap[coord[1], coord[0]] = score
            # print(heatmap.shape)
            # print(heatmap.max())
            # np.save(os.path.join(args.save_dir,f'{args.get_single_slide}_shift_{i}.npy'), heatmap)
            np.save(os.path.join(args.save_dir, f'{args.get_single_slide}_coords_shift_{i * 50}.npy'),
                    (img_coords + torch.tensor([256, 0], device=img_coords.device) * i * 50).cpu().detach().numpy())
            np.save(os.path.join(args.save_dir, f'{args.get_single_slide}_gradients_shift_{i * 50}.npy'),
                    grads.cpu().detach().numpy())


def test(test_loader, args):
    writer_dir = os.path.join(args.save_dir, 'tensorboard')
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir, exist_ok=True)

    # set up the writer
    writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)
    # set up writer
    if "wandb" in args.report_to:
        wandb.init(
            project='Finetune_Gigapath',
            name=args.exp_name,
            tags=[],
            config=vars(args),
        )
        writer = wandb
    elif "tensorboard" in args.report_to:
        writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)
    
    if args.add_virchow2_features:
        args.tile_input_dim += VIRCHOW2_FEATURE_DIM
    if args.add_uni2_features:
        args.tile_input_dim += UNI2_FEATURE_DIM
    if args.add_conch_features:
        args.tile_input_dim += CONCH_FEATURE_DIM

    # set up the model
    if (args.score_can_as_sb or args.score_can_as_bs or args.film) and not args.load_tile_logits and not args.tile_slide_model:  # Delete
        args.tile_model, model = get_model(**vars(args))
        args.tile_model = args.tile_model.to(args.device)
        args.tile_model.load_state_dict(torch.load(args.tile_model_ckpt), strict=False)
        args.tile_model.eval()
        print("Loaded tile model")
    else:
        model = get_model(**vars(args))
    model = model.to(args.device)
    # set up the loss function
    loss_fn = get_loss_function(args)
    # set up the fp16 scaler
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 training')

    print('Testing on {} samples'.format(len(test_loader.dataset)))

    test_records = None
    model.load_state_dict(torch.load(args.model_ckpt), strict=False)
    test_loader.dataset.update_clinical_features_params_and_handle_data(model)
    # test the model
    test_records, embeds = evaluate(test_loader, model, fp16_scaler, loss_fn, 0, args, save_embed=True) # 'test'
    results_df = pd.DataFrame({'slide_name': [], 'score': [], 'label': []})
    # save each embedding as a seperate filewith the name being the slide_id
    for idx, slide_id in enumerate(test_loader.dataset.slide_data[args.slide_key]):
        np.save(os.path.join(args.save_dir, f"{slide_id}.npy"), embeds[idx]) 
        if not args.task_config.get('setting', 'multi_class') == 'continuous':
            results = {'slide_name': slide_id, 'score': test_records['prob'][idx][1],
                       'label': np.argmax(test_records['label'][idx])}
                    #    'label': 0 if test_records['label'][idx][0] == 1 else 1}
        else:
            if not args.survival:
                results = {'slide_name': slide_id, 'score': test_records['prob'][idx][0],
                           'label': test_records['label'][idx][0]}
            else:
                results = {'slide_name': slide_id, 'score': test_records['prob'][idx][0],
                           'label': test_records['label'][idx][0], 'censoreship': test_records['censoreship'][idx][0]}
        print(f'results = {results}')
        print(f'pd.DataFrame([results]) = {pd.DataFrame([results])}')
        results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)

    print(results_df) # Delete
    results_df.to_csv(os.path.join(args.save_dir, 'slide_scores.csv'), index=False)
    # update the writer for test
    log_dict = {'test_' + k: v for k, v in test_records.items() if
                'prob' not in k and 'label' not in k and 'censoreship' not in k}
    log_writer(log_dict, 0, args.report_to, writer)
    wandb.finish() if "wandb" in args.report_to else None
    return test_records

def train_cycleGAN(dataloader, fold, args):
    train_loader, val_loader, test_loader = dataloader

    G_HE_IHC = ResNetGenerator().to(args.device)
    G_IHC_HE = ResNetGenerator().to(args.device)
    D_HE = PatchDiscriminator().to(args.device)
    D_IHC = PatchDiscriminator().to(args.device)
    cnn = torchvision.models.squeezenet1_1(pretrained=True).features.to(args.device)
    cnn.eval()
    for param in cnn.parameters():
        param.requires_grad = False

    opt_G = torch.optim.Adam(list(G_HE_IHC.parameters()) + list(G_IHC_HE.parameters()), lr=2e-4, betas=(0.5, 0.999))
    opt_D_HE = torch.optim.Adam(D_HE.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D_IHC = torch.optim.Adam(D_IHC.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion_GAN = nn.MSELoss()
    criterion_cycle = criterion_color = nn.L1Loss()
    real_label = 1.0
    fake_label = 0.0
    tile_bsz = 16
    style_layers = (1, 4, 6, 7)
    style_weights = (0.2, 0.005, 0.0002, 0.00001)

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            tiles, label, slide_id = batch['tiles'].squeeze(0), batch['labels'], batch['slide_id'][-1]
            print(f'batch_idx = {batch_idx}, slide_id = {slide_id}, label = {label}')            
            ihc_tiles, matching_tiles = batch['ihc_tiles'].squeeze(0), batch['matching_tiles'].squeeze(0) # Delete
            print(f'tiles.shape = {tiles.shape}, ihc_tiles.shape = {ihc_tiles.shape}, matching_tiles.shape = {matching_tiles.shape}') # Delete
            valid_matching_tiles = (~torch.isnan(matching_tiles)) & (matching_tiles < ihc_tiles.shape[0])
            if len(torch.nonzero(valid_matching_tiles)) == 0:
                print(f"{batch['slide_id']} has no valid matching tiles, skipping")
                continue
            # Select only valid matching tiles
            tiles = tiles[valid_matching_tiles]
            matching_tiles = matching_tiles[valid_matching_tiles].int()
            ihc_tiles = ihc_tiles[matching_tiles]

            for i in range(0, len(tiles), tile_bsz):
                curr_tiles = tiles[i:i+tile_bsz]
                curr_ihc_tiles = ihc_tiles[i:i+tile_bsz]

                real_HE = curr_tiles.to(args.device)
                real_IHC = curr_ihc_tiles.to(args.device)

                # ====================
                # Train Generators
                # ====================
                fake_IHC = G_HE_IHC(real_HE)
                rec_HE = G_IHC_HE(fake_IHC)
                fake_HE = G_IHC_HE(real_IHC)
                rec_IHC = G_HE_IHC(fake_HE)

                loss_idt_HE = criterion_cycle(G_IHC_HE(real_HE), real_HE)
                loss_idt_IHC = criterion_cycle(G_HE_IHC(real_IHC), real_IHC)

                pred_fake_IHC = D_IHC(fake_IHC)
                loss_GAN_HE_IHC = criterion_GAN(pred_fake_IHC, torch.ones_like(pred_fake_IHC))

                pred_fake_HE = D_HE(fake_HE)
                loss_GAN_IHC_HE = criterion_GAN(pred_fake_HE, torch.ones_like(pred_fake_HE))

                loss_cycle_HE = criterion_cycle(rec_HE, real_HE)
                loss_cycle_IHC = criterion_cycle(rec_IHC, real_IHC)

                # Compute style loss
                style_loss_HE = style_loss(fake_HE, real_HE, style_layers, style_weights, cnn)
                style_loss_IHC = style_loss(fake_IHC, real_IHC, style_layers, style_weights, cnn)

                loss_G = (
                    (style_loss_HE + style_loss_IHC) +
                    loss_GAN_HE_IHC + loss_GAN_IHC_HE +
                    10.0 * (loss_cycle_HE + loss_cycle_IHC) +
                    5.0 * (loss_idt_HE + loss_idt_IHC)
                )
                # print(f"loss_color_HE = {loss_color_HE.item()}, loss_color_IHC = {loss_color_IHC.item()}")
                print(f"style_loss_HE = {style_loss_HE.item()}, style_loss_IHC = {style_loss_IHC.item()}")
                print(f"loss_GAN_HE_IHC = {loss_GAN_HE_IHC.item()}, loss_GAN_IHC_HE = {loss_GAN_IHC_HE.item()}")
                print(f"loss_cycle_HE = {loss_cycle_HE.item()}, loss_cycle_IHC = {loss_cycle_IHC.item()}")
                print(f"loss_idt_HE = {loss_idt_HE.item()}, loss_idt_IHC = {loss_idt_IHC.item()}")
                print(f"loss_G = {loss_G.item()}")

                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

                # ====================
                # Train Discriminator A
                # ====================
                pred_real_HE = D_HE(real_HE)
                pred_fake_HE = D_HE(fake_HE.detach())

                loss_D_HE = (
                    criterion_GAN(pred_real_HE, torch.ones_like(pred_real_HE)) +
                    criterion_GAN(pred_fake_HE, torch.zeros_like(pred_fake_HE))
                ) * 0.5

                print(f"loss_D_HE = {loss_D_HE.item()}")

                opt_D_HE.zero_grad()
                loss_D_HE.backward()
                opt_D_HE.step()

                # ====================
                # Train Discriminator B
                # ====================
                pred_real_IHC = D_IHC(real_IHC)
                pred_fake_IHC = D_IHC(fake_IHC.detach())

                loss_D_IHC = (
                    criterion_GAN(pred_real_IHC, torch.ones_like(pred_real_IHC)) +
                    criterion_GAN(pred_fake_IHC, torch.zeros_like(pred_fake_IHC))
                ) * 0.5

                print(f"loss_D_IHC = {loss_D_IHC.item()}")

                opt_D_IHC.zero_grad()
                loss_D_IHC.backward()
                opt_D_IHC.step()

        # Save outputs
        os.makedirs(f"{args.save_dir}/samples", exist_ok=True)
        save_image(fake_IHC[0] * 0.5 + 0.5, f"{args.save_dir}/samples/fake_IHC_epoch{epoch}_slide_{slide_id}.png")
        save_image(fake_HE[0] * 0.5 + 0.5, f"{args.save_dir}/samples/fake_HE_epoch{epoch}_slide_{slide_id}.png")
        save_image(real_IHC[0], f"{args.save_dir}/samples/real_IHC_epoch{epoch}_slide_{slide_id}.png")
        save_image(real_HE[0], f"{args.save_dir}/samples/real_HE_epoch{epoch}_slide_{slide_id}.png")

        # Save models
        torch.save(G_HE_IHC.state_dict(), f"{args.save_dir}/G_HE_IHC_{epoch}.pth")
        torch.save(G_IHC_HE.state_dict(), f"{args.save_dir}/G_IHC_HE_{epoch}.pth")


def train_clinic_reg(dataloader, args):
    train_loader, val_loader, test_loader = dataloader

    model = get_model(**vars(args))
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = get_loss_function(args)
    val_losses = []
    min_val_loss = float('inf')

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            clinical_feats, score, label, slide_id = batch['clinical_feats'], batch['score'], batch['labels'], batch['slide_id'][-1]
            print(f'batch_idx = {batch_idx}, slide_id = {slide_id}, label = {label}')            

            clinical_feats = clinical_feats.to(args.device)
            score = score.to(args.device)
            label = label.to(args.device).flatten()
            logits = model(slide_score=score, clinical_features=clinical_feats.flatten()).view(-1, args.n_classes)

            loss = loss_fn(logits, label)
            print(f"loss = {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val_loss = test_clinic_reg(val_loader, model, epoch, args)
        val_losses.append(val_loss)

        # Save models
        torch.save(model.state_dict(), f"{args.save_dir}/model_epoch_{epoch}_checkpoint.pth")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.save_dir}/checkpoint.pth")
        
        # stop after 5 epochs with no improvement
        if len(val_losses) > 5 and val_losses[-1] > max(val_losses[-6:-1]):
            print("Early stopping due to no improvement in validation loss")
            break

def test_clinic_reg(test_loader, model=None, epoch=0, args=None):
    if args.run_inference:
        model = get_model(**vars(args))
        model = model.to(args.device)
        model.load_state_dict(torch.load(args.model_ckpt), strict=False)
    model.eval()

    loss_fn = get_loss_function(args)

    test_records = None
    test_loader.dataset.update_clinical_features_params_and_handle_data(model)
    results_df = pd.DataFrame({'slide_name': [], 'score': [], 'label': []})
    losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            slide_id = batch['slide_id'][-1]
            score = batch['score'].to(args.device)
            label = batch['labels'].to(args.device).flatten()
            clinical_feats = batch['clinical_feats'].to(args.device)
            logits = model(slide_score=score, clinical_features=clinical_feats.flatten()).view(-1, args.n_classes)
            loss = loss_fn(logits, label)
            losses.append(loss.item())
            print(f"loss = {loss.item()}")

            y_prob = torch.softmax(logits, dim=1).cpu().numpy()
            results = {'slide_name': slide_id, 'score': y_prob[0][1].item(),
                    'label': batch['labels'].to(args.device)[0].item()}
            results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
    
    print(f'results_df = {results_df}')

    # save results to csv
    results_df.to_csv(os.path.join(args.save_dir, f'results_epoch_{epoch}.csv'), index=False)

    avg_loss = np.mean(losses)
    print(f"Epoch {epoch+1}: Average {'test' if args.run_inference else 'val'} loss: {avg_loss}")
    return avg_loss


def train(dataloader, fold, args):
    train_loader, val_loader, test_loader = dataloader
    # set up the writer
    writer_dir = os.path.join(args.save_dir, f'fold_{fold}', 'tensorboard')
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir, exist_ok=True)

    # set up the writer
    writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)
    # set up writer
    if "wandb" in args.report_to:
        wandb.init(
            project='Finetune_Gigapath',
            name=args.exp_name,
            tags=[],
            config=vars(args),
        )
        writer = wandb
    elif "tensorboard" in args.report_to:
        writer = tensorboard.SummaryWriter(writer_dir, flush_secs=15)
    
    print("Before model setup:")
    print_free_gpu_memory(device=args.device)
    if args.add_virchow2_features:
        args.tile_input_dim += VIRCHOW2_FEATURE_DIM
    if args.add_uni2_features:
        args.tile_input_dim += UNI2_FEATURE_DIM
    if args.add_conch_features:
        args.tile_input_dim += CONCH_FEATURE_DIM

    # set up the model
    if (args.score_can_as_sb or args.score_can_as_bs or args.film) and not args.load_tile_logits and not args.tile_slide_model:  # Delete
        args.tile_model, model = get_model(**vars(args))
        args.tile_model = args.tile_model.to(args.device)
        if not args.tile_slide_training:
            args.tile_model.load_state_dict(torch.load(args.tile_model_ckpt), strict=False)
            args.tile_model.eval()
            print("Loaded tile model")
    else:        
        model = get_model(**vars(args))
        
    model = model.to(args.device)
    print("After model setup:")
    print_free_gpu_memory(device=args.device)
    
    # set up the optimizer
    if args.tile_slide_training and not args.tile_slide_model:
        args.tile_optimizer = torch.optim.AdamW(args.tile_model.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer = get_optimizer(args, model)
    if args.cnn_on_y or args.train_on_y:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        optimizer = get_optimizer(args, model)
    # set up the loss function
    loss_fn = get_loss_function(args)
    # set up the monitor
    monitor = Monitor_Score()
    # set up the fp16 scaler
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 training')

    if args.clinical_features:
        train_loader.dataset.update_clinical_features_params_and_handle_data(model)
        if val_loader is not None:
            val_loader.dataset.update_clinical_features_params_and_handle_data(model)
        if test_loader is not None:
            test_loader.dataset.update_clinical_features_params_and_handle_data(model)

    print('Training on {} samples'.format(len(train_loader.dataset)))
    print('Validating on {} samples'.format(len(val_loader.dataset))) if val_loader is not None else None
    print('Testing on {} samples'.format(len(test_loader.dataset))) if test_loader is not None else None
    print('Training starts!')

    val_records, test_records = None, None
    loss_pe, lr_pe, val_loss_pe = [], [], [] # Delete

    for i in range(args.epochs):
        print('Epoch: {}'.format(i))
        print_free_gpu_memory(device=args.device)
        args.output_dir = args.save_dir # Delete
        train_records = train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, i, args)
        try:
            loss_pe.append(train_records['loss'])
            lr_pe.append(train_records['lr'])
        except BaseException as e:
            print(e)
        print("********** Finished train_one_epoch **********")

        if val_loader is not None:
            val_records = evaluate(val_loader, model, fp16_scaler, loss_fn, i, args)
            val_loss_pe.append(val_records['loss'])

            # update the writer for train and val
            try:
                log_dict = {'train_' + k: v for k, v in train_records.items() if
                            'prob' not in k and 'label' not in k and 'censoreship' not in k}
                log_dict.update({'val_' + k: v for k, v in val_records.items() if
                                 'prob' not in k and 'label' not in k and 'censoreship' not in k})
                log_writer(log_dict, i, args.report_to, writer)
            except:
                for key in val_records:
                    print(key)
                    print(val_records[key])
                    try:
                        print(val_records[key].shape)
                    except:
                        pass
                raise
            # update the monitor scores
            if not args.task_config.get('setting', 'multi_class') == 'continuous':
                scores = val_records['macro_auroc']
            else:
                # the better loss is the lower one but we want to choose the epoch with the highest score
                scores = -val_records['loss']

        if args.model_select == 'val' and val_loader is not None:
            monitor(scores, model, ckpt_name=os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt"))
        elif args.model_select == 'last_epoch' and i == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt"))
            if args.tile_slide_training and not args.tile_slide_model:
                torch.save(args.tile_model.state_dict(), os.path.join(args.save_dir, 'fold_' + str(fold), "tile_checkpoint.pt"))
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, 'fold_' + str(fold), 'model_epoch_' + str(i) + '_checkpoint.pt'))
        if args.tile_slide_training and not args.tile_slide_model:
            torch.save(args.tile_model.state_dict(), os.path.join(args.save_dir, 'fold_' + str(fold), 'model_epoch_' + str(i) + '_tile_checkpoint.pt'))
        print_free_gpu_memory(device=args.device)

    # load model for test
    if test_loader is not None:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'fold_' + str(fold), "checkpoint.pt")),
                              strict=False)
        if args.task_config.get('setting', 'multi_class') == 'continuous' and not args.loss_fn == 'cox':
            print("in test time labels mean: %.2e" % model.mean.item())
            print("in test time labels std: %.2e" % model.std.item())
        # test the model
        test_records, embeds = evaluate(test_loader, model, fp16_scaler, loss_fn, i, args, save_embed=True)
        # save each embedding asa seperate filewith the name being the slide_id
        for idx, slide_id in enumerate(test_loader.dataset.slide_data[args.slide_key]):
            np.save(os.path.join(args.save_dir, 'fold_' + str(fold), f"{slide_id}.npy"), embeds[idx])
        # update the writer for test
        log_dict = {'test_' + k: v for k, v in test_records.items() if 'prob' not in k and 'label' not in k}
        log_writer(log_dict, fold, args.report_to, writer)
    wandb.finish() if "wandb" in args.report_to else None

    save_plot(x=range(args.epochs), y=loss_pe, xlabel='Epoch', ylabel='Train Loss', title="Train Loss per epoch", output_dir=args.output_dir)
    save_plot(x=range(args.epochs), y=lr_pe, xlabel='Epoch', ylabel='Learning rate', title="Learning rate per epoch", output_dir=args.output_dir)
    save_plot(x=range(args.epochs), y=val_loss_pe, xlabel='Epoch', ylabel='Val Loss', title="Val Loss per epoch", output_dir=args.output_dir)

    return val_records, test_records


def train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, epoch, args): 
    model.train()
    # set the start time
    start_time = time.time()

    # monitoring sequence length
    seq_len = 0
    curr_len = 0
    max_num_samples = 40000
    # max_num_samples = 25000
    max_num_valid_samples = 16000

    # setup the records
    records = get_records_array(len(train_loader), args.n_classes, args)
    if args.loss_fn == 'cox':
        train_loader.dataset.reorder_samples_for_cox(args.gc)
        # we are accumulating alot of memory by not performing backward at every iteration
        max_num_samples = 1000
        cum_logits = torch.zeros([args.gc])
        cum_labels = torch.zeros([args.gc])
        cum_censoreship = torch.zeros([args.gc])
        accumulate_another_batch = False
    batch_amplification = 1
    
    last_0_embed, last_3_embed = None, None
    last_neg_logits, last_pos_logits = None, None
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    mae = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    margin_rank = torch.nn.MarginRankingLoss(margin=2)
    loss_spear = SoftSpearmanLoss(regularization_strength=0.05)
    gnll = torch.nn.GaussianNLLLoss()
    a_mse = Assymmetric_MSE_Loss()
    
    for batch_idx, batch in enumerate(train_loader):
        slide_id = batch['slide_id'][-1] 

        # if '19-14590' not in slide_id:
        #     continue

        if args.window_training:            
            sliding_window_ds = SlidingWindowDataset(data_df=args.dataset, root_path=args.root_path, task_config=args.task_config, slide_key=args.slide_key, label=args.label, \
                                  dataset_name=args.test_dataset, folds=args.test_fold, use_clinical_features=args.clinical_features, test_on_all=args.test_on_all, 
                                  get_single_slide=slide_id, window_size=args.window_size, stride=args.window_size, val_fold=args.val_fold[-1], only_matching_tiles=True)
            window_loader = get_test_loader(sliding_window_ds, for_window=True, **vars(args))
            slide_windows = [{'imgs': wndw['imgs'].to(args.device, non_blocking=True), 
                              'coords': wndw['coords'].to(args.device, non_blocking=True), 
                              'labels': wndw['labels'].to(args.device, non_blocking=True),
                              'window_y': wndw['window_y'].to(args.device, non_blocking=True)} for wndw in window_loader]
            len_windows = len(slide_windows)
            print(f"Finished loading windows for slide {slide_id}, len(slide_windows) = {len_windows}")
            if len_windows == 0:
                continue
            
        # we use a per iteration lr scheduler
        if batch_idx % args.gc == 0 and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch, args)
            if args.tile_slide_training and not args.tile_slide_model:
               adjust_learning_rate(args.tile_optimizer, batch_idx / len(train_loader) + epoch, args) 
        num_tiles = batch['imgs'].shape[1]
        if num_tiles > max_num_samples:
            if not args.loss_fn == 'cox':
                print(f"slide {batch['slide_id']} has too many tiles, number of tiles is {num_tiles}. Trying with {max_num_samples} tiles.")
            indices = torch.randint(low=0, high=num_tiles, size=(max_num_samples,))
            slice_tiles(indices=indices, batch=batch)

        print('Before loading batch')
        print_free_gpu_memory(device=args.device)
        # ********** load the batch and transform this batch **********
        if not args.window_training:
            images, img_coords, label = batch['imgs'], batch['coords'], batch['labels']
            print(f'slide_id = {slide_id}, label = {label}')
            ihc_images, ihc_coords, matching_tiles = batch['ihc_imgs'], batch['ihc_coords'], batch['matching_tiles'] # Delete
            tumor_indices, non_tumor_indices, cancer_prob = batch['tumor_indices'], batch['non_tumor_indices'], batch.get('cancer_prob', None)
            tile_y = batch['tile_y']
            if args.cnn_on_y:
                tiles = batch['tiles']
                # check if tiles is a tensor larger than 0
                if tiles is None or not (isinstance(tiles, torch.Tensor) and tiles.numel() > 0):
                    print(f"tiles is None or not a valid tensor for slide {batch['slide_id']}, skipping")
                    continue
            synth_ihc_images = batch['synth_ihc_imgs'] # Delete
            if args.load_tile_logits:
                tile_logits = batch['tile_logits']
            if args.add_virchow2_features:
                virchow2_feats = batch['virchow2_feats']
            if args.add_uni2_features:
                uni2_feats = batch['uni2_feats']
            if args.add_conch_features:
                conch_feats = batch['conch_feats']
            if args.external_model_logits:
                external_tile_logits = batch['external_tile_logits']

            if args.survival:
                censoreship = batch['censoreship']
        # ********** data preperation and move to device **********
        if not args.window_training:
            images = images.to(args.device, non_blocking=True)
            img_coords = img_coords.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True) # .long()
            if ihc_images is not None:
                ihc_images = ihc_images.to(args.device, non_blocking=True)
                ihc_coords = ihc_coords.to(args.device, non_blocking=True)

            if args.synth_ihc_train:
                synth_ihc_images = synth_ihc_images.to(args.device, non_blocking=True)
            if args.predict_cancer and tumor_indices is not None:
                # init local_tumor_labels as zeros_like(images) and then fill the tumor indices with 1
                local_tumor_labels = torch.zeros(images.shape[:2], device=images.device)
                print(f"tumor_indices.shape = {tumor_indices.shape}")
                local_tumor_labels[:, tumor_indices] = 1
                tumor_indices = tumor_indices.to(args.device, non_blocking=True)
                if non_tumor_indices is not None:
                    non_tumor_indices = non_tumor_indices.to(args.device, non_blocking=True)

                if args.only_annotated_tiles:
                    annotated_indices = torch.cat([tumor_indices, non_tumor_indices], dim=-1).long()
                    images = images[:, annotated_indices].squeeze(0)
                    img_coords = img_coords[:, annotated_indices].squeeze(0)
                    local_tumor_labels = local_tumor_labels[:, annotated_indices].squeeze(0)
            
            if args.only_tumor_tiles:
                if tumor_indices is not None and tumor_indices.shape[-1] > 0:
                    tumor_indices = tumor_indices.squeeze(0)
                    images = images[tumor_indices]
                    img_coords = img_coords[tumor_indices]
                else:
                    print(f"{batch['slide_id']} has no tumor indices, skipping")
                    continue
                    
            if args.cat_y or args.paired_training_mw or args.tile_slide_training: # Delete:
                if matching_tiles is not None:
                    matching_tiles = matching_tiles.to(args.device, non_blocking=True)
                    valid_matching_tiles = ((~torch.isnan(matching_tiles)) & (matching_tiles < ihc_images.shape[1])).flatten()
                    if len(torch.nonzero(valid_matching_tiles)) == 0:
                        print(f"{batch['slide_id']} has no valid matching tiles, skipping")
                        continue

                    elif args.cat_y or args.paired_training_mw or args.tile_slide_training: # Delete
                    # else:
                        if args.malig_paired_mw and tumor_indices is not None and tumor_indices.shape[-1] > 0:
                            # get the indices of true values in valid_matching_tiles
                            valid_matching_indices = torch.nonzero(valid_matching_tiles.clone().flatten()).long().flatten()
                            tumor_in_non_nan = np.isin(valid_matching_indices.cpu().numpy(), tumor_indices.cpu().numpy())
                            tumor_matching_indices = np.where(tumor_in_non_nan)[0]
                            valid_tumor_matching_indices = valid_matching_indices[tumor_matching_indices].flatten()
                            if args.replace_benign_w_zero:
                                # get all indices which are in valid_matching_indices and not on valid_tumor_matching_indices without a for loop
                                non_tumor_matching_indices = np.setdiff1d(valid_matching_indices.cpu().numpy(), valid_tumor_matching_indices.cpu().numpy())
                                non_tumor_matching_tiles = matching_tiles.clone().flatten()[non_tumor_matching_indices].long().flatten()
                                images, img_coords = images[:, valid_matching_indices], img_coords[:, valid_matching_indices]
                                matching_tiles = matching_tiles.flatten()[valid_matching_indices].long().flatten()
                                # fill the non matching indices with zero vectors
                                ihc_images[:, non_tumor_matching_tiles] = 0
                            else:
                                matching_tiles = matching_tiles.flatten()[valid_tumor_matching_indices].long()
                                images, img_coords = images[:, valid_tumor_matching_indices], img_coords[:, valid_tumor_matching_indices]        
                        else:
                            images = images[:, valid_matching_tiles]
                            img_coords = img_coords[:, valid_matching_tiles]
                            matching_tiles = matching_tiles[:, valid_matching_tiles].long().flatten()
                            
                        if args.malig_paired_mw:
                            ihc_images = ihc_images[:, matching_tiles]

                        if cancer_prob is not None:
                            cancer_prob = cancer_prob.to(args.device, non_blocking=True)[:, valid_matching_tiles]

                        if args.cat_y or args.tile_slide_training:
                            if tile_y is not None:
                                tile_y = tile_y.to(args.device, non_blocking=True)
                                tile_y = tile_y.squeeze(0)
                            else:
                                print(f"{batch['slide_id']} tile_y is None, skipping")
                                continue

                        if args.cnn_on_y:
                            tiles = tiles.to(args.device, non_blocking=True)
                            # tiles = tiles[valid_matching_tiles]

                        if args.add_virchow2_features:
                            print(f'images.shape before adding virchow2 features = {images.shape}, virchow2_feats.shape = {virchow2_feats.shape}')
                            virchow2_feats = virchow2_feats.to(args.device, non_blocking=True)
                            virchow2_feats = virchow2_feats[valid_matching_tiles]
                            virchow2_feats = virchow2_feats.view(-1, VIRCHOW2_FEATURE_DIM)
                            images = torch.cat([images, virchow2_feats], dim=-1)
                        
                        if args.add_uni2_features:
                            print(f'images.shape before adding uni2 features = {images.shape}, uni2_feats.shape = {uni2_feats.shape}')
                            uni2_feats = uni2_feats.to(args.device, non_blocking=True)
                            uni2_feats = uni2_feats[valid_matching_tiles]
                            uni2_feats = uni2_feats.view(-1, UNI2_FEATURE_DIM)
                            images = torch.cat([images, uni2_feats], dim=-1)
                        
                        if args.add_conch_features:
                            print(f'images.shape before adding conch features = {images.shape}, conch_feats.shape = {conch_feats.shape}')
                            conch_feats = conch_feats.to(args.device, non_blocking=True)
                            conch_feats = conch_feats[valid_matching_tiles]
                            conch_feats = conch_feats.view(-1, CONCH_FEATURE_DIM)
                            images = torch.cat([images, conch_feats], dim=-1)

                        num_valid_tiles = images.shape[-2]
                        print(f'num_valid_tiles = {num_valid_tiles}')
                        if args.tile_slide_model and num_valid_tiles > max_num_valid_samples:
                            indices = torch.randint(low=0, high=num_valid_tiles, size=(max_num_valid_samples,))
                            images = images[indices]
                            img_coords = img_coords[indices]
                            if tile_y is not None:
                                tile_y = tile_y[indices]
                            if cancer_prob is not None:
                                cancer_prob = cancer_prob[indices]
                            if virchow2_feats is not None:
                                virchow2_feats = virchow2_feats[indices]
                            if uni2_feats is not None:
                                uni2_feats = uni2_feats[indices]
                            print(f"After slicing, images.shape = {images.shape}")
                        
                        if args.external_model_logits:
                            external_tile_logits = external_tile_logits.to(args.device, non_blocking=True)
                            external_tile_logits = external_tile_logits[valid_matching_tiles]
                        
                        images = images.view(1, -1, args.tile_input_dim)
                        
                        if args.y_and_tumor_from_map and not args.cnn_on_y:
                            # only keep tile_y values that are not -1
                            valid_tile_y_indices = (tile_y != -1) & (tile_y != 0)
                            images = images[valid_tile_y_indices]
                            if images.size(0) == 0:
                                print(f"{batch['slide_id']} has no valid tile_y indices, skipping")
                                continue
                            img_coords = img_coords[valid_tile_y_indices]
                            tile_y = tile_y[valid_tile_y_indices].view(-1, 1)
                            # if args.cnn_on_y:
                            #     tiles = tiles[valid_tile_y_indices]
                    else:
                        matching_tiles = matching_tiles[valid_matching_tiles].int()
                        
                            
                    # print(f"valid_matching_tiles = {valid_matching_tiles}, matching_tiles = {matching_tiles}, matching_tiles.shape = {matching_tiles.shape}")
                else:
                    print(f"{batch['slide_id']} matching_tiles is None, skipping")
                    continue    
        
            print('After loading batch')
            print_free_gpu_memory(device=args.device)

            # add the sequence length
            seq_len += images.shape[-2]
            curr_len += images.shape[-2]

        with torch.cuda.amp.autocast(dtype=torch.float16 if args.fp16 else torch.float32):
            # ********** get the logits **********
            if args.predict_cancer and tumor_indices is not None:
                images, img_coords = images.transpose(0, 1), img_coords.transpose(0, 1)
                local_tumor_labels = local_tumor_labels.squeeze(0).long()
                print(f'images.shape = {images.shape}, local_tumor_labels.shape = {local_tumor_labels.shape}')
                logits = model(images, img_coords)

            elif args.cat_y:
                if args.train_on_y:
                    images, img_coords = images.view(-1, 1, args.tile_input_dim), img_coords.view(-1, 1, 2)
                    print(f'images.shape = {images.shape}')
                    if images.shape[0] == 0:
                        print(f"slide {batch['slide_id']} has no matching ihc images, skipping")
                        continue
                    
                    tile_log_vars = None
                    if args.cnn_on_y:
                        tiles = tiles.view(-1, 3, args.tile_size, args.tile_size)
                        print(f'tiles.shape = {tiles.shape} before model call')
                        tile_logits = model(tiles)
                        print(f'tile_logits.shape = {tile_logits.shape} after model call')
                        logits = tile_logits.mean()

                    elif not args.conf_score:
                        logits, tile_logits = model(images, img_coords)
                    else:
                        logits, tile_logits, tile_log_vars = model(images, img_coords, return_conf=True)

            elif args.score_can_as_sb or args.score_can_as_bs or args.film or args.weighted_film:
                cancer_prob = cancer_prob.to(args.device, non_blocking=True)
                if args.tile_slide_model:
                    images, img_coords = images.view(-1, 1, args.tile_input_dim), img_coords.view(-1, 1, 2)
                    if args.use_tile_classifier:
                        tile_logits = model(images, img_coords, predict_tile_logits=True, tile_classifier=True)
                    else:
                        tile_logits = model(images, img_coords, predict_tile_logits=True)
                elif args.load_tile_logits:
                    tile_logits = tile_logits.to(args.device, non_blocking=True)
                else:
                    images, img_coords = images.view(-1, 1, args.tile_input_dim), img_coords.view(-1, 1, 2)
                    _, tile_logits = args.tile_model(images, img_coords)
                
                images = images[:,:, :args.input_dim]
                images, img_coords = images.view(1, -1, args.input_dim), img_coords.view(1, -1, 2)
                if args.weighted_film:
                    logits = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, weighted_film=args.weighted_film)
                elif not args.film:
                    logits = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, score_as_scale=args.score_can_as_sb)
                else:  # args.film
                    if args.const_tl:
                        tile_logits = torch.full_like(tile_logits, 3)
                    elif args.const_cp:
                        cancer_prob = torch.full_like(cancer_prob, 1)

                    if args.external_model_logits:
                        logits = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, external_model_scores=external_tile_logits, film=args.film)
                    else:
                        logits = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, film=args.film)

            elif args.synth_ihc_train: # Delete
                images = torch.cat([images, synth_ihc_images], dim=-1)
                logits = model(images, img_coords)
            elif args.window_training:
                window_logits = []
                for window in slide_windows:
                    wndw_logits = model(window['imgs'], window['coords'])
                    window_logits.append(wndw_logits)
                window_logits = torch.stack(window_logits).view(len_windows, -1)
            elif args.paired_training_mw:
                if args.malig_baseline:
                    logits = model(ihc_images, img_coords)
                elif args.paired_mw_w_cancer:
                    logits = model(images, img_coords, ihc_images, modality_weights=True, cancer_weights=True, tile_cancer_probs=cancer_prob)
                elif args.paired_mw_cancer_scale:
                    logits = model(images, img_coords, ihc_images, modality_weights=True, tile_cancer_probs=cancer_prob, cancer_scale=True)
                else:
                    logits = model(images, img_coords, ihc_images, modality_weights=True)
            elif args.paired_training_mb:
                logits = model(images, img_coords, ihc_images, ihc_coords, modality_bias=True)
            elif args.paired_training_cat_h:
                logits = model(images, img_coords, ihc_images, ihc_coords, concat_h=True)
            else:
                logits = model(images, img_coords)
                print(f"logits = {logits}, shape = {logits.shape}")

            # ********** Shapes and types before loss **********
            if not args.window_training:
                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    label = label.squeeze(-1).float()    
                elif (args.task_config.get('setting', 'multi_class') == 'continuous' and not args.loss_fn == 'cox') or args.task_config.get('setting', 'multi_class') == 'multi_class':
                    label = label.squeeze(-1).float()
                    logits = logits.squeeze(-1)
                else:
                    label = label.squeeze(-1) # .long()                 

            # ********** loss computation **********
            if not args.loss_fn == 'cox':
                print(f"batch_idx = {batch_idx}, slide_id = {batch['slide_id']}") # Delete
                if not args.survival:
                    if args.train_on_y:
                        if args.task_config.get('setting', 'multi_class') == 'binary':
                            tile_y = (tile_y >= 2).long().squeeze(-1)
                            label = label.long()
                            # print(f"tile_y = {tile_y}, shape = {tile_y.shape}")
                        # tile_y_scaled = tile_y / 3

                        elif not tile_logits.shape == tile_y.shape:
                            tile_logits = tile_logits.view(-1)
                            tile_y = tile_y.view(-1)

                        if not args.conf_score:
                            tile_loss = loss_fn(tile_logits, tile_y)
                        else:
                            tile_vars = torch.exp(tile_log_vars)
                            tile_loss = gnll(tile_logits, tile_y.float(), tile_vars) + tile_vars.mean()
                        
                        margin, spearman = None, None
                        spearman = loss_spear(tile_logits, tile_y) if not args.task_config.get('setting', 'multi_class') == 'binary' else None
                        
                        # if label.item() in [0, 0.5, 1]:
                        #     if len(tile_logits[tile_y < 0.5]) > 0:
                        #         last_neg_logits = tile_logits[tile_y < 0.5]
                        #         print(f"Setting last_neg_logits to {last_neg_logits}")
                        #         if last_pos_logits is not None:
                        #             min_len = min(len(last_pos_logits), len(last_neg_logits))
                        #             large, small = last_pos_logits.view(-1)[:min_len], last_neg_logits.view(-1)[:min_len]
                        #             margin = margin_rank(large, small, torch.ones_like(large))  # try ranking with last pos example
                        # elif label.item() in [3]:
                        #     if len(tile_logits[tile_y > 2]) > 0:
                        #         last_pos_logits = tile_logits[tile_y > 2]
                        #         print(f"Setting last_pos_logits to {last_pos_logits}")
                        #         if last_neg_logits is not None:
                        #             min_len = min(len(last_pos_logits), len(last_neg_logits))
                        #             large, small = last_pos_logits.view(-1)[:min_len], last_neg_logits.view(-1)[:min_len]
                        #             margin = margin_rank(large, small, torch.ones_like(large))  # try ranking with last neg example
                        
                        # margin = margin_loss(tile_logits, tile_y, margin_scale=0.8)
                        # tile_loss = loss_fn(tile_logits, tile_y_scaled)
                        # slide_loss = loss_fn(logits.unsqueeze(0), label)

                        # if epoch >= 2:
                        #     if margin is not None:
                        #         loss = tile_loss + spearman + margin
                        #     else:
                        #         loss = tile_loss + spearman
                        # else:
                        loss = tile_loss # + spearman # + slide_loss
                            
                        # loss = tile_loss + spearman if epoch >= 2 else tile_loss
                        print(f"tile_loss = {tile_loss}, margin_loss = {margin}, spearman = {spearman}, loss = {loss}") # slide_loss = {slide_loss},
                        print(f"tile_y = {tile_y}, tile_logits = {tile_logits}, logits = {logits}, label = {label}") # tile_vars = {tile_vars}, 
                    elif args.predict_cancer and tumor_indices is not None:
                        loss = loss_fn(logits, local_tumor_labels)
                    elif args.window_training:
                        window_labels = torch.stack([wndw['window_y'] for wndw in slide_windows])
                        print(f"window_logits = {window_logits}, window_labels = {window_labels}")
                        loss = loss_fn(window_logits, window_labels)
                        print(f"loss = {loss}")
                        logits = window_logits.mean(dim=0) # for evaluation records
                    elif args.tile_slide_training:
                        tile_loss = loss_fn(tile_logits.view(-1), tile_y.view(-1))
                        slide_loss = loss_fn(logits, label)
                        loss = tile_loss + slide_loss
                        print(f"tile_loss = {tile_loss}, slide_loss = {slide_loss}, loss = {loss}")
                        print(f"tile_y = {tile_y}, tile_logits = {tile_logits}, logits = {logits}, label = {label}")
                    else:
                        loss = loss_fn(logits, label)
                        print(f"loss = {loss}")
                else:
                    loss = loss_fn(logits, label, censoreship)
                loss /= args.gc
            else:
                curr_idx = batch_idx % args.gc
                cum_logits[curr_idx] = logits[0]
                cum_labels[curr_idx] = label[0]
                cum_censoreship[curr_idx] = censoreship[0]

            # ********** backward step **********
            if fp16_scaler is None:
                if not args.loss_fn == 'cox':
                    loss.backward()
                # update the parameters with gradient accumulation
                if (batch_idx + 1) % args.gc == 0:
                    if args.loss_fn == 'cox':
                        # if batch has no uncensored examples loss will be 0 so we accumulate the examples for another batch
                        if torch.sum(cum_censoreship) == 0:
                            raise ValueError("Found a batch without uncensored samples.")
                            print('no uncensored samples so will not perform step')
                            batch_amplification += 1
                            if not accumulate_another_batch:
                                prev_logits = torch.tensor(cum_logits, requires_grad=True)
                                prev_labels = torch.tensor(cum_labels, requires_grad=True)
                                prev_censoreship = torch.tensor(cum_censoreship, requires_grad=True)
                            else:
                                prev_logits = torch.cat((prev_logits, torch.tensor(cum_logits, requires_grad=True)))
                                prev_labels = torch.cat((prev_labels, torch.tensor(cum_labels, requires_grad=True)))
                                prev_censoreship = torch.cat(
                                    (prev_censoreship, torch.tensor(cum_censoreship, requires_grad=True)))
                            accumulate_another_batch = True
                        else:
                            if accumulate_another_batch:
                                raise ValueError("This should not be true.")
                                curr_logits = torch.cat((prev_logits, torch.tensor(cum_logits, requires_grad=True)))
                                curr_labels = torch.cat((prev_labels, torch.tensor(cum_labels, requires_grad=True)))
                                curr_censoreship = torch.cat(
                                    (prev_censoreship, torch.tensor(cum_censoreship, requires_grad=True)))
                                del prev_logits, prev_labels, prev_censoreship
                            loss = loss_fn(cum_logits, cum_labels, cum_censoreship) / args.gc
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            if args.tile_slide_training and not args.tile_slide_model:
                                args.tile_optimizer.step()
                                args.tile_optimizer.zero_grad()
                            print(f"performed step on {curr_len} tiles")
                            curr_len = 0
                            cum_logits = torch.zeros([args.gc])
                            cum_labels = torch.zeros([args.gc])
                            cum_censoreship = torch.zeros([args.gc])
                            # del curr_logits, curr_labels, curr_censoreship
                    else:
                        optimizer.step()
                        optimizer.zero_grad()
                        print(f"performed step on {curr_len} tiles")
                        curr_len = 0
            else:
                if not args.loss_fn == 'cox':                    
                    # fp16_scaler.scale(loss).backward(retain_graph=True)
                    fp16_scaler.scale(loss).backward()
                    # print("Classifier weight grad norm:", model.classifier[0].weight.grad) # Delete .norm()
                    
                # update the parameters with gradient accumulation
                if (batch_idx + 1) % args.gc == 0:
                    if args.loss_fn == 'cox':
                        # if batch has no uncensored examples loss will be 0 so we accumulate the examples for another batch
                        if torch.sum(cum_censoreship) == 0:
                            raise ValueError("Found a batch without uncensored samples.")
                            print('no uncensored samples so will not perform step')
                            batch_amplification += 1
                            if not accumulate_another_batch:
                                prev_logits = torch.tensor(cum_logits, requires_grad=True)
                                prev_labels = torch.tensor(cum_labels, requires_grad=True)
                                prev_censoreship = torch.tensor(cum_censoreship, requires_grad=True)
                            else:
                                prev_logits = torch.cat((prev_logits, torch.tensor(cum_logits, requires_grad=True)))
                                prev_labels = torch.cat((prev_labels, torch.tensor(cum_labels, requires_grad=True)))
                                prev_censoreship = torch.cat(
                                    (prev_censoreship, torch.tensor(cum_censoreship, requires_grad=True)))
                            accumulate_another_batch = True
                        else:
                            if accumulate_another_batch:
                                raise ValueError("This should not be true.")
                                curr_logits = torch.cat((prev_logits, torch.tensor(cum_logits, requires_grad=True)))
                                curr_labels = torch.cat((prev_labels, torch.tensor(cum_labels, requires_grad=True)))
                                curr_censoreship = torch.cat(
                                    (prev_censoreship, torch.tensor(cum_censoreship, requires_grad=True)))
                            loss = loss_fn(cum_logits, cum_labels, cum_censoreship) / args.gc
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                            print(f"performed step on {curr_len} tiles")
                            curr_len = 0
                            cum_logits = torch.zeros([args.gc])
                            cum_labels = torch.zeros([args.gc])
                            cum_censoreship = torch.zeros([args.gc])
                    else:
                        fp16_scaler.step(optimizer)
                        fp16_scaler.update()
                        optimizer.zero_grad()
                        if args.tile_slide_training and not args.tile_slide_model:
                            fp16_scaler.step(args.tile_optimizer)
                            fp16_scaler.update()
                            args.tile_optimizer.zero_grad()
                        print(f"performed step on {curr_len} tiles")
                        curr_len = 0

        # update the records
        if not args.loss_fn == 'cox':
            records['loss'] += loss.item() * args.gc
        elif (batch_idx + 1) % args.gc == 0:
            records['loss'] += loss.item() * args.gc
            # accumulate_another_batch = False
            # batch_amplification = 1

        if (batch_idx + 1) % 20 == 0:
            time_per_it = (time.time() - start_time) / (batch_idx + 1)
            print('Epoch: {}, Batch: {}, Loss: {:.4f}, LR: {:.6f}, Time: {:.4f} sec/it, Seq len: {:.1f}, Slide ID: {}' \
                  .format(epoch, batch_idx, records['loss'] / batch_idx, optimizer.param_groups[0]['lr'], time_per_it, \
                          seq_len / (batch_idx + 1), batch['slide_id'][-1] if 'slide_id' in batch else 'None'))

        if batch_idx == 0 or 'lr' not in records:
            records['lr'] = optimizer.param_groups[0]['lr'] # Delete

    records['loss'] = records['loss'] / len(train_loader)
    print('Epoch: {}, Train Loss: {:.4f}'.format(epoch, loss.item()))
    
    return records


def evaluate(loader, model, fp16_scaler, loss_fn, epoch, args, save_embed=False):
    model.eval()
    
    # set the evaluation records
    records = get_records_array(len(loader), args.n_classes, args)
    if args.predict_cancer or args.pred_y_baseline or args.pred_y:
        records['prob'] = []
        records['label'] = []
    embeds = np.zeros((len(loader), 768)) if not args.paired_training_cat_h else np.zeros((len(loader), 768*2)) # Delete

    # get the task setting
    task_setting = args.task_config.get('setting', 'multi_class')
    if task_setting == 'continuous' and not args.survival:
        regression_losses = get_regression_losses()
        for key in regression_losses:
            records[key] = 0
    with torch.no_grad():
        no_loss = False
        if args.loss_fn == 'cox':
            loader.dataset.reorder_samples_for_cox(args.gc)
            cum_logits = torch.zeros([args.gc])
            cum_labels = torch.zeros([args.gc])
            cum_censoreship = torch.zeros([args.gc])
            accumulate_another_batch = False
            max_num_samples = 1000
        for batch_idx, batch in enumerate(loader):
            slide_id = batch['slide_id'][-1] 
            # if not slide_id.startswith('19-14590_1_1'):
            #     continue
            # if batch_idx > 3:
            #     break

            if args.window_training:            
                sliding_window_ds = SlidingWindowDataset(data_df=args.dataset, root_path=args.root_path, task_config=args.task_config, slide_key=args.slide_key, label=args.label, \
                                    dataset_name=args.test_dataset, folds=args.test_fold, use_clinical_features=args.clinical_features, test_on_all=args.test_on_all, 
                                    get_single_slide=slide_id, window_size=args.window_size, stride=args.window_size, val_fold=args.val_fold[-1], only_matching_tiles=True)
                window_loader = get_test_loader(sliding_window_ds, for_window=True, **vars(args))
                slide_windows = [{'imgs': wndw['imgs'].to(args.device, non_blocking=True), 
                                'coords': wndw['coords'].to(args.device, non_blocking=True), 
                                'labels': wndw['labels'].to(args.device, non_blocking=True),
                                'window_y': wndw['window_y'].to(args.device, non_blocking=True)} for wndw in window_loader]
                label = batch['labels'].to(args.device, non_blocking=True).view(-1)
                len_windows = len(slide_windows)
                print(f"Finished loading windows for slide {slide_id}, len(slide_windows) = {len_windows}")
                if len_windows == 0:
                    continue

            # ********** load the batch and transform this batch **********
            if not args.window_training:
                num_tiles = batch['imgs'].shape[1]
                if args.loss_fn == 'cox' and num_tiles > max_num_samples:
                    indices = torch.randint(low=0, high=num_tiles, size=(max_num_samples,))
                    slice_tiles(indices=indices, batch=batch)

                images, img_coords, label, ihc_images, ihc_coords = batch['imgs'], batch['coords'], batch['labels'], batch['ihc_imgs'], batch['ihc_coords']
                matching_tiles, tile_y = batch['matching_tiles'], batch['tile_y']
                tumor_indices, non_tumor_indices, cancer_prob = batch['tumor_indices'], batch['non_tumor_indices'], batch.get('cancer_prob', None)
                synth_ihc_images = batch['synth_ihc_imgs'] # Delete
                if args.load_tile_logits:
                    tile_logits = batch['tile_logits']
                if args.add_virchow2_features:
                    virchow2_feats = batch['virchow2_feats']
                if args.add_uni2_features:
                    uni2_feats = batch['uni2_feats']
                if args.add_conch_features:
                    conch_feats = batch['conch_feats']
                if args.external_model_logits:
                    external_tile_logits = batch['external_tile_logits']
                if args.cnn_on_y:
                    tiles = batch['tiles']
                    # check if tiles is a tensor larger than 0
                    if tiles is None or not (isinstance(tiles, torch.Tensor) and tiles.numel() > 0):
                        print(f"tiles is None or not a valid tensor for slide {batch['slide_id']}, skipping")
                        continue
            # ********** data preperation and move to device **********
            print(f"slide = {slide_id}, batch_idx = {batch_idx}")  # , label = {label}
            # if slide_id != "21-8662_1_1_e":
                # continue
            if not args.window_training:
                images = images.to(args.device, non_blocking=True)
                img_coords = img_coords.to(args.device, non_blocking=True)
                label = label.to(args.device, non_blocking=True).long()
                if ihc_images is not None:
                    ihc_images = ihc_images.to(args.device, non_blocking=True)
                    ihc_coords = ihc_coords.to(args.device, non_blocking=True)

                if args.predict_cancer:
                    # init local_tumor_labels as zeros_like(images) and then fill the tumor indices with 1
                    local_tumor_labels = torch.zeros(images.shape[:2], device=images.device)
                    if tumor_indices is not None:
                        print(f"tumor_indices.shape = {tumor_indices.shape}")
                        if tumor_indices.shape[-1] > 0:
                            local_tumor_labels[:, tumor_indices] = 1
                        tumor_indices = tumor_indices.to(args.device, non_blocking=True)
                        if non_tumor_indices is not None:
                            non_tumor_indices = non_tumor_indices.to(args.device, non_blocking=True)

                    if args.only_annotated_tiles:
                        annotated_indices = torch.cat([tumor_indices, non_tumor_indices], dim=-1).long()
                        images = images[:, annotated_indices].squeeze(0)
                        img_coords = img_coords[:, annotated_indices].squeeze(0)
                        local_tumor_labels = local_tumor_labels[:, annotated_indices].squeeze(0)

                if args.synth_ihc_train:
                    synth_ihc_images = synth_ihc_images.to(args.device, non_blocking=True)

                if args.only_tumor_tiles:
                    if tumor_indices is not None and tumor_indices.shape[-1] > 0:
                        print(f'tumor_indices.shape = {tumor_indices.shape}, images.shape = {images.shape}, img_coords.shape = {img_coords.shape}')
                        print(f'tumor_indices = {tumor_indices}')
                        tumor_indices = tumor_indices.view(-1)
                        images = images[:, tumor_indices]
                        img_coords = img_coords[:, tumor_indices]
                    else:
                        print(f"{batch['slide_id']} has no tumor indices, skipping")
                        continue

                if (args.cat_y or args.paired_training_mw or args.tile_slide_training) and not args.all_he_tiles: # Delete:
                    if matching_tiles is not None:
                        matching_tiles = matching_tiles.to(args.device, non_blocking=True)
                        valid_matching_tiles = ((~torch.isnan(matching_tiles)) & (matching_tiles < ihc_images.shape[1])).flatten()
                        if len(torch.nonzero(valid_matching_tiles)) == 0:
                            print(f"{batch['slide_id']} has no valid matching tiles, skipping")
                            continue
                        if args.malig_paired_mw and tumor_indices is not None and tumor_indices.shape[-1] > 0:
                            # get the indices of true values in valid_matching_tiles
                            valid_matching_indices = torch.nonzero(valid_matching_tiles.clone().flatten()).long().flatten()
                            tumor_in_non_nan = np.isin(valid_matching_indices.cpu().numpy(), tumor_indices.cpu().numpy())
                            tumor_matching_indices = np.where(tumor_in_non_nan)[0]
                            valid_tumor_matching_indices = valid_matching_indices[tumor_matching_indices].flatten()
                            if args.replace_benign_w_zero:
                                # get all indices which are in valid_matching_indices and not on valid_tumor_matching_indices without a for loop
                                non_tumor_matching_indices = np.setdiff1d(valid_matching_indices.cpu().numpy(), valid_tumor_matching_indices.cpu().numpy())
                                non_tumor_matching_tiles = matching_tiles.clone().flatten()[non_tumor_matching_indices].long().flatten()
                                images, img_coords = images[:, valid_matching_indices], img_coords[:, valid_matching_indices]
                                matching_tiles = matching_tiles.flatten()[valid_matching_indices].long().flatten()
                                # fill the non matching indices with zero vectors
                                ihc_images[:, non_tumor_matching_tiles] = 0    
                            else:
                                matching_tiles = matching_tiles.flatten()[valid_tumor_matching_indices].long()
                                images, img_coords = images[:, valid_tumor_matching_indices], img_coords[:, valid_tumor_matching_indices]
                                    
                        else:
                            images = images[:, valid_matching_tiles]
                            img_coords = img_coords[:, valid_matching_tiles]
                            matching_tiles = matching_tiles[:, valid_matching_tiles].long().flatten()

                        if args.malig_paired_mw:
                            ihc_images = ihc_images[:, matching_tiles]

                        if cancer_prob is not None:
                            cancer_prob = cancer_prob.to(args.device, non_blocking=True)[:, valid_matching_tiles]
                        
                        if (args.cat_y and not args.create_y) or args.tile_slide_training:
                            if tile_y is not None:
                                tile_y = tile_y.to(args.device, non_blocking=True).squeeze(0)
                            else:
                                print(f"{batch['slide_id']} tile_y is None, skipping")
                                continue    

                        if args.cnn_on_y:
                            tiles = tiles.to(args.device, non_blocking=True)
                            # tiles = tiles[valid_matching_tiles]
                        
                        if args.add_virchow2_features:
                            print(f'images.shape before adding virchow2 features = {images.shape}, virchow2_feats.shape = {virchow2_feats.shape}')
                            virchow2_feats = virchow2_feats.to(args.device, non_blocking=True)
                            virchow2_feats = virchow2_feats[valid_matching_tiles]
                            virchow2_feats = virchow2_feats.view(-1, VIRCHOW2_FEATURE_DIM)
                            images = torch.cat([images, virchow2_feats], dim=-1)
                        
                        if args.add_uni2_features:
                            print(f'images.shape before adding uni2 features = {images.shape}, uni2_feats.shape = {uni2_feats.shape}')
                            uni2_feats = uni2_feats.to(args.device, non_blocking=True)
                            uni2_feats = uni2_feats[valid_matching_tiles]
                            uni2_feats = uni2_feats.view(-1, UNI2_FEATURE_DIM)
                            images = torch.cat([images, uni2_feats], dim=-1)
                        
                        if args.add_conch_features:
                            print(f'images.shape before adding conch features = {images.shape}, conch_feats.shape = {conch_feats.shape}')
                            conch_feats = conch_feats.to(args.device, non_blocking=True)
                            conch_feats = conch_feats[valid_matching_tiles]
                            conch_feats = conch_feats.view(-1, CONCH_FEATURE_DIM)
                            images = torch.cat([images, conch_feats], dim=-1)

                        if args.external_model_logits:
                            external_tile_logits = external_tile_logits.to(args.device, non_blocking=True)
                            external_tile_logits = external_tile_logits[valid_matching_tiles]
                        
                        images = images.view(1, -1, args.tile_input_dim)

                        if args.y_and_tumor_from_map and not args.cnn_on_y:
                            # only keep tile_y values that are not -1 and not 0
                            valid_tile_y_indices = (tile_y != -1) & (tile_y != 0)
                            images = images[valid_tile_y_indices]
                            if images.size(0) == 0:
                                print(f"{batch['slide_id']} has no valid tile_y indices, skipping")
                                continue
                            img_coords = img_coords[valid_tile_y_indices]
                            if not args.cnn_on_y:
                                tile_y = tile_y[valid_tile_y_indices].view(-1, 1)
                            # if args.cnn_on_y:
                            #     tiles = tiles[valid_tile_y_indices]
                            
                    else:
                        print(f"{batch['slide_id']} matching_tiles is None, skipping")
                        continue
                
                elif args.all_he_tiles:
                    no_loss = True
                    if args.add_virchow2_features:
                        print(f'images.shape before adding virchow2 features = {images.shape}, virchow2_feats.shape = {virchow2_feats.shape}')
                        virchow2_feats = virchow2_feats.to(args.device, non_blocking=True)
                        images = torch.cat([images, virchow2_feats], dim=-1)
                    
                    if args.add_uni2_features:
                        print(f'images.shape before adding uni2 features = {images.shape}, uni2_feats.shape = {uni2_feats.shape}')
                        uni2_feats = uni2_feats.to(args.device, non_blocking=True)
                        images = torch.cat([images, uni2_feats], dim=-1)
                    
                    if args.add_conch_features:
                        print(f'images.shape before adding conch features = {images.shape}, conch_feats.shape = {conch_feats.shape}')
                        conch_feats = conch_feats.to(args.device, non_blocking=True)
                        images = torch.cat([images, conch_feats], dim=-1)
                    
                    images = images.view(1, -1, args.tile_input_dim)

                if args.survival:
                    censoreship = batch['censoreship']

            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                # ********** get the logits **********
                if save_embed:
                    if args.cat_y: 
                        if args.train_on_y:
                            images, img_coords = images.view(-1, 1, args.tile_input_dim), img_coords.view(-1, 1, 2)
                            print(f'images.shape = {images.shape}')
                            if images.shape[0] == 0:
                                print(f"slide {batch['slide_id']} has no matching ihc images, skipping")
                                continue
                            
                            if not args.pred_y_baseline:
                                if args.cnn_on_y:
                                    tiles = tiles.view(-1, 3, args.tile_size, args.tile_size)
                                    print(f'tiles.shape = {tiles.shape} before model call')
                                    tile_logits = model(tiles)
                                    print(f'tile_logits.shape = {tile_logits.shape} after model call')
                                    logits = tile_logits.mean()
                                else:
                                    logits, tile_logits = model(images, img_coords)

                                if args.run_inference:
                                    save_pred_y_dir = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", f"tile_y_pred_{args.mpp}", slide_id)
                                    if not os.path.exists(save_pred_y_dir):
                                        os.makedirs(save_pred_y_dir, exist_ok=True)
                                    file_name = f"local_tile_y_pred_val{args.val_fold[-1]}.npy" if args.only_annotated else f"local_tile_y_pred_ad_val{args.val_fold[-1]}.npy"
                                    if args.y_and_tumor_from_map:
                                        file_name = file_name.replace('local_tile_y', 'regional_tile_y_from_map')
                                    if args.add_virchow2_features:
                                        file_name = file_name.replace('pred', 'pred_add_virchow2')
                                        if args.add_uni2_features:
                                            file_name = file_name.replace('virchow2', 'virchow2_uni2')
                                            if args.add_conch_features:
                                                file_name = file_name.replace('virchow2_uni2', 'virchow2_uni2_conch')
                                    if args.task_config.get('setting', 'multi_class') == 'binary':
                                        tile_logits = torch.softmax(tile_logits, dim=-1)[:, -1]
                                        file_name = file_name.replace('tile_y', 'binary_tile_y')

                                    np.save(os.path.join(save_pred_y_dir, file_name), tile_logits.cpu().detach().numpy())
                                    print(f'Saved tile_y_pred for slide {slide_id} to {os.path.join(save_pred_y_dir, file_name)}')
                            else:  # args.pred_y_baseline
                                tile_logits, embed = model(images, img_coords, return_embed=True)
                                embed = embed[0]  # Get the first tile's embedding
                                logits = tile_logits[0]

                                if args.run_inference:
                                    save_pred_bl_y_dir = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", f"tile_y_pred_{args.mpp}", slide_id)
                                    if not os.path.exists(save_pred_bl_y_dir):
                                        os.makedirs(save_pred_bl_y_dir, exist_ok=True)
                                    file_name = f"local_tile_y_pred_bl_val{args.val_fold[-1]}.npy" if args.only_annotated else f"local_tile_y_pred_bl_ad_val{args.val_fold[-1]}.npy"
                                    if args.y_and_tumor_from_map:
                                        file_name = file_name.replace('tile_y', 'tile_y_from_map')
                                    np.save(os.path.join(save_pred_bl_y_dir, file_name), tile_logits.cpu().detach().numpy())
                                    print(f'Saved tile_y_bl for slide {slide_id} to {os.path.join(save_pred_bl_y_dir, file_name)}')
                        else:
                            matching_ihc_images = ihc_images.squeeze(0)[matching_tiles]
                            # to create y
                            if args.create_y:
                                matching_ihc_images, img_coords = matching_ihc_images.unsqueeze(0).transpose(0, 1), img_coords.unsqueeze(0).transpose(0, 1)
                                print(f'matching_ihc_images.shape = {matching_ihc_images.shape}')
                                if matching_ihc_images.shape[0] == 0:
                                    print(f"slide {batch['slide_id']} has no matching ihc images, skipping")
                                    continue

                                logits = model(matching_ihc_images, img_coords)
                                
                                if args.run_inference:
                                    print(f'y = {logits}')
                                    save_y_dir = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", f"tile_y_{args.mpp}", slide_id)
                                    if not os.path.exists(save_y_dir):
                                        os.makedirs(save_y_dir, exist_ok=True)
                                    file_name = f"tile_y_val{args.val_fold[-1]}.npy" if args.only_annotated else f"tile_y_ad_val{args.val_fold[-1]}.npy"
                                    np.save(os.path.join(save_y_dir, file_name), logits.cpu().detach().numpy())
                                    print(f'Saved tile_y for slide {slide_id} to {os.path.join(save_y_dir, file_name)}')
                                    continue

                    elif args.score_can_as_sb or args.score_can_as_bs or args.film or args.weighted_film:
                        cancer_prob = cancer_prob.to(args.device, non_blocking=True)
                        if args.tile_slide_model:
                            images, img_coords = images.view(-1, 1, args.tile_input_dim), img_coords.view(-1, 1, 2)
                            if args.use_tile_classifier:
                                tile_logits = model(images, img_coords, predict_tile_logits=True, tile_classifier=True)
                            else:
                                tile_logits = model(images, img_coords, predict_tile_logits=True)
                        elif args.load_tile_logits:
                            tile_logits = tile_logits.to(args.device, non_blocking=True)
                        else:
                            images, img_coords = images.view(-1, 1, args.tile_input_dim), img_coords.view(-1, 1, 2)
                            _, tile_logits = args.tile_model(images, img_coords)
                        
                        images = images[:,:, :args.input_dim]
                        images, img_coords = images.view(1, -1, args.input_dim), img_coords.view(1, -1, 2)
                        if args.weighted_film:
                            logits = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, weighted_film=args.weighted_film)
                        elif not args.film:
                            logits = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, score_as_scale=args.score_can_as_sb)
                        else: # film
                            if args.const_tl:
                                tile_logits = torch.full_like(tile_logits, 3)
                            if args.const_cp:
                                cancer_prob = torch.full_like(cancer_prob, 1)

                            if args.external_model_logits:
                                logits, embed = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, external_model_scores=external_tile_logits, film=args.film, return_embed=True)
                            else:
                                logits, embed = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, film=args.film, return_embed=True)

                    elif args.synth_ihc_train: # Delete
                        images = torch.cat([images, synth_ihc_images], dim=-1)
                        logits, embed = model(images, img_coords, return_embed=True)

                    elif args.predict_cancer:
                        images, img_coords = images.transpose(0, 1), img_coords.transpose(0, 1)
                        local_tumor_labels = local_tumor_labels.squeeze(0).long()
                        print(f'images.shape = {images.shape}, local_tumor_labels.shape = {local_tumor_labels.shape}')
                        logits, embed = model(images, img_coords, return_embed=True)
                        embed = embed[0]  # Get the first tile's embedding
                    elif isinstance(model, TileClassificationHead):
                        logits, tile_logits, embed = model(images, img_coords, return_embed=True)
                        embeds[batch_idx] = embed.cpu().numpy()
                    elif args.window_training:
                        window_logits = []
                        for window in slide_windows:
                            wndw_logits = model(window['imgs'], window['coords'])
                            window_logits.append(wndw_logits)
                        window_logits = torch.stack(window_logits).view(len_windows, -1)
                    elif args.paired_training_mw:
                        if args.malig_baseline:
                            logits, embed = model(ihc_images, img_coords, return_embed=True)
                        elif args.paired_mw_w_cancer:
                            logits, embed = model(images, img_coords, ihc_images, modality_weights=True, cancer_weights=True, tile_cancer_probs=cancer_prob, return_embed=True)
                        elif args.paired_mw_cancer_scale:
                            logits, embed = model(images, img_coords, ihc_images, modality_weights=True, cancer_scale=True, tile_cancer_probs=cancer_prob, return_embed=True)
                        else:
                            logits, embed = model(images, img_coords, ihc_images, modality_weights=True, return_embed=True)
                        embeds[batch_idx] = embed.cpu().numpy()
                    elif args.paired_training_mb:
                        logits, embed = model(images, img_coords, ihc_images, ihc_coords, modality_bias=True, return_embed=True)
                        embeds[batch_idx] = embed.cpu().numpy()
                    elif args.paired_training_cat_h:
                        logits, embed = model(images, img_coords, ihc_images, ihc_coords, concat_h=True, return_embed=True)
                        embeds[batch_idx] = embed.cpu().numpy()
                    else:
                        logits, embed = model(images, img_coords, return_embed=True)
                        embeds[batch_idx] = embed.cpu().numpy()
                else: # no save_embed
                    if args.cat_y: # Delete
                        if args.train_on_y:
                            images, img_coords = images.view(-1, 1, args.tile_input_dim), img_coords.view(-1, 1, 2)
                            print(f'images.shape = {images.shape}')
                            if images.shape[0] == 0:
                                print(f"slide {batch['slide_id']} has no matching ihc images, skipping")
                                continue
                            
                            if not args.pred_y_baseline:
                                if args.cnn_on_y:
                                    tiles = tiles.view(-1, 3, args.tile_size, args.tile_size)
                                    print(f'tiles.shape = {tiles.shape} before model call')
                                    tile_logits = model(tiles)
                                    print(f'tile_logits.shape = {tile_logits.shape} after model call')
                                    # in case of binary classification keep dimension
                                    logits = tile_logits.mean(dim=0, keepdim=True) if args.task_config.get('setting', 'multi_class') == 'binary' else tile_logits.mean()
                                else:
                                    logits, tile_logits = model(images, img_coords)
                            else:
                                tile_logits = model(images, img_coords)
                        else:
                            matching_ihc_images = ihc_images.squeeze(0)[matching_tiles]
                            # to create y
                            if args.create_y:
                                matching_ihc_images, img_coords = matching_ihc_images.unsqueeze(0).transpose(0, 1), img_coords.unsqueeze(0).transpose(0, 1)
                                print(f'matching_ihc_images.shape = {matching_ihc_images.shape}')
                                if matching_ihc_images.shape[0] == 0:
                                    print(f"slide {batch['slide_id']} has no matching ihc images, skipping")
                                    continue

                                logits = model(matching_ihc_images, img_coords).unsqueeze(-1)
                                
                                if args.run_inference:
                                    print(f'y = {logits}')
                                    save_y_dir = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", f"tile_y_{args.mpp}_val{args.val_fold[-1]}", slide_id)
                                    if not os.path.exists(save_y_dir):
                                        os.makedirs(save_y_dir, exist_ok=True)
                                    file_name = f"tile_y_val{args.val_fold[-1]}.npy" if args.only_annotated else f"tile_y_ad_val{args.val_fold[-1]}.npy"
                                    np.save(os.path.join(save_y_dir, file_name), logits.cpu().detach().numpy())
                                    print(f'Saved tile_y for slide {slide_id} to {os.path.join(save_y_dir, file_name)}')
                                    continue

                    elif args.score_can_as_sb or args.score_can_as_bs or args.film or args.weighted_film:
                        cancer_prob = cancer_prob.to(args.device, non_blocking=True)
                        if args.tile_slide_model:
                            images, img_coords = images.view(-1, 1, args.tile_input_dim), img_coords.view(-1, 1, 2)
                            if args.use_tile_classifier:
                                tile_logits = model(images, img_coords, predict_tile_logits=True, tile_classifier=True)
                            else:
                                tile_logits = model(images, img_coords, predict_tile_logits=True)
                        elif args.load_tile_logits:
                            tile_logits = tile_logits.to(args.device, non_blocking=True)
                        else:
                            _, tile_logits = args.tile_model(images, img_coords)                        

                        images = images[:,:, :args.input_dim]
                        images, img_coords = images.view(1, -1, args.input_dim), img_coords.view(1, -1, 2)
                        if args.weighted_film:
                            logits = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, weighted_film=args.weighted_film)
                        elif not args.film:
                            logits = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, score_as_scale=args.score_can_as_sb)
                        else:
                            if args.const_tl:
                                tile_logits = torch.full_like(tile_logits, 3)
                            if args.const_cp:
                                cancer_prob = torch.full_like(cancer_prob, 1)
                            
                            if args.external_model_logits:
                                logits = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, external_model_scores=external_tile_logits, film=args.film)
                            else:
                                logits = model(images, img_coords, tile_scores=tile_logits, tile_cancer_probs=cancer_prob, film=args.film)

                    elif args.synth_ihc_train: # Delete
                        images = torch.cat([images, synth_ihc_images], dim=-1)
                        logits = model(images, img_coords)
                    elif args.predict_cancer:
                        images, img_coords = images.transpose(0, 1), img_coords.transpose(0, 1)
                        local_tumor_labels = local_tumor_labels.squeeze(0).long()
                        print(f'images.shape = {images.shape}, local_tumor_labels.shape = {local_tumor_labels.shape}')
                        logits = model(images, img_coords)
                    elif args.window_training:
                        window_logits = []
                        for window in slide_windows:
                            wndw_logits = model(window['imgs'], window['coords'])
                            window_logits.append(wndw_logits)
                        window_logits = torch.stack(window_logits).view(len_windows, -1)
                        logits = window_logits.mean(dim=0) # for evaluation records
                    elif args.paired_training_mw:
                        if args.malig_baseline:
                            logits = model(ihc_images, img_coords)
                        elif args.paired_mw_w_cancer:
                            logits = model(images, img_coords, ihc_images, modality_weights=True, cancer_weights=True, tile_cancer_probs=cancer_prob)
                        elif args.paired_mw_cancer_scale:
                            logits = model(images, img_coords, ihc_images, modality_weights=True, cancer_scale=True, tile_cancer_probs=cancer_prob)
                        else:
                            logits = model(images, img_coords, ihc_images, modality_weights=True)
                    elif args.paired_training_mb:
                        logits = model(images, img_coords, ihc_images, ihc_coords, modality_bias=True)
                    elif args.paired_training_cat_h:
                        logits = model(images, img_coords, ihc_images, ihc_coords, concat_h=True)
                    else:
                        logits = model(images, img_coords)
                        
                if logits is None:
                    continue
                
                # ********** Shapes and types before loss **********
                if not args.window_training:
                    if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                        label = label.squeeze(-1).float()
                    elif task_setting == 'continuous' and not args.loss_fn == 'cox':
                        label = label.squeeze(-1).float()
                        logits = logits.squeeze(-1)

                    else:
                        label = label.squeeze(-1).long()
                
                # ********** loss computation **********
                if not no_loss:
                    try:
                        if not args.loss_fn == 'cox':
                            if not args.survival:
                                print(f'logits = {logits}, label = {label}') # Delete
                                if args.train_on_y:
                                    if args.task_config.get('setting', 'multi_class') == 'binary':
                                        print(f"tile_y = {tile_y}")
                                        tile_y = (tile_y >= 2).long().squeeze(-1)

                                    elif not tile_logits.shape == tile_y.shape:
                                        tile_logits = tile_logits.view(-1)
                                        tile_y = tile_y.view(-1)
                                    # pred_class = logits.argmax()
                                    # loss = loss_fn(pred_class, label)
                                    
                                    loss = loss_fn(tile_logits, tile_y)
                                    print(f"loss = {loss}")
                                    print(f"tile_y = {tile_y}, tile_logits = {tile_logits}, logits = {logits}, label = {label}")
                                elif args.predict_cancer:
                                    loss = loss_fn(logits, local_tumor_labels)
                                elif args.window_training:
                                    window_labels = torch.stack([wndw['window_y'] for wndw in slide_windows])
                                    print(f"window_logits = {window_logits}, window_labels = {window_labels}")
                                    loss = loss_fn(window_logits, window_labels)
                                    print(f"loss = {loss}")
                                elif args.tile_slide_training:
                                    tile_loss = loss_fn(tile_logits.view(-1), tile_y.view(-1))
                                    slide_loss = loss_fn(logits, label)
                                    loss = tile_loss + slide_loss
                                    print(f"tile_loss = {tile_loss}, slide_loss = {slide_loss}, loss = {loss}")
                                else:
                                    loss = loss_fn(logits, label)
                            else:
                                loss = loss_fn(logits, label, censoreship)
                        else:
                            curr_idx = batch_idx % args.gc
                            cum_logits[curr_idx] = logits[0]
                            cum_labels[curr_idx] = label[0]
                            cum_censoreship[curr_idx] = censoreship[0]
                            if (batch_idx + 1) % args.gc == 0:
                                # if batch has no uncensored examples loss will be 0 so we accumulate the examples for another batch
                                if torch.sum(cum_censoreship) == 0:
                                    print('no uncensored samples so will calculate loss')
                                    if not accumulate_another_batch:
                                        prev_logits = torch.tensor(cum_logits, requires_grad=True)
                                        prev_labels = torch.tensor(cum_labels, requires_grad=True)
                                        prev_censoreship = torch.tensor(cum_censoreship, requires_grad=True)
                                    else:
                                        prev_logits = torch.cat(
                                            (prev_logits, torch.tensor(cum_logits, requires_grad=True)))
                                        prev_labels = torch.cat(
                                            (prev_labels, torch.tensor(cum_labels, requires_grad=True)))
                                        prev_censoreship = torch.cat(
                                            (prev_censoreship, torch.tensor(cum_censoreship, requires_grad=True)))
                                    accumulate_another_batch = True
                                else:
                                    if accumulate_another_batch:
                                        curr_logits = torch.cat(
                                            (prev_logits, torch.tensor(cum_logits, requires_grad=True)))
                                        curr_labels = torch.cat(
                                            (prev_labels, torch.tensor(cum_labels, requires_grad=True)))
                                        curr_censoreship = torch.cat(
                                            (prev_censoreship, torch.tensor(cum_censoreship, requires_grad=True)))
                                    else:
                                        curr_logits = cum_logits
                                        curr_labels = cum_labels
                                        curr_censoreship = cum_censoreship
                                    loss = loss_fn(curr_logits, curr_labels, curr_censoreship)
                                # loss = loss_fn(cum_logits, cum_labels, cum_censoreship)
                    except:
                        if not loader.dataset.test_on_all:
                            raise
                        no_loss = True
                        print("Evaluating model on slides with no label so loss will be meaningless")

            # update the records
            if not no_loss:
                if not args.loss_fn == 'cox':
                    records['loss'] += loss.item()
                elif (batch_idx + 1) % args.gc == 0 and torch.sum(cum_censoreship) != 0:
                    records['loss'] += loss.item()
            if task_setting == 'multi_label':
                Y_prob = torch.sigmoid(logits)
                records['prob'][batch_idx] = Y_prob.cpu().numpy()
                records['label'][batch_idx] = label.cpu().numpy()
            elif task_setting == 'multi_class' or task_setting == 'binary':
                Y_prob = torch.softmax(logits, dim=-1).cpu()
                print(f"Y_prob = {Y_prob}, shape = {Y_prob.shape}")
                if not (args.predict_cancer or args.pred_y):
                    # convert label to one-hot
                    records['prob'][batch_idx] = Y_prob.numpy() # if args.n_classes == 1 else np.argmax(Y_prob.numpy())
                    try:
                        label_ = torch.zeros_like(Y_prob).scatter_(-1, label.cpu().unsqueeze(1), 1)
                    except BaseException as e:
                        label_ = torch.zeros_like(Y_prob)
                        label_[label.item()] = 1
                    records['label'][batch_idx] = label_.numpy()
                else:
                    if tumor_indices is not None and args.predict_cancer:
                        print(f"local_tumor_labels.shape = {local_tumor_labels.shape}, y_prob.shape = {Y_prob.shape}")
                        # convert label to one-hot
                        label_ = torch.zeros_like(Y_prob).scatter_(1, local_tumor_labels.cpu().unsqueeze(1), 1)
                        records['prob'].append(Y_prob.numpy())
                        records['label'].append(label_.numpy())
                    else:
                        # print(f"label.shape = {local_tumor_labels.shape}, y_prob.shape = {Y_prob.shape}")
                        label_ = torch.zeros_like(Y_prob).scatter_(1, label.cpu().unsqueeze(1), 1)
                        records['prob'].append(Y_prob.numpy())
                        records['label'].append(label_.numpy())

                    if args.run_inference and args.predict_cancer:
                        y_prob_to_save = Y_prob[:, 1].numpy()
                        save_prob_dir = os.path.join(args.root_path.replace("gigapath_features", "cancer_probs"), slide_id)
                        if not os.path.exists(save_prob_dir):
                            os.makedirs(save_prob_dir, exist_ok=True)
                        np.save(os.path.join(save_prob_dir, f"cancer_prob_val{args.val_fold[0]}.npy"), y_prob_to_save)
                    
                print(f"label_ = {label_}, label = {label}, np.argmax(label_.numpy()) = {np.argmax(label_.numpy())}") # Delete
                # print(f'records = {records}') # Delete
            elif task_setting == 'continuous':
                if not args.loss_fn == 'cox':
                    #     records['prob'][batch_idx] = ((logits * model.std.item()) + model.mean.item()).cpu().numpy()
                    #     records['label'][batch_idx] = ((label * model.std.item()) + model.mean.item()).cpu().numpy()
                    if args.pred_y_baseline or args.pred_y:
                        if args.all_he_tiles:
                            tile_y, tile_logits = tile_y.view(-1), tile_logits.view(-1)
                            valid_matching_tiles = ((~torch.isnan(matching_tiles)) & (matching_tiles < ihc_images.shape[1])).view(-1)
                            # get the indices of true values in valid_matching_tiles
                            valid_matching_indices = torch.nonzero(valid_matching_tiles).view(-1)

                            tumor_in_non_nan = np.isin(valid_matching_indices, tumor_indices)
                            tumor_matching_indices = np.where(tumor_in_non_nan)[0]
                            tile_y = tile_y[tumor_matching_indices]

                            tile_logits = tile_logits[valid_matching_tiles]
                            tile_logits = tile_logits[tumor_matching_indices]

                        records['prob'].append(tile_logits.view(-1, 1).cpu().numpy())
                        records['label'].append(tile_y.view(-1, 1).cpu().numpy())
                    elif args.train_on_y:
                        pred_class = logits.argmax()
                        records['prob'][batch_idx] = pred_class.cpu().numpy()
                    elif not args.loss_fn == 'coral':
                        records['prob'][batch_idx] = logits.cpu().numpy()
                        records['label'][batch_idx] = label.cpu().numpy()
                    else:
                        prob = torch.sigmoid(logits)
                        pred = torch.sum(prob > 0.5, dim=-1)
                        records['prob'][batch_idx] = pred.cpu().numpy()
                        records['label'][batch_idx] = label.cpu().numpy()    
                    if not no_loss and not args.survival:
                        for key in regression_losses:
                            try:
                                if args.train_on_y:
                                    if key in ['spearmanr', 'pearsonr']:
                                        np_tile_y = tile_y.squeeze(-1).clone().detach().cpu().numpy()
                                        np_tile_logits = tile_logits.squeeze(-1).clone().detach().cpu().numpy()
                                        if np_tile_y.size < 2:
                                            print(f"Only one tumor tile for slide {slide_id}, cannot compute {key}")
                                            continue
                                        rho, _ = regression_losses[key](np_tile_logits, np_tile_y)
                                        if not np.isnan(rho):
                                            records[key] += rho
                                            print(f"Computed {key} for slide {slide_id}: {rho}")
                                    else:
                                        records[key] += regression_losses[key](tile_logits, tile_y).item()
                                else:
                                    records[key] += regression_losses[key](logits, label).item()
                            except BaseException as e:
                                print(f'Cannot calculate {key}\n{e}')
                                continue
                else:
                    records['prob'][batch_idx] = logits.cpu().numpy()
                    records['label'][batch_idx] = label.cpu().numpy()
                    records['censoreship'][batch_idx] = censoreship[0].cpu().numpy()

        ### end of for batch loop ###
        if (args.predict_cancer or args.pred_y_baseline or args.pred_y or (args.train_on_y and not task_setting == 'binary')) and not args.cnn_on_y:
            records['prob'] = np.concatenate(records['prob'])
            records['label'] = np.concatenate(records['label'])

    if not no_loss:
        records['loss'] = records['loss'] / len(loader)
        if task_setting == 'continuous' and not args.survival:
            for key in regression_losses:
                records[key] = records[key] / len(loader)
    else:
        records['loss'] = 0
        if task_setting == 'continuous':
            for key in regression_losses:
                records[key] = 0
    try:
        if not task_setting == 'continuous':
            records.update(calculate_metrics_with_task_cfg(records['prob'], records['label'], args.task_config))

        if task_setting == 'multi_label':
            info = 'Epoch: {}, Loss: {:.4f}, Micro AUROC: {:.4f}, Macro AUROC: {:.4f}, Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}'.format(
                epoch, records['loss'], records['micro_auroc'], records['macro_auroc'], records['micro_auprc'],
                records['macro_auprc'])
        elif task_setting == 'multi_class' or task_setting == 'binary':
            info = 'Epoch: {}, Loss: {:.4f}, AUROC: {:.4f}, ACC: {:.4f}, BACC: {:.4f}'.format(epoch, records['loss'],
                                                                                              records['macro_auroc'],
                                                                                              records['acc'],
                                                                                              records['bacc'])
            for metric in args.task_config.get('add_metrics', []):
                info += ', {}: {:.4f}'.format(metric, records[metric])
        else:
            info = 'Epoch: {},'.format(epoch)
            if not args.survival:
                for key in regression_losses:
                    info += ' Eval {} Loss: {:.4f}'.format(key, records[key])

                if args.pred_y_baseline or args.pred_y:
                    tile_y_bin = (records['label'] >= 2).astype(int)
                    # create a DataFrame to hold the values and save it to csv
                    df = pd.DataFrame({'tile_y': records['label'].flatten(), 'tile_y_bin': tile_y_bin.flatten(), 'tile_pred': records['prob'].flatten()})
                    df.to_csv(os.path.join(args.save_dir, f"tumor_tile_valid_y_preds_{args.mpp}_val{args.val_fold[0]}.csv"), index=False)
                    
                    print(f'Saved tile_y predictions to {os.path.join(args.save_dir, f"tumor_tile_valid_y_preds_{args.mpp}_val{args.val_fold[0]}.csv")}')

                    print(f"tile_y_bin = {tile_y_bin}, records['label'] = {records['label']}")
                    records.update(calculate_metrics_with_task_cfg(records['prob'], tile_y_bin, {'name': 'binary_y', 'setting': 'binary', 'label_dict': {0: 0, 1: 1}}))
                    info = 'Epoch: {}, Loss: {:.4f}, AUROC: {:.4f}, ACC: {:.4f}, BACC: {:.4f}'.format(epoch, records['loss'],
                                                                                              records['macro_auroc'],
                                                                                              records['acc'],
                                                                                              records['bacc'])
                    for metric in args.task_config.get('add_metrics', []):
                        info += ', {}: {:.4f}'.format(metric, records[metric])
            else:
                info += ' Eval Loss: {:.4f}'.format(records['loss'])
        print(info)
    except BaseException as e:
        print("Failed to get metrics, this should only happen on test set.")
        print(e)

    # return the embeddings
    if save_embed:
        return records, embeds
    return records
