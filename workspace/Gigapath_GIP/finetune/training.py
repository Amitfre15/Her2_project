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

from gigapath.classification_head import get_model, get_regressor, TileClassificationHead
from metrics import calculate_metrics_with_task_cfg
from utils import (get_optimizer, get_loss_function, \
                   Monitor_Score, get_records_array,
                   log_writer, adjust_learning_rate,
                   get_regression_losses, LogCoshLoss, get_test_loader, 
                   Contrastive_Loss, DifferentiableHistogramKL, CLIPLoss, MidPenalizedLoss, margin_loss)
from slides_to_thumbs import patch_weighted_score_matrices, show_score_matrix, IHC_THUMB_WIDTH, IHC_THUMB_HEIGHT, SEG_THUMB_WIDTH, SEG_THUMB_HEIGHT
from datasets.slide_datatset import SlidingWindowDataset
from finetune.cycle_gan import ResNetGenerator, PatchDiscriminator, style_loss

def print_free_gpu_memory(device):
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

def compute_style_stats(emb):
    mean = emb.mean(dim=-2, keepdim=True)
    std = emb.std(dim=-2, keepdim=True, unbiased=False)
    return mean, std

def slice_tiles(indices, batch):
    batch['imgs'] = batch['imgs'][:, indices, :]
    batch['coords'] = batch['coords'][:, indices, :]
    if batch['local_labels'] is not None:
        batch['local_labels'] = batch['local_labels'][:, indices, :]
    if batch['ihc_imgs'] is not None:
        batch['ihc_imgs'] = batch['ihc_imgs'][:, indices, :]
        batch['ihc_coords'] = batch['ihc_coords'][:, indices, :]
    if batch['teacher_labels'] is not None:
        batch['teacher_labels'] = batch['teacher_labels'][:, indices, :]
    if batch['matching_tiles'] is not None:
        batch['matching_tiles'] = batch['matching_tiles'][:, indices, :]

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
    for slide in data_df[args.slide_key]:
        if not slide.startswith("21-1617_2_8"):
            continue
        args.get_single_slide = slide
        print(f'slide = {slide}.mrxs')
        test_data = dataset_class(data_df, args.root_path, args.task_config, slide_key=args.slide_key, label=args.label, \
                                  dataset_name=args.test_dataset, folds=args.test_fold,
                                  use_clinical_features=args.clinical_features, \
                                  test_on_all=args.test_on_all, get_single_slide=args.get_single_slide,
                                  window_size=args.window_size, stride=args.stride)
        test_loader = get_test_loader(test_data, for_heatmap=True, **vars(args))

        # Initialize score and weight matrices
        # weighted_score_matrix = torch.zeros((IHC_THUMB_HEIGHT, IHC_THUMB_WIDTH), dtype=torch.float64).to(args.device)
        # weight_matrix = torch.zeros((IHC_THUMB_HEIGHT, IHC_THUMB_WIDTH), dtype=torch.float64).to(args.device)
        # init both matrices with full matrix of minus 1
        weighted_score_matrix = torch.zeros((SEG_THUMB_HEIGHT, SEG_THUMB_WIDTH), dtype=torch.float64).to(args.device)
        weight_matrix = torch.zeros((SEG_THUMB_HEIGHT, SEG_THUMB_WIDTH), dtype=torch.float64).to(args.device)

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
                            score = logits * model.std.item() + model.mean.item()
                            results = {'window_name': window_name, 'score': score.item(),
                                       'label': test_data.label.item()}

                    results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)

                patch_weighted_score_matrices(weighted_score_matrix=weighted_score_matrix, weight_matrix=weight_matrix,
                                              img_coords=img_coords, window_score=score)

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
        npy_output_path = os.path.join(args.save_dir, f"score_matrix.npy")
        np.save(npy_output_path, compressed_score_matrix)
        print(f"Saved score matrix for slide: {slide} at {npy_output_path}")

        show_score_matrix(final_score_matrix, slide, save_dir=args.save_dir, vmin=-1, vmax=final_score_matrix.max())

        # npz_output_path = os.path.join(args.save_dir, f"{slide}_score_matrix.npz")
        # np.savez_compressed(npz_output_path, compressed_score_matrix)

        # print(f"Saved score matrix for slide: {slide} at {npz_output_path}")

    results_df.to_csv(os.path.join(args.save_dir, 'slide_scores.csv'), index=False)
    print(f'Saved slide_scores.csv')

    return


def get_batch_weights(labels, weights, bin_edges):
    # Digitize target values to find their corresponding bins
    target_bins = torch.bucketize(labels, bin_edges) - 1  # Get bin indices

    # Fetch weights based on the computed bins
    batch_weights = weights[target_bins]

    return batch_weights


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

    if args.student_training:  # Delete
        args.student_net = True
        
    # set up the model
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
    if not args.use_tile_classification:
        test_records, embeds = evaluate(test_loader, model, fp16_scaler, loss_fn, 0, args, save_embed=True) # 'test'
    else:
        test_records = evaluate(test_loader, model, fp16_scaler, loss_fn, 0, args, save_embed=False) # 'test'
    results_df = pd.DataFrame({'slide_name': [], 'score': [], 'label': []})
    # save each embedding asa seperate filewith the name being the slide_id
    for idx, slide_id in enumerate(test_loader.dataset.slide_data[args.slide_key]):
        if not args.use_tile_classification:
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

                # loss_color_HE = criterion_color(fake_HE.mean(dim=[2, 3]), 
                #                                 real_HE.mean(dim=[2, 3]))
                # loss_color_IHC = criterion_color(fake_IHC.mean(dim=[2, 3]), 
                #                                  real_IHC.mean(dim=[2, 3]))
                # loss_color_std_HE = criterion_color(fake_HE.std(dim=[2, 3]), 
                #                                 real_HE.std(dim=[2, 3]))
                # loss_color_std_IHC = criterion_color(fake_IHC.std(dim=[2, 3]), 
                #                                  real_IHC.std(dim=[2, 3]))
                # loss_color_max_HE = criterion_color(fake_HE.amax(dim=[2, 3]), 
                #                                 real_HE.max(dim=[2, 3]))
                # loss_color_max_IHC = criterion_color(fake_IHC.amax(dim=[2, 3]), 
                #                                  real_IHC.max(dim=[2, 3]))

                loss_G = (
                    # 10.0 * (loss_color_HE + loss_color_IHC) +
                    # 10.0 * (loss_color_std_HE + loss_color_std_IHC) +
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

    if args.student_training:  # Delete
        args.teacher_model = get_model(**vars(args))
        args.teacher_model = args.teacher_model.to(args.device)
        args.teacher_model.load_state_dict(torch.load(args.teacher_model_ckpt), strict=False)
        print("Loaded teacher model")
        args.student_net = True

    if args.matching_tiles_training:  # Delete
        args.ihc_regressor = get_regressor(**vars(args))
        args.ihc_regressor = args.ihc_regressor.to(args.device)
        args.ihc_reconstructor = get_regressor(**vars(args))
        args.ihc_reconstructor = args.ihc_reconstructor.to(args.device)
        print("Loaded ihc_regressor")

    if args.compress_features:
        print(f'args.input_dim // (args.reduction ** args.comp_power) = {args.input_dim // (args.reduction ** args.comp_power)}')
    
    # set up the model
    model = get_model(**vars(args))
    model = model.to(args.device)
    if args.student_training and args.model_ckpt != '':  # Delete
        print("Loaded student model")
        model.load_state_dict(torch.load(args.model_ckpt), strict=False)
    # set up the optimizer
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

    # test evaluate function
    # val_records = evaluate(val_loader, model, fp16_scaler, loss_fn, 0, args)

    val_records, test_records = None, None
    loss_pe, lr_pe, val_loss_pe = [], [], [] # Delete
    if args.use_tile_classification or args.window_training:
        ihc_gigapath_dir = os.path.join("Carmel", "Her2", "gigapath_IHC")
        local_labels_dir = args.root_path.replace("gigapath_features", "tile_labels") if args.use_tile_classification \
            else args.root_path.replace("gigapath_features", "window_labels")
        local_labels_dir = local_labels_dir.replace("gigapath_CAT_features", ihc_gigapath_dir)
        local_labels_hist = os.path.join(local_labels_dir, "local_labels_hist.npy")
        local_labels_bins = os.path.join(local_labels_dir, "local_labels_bins.npy")
        print(f"local_labels_hist = {local_labels_hist}")
        if os.path.exists(local_labels_hist):
            args.local_labels_hist = np.load(local_labels_hist)
        if os.path.exists(local_labels_bins):
            args.local_labels_bins = np.load(local_labels_bins)

    for i in range(args.epochs):
        print('Epoch: {}'.format(i))
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
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, 'fold_' + str(fold), 'model_epoch_' + str(i) + '_checkpoint.pt'))

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
    # if args.use_tile_classification and args.matching_tiles_training and epoch >= 5:
    #     model.freeze_regressors()

    if args.student_training and 'teacher_model' in args:
        args.teacher_model.eval()
        # Memory Bank to store embeddings (key: label, value: list of embeddings)
        # CLIP loss
        HE_embedding_memory, IHC_embedding_memory = {}, {}
    
    if args.cl_HE_feat:
        HE_embedding_memory = {}
        
    model.train()
    # set the start time
    start_time = time.time()

    # monitoring sequence length
    seq_len = 0
    curr_len = 0
    max_num_samples = 40000

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

    # if args.use_tile_classification or args.window_training:
    all_local_labels = None # Delete
    all_local_logits = None
    batch_local_labels = None
    batch_local_logits = None
    all_slide_labels = None
    all_slide_logits = None
    slide_windows = None
    # local_labels_mean = 0.6577 if args.use_tile_classification else 0.6782 # Delete
    # local_labels_std = 0.7558 if args.use_tile_classification else 0.7782
    # local_labels_mean = 0
    # local_labels_std = 1
    # local_labels_hist = torch.from_numpy(args.local_labels_hist).to(args.device, non_blocking=True)
    # local_labels_hist = 1.0 / (1.0 + local_labels_hist)  # Inverse frequency
    # local_labels_hist = local_labels_hist / local_labels_hist.sum()
    # local_labels_bins = (torch.from_numpy(args.local_labels_bins).to(args.device, non_blocking=True) - local_labels_mean) / local_labels_std
    # min_local_labels_bins, max_local_labels_bins = local_labels_bins.min(), local_labels_bins.max()
    # print(f"local_labels_bins = {local_labels_bins}, local_labels_hist = {local_labels_hist}")
    
    last_0_embed, last_3_embed = None, None
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    mae = torch.nn.L1Loss()
    ce = torch.nn.CrossEntropyLoss(reduction='none') 
    
    for batch_idx, batch in enumerate(train_loader):
        slide_id = batch['slide_id'][-1] 
        if args.use_tile_classification and batch['local_labels'] is None: 
            continue

        if args.window_training:
            slide_id = batch['slide_id'][-1]
            sliding_window_ds = SlidingWindowDataset(data_df=args.dataset, root_path=args.root_path, task_config=args.task_config, slide_key=args.slide_key, label=args.label, \
                                  dataset_name=args.test_dataset, folds=args.test_fold,
                                  use_clinical_features=args.clinical_features, \
                                  test_on_all=args.test_on_all, get_single_slide=slide_id,
                                  window_size=args.window_size, stride=args.window_size)
            window_loader = get_test_loader(sliding_window_ds, for_heatmap=True, **vars(args))
            slide_windows = [{'imgs': wndw['imgs'].to(args.device, non_blocking=True), 
                              'coords': wndw['coords'].to(args.device, non_blocking=True)} for wndw in window_loader]
            len_windows = len(slide_windows)
            print(f"Finished loading windows for slide {slide_id}, len(slide_windows) = {len_windows}")
            if len_windows == 0:
                continue
            

        # we use a per iteration lr scheduler
        if batch_idx % args.gc == 0 and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch, args)
        num_tiles = batch['imgs'].shape[1]
        if num_tiles > max_num_samples:
            if not args.loss_fn == 'cox':
                print(f"slide {batch['slide_id']} has too many tiles, number of tiles is {num_tiles}. Trying with {max_num_samples} tiles.")
            indices = torch.randint(low=0, high=num_tiles, size=(max_num_samples,))
            slice_tiles(indices=indices, batch=batch)

        # ********** load the batch and transform this batch **********
        images, img_coords, label, local_labels = batch['imgs'], batch['coords'], batch['labels'], batch['local_labels']
        print(f'slide_id = {slide_id}, label = {label}')
        local_labels = torch.clamp(local_labels.float(), min=0, max=3) if local_labels is not None else None
        ihc_images, ihc_coords, matching_tiles = batch['ihc_imgs'], batch['ihc_coords'], batch['matching_tiles'] # Delete
        tumor_indices, non_tumor_indices = batch['tumor_indices'], batch['non_tumor_indices']
        teacher_labels, tile_y, tile_x, tile_reg_x = batch['teacher_labels'], batch['tile_y'], batch['tile_x'], batch['tile_reg_x'] # Delete
        synth_ihc_images = batch['synth_ihc_imgs'] # Delete
        teacher_logits = None

        if args.survival:
            censoreship = batch['censoreship']
        # ********** data preperation and move to device **********
        images = images.to(args.device, non_blocking=True)
        if args.compress_features:
            orig_cos_sim = cos_sim(images[0][0], images[0])
        img_coords = img_coords.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True) # .long()
        if ihc_images is not None:
            ihc_images = ihc_images.to(args.device, non_blocking=True)
            ihc_coords = ihc_coords.to(args.device, non_blocking=True)
        if teacher_labels is not None:
            teacher_labels = teacher_labels.to(args.device, non_blocking=True)
        if local_labels is not None and (args.use_tile_classification or args.window_training or args.cl_HE_feat): # Delete:
            local_labels = local_labels.to(args.device, non_blocking=True)
            valid_label_indices = ~torch.isnan(local_labels)
            print(f"valid_label_indices = {valid_label_indices}")
            if valid_label_indices.sum() == 0:
                print(f"{slide_id} has no valid local labels, skipping")
                continue
            if args.use_tile_classification or args.cl_HE_feat:
                if not args.matching_tiles_training:
                    images = images[valid_label_indices]
                    img_coords = img_coords[valid_label_indices]
                    local_labels = local_labels[valid_label_indices]
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
        if args.matching_tiles_training or args.cat_ihc or args.cat_y or args.cat_x or args.cat_reg_x or args.pred_mean_std or args.pred_from_tumor: # Delete:
            if matching_tiles is not None:
                matching_tiles = matching_tiles.to(args.device, non_blocking=True)
                valid_matching_tiles = (~torch.isnan(matching_tiles)) & (matching_tiles < ihc_images.shape[1])
                if len(torch.nonzero(valid_matching_tiles)) == 0:
                    print(f"{batch['slide_id']} has no valid matching tiles, skipping")
                    continue
                if args.use_tile_classification:
                    valid_indices = valid_label_indices & valid_matching_tiles
                    images = images[valid_indices]
                    img_coords = img_coords[valid_indices]
                    local_labels = local_labels[valid_indices]
                    matching_tiles = matching_tiles[valid_indices].int()
                elif args.cat_ihc or args.cat_y or args.cat_x or args.pred_mean_std or args.pred_from_tumor: # Delete
                    images = images[valid_matching_tiles]
                    img_coords = img_coords[valid_matching_tiles]
                    matching_tiles = matching_tiles[valid_matching_tiles].int()
                    if args.cat_y:
                        if tile_y is not None:
                            tile_y = tile_y.to(args.device, non_blocking=True)
                            tile_y = tile_y.squeeze(0)
                        else:
                            print(f"{batch['slide_id']} tile_y is None, skipping")
                            continue
                    elif args.cat_x:
                        if tile_x is not None:
                            tile_x = tile_x.to(args.device, non_blocking=True)
                            tile_x = tile_x.squeeze(0).squeeze(0)
                        else:
                            print(f"{batch['slide_id']} tile_x is None, skipping")
                            continue
                    elif args.pred_mean_std:
                        ihc_images = ihc_images.squeeze(0)
                        matching_ihc_images = ihc_images[matching_tiles]
                        if matching_ihc_images.shape[0] == 0:
                            print(f"slide {batch['slide_id']} has no matching ihc images, skipping")
                            continue
                    if args.only_tumor_tiles:
                        if tumor_indices is not None and tumor_indices.shape[-1] > 0:
                            tumor_indices = tumor_indices.squeeze(0)
                            images = images[tumor_indices]
                            img_coords = img_coords[tumor_indices]
                            if args.cat_y:
                                tile_y = tile_y[tumor_indices]
                        else:
                            print(f"{batch['slide_id']} has no tumor indices, skipping")
                            continue
                    
                elif args.cat_reg_x:
                    if tile_reg_x is not None:
                        tile_reg_x = tile_reg_x.to(args.device, non_blocking=True).squeeze(0)
                    else:
                        print(f"{batch['slide_id']} tile_reg_x is None, skipping")
                        continue
                else:
                    matching_tiles = matching_tiles[valid_matching_tiles].int()
                # print(f"valid_matching_tiles = {valid_matching_tiles}, matching_tiles = {matching_tiles}, matching_tiles.shape = {matching_tiles.shape}")
            else:
                print(f"{batch['slide_id']} matching_tiles is None, skipping")
                continue    
        elif (args.window_training or args.use_tile_classification) and (local_labels is None or local_labels.size == 0):
            print(f"{slide_id} local labels is None, skipping")
            continue
            

        # add the sequence length
        seq_len += images.shape[-2]
        curr_len += images.shape[-2]

        with torch.cuda.amp.autocast(dtype=torch.float16 if args.fp16 else torch.float32):
            # ********** get the logits **********
            if args.use_tile_classification or args.window_training:
                if valid_label_indices.sum() == 0:
                    print(f"slide {batch['slide_id']} has no valid labels, skipping")
                    continue
                if args.window_training and slide_windows is not None:
                    valid_label_indices = torch.nonzero(valid_label_indices.flatten(), as_tuple=True)[0]
                    print(f"valid_label_indices = {valid_label_indices}")
                    slide_windows = [slide_windows[i] for i in valid_label_indices]
                    logits, local_logits = model(slide_windows)
                elif args.use_tile_classification:
                    if args.matching_tiles_training:
                        ihc_images = ihc_images.squeeze(0)
                        matching_ihc_images = ihc_images[matching_tiles]
                        logits, local_logits, regressed_he = model(images, img_coords, regress_HE=True, return_regress=True)
                        # logits, local_logits = model(images, img_coords)
                        ihc_logits, ihc_local_logits, regressed_ihc = model(matching_ihc_images, img_coords, regress_IHC=True, return_regress=True)
                    else:
                        logits, local_logits = model(images, img_coords)

                    if args.loss_fn == 'coral':
                        print(f"local_logits = {local_logits}, logits = {logits}")
                        prob = torch.sigmoid(logits)
                        pred = torch.sum(prob > 0.5, dim=-1)  # predicted class ∈ {0, 1, 2, 3}
                        local_prob = torch.sigmoid(local_logits)
                        local_preds = torch.sum(local_prob > 0.5, dim=-1)  # predicted class ∈ {0, 1, 2, 3}
                        local_preds = local_preds.squeeze(0)                        
                        print(f"local_prob = {local_prob}, local_preds = {local_preds}, prob = {prob}, pred = {pred}")
                    
                    if logits is None:
                        continue

            elif args.predict_cancer and tumor_indices is not None:
                images, img_coords = images.transpose(0, 1), img_coords.transpose(0, 1)
                local_tumor_labels = local_tumor_labels.squeeze(0).long()
                print(f'images.shape = {images.shape}, local_tumor_labels.shape = {local_tumor_labels.shape}')
                logits = model(images, img_coords)    
            elif args.student_training:
                # logits, regressor_output = model(images, img_coords)
                logits, regressor_output = model(images, img_coords)
                with torch.no_grad():
                    teacher_logits, teacher_feature_map = args.teacher_model(ihc_images, ihc_coords, return_embed=True)
            elif args.matching_tiles_training and not args.use_tile_classification:
                logits, regressor_output = model(images, img_coords)
            elif args.cat_random:
                random_features = torch.randn_like(images)
                images = torch.cat([images, random_features], dim=-1)
                logits = model(images, img_coords)
            elif args.cat_ihc or args.cat_y: # Delete
                matching_ihc_images = ihc_images.squeeze(0)[matching_tiles]
                if args.train_on_y:
                    images, img_coords = images.unsqueeze(0).transpose(0, 1), img_coords.unsqueeze(0).transpose(0, 1)
                    print(f'images.shape = {images.shape}')
                    if images.shape[0] == 0:
                        print(f"slide {batch['slide_id']} has no matching ihc images, skipping")
                        continue

                    logits, tile_logits = model(images, img_coords)
                elif not args.cat_y:
                    images = torch.cat([images, matching_ihc_images], dim=-1)
                    logits = model(images, img_coords)
                else:                    
                    print(f'tile_y = {tile_y}')
                    if not args.trim_images:
                        images = torch.cat([images, tile_y], dim=-1)
                        logits = model(images, img_coords)
                    elif "hf" in args.pretrained:
                        # use pretrained slide encoder
                        logits = model(images, img_coords, y=tile_y, trim_images=args.trim_images)
            elif args.cat_x: # Delete
                tile_x = tile_x[matching_tiles]
                if not args.compress_features:
                    images = torch.cat([images, tile_x], dim=-1)
                    logits = model(images, img_coords)
                else:
                    if "hf" in args.pretrained: # use pretrained slide encoder
                        if args.trim_images: 
                            logits = model(images, img_coords, x=tile_x, trim_images=args.trim_images)
                        elif args.x_as_sb:
                            logits = model(images, img_coords, x=tile_x, x_as_sb=args.x_as_sb)
                        else: # compress images
                            logits = model(images, img_coords, x=tile_x)             
                    else: # regress x
                        logits, regressed_x = model(images, img_coords, return_images=True)
                        regressed_x = regressed_x.squeeze(0)                    
            elif args.cat_reg_x: # Delete
                if args.x_as_sb:
                    logits = model(images, img_coords, x=tile_reg_x, x_as_sb=args.x_as_sb)
                else:
                    # # only reg_x
                    # images = tile_reg_x

                    tile_reg_x = tile_reg_x.view(images.shape[0], images.shape[1], -1)
                    images = torch.cat([images, tile_reg_x], dim=-1)
                    logits = model(images, img_coords)
            elif args.synth_ihc_train: # Delete
                images = torch.cat([images, synth_ihc_images], dim=-1)
                logits = model(images, img_coords)
            elif args.cl_HE_feat and args.compress_features: # Delete
                logits, regressed_x = model(images, img_coords, return_images=True)
                regressed_x = regressed_x.squeeze(0)
            elif args.pred_mean_std and args.compress_features: # Delete
                logits, mean_pred, std_pred = model(images, img_coords, pred_mean_std=True)
            elif args.compress_features:
                logits, comp_images = model(images, img_coords, return_images=True)
            else:
                logits = model(images, img_coords)
                print(f"logits = {logits}, shape = {logits.shape}")

            # ********** Save logits **********
            if not (args.cat_x and args.compress_features) and not (args.cl_HE_feat or args.pred_mean_std or args.train_on_y or args.predict_cancer or args.pred_from_tumor):
                if args.use_tile_classification or args.window_training:
                    if not args.loss_fn == 'coral':
                        local_logits_to_add = local_logits.flatten() if args.n_classes == 1 else torch.argmax(local_logits, dim=-1).view(-1, 1)
                        logits_to_add = logits if args.n_classes == 1 else torch.argmax(logits).unsqueeze(0)
                    else:
                        local_logits_to_add, logits_to_add = local_preds, pred
                        
                    if all_local_logits is None:
                        if not args.loss_fn == 'coral':
                            all_local_logits = local_logits.flatten() if args.n_classes == 1 else torch.argmax(local_logits, dim=-1).view(-1, 1)
                        else:
                            all_local_logits = local_preds
                    else:
                        all_local_logits = torch.cat([all_local_logits, local_logits_to_add])
                else:
                    logits_to_add = logits if args.n_classes == 1 else torch.argmax(logits).unsqueeze(0)

                if all_slide_logits is None:
                    if not args.loss_fn == 'coral':
                        all_slide_logits = logits if args.n_classes == 1 else torch.argmax(logits).unsqueeze(0)
                    else:
                        all_slide_logits = pred
                else:
                    all_slide_logits = torch.cat([all_slide_logits, logits_to_add])

            # ********** Shapes and types before loss **********
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                label = label.squeeze(-1).float()    
            elif (args.task_config.get('setting', 'multi_class') == 'continuous' and not args.loss_fn == 'cox') or args.task_config.get('setting', 'multi_class') == 'multi_class':
                label = label.squeeze(-1).float()
                logits = logits.squeeze(-1)

                if args.use_tile_classification or args.window_training: # Delete
                    if local_labels is not None:
                        local_logits = local_logits.squeeze(-1)
                        local_labels = local_labels.squeeze(-1).float()
                    if args.task_config.get('setting', 'multi_class') == 'multi_class':
                        local_labels = torch.round(local_labels)

                if args.student_training and args.model_ckpt != '':  # Delete
                    if teacher_labels is not None:
                        teacher_labels = teacher_labels.squeeze(-1).float()
                    if teacher_logits is not None:
                        teacher_logits = teacher_logits.squeeze(-1).float()
                
                # ********** Normalizations and label gathering **********

                # if not args.window_training:
                #     label = (label - model.mean.item()) / model.std.item()
                if not (args.cat_x and args.compress_features) and not (args.cl_HE_feat or args.pred_mean_std or args.train_on_y or args.pred_from_tumor):
                    if all_slide_labels is None:
                        all_slide_labels = label
                    else:
                        all_slide_labels = torch.cat([all_slide_labels, label])

                    if args.use_tile_classification or args.window_training:
                        if local_labels is not None:
                            # local_labels = (local_labels - local_labels_mean) / local_labels_std
                            if all_local_labels is None:
                                all_local_labels = local_labels.flatten()
                            else:
                                all_local_labels = torch.cat([all_local_labels, local_labels.flatten()])

                            if batch_local_labels is None:
                                batch_local_labels = local_labels.flatten()
                            else:
                                batch_local_labels = torch.cat([batch_local_labels, local_labels.flatten()])
                            if batch_local_logits is None:
                                batch_local_logits = local_logits.flatten()
                            else:
                                batch_local_logits = torch.cat([batch_local_logits, local_logits.flatten()])

                if args.student_training and args.model_ckpt != '':  # Delete
                    if teacher_labels is not None:
                        teacher_labels = (teacher_labels - model.mean.item()) / model.std.item()

            else:
                label = label.squeeze(-1) # .long()     
            

            # ********** loss computation **********
            if not args.loss_fn == 'cox':
                print(f"batch_idx = {batch_idx}, slide_id = {batch['slide_id']}") # Delete
                if not args.survival:
                    if args.use_tile_classification or args.window_training:
                        kl_loss, ihc_similarity_loss, ihc_local_loss, ihc_slide_loss, local_logits_sim_loss, local_logits_loss = None, None, None, None, None, None
                        if local_labels.shape != local_logits.shape:
                            if args.n_classes == 1:
                                if not args.loss_fn == 'coral':
                                    local_labels = local_labels.view(local_logits.shape)
                                else:
                                    local_logits = local_logits.squeeze(0)
                            else: # CE_loss
                                local_logits = local_logits.squeeze(0)
                                local_labels = local_labels.long()
                                label = label.long()
                        if args.loss_fn == "weighted_mse":
                            weights = get_batch_weights(local_labels, local_labels_hist, local_labels_bins)
                            local_loss = loss_fn(local_logits, local_labels, weights)
                            slide_loss_fn = torch.nn.MSELoss()
                            slide_loss = slide_loss_fn(logits, label)
                        else:
                            # print(f"logits.shape = {logits.shape}, label.shape = {label.shape}, local_logits.shape = {local_logits.shape}, local_labels.shape = {local_labels.shape}")
                            local_loss = loss_fn(local_logits, local_labels)
                            slide_loss = loss_fn(logits, label)
                            ihc_local_logits = ihc_local_logits.view(local_logits.shape)
                            ihc_logits = ihc_logits.view(logits.shape)

                            if args.matching_tiles_training:
                                # ihc_similarity_loss = 1 - cos_sim(regressed_ihc, matching_ihc_images).mean()
                                # half_num_tiles = regressed_he.shape[1] // 2
                                ihc_similarity_loss = 1 - cos_sim(regressed_he, regressed_ihc).mean()
                                ihc_local_loss = loss_fn(ihc_local_logits, local_labels)
                                ihc_slide_loss = loss_fn(ihc_logits, label)
                                local_logits_sim_loss = 1 - cos_sim(local_logits ** 5, ihc_local_logits ** 5).mean()
                                # local_logits_loss = loss_fn(local_logits, ihc_local_logits)
                                # logits_loss = loss_fn(logits, ihc_logits)

                            # mid_loss_fn = MidPenalizedLoss(loss_fn)
                            # mid_loss = mid_loss_fn(local_logits, local_labels)

                            # Per slide - tile score dist loss (KL)
                            # kl = DifferentiableHistogramKL()
                            # kl_loss = kl(local_logits, local_labels)

                            # # Calculate the KL divergence between the local_logits and the normal tensor
                            # normal_kl_loss = kl(local_logits, normal_tensor)
                        #TODO: try with only local loss
                        # loss = local_loss # Best (eval mae - 0.88)
                        # loss = kl_loss
                        
                        # TODO: try common space for tile_features
                        # loss = local_loss + slide_loss + ihc_similarity_loss if ihc_similarity_loss is not None else local_loss + slide_loss
                        # loss = ihc_local_loss + ihc_slide_loss # + ihc_similarity_loss # + 7 * local_logits_sim_loss + slide_loss
                        if epoch >= 5:
                            loss = ihc_similarity_loss
                        else:
                            loss = ihc_local_loss + ihc_slide_loss
                        
                        # loss = 0.1 * slide_loss + 0.9 * local_loss
                        # loss = loss_fn(logits, torch.tensor(100).float().to(args.device))
                        # loss = local_loss + var_penalty
                        
                        if batch['slide_id'][0] == "21-5467_2_1_b":
                            torch.set_printoptions(threshold=torch.inf) # Delete
                        else:
                            torch.set_printoptions(threshold=1000)

                        print(f"loss = {loss}, slide_loss = {slide_loss}, local_loss = {local_loss}, ihc_local_loss = {ihc_local_loss}, ihc_slide_loss = {ihc_slide_loss}") # Delete kl_loss = {kl_loss}, 
                        print(f"local_logits_sim_loss = {local_logits_sim_loss}, ihc_similarity_loss = {ihc_similarity_loss}")
                        print(f"regressed_he[0][0] = {regressed_he[0][0]}, regressed_ihc[0][0] = {regressed_ihc[0][0]}") # Delete
                        print(f"logits = {logits}, label = {label}, local_logits = {local_logits}, shape = {local_logits.shape}, local_labels = {local_labels}, shape = {local_labels.shape}")
                        print(f"ihc_local_logits = {ihc_local_logits}, ihc_logits = {ihc_logits}, ")

                    elif args.student_training:
                        label_loss = loss_fn(logits, label)
                        teacher_tile_label_loss = loss_fn(logits, teacher_labels) if teacher_labels is not None else None
                        teacher_slide_label_loss = loss_fn(logits, teacher_logits) if teacher_logits is not None else None
                        hidden_rep_loss = loss_fn(regressor_output, teacher_feature_map)
                        # contr_loss = Contrastive_Loss()
                        # contrastive_loss = contr_loss(embedding_memory, regressor_output, label)
                        # contrastive_loss = contr_loss(embedding_memory, regressor_output, teacher_feature_map, label)
                        clip_loss = CLIPLoss(embed_dim=regressor_output.shape[-1])
                        cl_loss = clip_loss(regressor_output, teacher_feature_map, label, HE_embedding_memory, IHC_embedding_memory)
                        
                        if args.model_ckpt != '': # KD training
                            if teacher_tile_label_loss is not None:
                                loss = 0.5 * label_loss + 0.5 * teacher_tile_label_loss
                            else:
                                loss = label_loss
                        else: # HT
                            # loss = hidden_rep_loss
                            # loss = 0.5 * hidden_rep_loss + 0.5 * label_loss
                            # loss = 0.5 * hidden_rep_loss + 0.5 * contrastive_loss if epoch > 1 else hidden_rep_loss
                            # loss = contrastive_loss
                            loss = 0.5 * cl_loss + 0.5 * teacher_slide_label_loss

                        print(f"loss = {loss}, label_loss = {label_loss}, hidden_rep_loss = {hidden_rep_loss}, teacher_tile_label_loss={teacher_tile_label_loss}, \
                              teacher_slide_label_loss={teacher_slide_label_loss}, clip_loss={cl_loss}") # Delete

                    elif args.matching_tiles_training and not args.tile_classification:
                        label_loss = loss_fn(logits, label)
                        ihc_images = ihc_images.squeeze(0)
                        teacher_matching_images = ihc_images[matching_tiles]
                        ihc_regressor_features = 0.5 * args.ihc_regressor(teacher_matching_images) + 0.5 * teacher_matching_images
                        # recon_teacher_matching_images = args.ihc_reconstructor(ihc_regressor_features)
                        valid_matching_tiles = valid_matching_tiles.flatten()
                        print(f"regressor_output.shape = {regressor_output.shape}, valid_matching_tiles.shape = {valid_matching_tiles.shape}") # Delete
                        regressor_images = regressor_output[valid_matching_tiles]
                        print(f"teacher_matching_images.shape = {teacher_matching_images.shape}, regressor_images.shape = {regressor_images.shape}") # Delete
                        # features_loss = loss_fn(regressor_images, teacher_matching_images) if teacher_matching_images is not None else None
                        features_loss = loss_fn(regressor_images, ihc_regressor_features) if ihc_regressor_features is not None else None
                        # self_similarity_loss = loss_fn(regressor_output, images)
                        # ihc_ss_loss = loss_fn(ihc_regressor_features, teacher_matching_images)
                        # recon_loss = loss_fn(recon_teacher_matching_images, teacher_matching_images)
                        # loss = 0.1 * label_loss + 0.3 * features_loss + 0.3 * self_similarity_loss + 0.3 * ihc_ss_loss
                        loss = 0.5 * label_loss + 0.5 * features_loss
                        try:
                            # print(f"teacher_matching_images[0] = {teacher_matching_images[0]}, ihc_regressor_features[0] = {ihc_regressor_features[0]}, regressor_images[0] = {regressor_images[0]}, batch_idx = {batch_idx}") # Delete
                            print(f"teacher_matching_images[0] = {teacher_matching_images[0]}, regressor_images[0] = {regressor_images[0]}, batch_idx = {batch_idx}") # Delete
                        except BaseException as e:
                            pass
                        # print(f"loss = {loss}, label_loss = {label_loss}, features_loss = {features_loss}, self_similarity_loss = {self_similarity_loss}, ihc_ss_loss = {ihc_ss_loss}") # Delete
                        print(f"loss = {loss}, label_loss = {label_loss}, features_loss = {features_loss}") # Delete
                        if loss.item() > 0.5:
                            print(f"logits = {logits}, label = {label}")
                        # loss = label_loss
                    elif args.cat_x and args.compress_features and args.x_dim == 0: # regress x
                        neg_cos_loss = None

                        # if label == 0:
                        #     last_0_embed = tile_x.mean(dim=0)
                        #     if last_3_embed is not None:
                        #         neg_cos_loss = cos_sim(regressed_x, last_3_embed).mean()
                        # if label == 3:
                        #     last_3_embed = tile_x.mean(dim=0)
                        #     if last_0_embed is not None:
                        #         neg_cos_loss = cos_sim(regressed_x, last_0_embed).mean()

                        tile_x_cos = cos_sim(tile_x[0], tile_x)
                        reg_x_cos = cos_sim(regressed_x[0], regressed_x)
                        # cos_loss = 1 - cos_sim(regressed_x, tile_x).mean() # tile_x
                        # mse_loss = loss_fn(regressed_x, tile_x) # tile_x
                        mae_loss = mae(regressed_x, tile_x)
                        mae_cos_loss = mae(tile_x_cos, reg_x_cos)

                        # loss = cos_loss + neg_cos_loss if neg_cos_loss is not None else cos_loss
                        loss = mae_cos_loss + mae_loss # mse_loss
                        print(f"loss = {loss}, mae_cos_loss = {mae_cos_loss}, mae_loss = {mae_loss}, neg_cos_loss = {neg_cos_loss}") # Delete
                        print(f'regressed_x = {regressed_x}')
                        print(f'tile_x = {tile_x}')
                    elif args.compress_features and args.cl_HE_feat:
                        clip_loss = CLIPLoss()
                        loss = clip_loss(HE_tile_features=regressed_x, HE_tile_labels=local_labels, HE_memory=HE_embedding_memory)
                        print(f"clip loss = {loss}")
                        if loss == 0:
                            print("clip loss is 0, skipping step")
                            continue
                    elif args.compress_features and args.pred_mean_std:   
                        mean_gt, std_gt = compute_style_stats(matching_ihc_images)  # [B]
                        print(f"mean_gt.shape = {mean_gt.shape}, std_gt.shape = {std_gt.shape}")
                        mean_loss = mae(mean_pred, mean_gt)
                        std_loss = mae(std_pred, std_gt)
                        logits_loss = loss_fn(logits, label)
                        loss = mean_loss + std_loss # logits_loss + 
                        print(f"mean_gt = {mean_gt}, mean_pred = {mean_pred}, std_gt = {std_gt}, std_pred = {std_pred}")
                        print(f"loss = {loss}, logits_loss = {logits_loss}, mean_loss = {mean_loss}, std_loss = {std_loss}")
                    elif args.compress_features and not (args.cat_x or args.cat_reg_x) : # training a model to create x
                        print(f'comp_images.shape = {comp_images.shape}')
                        comp_cos_sim = cos_sim(comp_images[0][0], comp_images[0])
                        mse_loss = loss_fn(logits, label)
                        # cos_loss = 1 - cos_sim(orig_cos_sim, comp_cos_sim).mean()
                        mae_cos_loss = mae(orig_cos_sim, comp_cos_sim)
                        loss = mse_loss + mae_cos_loss # + cos_loss
                        print(f"comp_cos_sim = {comp_cos_sim}, orig_cos_sim = {orig_cos_sim}, mse_loss = {mse_loss}, mae_cos_loss = {mae_cos_loss}")
                    elif args.train_on_y:
                        if args.task_config.get('setting', 'multi_class') == 'binary':
                            tile_y = (tile_y >= 2).long().squeeze(-1)
                            label = label.long()
                            # print(f"tile_y = {tile_y}, shape = {tile_y.shape}")
                        # tile_y_scaled = tile_y / 3
                        tile_loss = loss_fn(tile_logits, tile_y) # ** 2
                        # tile_loss = mae(tile_logits, tile_y) 
                        # margin = margin_loss(tile_logits, tile_y, margin_scale=0.8)
                        # tile_loss = loss_fn(tile_logits, tile_y_scaled)
                        # slide_loss = loss_fn(logits.unsqueeze(0), label)
                        loss = tile_loss # + slide_loss #+ 5 * margin # 
                        print(f"tile_loss = {tile_loss}, loss = {loss}") # slide_loss = {slide_loss},
                        print(f"tile_y = {tile_y}, tile_logits = {tile_logits}, logits = {logits}, label = {label}")
                    elif args.predict_cancer and tumor_indices is not None:
                        loss = loss_fn(logits, local_tumor_labels)
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
                            # else:
                            #    curr_logits = cum_logits
                            #    curr_labels = cum_labels
                            #    curr_censoreship = cum_censoreship
                            # loss = loss_fn(curr_logits, curr_labels, curr_censoreship) / (args.gc * batch_amplification)
                            loss = loss_fn(cum_logits, cum_labels, cum_censoreship) / args.gc
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
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
                if not args.loss_fn == 'cox' and not (args.student_training and not loss.requires_grad):
                    # if (batch_idx + 1) % (args.gc * 3) == 0:
                    #     kl = DifferentiableHistogramKL()
                    #     # random_binary = torch.randint(0, 2, size=batch_local_logits.shape, device=batch_local_logits.device)
                    #     # # Scale 0 -> 0 and 1 -> 3
                    #     # batch_local_labels = random_binary * 3
                    #     kl_loss = kl(batch_local_logits, batch_local_labels)
                    #     loss = loss_fn(batch_local_logits, batch_local_labels)
                    #     while(kl_loss < loss * 2):
                    #         kl_loss = kl_loss * 2
                    #     print(f"Batch loss = {loss}, kl_loss = {kl_loss}") # Delete
                    #     loss = loss + kl_loss # * args.gc
                    #     # loss = kl_loss
                    #     fp16_scaler.scale(loss).backward()
                    #     print("Classifier weight grad norm:", model.classifier[0].weight.grad.norm())
                    #     batch_local_logits, batch_local_labels = None, None
                    
                    fp16_scaler.scale(loss).backward()
                    print("Classifier weight grad norm:", model.classifier[0].weight.grad) # Delete .norm()
                    
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
                            # else:
                            #    curr_logits = cum_logits
                            #    curr_labels = cum_labels
                            #    curr_censoreship = cum_censoreship
                            # loss = loss_fn(curr_logits, curr_labels, curr_censoreship) / (args.gc * batch_amplification)
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
                        print(f"performed step on {curr_len} tiles")
                        curr_len = 0
                        print(f"Epoch {epoch}: Classifier Weights: {model.classifier[0].weight}")
                        try:
                            print(f"Epoch {epoch}: Tile_Classifier Weights: {model.tile_classifier[0].weight}")
                        except BaseException as e:
                            pass
                        if args.cl_HE_feat:
                            print(f"Epoch {epoch}: Compressor Weights: {model.compressor.W_v.weight}")

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

    tile_or_window_str = 'Window' if args.window_training else "Tile"
    # Delete
    
    if args.save_plots and not (args.cat_x and args.compress_features) and not (args.cl_HE_feat or args.pred_mean_std or  args.train_on_y or args.pred_from_tumor):
        if epoch == 0:
            if args.use_tile_classification or args.window_training:
                save_plot(x=None, y=all_local_labels, xlabel="Values", ylabel="Frequency", title=f"{tile_or_window_str} labels disstribution epoch={epoch}",
                    hist=True, output_dir=args.output_dir)
            save_plot(x=None, y=all_slide_labels, xlabel="Values", ylabel="Frequency", title=f"Slide labels disstribution epoch={epoch}", hist=True, output_dir=args.output_dir)

        if args.use_tile_classification or args.window_training:
            save_plot(x=None, y=all_local_logits, xlabel="Values", ylabel="Frequency", title=f"{tile_or_window_str} logits disstribution epoch={epoch}", 
                    hist=True, output_dir=args.output_dir)
        save_plot(x=None, y=all_slide_logits, xlabel="Values", ylabel="Frequency", title=f"Slide logits disstribution epoch={epoch}", hist=True, output_dir=args.output_dir)
    return records


def evaluate(loader, model, fp16_scaler, loss_fn, epoch, args, save_embed=False):
    if args.student_training and 'teacher_model' in args:
        args.teacher_model.eval()

    model.eval()
    
    mae = torch.nn.L1Loss()

    # set the evaluation records
    records = get_records_array(len(loader), args.n_classes, args)
    if args.predict_cancer or args.pred_y_baseline or args.pred_y:
        records['prob'] = []
        records['label'] = []
    embeds = np.zeros((len(loader), 768))

    if args.cl_HE_feat:
        return records

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
            if args.use_tile_classification and batch['local_labels'] is None: 
                continue
            
            slide_id = batch['slide_id'][-1] # if args.window_training else batch['slide_id']
            # if not slide_id.startswith('21-6140'):
            #     continue
            if args.window_training:
                sliding_window_ds = SlidingWindowDataset(data_df=args.dataset, root_path=args.root_path, task_config=args.task_config, slide_key=args.slide_key, label=args.label, \
                                    dataset_name=args.test_dataset, folds=args.test_fold,
                                    use_clinical_features=args.clinical_features, \
                                    test_on_all=args.test_on_all, get_single_slide=slide_id,
                                    window_size=args.window_size, stride=args.window_size)
                window_loader = get_test_loader(sliding_window_ds, for_heatmap=True, **vars(args))
                slide_windows = [{'imgs': wndw['imgs'].to(args.device, non_blocking=True), 
                                'coords': wndw['coords'].to(args.device, non_blocking=True)} for wndw in window_loader]

            # ********** load the batch and transform this batch **********
            num_tiles = batch['imgs'].shape[1]
            if args.loss_fn == 'cox' and num_tiles > max_num_samples:
                indices = torch.randint(low=0, high=num_tiles, size=(max_num_samples,))
                slice_tiles(indices=indices, batch=batch)

            images, img_coords, label, local_labels, ihc_images, ihc_coords = batch['imgs'], batch['coords'], batch['labels'], batch['local_labels'], batch['ihc_imgs'], batch['ihc_coords']
            teacher_labels, matching_tiles, tile_y, tile_x, tile_reg_x = batch['teacher_labels'], batch['matching_tiles'], batch['tile_y'], batch['tile_x'], batch['tile_reg_x']
            tumor_indices, non_tumor_indices = batch['tumor_indices'], batch['non_tumor_indices']
            local_labels = torch.clamp(local_labels.float(), min=0, max=3) if local_labels is not None else None
            synth_ihc_images = batch['synth_ihc_imgs'] # Delete           
            # ********** data preperation and move to device **********
            print(f"slide = {slide_id}, batch_idx = {batch_idx}, label = {label}")
            # if slide_id != "21-8662_1_1_e":
                # continue
            images = images.to(args.device, non_blocking=True)
            img_coords = img_coords.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True).long()
            if ihc_images is not None:
                ihc_images = ihc_images.to(args.device, non_blocking=True)
                ihc_coords = ihc_coords.to(args.device, non_blocking=True)
            if local_labels is not None:
                local_labels = local_labels.to(args.device, non_blocking=True)
                valid_label_indices = ~torch.isnan(local_labels)
                local_labels = local_labels[valid_label_indices]
                if args.use_tile_classification:
                    images = images[valid_label_indices]
                    img_coords = img_coords[valid_label_indices]
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
            if args.cat_ihc or args.cat_y or args.cat_x or args.pred_mean_std or args.pred_from_tumor: # Delete:
                if matching_tiles is not None:
                    matching_tiles = matching_tiles.to(args.device, non_blocking=True)
                    valid_matching_tiles = (~torch.isnan(matching_tiles)) & (matching_tiles < ihc_images.shape[1])
                    if len(torch.nonzero(valid_matching_tiles)) == 0:
                        print(f"{batch['slide_id']} has no valid matching tiles, skipping")
                        continue
                    images = images[valid_matching_tiles]
                    img_coords = img_coords[valid_matching_tiles]
                    matching_tiles = matching_tiles[valid_matching_tiles].int()
                    if args.cat_y and not args.create_y:
                        if tile_y is not None:
                            tile_y = tile_y.to(args.device, non_blocking=True).squeeze(0)
                        else:
                            print(f"{batch['slide_id']} tile_y is None, skipping")
                            continue
                    if args.cat_x and not args.create_pred_y:
                        if tile_x is not None:
                            tile_x = tile_x.to(args.device, non_blocking=True).squeeze(0).squeeze(0)
                            tile_x = tile_x[matching_tiles]
                        else:
                            print(f"{batch['slide_id']} tile_x is None, skipping")
                            continue
                    elif args.pred_mean_std:
                        ihc_images = ihc_images.squeeze(0)
                        matching_ihc_images = ihc_images[matching_tiles]
                        if matching_ihc_images.shape[0] == 0:
                            print(f"slide {batch['slide_id']} has no matching ihc images, skipping")
                            continue
                    if args.only_tumor_tiles:
                        if tumor_indices is not None and tumor_indices.shape[-1] > 0:
                            print(f'tile_y.shape = {tile_y.shape}, tumor_indices.shape = {tumor_indices.shape}, images.shape = {images.shape}, img_coords.shape = {img_coords.shape}')
                            print(f'tumor_indices = {tumor_indices}')
                            tumor_indices = tumor_indices.squeeze(0)
                            images = images[tumor_indices]
                            img_coords = img_coords[tumor_indices]
                            if args.cat_y:
                                tile_y = tile_y[tumor_indices]
                        else:
                            print(f"{batch['slide_id']} has no tumor indices, skipping")
                            continue
                else:
                    print(f"{batch['slide_id']} matching_tiles is None, skipping")
                    continue
            if args.cat_reg_x:
                if tile_reg_x is not None:
                    tile_reg_x = tile_reg_x.to(args.device, non_blocking=True).squeeze(0)
                else:
                    print(f"{batch['slide_id']} tile_reg_x is None, skipping")
                    continue
                
            if args.survival:
                censoreship = batch['censoreship']


            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                # ********** get the logits **********
                if args.use_tile_classification or args.window_training:
                    if valid_label_indices.sum() == 0:
                        print(f"slide {slide_id} has no valid labels, skipping")
                        np.delete(records['prob'], batch_idx, axis=0)
                        np.delete(records['label'], batch_idx, axis=0)
                        continue
                    if args.window_training and slide_windows is not None:
                        valid_label_indices = torch.nonzero(valid_label_indices.flatten(), as_tuple=True)[0]
                        print(f"valid_label_indices = {valid_label_indices}")
                        slide_windows = [slide_windows[i] for i in valid_label_indices]

                if save_embed:
                    if args.window_training and slide_windows is not None:
                        logits, local_logits, embed = model(slide_windows, return_embed=True)
                        print(f"local_logits = {local_logits}, logits = {logits}, local_labels = {local_labels}, label = {label}")

                    elif args.use_tile_classification:
                        if args.matching_tiles_training:
                            logits, local_logits, embed = model(images, img_coords, regress=True, return_embed=True)
                        else:
                            logits, local_logits, embed = model(images, img_coords, return_embed=True)

                    elif args.student_training or (args.matching_tiles_training and not args.use_tile_classification):
                        # logits, _, embed = model(images, img_coords, return_embed=True)
                        logits, embed = model(images, img_coords)
                    elif args.cat_random:
                        random_features = torch.randn_like(images)
                        images = torch.cat([images, random_features], dim=-1)
                        logits, embed = model(images, img_coords, return_embed=True)
                    elif args.cat_ihc or args.cat_y: # Delete
                        if args.train_on_y:
                            images, img_coords = images.unsqueeze(0).transpose(0, 1), img_coords.unsqueeze(0).transpose(0, 1)
                            print(f'images.shape = {images.shape}')
                            if images.shape[0] == 0:
                                print(f"slide {batch['slide_id']} has no matching ihc images, skipping")
                                continue
                            if not args.pred_y_baseline:
                                logits, tile_logits = model(images, img_coords)
                            else:
                                tile_logits, embed = model(images, img_coords, return_embed=True)
                                embed = embed[0]  # Get the first tile's embedding
                                logits = tile_logits[0]
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
                                
                                if args.create_y:
                                    print(f'y = {logits}')
                                    save_y_dir = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", f"tile_y_{args.mpp}", slide_id)
                                    if not os.path.exists(save_y_dir):
                                        os.makedirs(save_y_dir, exist_ok=True)
                                    np.save(os.path.join(save_y_dir, f"tile_y_val{args.val_fold[-1]}.npy"), logits.cpu().detach().numpy())
                                    print(f'Saved tile_y for slide {slide_id} to {os.path.join(save_y_dir, f"tile_y_val{args.val_fold[-1]}.npy")}')
                                    continue


                            elif not args.cat_y:
                                images = torch.cat([images, matching_ihc_images], dim=-1)
                                logits, embed = model(images, img_coords, return_embed=True)
                            else:
                                if not args.trim_images:
                                    images = torch.cat([images, tile_y], dim=-1)
                                    logits, embed = model(images, img_coords, return_embed=True)
                                elif "hf" in args.pretrained:
                                    # use pretrained slide encoder
                                    logits, embed = model(images, img_coords, y=tile_y, trim_images=args.trim_images, return_embed=True)
                    elif args.cat_x: # Delete
                        if args.create_pred_y:
                            images, img_coords = images.unsqueeze(0).transpose(0, 1), img_coords.unsqueeze(0).transpose(0, 1)
                            print(f'images.shape = {images.shape}')
                            if images.shape[0] == 0:
                                print(f"slide {batch['slide_id']} has no matching ihc images, skipping")
                                continue

                            with torch.no_grad():
                                pred_y = model(images, img_coords)
                                print(f'pred_y = {pred_y}')
                            save_pred_y_dir = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_HE", f"tile_pred_y_{args.mpp}", slide_id)
                            if not os.path.exists(save_pred_y_dir):
                                os.makedirs(save_pred_y_dir, exist_ok=True)
                            np.save(os.path.join(save_pred_y_dir, "tile_pred_y.npy"), pred_y.cpu().detach().numpy())
                            print(f'Saved tile_pred_y for slide {slide_id} to {os.path.join(save_pred_y_dir, "tile_pred_y.npy")}')
                            continue

                        if not args.compress_features:
                            images = torch.cat([images, tile_x], dim=-1)
                            logits, embed = model(images, img_coords, return_embed=True)
                        else:
                            if "hf" in args.pretrained: # use pretrained slide encoder
                                if args.trim_images: 
                                    logits, embed = model(images, img_coords, x=tile_x, trim_images=args.trim_images, return_embed=True)
                                elif args.x_as_sb:
                                    logits, embed = model(images, img_coords, x=tile_x, x_as_sb=args.x_as_sb, return_embed=True)
                                else: # compress images
                                    logits, embed = model(images, img_coords, x=tile_x, return_embed=True) 
                            else: # regress x
                                logits, embed = model(images, img_coords, return_embed=True)
                    elif args.cat_reg_x: # Delete
                        if args.x_as_sb:
                            logits, embed = model(images, img_coords, x=tile_reg_x, x_as_sb=args.x_as_sb, return_embed=True)
                        else:
                            images = torch.cat([images, tile_reg_x], dim=-1)
                            logits, embed = model(images, img_coords, return_embed=True)
                    elif args.synth_ihc_train: # Delete
                        images = torch.cat([images, synth_ihc_images], dim=-1)
                        logits, embed = model(images, img_coords, return_embed=True)
                    elif args.compress_features and args.save_x:
                        logits, x = model(images, img_coords, return_images=True)
                        print(f"x = {x}, shape = {x.shape}")
                        save_x_dir = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", 
                        f"tile_x_cos_{args.reduction}_{args.comp_power}_{args.mpp}", slide_id)
                        if not os.path.exists(save_x_dir):
                            os.makedirs(save_x_dir, exist_ok=True)
                        np.save(os.path.join(save_x_dir, "tile_x.npy"), x.cpu().detach().numpy())
                        continue
                    elif args.compress_features and args.save_reg_x:
                        logits, regressed_x = model(images, img_coords, return_images=True)
                        save_reg_x_dir = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", f"tile_cos_reg_x_{args.mpp}", slide_id)
                        if not os.path.exists(save_reg_x_dir):
                            os.makedirs(save_reg_x_dir, exist_ok=True)
                        np.save(os.path.join(save_reg_x_dir, "tile_reg_x.npy"), regressed_x.cpu().detach().numpy())
                        
                        if tile_x is not None:
                            cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
                            
                            tile_x = tile_x.to(args.device, non_blocking=True).squeeze(0).squeeze(0)
                            matching_tiles = matching_tiles.to(args.device, non_blocking=True)
                            valid_matching_tiles = (~torch.isnan(matching_tiles)) & (matching_tiles < ihc_images.shape[1])
                            if len(torch.nonzero(valid_matching_tiles)) == 0:
                                print(f"{batch['slide_id']} has no valid matching tiles, skipping")
                                continue
                            matching_tiles = matching_tiles[valid_matching_tiles].int()
                            regressed_x = regressed_x[valid_matching_tiles]
                            tile_x = tile_x[matching_tiles]

                            cos_loss = 1 - cos_sim(regressed_x, tile_x).mean() # tile_x
                            mse_loss = loss_fn(regressed_x, tile_x) # tile_x

                            print(f"cos_loss = {cos_loss}, mse_loss = {mse_loss}")
                            print(f"regressed_x = {regressed_x}, shape = {regressed_x.shape}")
                            print(f"tile_x = {tile_x}, shape = {tile_x.shape}")
                        continue
                    elif args.predict_cancer:
                        images, img_coords = images.transpose(0, 1), img_coords.transpose(0, 1)
                        local_tumor_labels = local_tumor_labels.squeeze(0).long()
                        print(f'images.shape = {images.shape}, local_tumor_labels.shape = {local_tumor_labels.shape}')
                        logits, embed = model(images, img_coords, return_embed=True)
                        embed = embed[0]  # Get the first tile's embedding
                    elif model.isinstanceof(TileClassificationHead):
                        logits, tile_logits, embed = model(images, img_coords, return_embed=True)
                        embeds[batch_idx] = embed.cpu().numpy()
                    else:
                        logits, embed = model(images, img_coords, return_embed=True)
                        embeds[batch_idx] = embed.cpu().numpy()
                else: # no save_embed
                    if args.window_training and slide_windows is not None:
                        logits, local_logits = model(slide_windows)

                    elif args.use_tile_classification:
                        if args.matching_tiles_training:
                            logits, local_logits = model(images, img_coords, regress_HE=True)
                        else:
                            logits, local_logits = model(images, img_coords)

                    elif args.student_training or (args.matching_tiles_training and not args.use_tile_classification):
                        logits, _ = model(images, img_coords)
                    elif args.cat_random:
                        random_features = torch.randn_like(images)
                        images = torch.cat([images, random_features], dim=-1)
                        logits = model(images, img_coords)
                    elif args.cat_ihc or args.cat_y: # Delete
                        if args.train_on_y:
                            images, img_coords = images.unsqueeze(0).transpose(0, 1), img_coords.unsqueeze(0).transpose(0, 1)
                            print(f'images.shape = {images.shape}')
                            if images.shape[0] == 0:
                                print(f"slide {batch['slide_id']} has no matching ihc images, skipping")
                                continue

                            if not args.pred_y_baseline:
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
                                
                                if args.create_y:
                                    print(f'y = {logits}')
                                    save_y_dir = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", f"tile_y_{args.mpp}_val{args.val_fold[-1]}", slide_id)
                                    if not os.path.exists(save_y_dir):
                                        os.makedirs(save_y_dir, exist_ok=True)
                                    np.save(os.path.join(save_y_dir, "tile_y.npy"), logits.cpu().detach().numpy())
                                    print(f'Saved tile_y for slide {slide_id} to {os.path.join(save_y_dir, "tile_y.npy")}')
                                    continue
                            
                            elif not args.cat_y:
                                images = torch.cat([images, matching_ihc_images], dim=-1)
                                logits = model(images, img_coords)
                            else:
                                if not args.trim_images:
                                    images = torch.cat([images, tile_y], dim=-1)
                                    logits = model(images, img_coords)
                                elif "hf" in args.pretrained:
                                    # use pretrained slide encoder
                                    logits = model(images, img_coords, y=tile_y, trim_images=args.trim_images)
                    elif args.cat_x:
                        if not args.compress_features:
                            images = torch.cat([images, tile_x], dim=-1)
                            logits = model(images, img_coords)
                        else:
                            if "hf" in args.pretrained: # use pretrained slide encoder
                                if args.trim_images: 
                                    logits = model(images, img_coords, x=tile_x, trim_images=args.trim_images)
                                elif args.x_as_sb:
                                    logits = model(images, img_coords, x=tile_x, x_as_sb=args.x_as_sb)
                                else: # compress images
                                    logits = model(images, img_coords, x=tile_x)             
                            else: # regress x
                                logits = model(images, img_coords)
                    elif args.cat_reg_x:
                        if args.x_as_sb:
                            logits = model(images, img_coords, x=tile_reg_x, x_as_sb=args.x_as_sb)
                        else:
                            tile_reg_x = tile_reg_x.view(images.shape[0], images.shape[1], -1)
                            images = torch.cat([images, tile_reg_x], dim=-1)
                    elif args.synth_ihc_train: # Delete
                        images = torch.cat([images, synth_ihc_images], dim=-1)
                        logits = model(images, img_coords)
                    elif args.compress_features and args.pred_mean_std:
                        logits, mean_pred, std_pred = model(images, img_coords, pred_mean_std=True)
                        mean_gt, std_gt = compute_style_stats(matching_ihc_images)  # [B]
                        mean_loss = mae(mean_pred, mean_gt)
                        std_loss = mae(std_pred, std_gt)
                        logits_loss = loss_fn(logits, label)
                        loss = logits_loss + mean_loss + std_loss
                        print(f"mean_gt = {mean_gt}, mean_pred = {mean_pred}, std_gt = {std_gt}, std_pred = {std_pred}")
                        print(f"loss = {loss}, logits_loss = {logits_loss}, mean_loss = {mean_loss}, std_loss = {std_loss}")
                    elif args.predict_cancer:
                        images, img_coords = images.transpose(0, 1), img_coords.transpose(0, 1)
                        local_tumor_labels = local_tumor_labels.squeeze(0).long()
                        print(f'images.shape = {images.shape}, local_tumor_labels.shape = {local_tumor_labels.shape}')
                        logits = model(images, img_coords)
                    else:
                        logits = model(images, img_coords)

                if args.loss_fn == 'coral':
                    print(f"local_logits = {local_logits}, logits = {logits}")
                    prob = torch.sigmoid(logits)
                    pred = torch.sum(prob > 0.5, dim=-1)  # predicted class ∈ {0, 1, 2, 3}
                    local_prob = torch.sigmoid(local_logits)
                    local_preds = torch.sum(local_prob > 0.5, dim=-1)  # predicted class ∈ {0, 1, 2, 3}
                    local_preds = local_preds.squeeze(0)                        
                    print(f"local_prob = {local_prob}, local_preds = {local_preds}, prob = {prob}, pred = {pred}")
                        
                if logits is None:
                    continue
                
                # ********** Shapes and types before loss **********
                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    label = label.squeeze(-1).float()
                elif task_setting == 'continuous' and not args.loss_fn == 'cox':
                    label = label.squeeze(-1).float()
                    logits = logits.squeeze(-1)

                else:
                    label = label.squeeze(-1).long()
                
                # ********** Normalizations and label gathering **********
                # if task_setting == 'continuous' and not args.loss_fn == 'cox':
                    # if not args.window_training:
                        # label = (label - model.mean.item()) / model.std.item()
                
                # ********** loss computation **********
                if not no_loss:
                    try:
                        if not args.loss_fn == 'cox':
                            if not args.survival:
                                print(f'logits = {logits}, label = {label}') # Delete
                                if args.train_on_y:
                                    # pred_class = logits.argmax()
                                    # loss = loss_fn(pred_class, label)
                                    if args.task_config.get('setting', 'multi_class') == 'binary':
                                        print(f"tile_y = {tile_y}")
                                        tile_y = (tile_y >= 2).long().squeeze(-1)
                                    loss = loss_fn(tile_logits, tile_y)
                                    print(f"loss = {loss}")
                                    print(f"tile_y = {tile_y}, tile_logits = {tile_logits}, logits = {logits}, label = {label}")
                                elif args.predict_cancer:
                                    loss = loss_fn(logits, local_tumor_labels)
                                elif not args.use_tile_classification and not args.window_training:
                                    loss = loss_fn(logits, label)
                                else:
                                    # if args.loss_fn == "weighted_mse":
                                    #     loss_fn = torch.nn.MSELoss()   
                                    if local_labels.shape != local_logits.shape:
                                        if args.n_classes == 1:
                                            if not args.loss_fn == 'coral':
                                                local_labels = local_labels.view(local_logits.shape)
                                            else:
                                                local_logits = local_logits.squeeze(0)
                                        else: # CE_loss
                                            local_logits = local_logits.squeeze(0)
                                            local_labels = local_labels.long()
                                            label = label.long()
                                    local_loss = loss_fn(local_logits, local_labels)
                                    print(f'local_logits = {local_logits}, local_labels = {local_labels}, local_loss = {local_loss}') # Delete
                                    # slide_loss = loss_fn(logits, label)
                                    # loss = (slide_loss + local_loss) / 2
                                    # loss = slide_loss
                                    loss = local_loss
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
                Y_prob = torch.softmax(logits, dim=-1).cpu() # .view(1, -1)
                print(f"Y_prob = {Y_prob}, shape = {Y_prob.shape}")
                if not args.predict_cancer:
                    # convert label to one-hot
                    label_ = torch.zeros_like(Y_prob).scatter_(1, label.cpu().unsqueeze(1), 1)
                    records['prob'][batch_idx] = Y_prob.numpy() # if args.n_classes == 1 else np.argmax(Y_prob.numpy())
                    records['label'][batch_idx] = label_.numpy()
                else:
                    if tumor_indices is not None:
                        print(f"local_tumor_labels.shape = {local_tumor_labels.shape}, y_prob.shape = {Y_prob.shape}")
                        # convert label to one-hot
                        label_ = torch.zeros_like(Y_prob).scatter_(1, local_tumor_labels.cpu().unsqueeze(1), 1)
                        records['prob'].append(Y_prob.numpy())
                        records['label'].append(label_.numpy())
                    else:
                        print(f"label.shape = {local_tumor_labels.shape}, y_prob.shape = {Y_prob.shape}")
                        label_ = torch.zeros_like(Y_prob).scatter_(1, label.cpu().unsqueeze(1), 1)
                        records['prob'].append(Y_prob.numpy())
                        records['label'].append(label_.numpy())

                    if args.run_inference:
                        y_prob_to_save = Y_prob[:, 1].numpy()
                        save_prob_dir = os.path.join(args.root_path.replace("gigapath_features", "cancer_probs"), slide_id)
                        if not os.path.exists(save_prob_dir):
                            os.makedirs(save_prob_dir, exist_ok=True)
                        np.save(os.path.join(save_prob_dir, f"cancer_prob_val{args.val_fold[0]}.npy"), y_prob_to_save)
                    
                print(f"label_ = {label_}, label = {label}, np.argmax(label_.numpy()) = {np.argmax(label_.numpy())}") # Delete
                # print(f'records = {records}') # Delete
            elif task_setting == 'continuous':
                if not args.loss_fn == 'cox':
                    # if not args.window_training:
                    #     records['prob'][batch_idx] = ((logits * model.std.item()) + model.mean.item()).cpu().numpy()
                    #     records['label'][batch_idx] = ((label * model.std.item()) + model.mean.item()).cpu().numpy()
                    # else:
                    if args.pred_y_baseline or args.pred_y:
                        records['prob'].append(tile_logits.cpu().numpy())
                        records['label'].append(tile_y.cpu().numpy())
                    elif args.train_on_y:
                        pred_class = logits.argmax()
                        records['prob'][batch_idx] = pred_class.cpu().numpy()
                    elif not args.loss_fn == 'coral':
                        records['prob'][batch_idx] = logits.cpu().numpy()
                    else:
                        prob = torch.sigmoid(logits)
                        pred = torch.sum(prob > 0.5, dim=-1)
                        records['prob'][batch_idx] = pred.cpu().numpy()
                        records['label'][batch_idx] = label.cpu().numpy()    
                    if not no_loss and not args.survival:
                        for key in regression_losses:
                            try:
                                if args.use_tile_classification and key in ['cox', 'trunc_mse']:
                                    continue
                                elif args.train_on_y:
                                    if key in ['spearmanr', 'pearsonr']:
                                        np_tile_y = tile_y.squeeze(-1).clone().detach().cpu().numpy()
                                        np_tile_logits = tile_logits.squeeze(-1).clone().detach().cpu().numpy()
                                        if np_tile_y.size < 2:
                                            print(f"Only one tumor tile for slide {slide_id}, cannot compute {key}")
                                            continue
                                        rho, _ = regression_losses[key](np_tile_logits, np_tile_y)
                                        records[key] += rho
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
        if args.predict_cancer or args.pred_y_baseline or args.pred_y:
            records['prob'] = np.concatenate(records['prob'], axis=0)
            records['label'] = np.concatenate(records['label'], axis=0)
    if not no_loss:
        records['loss'] = records['loss'] / len(loader)
        if task_setting == 'continuous' and not args.survival:
            for key in regression_losses:
                records[key] = records[key] / len(loader)
    else:
        records['loss'] = 0
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
                    df.to_csv(os.path.join(args.save_dir, f"tile_y_preds_{args.mpp}_val{args.val_fold[0]}.csv"), index=False)
                    print(f'Saved tile_y predictions to {os.path.join(args.save_dir, f"tile_y_preds_{args.mpp}_val{args.val_fold[0]}.csv")}')

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
