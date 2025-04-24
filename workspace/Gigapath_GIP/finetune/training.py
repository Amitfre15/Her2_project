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

from gigapath.classification_head import get_model, get_regressor
from metrics import calculate_metrics_with_task_cfg
from utils import (get_optimizer, get_loss_function, \
                   Monitor_Score, get_records_array,
                   log_writer, adjust_learning_rate,
                   get_regression_losses, LogCoshLoss, get_test_loader, Contrastive_Loss)
from slides_to_thumbs import patch_weighted_score_matrices, IHC_THUMB_WIDTH, IHC_THUMB_HEIGHT
from datasets.slide_datatset import SlidingWindowDataset


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
        args.get_single_slide = slide
        print(f'slide = {slide}.mrxs')
        test_data = dataset_class(data_df, args.root_path, args.task_config, slide_key=args.slide_key, label=args.label, \
                                  dataset_name=args.test_dataset, folds=args.test_fold,
                                  use_clinical_features=args.clinical_features, \
                                  test_on_all=args.test_on_all, get_single_slide=args.get_single_slide,
                                  window_size=args.window_size, stride=args.stride)
        test_loader = get_test_loader(test_data, for_heatmap=True, **vars(args))

        # Initialize score and weight matrices
        weighted_score_matrix = torch.zeros((IHC_THUMB_HEIGHT, IHC_THUMB_WIDTH), dtype=torch.float64).to(args.device)
        weight_matrix = torch.zeros((IHC_THUMB_HEIGHT, IHC_THUMB_WIDTH), dtype=torch.float64).to(args.device)

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
                        results = {'window_name': window_name, 'score': logits.item(),
                                   'label': test_data.label.item()}
                        score = logits
                    else:
                        if not args.survival:
                            # un-normalize
                            score = logits * model.std.item() + model.mean.item()
                            results = {'window_name': window_name, 'score': score.item(),
                                       'label': test_data.label.item()}

                    results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)

                patch_weighted_score_matrices(weighted_score_matrix=weighted_score_matrix, weight_matrix=weight_matrix,
                                              img_coords=img_coords, window_score=score.item())

        # Normalize the weighted score matrix by the weight matrix
        final_score_matrix = np.divide(
            weighted_score_matrix.cpu().numpy(),
            weight_matrix.cpu().numpy(),
            out=np.zeros_like(weighted_score_matrix.cpu().numpy()),
            where=weight_matrix.cpu().numpy() > 0  # Avoid division by zero
        )

        # Save the score matrix as a .npz file (NumPy compressed format)
        compressed_score_matrix = final_score_matrix.astype(np.float16)
        npz_output_path = os.path.join(args.save_dir, f"{slide}_score_matrix.npz")
        np.savez_compressed(npz_output_path, compressed_score_matrix)

        print(f"Saved score matrix for slide: {slide} at {npz_output_path}")

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
    test_records, embeds = evaluate(test_loader, model, fp16_scaler, loss_fn, 'test', args, save_embed=True)
    results_df = pd.DataFrame({'slide_name': [], 'score': [], 'label': []})
    # save each embedding asa seperate filewith the name being the slide_id
    for idx, slide_id in enumerate(test_loader.dataset.slide_data[args.slide_key]):
        np.save(os.path.join(args.save_dir, f"{slide_id}.npy"), embeds[idx])
        if not args.task_config.get('setting', 'multi_class') == 'continuous':
            results = {'slide_name': slide_id, 'score': test_records['prob'][idx][1],
                       'label': 0 if test_records['label'][idx][0] == 1 else 1}
        else:
            if not args.survival:
                results = {'slide_name': slide_id, 'score': test_records['prob'][idx][0],
                           'label': test_records['label'][idx][0]}
            else:
                results = {'slide_name': slide_id, 'score': test_records['prob'][idx][0],
                           'label': test_records['label'][idx][0], 'censoreship': test_records['censoreship'][idx][0]}
        results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)

    results_df.to_csv(os.path.join(args.save_dir, 'slide_scores.csv'), index=False)
    # update the writer for test
    log_dict = {'test_' + k: v for k, v in test_records.items() if
                'prob' not in k and 'label' not in k and 'censoreship' not in k}
    log_writer(log_dict, 0, args.report_to, writer)
    wandb.finish() if "wandb" in args.report_to else None
    return test_records


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
    loss_pe, lr_pe = [], [] # Delete
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

            # update the writer for train and val
            try:
                log_dict = {'train_' + k: v for k, v in train_records.items() if
                            'prob' not in k and 'label' not in k and 'censoreship' not in k}
                log_dict.update({'val_' + k: v for k, v in val_records.items() if
                                 'prob' not in k and 'label' not in k and 'censoreship' not in k})
                log_writer(log_dict, i, args.report_to, writer)
            except:
                for key in records:
                    print(key)
                    print(records[key])
                    try:
                        print(records[key].shape)
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

    save_plot(x=range(args.epochs), y=loss_pe, xlabel='Epoch', ylabel='Loss', title="Loss per epoch", output_dir=args.output_dir)
    save_plot(x=range(args.epochs), y=lr_pe, xlabel='Epoch', ylabel='Learning rate', title="Learning rate per epoch", output_dir=args.output_dir)

    return val_records, test_records


def train_one_epoch(train_loader, model, fp16_scaler, optimizer, loss_fn, epoch, args):
    if args.student_training and 'teacher_model' in args:
        args.teacher_model.eval()
        # Memory Bank to store embeddings (key: label, value: list of embeddings)
        embedding_memory = {}
        
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

    if args.use_tile_classification or args.window_training:
        all_local_labels = None # Delete
        all_local_logits = None
        batch_local_labels = None
        batch_local_logits = None
        all_slide_labels = None
        all_slide_logits = None
        slide_windows = None
        local_labels_mean = 0.6577 if args.use_tile_classification else 0.6782 # Delete
        local_labels_std = 0.7558 if args.use_tile_classification else 0.7782
        # local_labels_mean = 0
        # local_labels_std = 1
        # local_labels_hist = torch.from_numpy(args.local_labels_hist).to(args.device, non_blocking=True)
        # local_labels_hist = 1.0 / (1.0 + local_labels_hist)  # Inverse frequency
        # local_labels_hist = local_labels_hist / local_labels_hist.sum()
        # local_labels_bins = (torch.from_numpy(args.local_labels_bins).to(args.device, non_blocking=True) - local_labels_mean) / local_labels_std
        # min_local_labels_bins, max_local_labels_bins = local_labels_bins.min(), local_labels_bins.max()
        # print(f"local_labels_bins = {local_labels_bins}, local_labels_hist = {local_labels_hist}")
    
    for batch_idx, batch in enumerate(train_loader):         
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

        # load the batch and transform this batch
        images, img_coords, label, local_labels = batch['imgs'], batch['coords'], batch['labels'], batch['local_labels']
        # TODO: truncate negative local labels
        ihc_images, ihc_coords = batch['ihc_imgs'], batch['ihc_coords'] # Delete
        teacher_labels, matching_tiles = batch['teacher_labels'], batch['matching_tiles']
        print(f'teacher_labels = {teacher_labels}')
        if args.survival:
            censoreship = batch['censoreship']
        images = images.to(args.device, non_blocking=True)
        img_coords = img_coords.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True).long()
        if ihc_images is not None:
            ihc_images = ihc_images.to(args.device, non_blocking=True)
            ihc_coords = ihc_coords.to(args.device, non_blocking=True)
        if teacher_labels is not None:
            teacher_labels = teacher_labels.to(args.device, non_blocking=True)
        if local_labels is not None and (args.use_tile_classification or args.window_training): # Delete:
            local_labels = local_labels.to(args.device, non_blocking=True)
            valid_label_indices = ~torch.isnan(local_labels)
            print(f"valid_label_indices = {valid_label_indices}")
            local_labels = local_labels[valid_label_indices]
            if args.use_tile_classification:
                images = images[valid_label_indices]
                img_coords = img_coords[valid_label_indices]
        if args.matching_tiles_training: # Delete:
            if matching_tiles is not None:
                matching_tiles = matching_tiles.to(args.device, non_blocking=True)
                # valid_matching_tiles = ~torch.isnan(matching_tiles)
                valid_matching_tiles = (~torch.isnan(matching_tiles)) & (matching_tiles < ihc_images.shape[1])
                matching_tiles = matching_tiles[valid_matching_tiles].int()
                # print(f"valid_matching_tiles = {valid_matching_tiles}, matching_tiles = {matching_tiles}, matching_tiles.shape = {matching_tiles.shape}")
            else:
                print(f"{batch['slide_id']} matching_tiles is None, skipping")
                continue    
        elif (args.window_training or args.use_tile_classification) and (local_labels is None or local_labels.size == 0):
            print(f"{slide_id} local labels is None, skipping")
            continue
            

        # add the sequence length
        seq_len += images.shape[1]
        curr_len += images.shape[1]

        with torch.cuda.amp.autocast(dtype=torch.float16 if args.fp16 else torch.float32):
            # get the logits
            if args.use_tile_classification or args.window_training:
                if args.window_training and slide_windows is not None:
                    valid_label_indices = torch.nonzero(valid_label_indices.flatten(), as_tuple=True)[0]
                    print(f"valid_label_indices = {valid_label_indices}")
                    if len(valid_label_indices) == 0:
                        continue
                    slide_windows = [slide_windows[i] for i in valid_label_indices]
                    logits, local_logits = model(slide_windows)
                elif args.use_tile_classification:
                    logits, local_logits = model(images, img_coords)
                    if logits is None:
                        continue

                if all_local_logits is None:
                    all_local_logits = local_logits.flatten()
                else:
                    all_local_logits = torch.cat([all_local_logits, local_logits.flatten()])
                if batch_local_logits is None:
                    batch_local_logits = local_logits.flatten()
                else:
                    batch_local_logits = torch.cat([batch_local_logits, local_logits.flatten()])
                if all_slide_logits is None:
                    all_slide_logits = logits
                else:
                    all_slide_logits = torch.cat([all_slide_logits, logits])
            elif args.student_training:
                logits, regressor_output = model(images, img_coords)
                with torch.no_grad():
                    _, teacher_feature_map = args.teacher_model(ihc_images, ihc_coords, return_embed=True)
            elif args.matching_tiles_training:
                logits, regressor_output = model(images, img_coords)
            else:
                logits = model(images, img_coords)
                print(f"logits = {logits}, shape = {logits.shape}")

            # except RuntimeError as e:
            #        if "out of memory" in str(e):
            #            images.detach_()
            #            img_coords.detach_()
            #            del images
            #            del img_coords
            #            gc.collect()
            #            torch.cuda.empty_cache()
            #            num_tiles = batch['imgs'].shape[1]
            #            num_samples = 20000
            #            print(f"slide {batch['slide_id']} has too many tiles, number of tiles is {num_tiles}. Trying again with {num_samples} tiles.")
            #            print(f"so far accumulated {curr_len} tiles, will instead use {curr_len - num_tiles + num_samples}")
            #            raise
            #            indices = torch.randint(low=0, high=num_tiles, size=(num_samples,))
            #            batch['imgs'] = batch['imgs'][:,indices,:]
            #            batch['coords'] = batch['coords'][:,indices,:]
            #            images, img_coords = batch['imgs'], batch['coords']
            #            images = images.to(args.device, non_blocking=True)
            #            img_coords = img_coords.to(args.device, non_blocking=True)
            #            seq_len += images.shape[1]
            #            seq_len -= num_tiles
            #            curr_len = curr_len - num_tiles + num_samples
            #            logits = model(images, img_coords)
            #        else:
            #            raise
            # get the loss
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                label = label.squeeze(-1).float()
            elif args.task_config.get('setting', 'multi_class') == 'continuous' and not args.loss_fn == 'cox':
                label = label.squeeze(-1).float()
                logits = logits.squeeze(-1)

                if args.use_tile_classification or args.window_training: # Delete
                    if local_labels is not None:
                        local_logits = local_logits.squeeze(-1)
                        local_labels = local_labels.squeeze(-1).float()

                if args.student_training and args.model_ckpt != '':  # Delete
                    if teacher_labels is not None:
                        teacher_labels = teacher_labels.squeeze(-1).float()
            else:
                label = label.squeeze(-1).long()
            if args.task_config.get('setting', 'multi_class') == 'continuous' and not args.loss_fn == 'cox':
                if not args.window_training:
                    label = (label - model.mean.item()) / model.std.item()

                if args.use_tile_classification or args.window_training: # Delete
                    if all_slide_labels is None:
                        all_slide_labels = label
                    else:
                        all_slide_labels = torch.cat([all_slide_labels, label])
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

                if args.student_training and args.model_ckpt != '':  # Delete
                    if teacher_labels is not None:
                        teacher_labels = (teacher_labels - model.mean.item()) / model.std.item()

            if not args.loss_fn == 'cox':
                print(f"batch_idx = {batch_idx}, slide_id = {batch['slide_id']}") # Delete
                if not args.survival:
                    if args.use_tile_classification or args.window_training:
                        kl_loss = None
                        if local_labels.shape != local_logits.shape and args.n_classes == 1:
                            print(f"local_labels = {local_labels}, shape = {local_labels.shape}, local_logits = {local_logits}, shape = {local_logits.shape}")
                            local_labels = local_labels.view(local_logits.shape)
                        if args.loss_fn == "weighted_mse":
                            weights = get_batch_weights(local_labels, local_labels_hist, local_labels_bins)
                            local_loss = loss_fn(local_logits, local_labels, weights)
                            slide_loss_fn = torch.nn.MSELoss()
                            slide_loss = slide_loss_fn(logits, label)
                        else:
                            # print(f"logits.shape = {logits.shape}, label.shape = {label.shape}, local_logits.shape = {local_logits.shape}, local_labels.shape = {local_labels.shape}")
                            local_loss = loss_fn(local_logits, local_labels)
                            slide_loss = loss_fn(logits, label)
                        #TODO: try with only local loss
                        loss = (slide_loss + local_loss) / 2
                        # loss = 0.1 * slide_loss + 0.9 * local_loss
                        # loss = loss_fn(logits, torch.tensor(100).float().to(args.device))

                        # # kl loss trial
                        # if (batch_idx + 1) % args.gc == 0:
                        #     eps = 1e-8
                        #     logits_dist = torch.histc(batch_local_logits.float(), bins=10, min=min_local_labels_bins, max=max_local_labels_bins)
                        #     label_dist = torch.histc(batch_local_labels.float(), bins=10, min=min_local_labels_bins, max=max_local_labels_bins)
                        #     kl_loss = torch.nn.functional.kl_div((logits_dist + eps).log(), label_dist, reduction="batchmean") / args.gc
                        #     loss = (local_loss + kl_loss) / 2
                        # else:
                        #     loss = local_loss
                        
                        print(f"loss = {loss}, slide_loss = {slide_loss}, local_loss = {local_loss}, kl_loss = {kl_loss}") # Delete
                        if loss.item() > 0.5:
                            print(f"slide_id = {batch['slide_id']}")
                            print(f"logits = {logits}, label = {label}, local_logits = {local_logits}, local_labels = {local_labels}")

                    elif args.student_training:
                        label_loss = loss_fn(logits, label)
                        teacher_label_loss = loss_fn(logits, teacher_labels) if teacher_labels is not None else None
                        hidden_rep_loss = loss_fn(regressor_output, teacher_feature_map)
                        contr_loss = Contrastive_Loss()
                        contrastive_loss = contr_loss(embedding_memory, regressor_output, label)
                        
                        if args.model_ckpt != '': # KD training
                            if teacher_label_loss is not None:
                                loss = 0.5 * label_loss + 0.5 * teacher_label_loss
                            else:
                                loss = label_loss
                        else: # HT
                            # loss = hidden_rep_loss
                            # loss = 0.5 * hidden_rep_loss + 0.5 * label_loss
                            # loss = 0.5 * hidden_rep_loss + 0.5 * contrastive_loss if epoch > 1 else hidden_rep_loss
                            loss = contrastive_loss
                        print(f"loss = {loss}, label_loss = {label_loss}, hidden_rep_loss = {hidden_rep_loss}, teacher_label_loss={teacher_label_loss}, \
                              contrastive_loss={contrastive_loss}") # Delete

                    elif args.matching_tiles_training:
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
                    else:
                        print(f"logits = {logits}, shape = {logits.shape}, label = {label}, shape = {label.shape}")
                        loss = loss_fn(logits, label)  
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
                if not args.loss_fn == 'cox' and not (args.student_training and batch_idx == 0):
                    fp16_scaler.scale(loss).backward()
                    
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
    print('Epoch: {}, Train Loss: {:.4f}'.format(epoch, loss))

    tile_or_window_str = 'Window' if args.window_training else "Tile"
    # Delete
    if args.use_tile_classification or args.window_training:
        print(f'args.output_dir = {args.output_dir}')
        save_plot(x=None, y=all_local_labels, xlabel="Values", ylabel="Frequency", title=f"{tile_or_window_str} labels disstribution epoch={epoch}",
                hist=True, output_dir=args.output_dir)
        save_plot(x=None, y=all_local_logits, xlabel="Values", ylabel="Frequency", title=f"{tile_or_window_str} logits disstribution epoch={epoch}", 
                hist=True, output_dir=args.output_dir)
        save_plot(x=None, y=all_slide_labels, xlabel="Values", ylabel="Frequency", title=f"Slide labels disstribution epoch={epoch}", hist=True, output_dir=args.output_dir)
        save_plot(x=None, y=all_slide_logits, xlabel="Values", ylabel="Frequency", title=f"Slide logits disstribution epoch={epoch}", hist=True, output_dir=args.output_dir)
    return records


def evaluate(loader, model, fp16_scaler, loss_fn, epoch, args, save_embed=False):
    if args.student_training and 'teacher_model' in args:
        args.teacher_model.eval()

    model.eval()

    # set the evaluation records
    records = get_records_array(len(loader), args.n_classes, args)
    embeds = np.zeros((len(loader), 768))
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
            
            slide_id = batch['slide_id'][-1] if args.window_training else batch['slide_id']
            if args.window_training:
                sliding_window_ds = SlidingWindowDataset(data_df=args.dataset, root_path=args.root_path, task_config=args.task_config, slide_key=args.slide_key, label=args.label, \
                                    dataset_name=args.test_dataset, folds=args.test_fold,
                                    use_clinical_features=args.clinical_features, \
                                    test_on_all=args.test_on_all, get_single_slide=slide_id,
                                    window_size=args.window_size, stride=args.window_size)
                window_loader = get_test_loader(sliding_window_ds, for_heatmap=True, **vars(args))
                slide_windows = [{'imgs': wndw['imgs'].to(args.device, non_blocking=True), 
                                'coords': wndw['coords'].to(args.device, non_blocking=True)} for wndw in window_loader]

            # load the batch and transform this batch
            num_tiles = batch['imgs'].shape[1]
            if args.loss_fn == 'cox' and num_tiles > max_num_samples:
                indices = torch.randint(low=0, high=num_tiles, size=(max_num_samples,))
                batch['imgs'] = batch['imgs'][:, indices, :]
                batch['coords'] = batch['coords'][:, indices, :]
                if batch['local_labels'] is not None:
                    batch['local_labels'] = batch['local_labels'][:, indices, :]
                if batch['ihc_imgs'] is not None:
                    batch['ihc_imgs'] = batch['ihc_imgs'][:, indices, :]
                    batch['ihc_coords'] = batch['ihc_coords'][:, indices, :]

            images, img_coords, label, local_labels, ihc_images, ihc_coords = batch['imgs'], batch['coords'], batch['labels'], batch['local_labels'], batch['ihc_imgs'], batch['ihc_coords']
            images = images.to(args.device, non_blocking=True)
            img_coords = img_coords.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True).long()
            if ihc_images is not None:
                ihc_images = ihc_images.to(args.device, non_blocking=True)
                ihc_coords = ihc_coords.to(args.device, non_blocking=True)
            # valid_label_indices = None
            # if local_labels is not None:
            #     local_labels = local_labels.to(args.device, non_blocking=True)
            #     valid_label_indices = ~torch.isnan(local_labels)
            #     local_labels = local_labels[valid_label_indices]
            #     if args.use_tile_classification:
            #         images = images[valid_label_indices]
            #         img_coords = img_coords[valid_label_indices]
            if args.survival:
                censoreship = batch['censoreship']
            # if (args.window_training or args.use_tile_classification) and (local_labels is None or local_labels.size == 0):
            #     print(f"{slide_id} local labels is None, skipping")
            #     continue

            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                # if args.window_training and slide_windows is not None and valid_label_indices is not None:
                #     valid_label_indices = torch.nonzero(valid_label_indices.flatten(), as_tuple=True)[0]
                #     print(f"valid_label_indices = {valid_label_indices}, batch_idx = {batch_idx}")
                #     if len(valid_label_indices) == 0:
                #         continue
                #     slide_windows = [slide_windows[i] for i in valid_label_indices]

                print(f"slide = {slide_id}, batch_idx = {batch_idx}")
                # get the logits
                if save_embed:
                    if args.window_training and slide_windows is not None:
                        logits, local_logits, embed = model(slide_windows, return_embed=True)
                        print(f"local_logits = {local_logits}, logits = {logits}, local_labels = {local_labels}, label = {label}")
                    elif args.use_tile_classification:
                        logits, local_logits, embed = model(images, img_coords, return_embed=True)
                    elif args.student_training or args.matching_tiles_training:
                        logits, _, embed = model(images, img_coords, return_embed=True)
                    else:
                        logits, embed = model(images, img_coords, return_embed=True)
                    embeds[batch_idx] = embed.cpu().numpy()
                else:
                    if args.window_training and slide_windows is not None:
                        logits, local_logits = model(slide_windows)
                    elif args.use_tile_classification:
                        logits, local_logits = model(images, img_coords)
                    elif args.student_training or args.matching_tiles_training:
                        logits, _ = model(images, img_coords)
                    else:
                        logits = model(images, img_coords)

                if logits is None:
                    continue
                
                # get the loss
                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    label = label.squeeze(-1).float()
                elif task_setting == 'continuous' and not args.loss_fn == 'cox':
                    label = label.squeeze(-1).float()
                    logits = logits.squeeze(-1)

                    # if args.use_tile_classification or args.window_training: # Delete
                    #     if local_labels is not None:
                    #         local_logits = local_logits.squeeze(-1)
                    #         local_labels = local_labels.squeeze(-1).float()
                else:
                    label = label.squeeze(-1).long()
                if task_setting == 'continuous' and not args.loss_fn == 'cox':
                    if not args.window_training:
                        label = (label - model.mean.item()) / model.std.item()
                if not no_loss:
                    try:
                        if not args.loss_fn == 'cox':
                            if not args.survival:
                                if not args.use_tile_classification and not args.window_training:
                                    loss = loss_fn(logits, label)
                                else:
                                    # if args.loss_fn == "weighted_mse":
                                    #     loss_fn = torch.nn.MSELoss()   
                                    # if local_labels.shape != local_logits.shape:
                                    #     local_labels = local_labels.view(local_logits.shape)
                                    # local_loss = loss_fn(local_logits, local_labels)
                                    slide_loss = loss_fn(logits, label)
                                    # loss = (slide_loss + local_loss) / 2
                                    loss = slide_loss
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
                Y_prob = torch.softmax(logits, dim=1).cpu()
                records['prob'][batch_idx] = Y_prob.numpy()
                # convert label to one-hot
                label_ = torch.zeros_like(Y_prob).scatter_(1, label.cpu().unsqueeze(1), 1)
                records['label'][batch_idx] = label_.numpy()
            elif task_setting == 'continuous':
                if not args.loss_fn == 'cox':
                    if not args.window_training:
                        records['prob'][batch_idx] = ((logits * model.std.item()) + model.mean.item()).cpu().numpy()
                        records['label'][batch_idx] = ((label * model.std.item()) + model.mean.item()).cpu().numpy()
                    else:
                        records['prob'][batch_idx] = logits.cpu().numpy()
                        records['label'][batch_idx] = label.cpu().numpy()    
                    if not no_loss and not args.survival:
                        for key in regression_losses:
                            if args.use_tile_classification and key in ['cox', 'trunc_mse']:
                                continue
                            else:
                                try:
                                    records[key] += regression_losses[key](logits, label).item()
                                except BaseException as e:
                                    print(f'Cannot calculate {key}\n{e}')
                                    continue
                else:
                    records['prob'][batch_idx] = logits.cpu().numpy()
                    records['label'][batch_idx] = label.cpu().numpy()
                    records['censoreship'][batch_idx] = censoreship[0].cpu().numpy()

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
            else:
                info += ' Eval Loss: {:.4f}'.format(records['loss'])
        print(info)
    except:
        print("Failed to get metrics, this should only happen on test set.")

    # return the embeddings
    if save_embed:
        return records, embeds
    return records
