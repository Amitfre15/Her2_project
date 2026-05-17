import os
import re
import math
import torch
import pickle
import random
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler

DESIRED_MPP = 2
SLIDE_PATCH_SIZE = 256
SLIDE_MPP = 0.242797397769517
SLIDE_HEIGHT = 204614
SLIDE_WIDTH = 89484
HE_THUMB_HEIGHT = 10230
HE_THUMB_WIDTH = 4473
IHC_THUMB_HEIGHT = 5115
IHC_THUMB_WIDTH = 2237

def map_colors(patches, rate = 1):
    map_file = '/home/shacharcohen/workspace/WSI/legacy/color_map_FinHer2Carmel.csv'
    colormap = np.genfromtxt(map_file, delimiter=',')
    colormap_red = colormap[:,0]
    colormap_green = colormap[:,1]
    colormap_blue = colormap[:,2]
    #patches = F.interpolate(torch.tensor(patches), scale_factor = rate, mode = 'bilinear')
    patches_int = (patches*999/255).astype(int)
    patches[0,:,:] = np.array(colormap_red[patches_int[0,:,:]])
    patches[1,:,:] = np.array(colormap_green[patches_int[1,:,:]])
    patches[2,:,:] = np.array(colormap_blue[patches_int[2,:,:]])
    #patches = np.array(F.interpolate(patches, scale_factor = 1/rate, mode = 'bilinear'))
    return patches


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)
        

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def load_npy_file(file_path):
    try:
        npy = np.load(file_path) # load the numpy file
        return npy
    except BaseException as e:
        print(e)


def seed_torch(device, seed=7):
    # ------------------------------------------------------------------------------------------
    # References:
    # HIPT: https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/main.py
    # ------------------------------------------------------------------------------------------
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_exp_code(args):
    '''Get the experiment code for the current run.'''
    # set up the model code
    model_code = 'eval'
    if len(args.pretrained) > 0:
        model_code += '_pretrained'
    if args.freeze:
        model_code += '_freeze'
        
    # set up the task code
    task_code = args.task
    if args.pat_strat:
        task_code += '_pat_strat'

    # set up the experiment code
    exp_code = '{model_code}_{task_code}'

    return model_code, task_code, exp_code.format(model_code=model_code, task_code=task_code)


def pad_tensors(imgs, coords):
    # ------------------------------------------------------------------------------------------
    # References:
    # mae: https://github.com/facebookresearch/mae/tree/main
    # ------------------------------------------------------------------------------------------
    max_len = max([t.size(0) for t in imgs])  # get the maximum length
    padded_tensors = []  # list to store all padded tensors
    padded_coords = []  # list to store all padded coords
    masks = []  # list to store all masks
    for i in range(len(imgs)):
        # tensor: [L, d]
        tensor = imgs[i]
        # coords: [L, 2]
        coord = coords[i]
        N_i = tensor.size(0)  # get the original length
        # create a new tensor of shape (max_len, d) filled with zeros
        padded_tensor = torch.zeros(max_len, tensor.size(1))
        padded_coord = torch.zeros(max_len, 2)
        # create a new tensor of shape (max_len) filled with zeros for mask
        mask = torch.zeros(max_len)
        # place the original tensor into the padded tensor
        padded_tensor[:N_i] = tensor
        padded_coord[:N_i] = coord
        # the mask is filled with ones at the same indices as the original tensor
        mask[:N_i] = torch.ones(N_i)
        padded_tensors.append(padded_tensor)
        padded_coords.append(padded_coord)
        masks.append(mask)

    # concatenate all tensors along the 0th dimension
    padded_tensors = torch.stack(padded_tensors)
    padded_coords = torch.stack(padded_coords)
    masks = torch.stack(masks)
    # convert masks to bool type
    masks = masks.bool()
    return padded_tensors, padded_coords, masks


def slide_collate_fn(samples):
    '''Separate the inputs and targets into separate lists
    Return value {imgs: [N, L, 256, 384], pad_mask: [N, L]}'''
    image_list = [s.get('imgs', None) for s in samples]
    ihc_image_list = [s.get('ihc_imgs', None) for s in samples]
    synth_ihc_image_list = [s.get('synth_ihc_imgs', None) for s in samples]
    img_len_list = [s.get('imgs', None).size(0) if s.get('imgs', None) is not None else 0 for s in samples]
    coord_list = [s.get('coords', None) for s in samples]
    ihc_coord_list = [s.get('ihc_coords', None) for s in samples]
    label_list = [s.get('labels', None) for s in samples]
    matching_tiles_list = [s.get('matching_tiles', None) for s in samples]
    tumor_indices_list = [s.get('tumor_indices', None) for s in samples]
    non_tumor_indices_list = [s.get('non_tumor_indices', None) for s in samples]
    cancer_prob_list = [s.get('cancer_prob', None) for s in samples]
    tile_logits_list = [s.get('tile_logits', None) for s in samples]
    external_tile_logits_list = [s.get('external_tile_logits', None) for s in samples]
    # tiles_list = [s.get('tiles', None) for s in samples]
    ihc_tiles_list = [s.get('ihc_tiles', None) for s in samples]
    tile_y_list = [s.get('tile_y', None) for s in samples]
    virchow2_feats_list = [s.get('virchow2_feats', None) for s in samples]
    uni2_feats_list = [s.get('uni2_feats', None) for s in samples]
    conch_feats_list = [s.get('conch_feats', None) for s in samples]
    clinical_feats_list = [s.get('clinical_feats', None) for s in samples]
    score_list = [s.get('score', None) for s in samples]
    slide_id_list = [s.get('slide_id', None) for s in samples]
    labels = torch.stack(label_list)
    matching_tiles = torch.stack(matching_tiles_list) if matching_tiles_list[0] is not None else None
    tumor_indices = torch.stack(tumor_indices_list) if tumor_indices_list[0] is not None else None
    non_tumor_indices = torch.stack(non_tumor_indices_list) if non_tumor_indices_list[0] is not None else None
    cancer_prob = torch.stack(cancer_prob_list) if cancer_prob_list[0] is not None else None
    tile_logits = torch.stack(tile_logits_list) if tile_logits_list[0] is not None else None
    external_tile_logits = torch.stack(external_tile_logits_list) if external_tile_logits_list[0] is not None else None
    virchow2_feats = torch.stack(virchow2_feats_list) if virchow2_feats_list[0] is not None else None
    uni2_feats = torch.stack(uni2_feats_list) if uni2_feats_list[0] is not None else None
    conch_feats = torch.stack(conch_feats_list) if conch_feats_list[0] is not None else None
    clinical_feats = torch.stack(clinical_feats_list) if clinical_feats_list[0] is not None else None
    score = torch.stack(score_list) if score_list[0] is not None else None
    # tiles = torch.stack(tiles_list) if tiles_list[0] is not None else None
    tiles = samples[0]['tiles'] if samples[0].get('tiles', None) is not None else None
    ihc_tiles = torch.stack(ihc_tiles_list) if ihc_tiles_list[0] is not None else None
    tile_y = torch.stack(tile_y_list) if tile_y_list[0] is not None else None
    censoreship = [s.get('censoreship', None) for s in samples]
    if image_list[0] is not None:
        pad_imgs, pad_coords, pad_mask = pad_tensors(image_list, coord_list)
        
        res_tuple = pad_tensors(ihc_image_list, ihc_coord_list) if ihc_image_list[0] is not None else None, None, None
        res_tuple = res_tuple[0]
        if res_tuple is not None:
            ihc_pad_imgs, ihc_pad_coords, ihc_pad_mask = res_tuple[0], res_tuple[1], res_tuple[2]
        else:
            ihc_pad_imgs, ihc_pad_coords, ihc_pad_mask = None, None, None

        if synth_ihc_image_list[0] is not None:
            synth_pad_imgs, _, _ = pad_tensors(synth_ihc_image_list, coord_list)
        else:
            synth_pad_imgs = None
    else:
        pad_imgs, pad_coords, pad_mask, ihc_pad_imgs, ihc_pad_coords, ihc_pad_mask, synth_pad_imgs = None, None, None, None, None, None, None
    
    data_dict = {'imgs': pad_imgs, 
            'ihc_imgs': ihc_pad_imgs,
            'synth_ihc_imgs': synth_pad_imgs,
            'img_lens': img_len_list,
            'coords': pad_coords,
            'ihc_coords': ihc_pad_coords,
            'slide_id': slide_id_list,
            'pad_mask': pad_mask,
            'ihc_pad_mask': ihc_pad_mask,
            'labels': labels,
            'matching_tiles': matching_tiles,
            'tumor_indices': tumor_indices,
            'non_tumor_indices': non_tumor_indices,
            'cancer_prob': cancer_prob,
            'tile_logits': tile_logits,
            'external_tile_logits': external_tile_logits,
            'virchow2_feats': virchow2_feats,
            'uni2_feats': uni2_feats,
            'conch_feats': conch_feats,
            'clinical_feats': clinical_feats,
            'score': score,
            'tiles': tiles,
            'ihc_tiles': ihc_tiles,
            'tile_y': tile_y,
            'censoreship': censoreship}
    return data_dict

def slide_collate_fn_for_heatmap(samples):
    '''Separate the inputs and targets into separate lists
    Return value {imgs: [N, L, 256, 384], pad_mask: [N, L]}'''
    image_list = [s['imgs'] for s in samples]
    img_len_list = [s['imgs'].size(0) for s in samples]
    coord_list = [s['coords'] for s in samples]
    label_list = [s['labels'] for s in samples]
    labels = torch.stack(label_list)
    pad_imgs, pad_coords, pad_mask = pad_tensors(image_list, coord_list)
    
    data_dict = {'imgs': pad_imgs, 
            'img_lens': img_len_list,
            'coords': pad_coords,
            'pad_mask': pad_mask,
            'labels': labels}
    return data_dict


def slide_collate_fn_for_window(samples):
    '''Separate the inputs and targets into separate lists
    Return value {imgs: [N, L, 256, 384], pad_mask: [N, L]}'''
    image_list = [s['imgs'] for s in samples]
    img_len_list = [s['imgs'].size(0) for s in samples]
    coord_list = [s['coords'] for s in samples]
    label_list = [s['labels'] for s in samples]
    window_y_list = [s['window_y'] for s in samples]
    labels = torch.stack(label_list)
    window_y = torch.stack(window_y_list)
    pad_imgs, pad_coords, pad_mask = pad_tensors(image_list, coord_list)
    
    data_dict = {'imgs': pad_imgs, 
            'img_lens': img_len_list,
            'coords': pad_coords,
            'pad_mask': pad_mask,
            'labels': labels,
            'window_y': window_y}
    return data_dict


def get_splits(df: pd.DataFrame, 
               val_r: float=0.1, test_r: float=0.2, 
               fold: int=0, 
               split_dir: str='', 
               fetch_splits: bool=True, 
               prop: int=1, 
               split_key='slide_id', 
               **kwargs) -> Tuple[List[str], List[str], List[str]]:
    '''Get the splits for the dataset. The default train/val/test split is 70/10/20.'''
    # get the split names
    files = os.listdir(split_dir)
    train_name, val_name, test_name = f'train_{fold}.csv', f'val_{fold}.csv', f'test_{fold}.csv'
    # check split_key is in the columns
    assert split_key in df.columns, f'{split_key} not in the columns of the dataframe'
    # make sure the dataset exists, otherwise create new datasets
    if train_name not in files or val_name not in files or test_name not in files or not fetch_splits:
        samples = df.drop_duplicates(split_key)[split_key].to_list()
        train_samples, temp_samples = train_test_split(samples, test_size=(val_r + test_r), random_state=fold)
        if val_r > 0:
            val_samples, test_samples = train_test_split(temp_samples, test_size=(test_r / (val_r + test_r)), random_state=fold)
        else:
            val_samples, test_samples = [], temp_samples
        train_data = df[df[split_key].isin(train_samples)]
        val_data = df[df[split_key].isin(val_samples)]
        test_data = df[df[split_key].isin(test_samples)]

        # sample the training data
        if prop > 0:
            train_data = train_data.sample(frac=prop, random_state=fold).reset_index(drop=True)
        # save datasets
        train_data.to_csv(os.path.join(split_dir, train_name))
        val_data.to_csv(os.path.join(split_dir, val_name))
        test_data.to_csv(os.path.join(split_dir, test_name))
    # load the dataframe
    train_splits = pd.read_csv(os.path.join(split_dir, train_name))[split_key].to_list()
    val_splits = pd.read_csv(os.path.join(split_dir, val_name))[split_key].to_list()
    test_splits = pd.read_csv(os.path.join(split_dir, test_name))[split_key].to_list()

    return train_splits, val_splits, test_splits


def get_loader(train_data, val_data, test_data, 
               task_config, weighted_sample=False, 
               batch_size=1, num_workers=10, seed=0, loss_fn='mse',
               **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''Get the dataloader for the dataset.'''
    if weighted_sample and not task_config.get('setting', 'multi_class') == 'multi_label':
        # get the weights for each class, we only do this for multi-class classification
        N = len(train_data)
        weights = {}
        for idx in range(N):
            label = int(train_data.labels[idx][0])
            if label not in weights: weights[label] = 0
            weights[label] += 1.0 / N
        for l in weights.keys(): weights[l] = 1.0 / weights[l]
        sample_weights = [weights[int(train_data.labels[i][0])] for i in range(N)]
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    else:
        if loss_fn!='cox':
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = SequentialSampler(val_data)

    # set up generator and worker_init_fn
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # if it's the sequence based model, we use the slide collate function to pad
    train_loader = DataLoader(train_data, \
                            num_workers=num_workers, \
                            batch_size=batch_size, sampler=train_sampler, \
                            generator=g, worker_init_fn=seed_worker, \
                            collate_fn=slide_collate_fn)
    val_loader = DataLoader(val_data, \
                            num_workers=num_workers, \
                            batch_size=1, sampler=SequentialSampler(val_data), \
                            worker_init_fn=seed_worker, \
                            collate_fn=slide_collate_fn) if val_data is not None else None
    test_loader = DataLoader(test_data, \
                            num_workers=num_workers, \
                            batch_size=1, sampler=SequentialSampler(test_data), \
                            worker_init_fn=seed_worker, \
                            collate_fn=slide_collate_fn) if test_data is not None else None

    return train_loader, val_loader, test_loader

def get_test_loader(test_data, num_workers=10, for_heatmap = False, for_window = False, **kwargs):
    if for_heatmap:
        collate_fn = slide_collate_fn_for_heatmap
    elif for_window:
        collate_fn = slide_collate_fn_for_window
    else:
        collate_fn = slide_collate_fn
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    test_loader = DataLoader(test_data, \
                            num_workers=num_workers, \
                            batch_size=1, sampler=SequentialSampler(test_data), \
                            worker_init_fn=seed_worker, \
                            collate_fn=collate_fn)
    return test_loader


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    # ------------------------------------------------------------------------------------------
    # References:
    # BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    # ------------------------------------------------------------------------------------------
    param_group_names = {}
    param_groups = {}

    num_layers = model.slide_encoder.encoder.num_layers + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        if 'mask_token' in n or 'slide_encoder.decoder' in n:
            continue


        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
        
        layer_id = get_layer_id(n, num_layers)

        group_name = n + "_%d_%s" % (layer_id + 1, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
            # if n.startswith('classifier'):
            #     this_scale = 100.0
            
            # if n.startswith('tile_classifier'):
            #     this_scale = 10.0

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id(name, num_layers):
    # ------------------------------------------------------------------------------------------
    # References:
    # BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    # ------------------------------------------------------------------------------------------
    if 'cls_token' in name or 'pos_embed' in name:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('slide_encoder.encoder.layers'):
        return int(name.split('.')[3]) + 1
    else:
        return num_layers


def adjust_learning_rate(optimizer, epoch, args):
    # ------------------------------------------------------------------------------------------
    # References:
    # mae: https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
    # ------------------------------------------------------------------------------------------
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_optimizer(args, model):
    '''Set up the optimizer for the model.'''
    # make the optimizer
    optim_func = torch.optim.AdamW if args.optim == 'adamw' else torch.optim.Adam

    if not args.use_tile_classification:
        param_groups = param_groups_lrd(model, args.optim_wd,
            layer_decay=args.layer_decay)    
        # print(f'param_groups = {param_groups}')
        optimizer = optim_func(param_groups, lr=args.lr)

    else:
        optimizer = optim_func(model.parameters(), lr=args.lr, weight_decay=args.optim_wd)

    return optimizer

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)

def PartialLogLikelihood(logits, fail_indicator):
    '''
    fail_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model 
    taken from https://github.com/runopti/stg
    '''
    logL = 0
    # pre-calculate cumsum
    cumsum_y_pred = torch.cumsum(logits, 0)
    hazard_ratio = torch.exp(logits)
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)
    log_risk = torch.log(cumsum_hazard_ratio)
    likelihood = logits - log_risk
    # dimension for E: np.array -> [None, 1]
    uncensored_likelihood = likelihood * fail_indicator
    logL = -torch.sum(uncensored_likelihood)
    # negative average log-likelihood
    observations = torch.sum(fail_indicator, 0)
    return 1.0*logL / observations

class PartialLogLikelihoodLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, y_censoreship: torch.Tensor
    ) -> torch.Tensor:
        indices = torch.sort(y_true).indices
        y_pred = y_pred[indices]
        y_censoreship = y_censoreship[indices]
        return PartialLogLikelihood(y_pred, y_censoreship)

class Truncated_MSE_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, y_censoreship: torch.Tensor
    ) -> torch.Tensor:
        y_censoreship = torch.tensor(y_censoreship, device = y_pred.device)
        mask = (y_censoreship == 1) | (y_pred <= y_true)
        if mask.sum() == 0:
            return torch.sum(y_pred * 0)
        y_pred_masked = y_pred[mask]
        y_true_masked = y_true[mask]
        loss = self.loss_fn(y_pred_masked, y_true_masked)
        return loss

class Weighted_MSE_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        return (weights * (y_pred - y_true) ** 2).mean()


class Assymmetric_MSE_Loss(torch.nn.Module):
    def __init__(self, w_over=1.0, w_under=1.0):
        super().__init__()
        self.w_over = w_over
        self.w_under = w_under

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        diff = y_pred - y_true
        loss = torch.where(diff > 0,
                        self.w_over * diff * diff,   # overestimates
                        self.w_under * diff * diff)  # underestimates
        return loss.mean()


class PenalizedCELoss(torch.nn.Module):
    def __init__(self, n_classes, penalty_weight=1.0):
        """
        Penalized Cross-Entropy Loss to penalize high logits for distant classes.

        Args:
            n_classes (int): The number of classes.
            penalty_weight (float): Weight of the penalty term.
        """
        super(PenalizedCELoss, self).__init__()
        self.n_classes = n_classes
        self.penalty_weight = penalty_weight

    def forward(self, logits, labels):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape [N, n_classes].
            labels (torch.Tensor): True labels of shape [N].

        Returns:
            torch.Tensor: Combined loss (CE loss + distance penalty).
        """
        # Compute the standard Cross-Entropy Loss
        ce_loss = F.cross_entropy(logits, labels)

        # Create a distance matrix for the classes
        class_indices = torch.arange(self.n_classes, device=logits.device).float()  # Shape: [n_classes]
        distance_matrix = torch.abs(class_indices.unsqueeze(0) - class_indices.unsqueeze(1))  # Shape: [n_classes, n_classes]

        # Get the predicted probabilities
        probs = F.softmax(logits, dim=-1)  # Shape: [N, n_classes]

        # Gather the distances for the true labels
        true_distances = distance_matrix[labels]  # Shape: [N, n_classes]

        # Compute the penalty term
        penalty = (probs * true_distances).sum(dim=-1).mean()  # Penalize high probabilities for distant classes
        print(f'probs * true_distances = {probs * true_distances}')

        # Combine the CE loss and the penalty
        total_loss = ce_loss + self.penalty_weight * penalty
        return total_loss


class KL_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred, y_true = y_pred.float(), y_true.float()

        min_label, max_label = -0.5, 3.5
        logits_hist = torch.histc(y_pred, bins=20, min=min_label, max=max_label)
        labels_hist = torch.histc(y_true, bins=20, min=min_label, max=max_label)
        print(f'logits_hist = {logits_hist}, labels_hist = {labels_hist}')

        # Normalize to get probability distributions
        logits_dist = logits_hist / logits_hist.sum()
        labels_dist = labels_hist / labels_hist.sum()

        # Convert to log-probabilities for KLDiv
        log_logits_dist = torch.log(logits_dist + 1e-8)  # Avoid log(0)

        # Compute KL divergence (from labels → logits)
        kl_loss = F.kl_div(log_logits_dist, labels_dist, reduction='batchmean')
        if torch.isnan(kl_loss):
            print(f'y_pred = {y_pred}, y_true = {y_true}, logits_hist = {logits_hist}, labels_hist = {labels_hist}, log_logits_dist = {log_logits_dist}')
        return kl_loss

class DifferentiableHistogramKL(torch.nn.Module):
    def __init__(self, bins=10, min_val=2, max_val=5):
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        self.bin_width = (max_val - min_val) / bins
        self.bin_centers = torch.linspace(min_val + self.bin_width / 2, max_val - self.bin_width / 2, bins)

    def forward(self, y_pred, y_true):
        device = y_pred.device
        bin_centers = self.bin_centers.to(device)

        def soft_histogram(x):
            # x: (N,)
            x = x.view(-1, 1)  # (N, 1)
            bin_centers_ = bin_centers.view(1, -1)  # (1, B)
            # Gaussian kernel to approximate histogram binning
            kernel = torch.exp(-0.5 * ((x - bin_centers_) / self.bin_width)**2)  # (N, B)
            hist = kernel.sum(dim=0)  # (B,)
            return hist / hist.sum()  # Normalize to sum to 1

        pred_dist = soft_histogram(y_pred)
        true_dist = soft_histogram(y_true)

        log_pred = torch.log(pred_dist + 1e-8)
        kl = F.kl_div(log_pred, true_dist, reduction='batchmean')
        return kl


class MidPenalizedLoss(torch.nn.Module):
    def __init__(self, base_loss_fn, label_range=(0, 3), penalty_weight=1.0):
        """
        Penalized loss to punish predictions near the middle of the label range.

        Args:
            base_loss_fn: The base loss function (e.g., MSELoss, CrossEntropyLoss).
            label_range: Tuple indicating the range of labels (e.g., (0, 3)).
            penalty_weight: Weight of the penalty term.
        """
        super(MidPenalizedLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.middle = (label_range[0] + label_range[1]) / 2  # Middle of the range
        self.penalty_weight = penalty_weight

    def forward(self, logits, labels):
        # Base loss (e.g., MSE or CrossEntropy)
        base_loss = self.base_loss_fn(logits, labels)

        # Penalize predictions near the middle of the range
        penalty = torch.abs(logits - self.middle)  # Distance from the middle
        penalty = 1.0 / (penalty + 1e-8)  # Inverse distance (higher penalty near the middle)

        # Apply the penalty weight
        penalty_loss = self.penalty_weight * penalty.mean()

        # Combine the base loss and the penalty
        total_loss = base_loss + penalty_loss
        return total_loss


class CLIPLoss(torch.nn.Module):
    def __init__(self, num_per_class: int = 500):
        super(CLIPLoss, self).__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))
        self.num_per_class = num_per_class

    def forward(self, HE_tile_features: torch.Tensor, HE_tile_labels: torch.Tensor, HE_memory: dict) -> torch.Tensor:
        if HE_tile_features is None or HE_tile_labels is None:
            return torch.tensor(0.0, device=HE_tile_features.device)

        device = HE_tile_features.device
        num_samples = min(self.num_per_class, HE_tile_features.size(0))
        HE_tile_labels = torch.clamp(HE_tile_labels.squeeze(0).round(), min=0, max=3)
        # HE_tile_labels[HE_tile_labels == 1] = 0
        # HE_tile_labels[HE_tile_labels == 2] = 1
        HE_tile_labels[HE_tile_labels == 3] = 1
        print(f'HE_tile_features.shape = {HE_tile_features.shape}, HE_tile_labels.shape = {HE_tile_labels.shape}')
        # bucket_ids = torch.clamp(HE_tile_labels, min=0, max=3)
        bucket_ids = torch.clamp(HE_tile_labels, min=0, max=1)
        print(f'bucket_ids = {bucket_ids}')
        bucket_keys = ['0', '1'] # ['0', '1', '2', '3']
        num_keys = len(bucket_keys)

        # Ensure all memory keys exist
        for key in bucket_keys:
            if key not in HE_memory:
                HE_memory[key] = []

        # Update memory bank with current features (detached, per class)
        for i, key in enumerate(bucket_keys):
            mask = (bucket_ids == i).squeeze(0)
            if mask.any():
                HE_memory[key].extend([f.detach().cpu() for f in HE_tile_features[mask]])
                if len(HE_memory[key]) > self.num_per_class * 5:
                    HE_memory[key] = HE_memory[key][-self.num_per_class * 5:]

        # Check all memory buckets are non-empty before proceeding
        if any(len(HE_memory[k]) == 0 for k in bucket_keys):
            return torch.tensor(0.0, device=device)

        # Sample num_samples anchors from current batch
        anchor_idxs = torch.randperm(HE_tile_features.size(0))[:num_samples]
        anchors = HE_tile_features[anchor_idxs]
        anchor_labels = bucket_ids[anchor_idxs]

        all_feats = []

        for anchor, anchor_label in zip(anchors, anchor_labels):
            anchor_lbl = anchor_label.int().item() # Convert to int for indexing
            group_feats = [None] * num_keys  # one for each label
            group_feats[anchor_lbl] = anchor  # place anchor in correct slot

            for lbl in range(num_keys):
                if lbl == anchor_lbl:
                    continue
                lbl_key = str(lbl)
                sample = random.choice(HE_memory[lbl_key])
                group_feats[lbl] = sample.to(device)

            all_feats.append(torch.stack(group_feats))  # shape (num_keys, feat_dim)

        all_feats = torch.stack(all_feats, dim=0)  # (num_samples, num_keys, feat_dim)

        # ---- Permute all_feats into all_feats2 with no index i = i ----
        def generate_derangement(n):
            while True:
                idx = torch.randperm(n)
                if not torch.any(idx == torch.arange(n)):
                    return idx

        deranged_idx = generate_derangement(num_samples)
        all_feats2 = all_feats[deranged_idx]  # (num_samples, num_keys, feat_dim)

        print(f'all_feats[0] = {all_feats[0]}, all_feats[0].requires_grad = {all_feats[0].requires_grad}')
        print(f'all_feats2[0] = {all_feats2[0]}')
        # Normalize both
        all_feats = F.normalize(all_feats, dim=2)    # (N, num_keys, D)
        all_feats2 = F.normalize(all_feats2, dim=2)  # (N, num_keys, D)
        print(f'all_feats[0] = {all_feats[0]}')
        print(f'all_feats2[0] = {all_feats2[0]}')

        # Compute logits: (N, num_keys, num_keys)
        logits = torch.bmm(all_feats, all_feats2.transpose(1, 2)) # * self.temperature.exp()
        print(f'logits = {logits}')

        # Targets: identity across num_keys positions
        targets = torch.arange(num_keys, device=device).unsqueeze(0).expand(num_samples, -1)  # (N, num_keys)
        print(f'logits.shape = {logits.shape}, targets.shape = {targets.shape}')

        # Compute cross-entropy loss for each example in batch
        loss1 = F.cross_entropy(logits.reshape(-1, num_keys), targets.reshape(-1))
        loss2 = F.cross_entropy(logits.transpose(1, 2).reshape(-1, num_keys), targets.reshape(-1))

        # # Step 1: Create soft targets of shape (N, 4, 4)
        # soft_targets = torch.tensor([
        #     [0.75, 0.25, 0.00, 0.00],
        #     [0.25, 0.50, 0.25, 0.00],
        #     [0.00, 0.25, 0.50, 0.25],
        #     [0.00, 0.00, 0.25, 0.75]
        # ], device=logits.device).unsqueeze(0).expand(logits.size(0), -1, -1)  # (N, 4, 4)

        # # Step 2: Compute log-probs over last dim
        # log_probs = F.log_softmax(logits, dim=-1)  # (N, 4, 4)
        # log_probs_T = F.log_softmax(logits.transpose(1, 2), dim=-1)

        # # Step 3: KL-div between log_probs and soft_targets
        # loss1 = F.kl_div(log_probs, soft_targets, reduction='batchmean')
        # loss2 = F.kl_div(log_probs_T, soft_targets, reduction='batchmean')

        return (loss1 + loss2) / 2


class Contrastive_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    # Contrastive Loss Function (from the provided image)
    def contrastive_loss(self, dif_class, dist, margin=50):
        loss = ((1 - dif_class) * 0.5 * dist**2 + dif_class * 0.5 * (torch.clamp(margin - dist, min=0) ** 2))
        print(f'loss = {loss}')

        return loss.mean()
        
    def forward(self, embedding_memory: dict, embedding: torch.Tensor, teacher_embedding: torch.Tensor, label: torch.Tensor, margin=50) -> torch.Tensor:
        str_label = str(label.item())
        print(f'str_label = {str_label}')

        loss = torch.tensor(0.0, device=embedding.device)

        # If we already have embeddings stored
        if len(embedding_memory) > 0:
            pos_samples = []
            neg_samples = []

            # Collect positive and negative pairs
            for stored_label, stored_embs in embedding_memory.items():
                stored_embs = torch.stack(stored_embs)  # Convert list to tensor
                distances = torch.norm(embedding - stored_embs, dim=1)  # Compute distances

                if stored_label == str_label:
                    # pos_samples.append(distances)  # Positive pairs
                    pos_samples.append(torch.norm(embedding - teacher_embedding, dim=1))  # Positive pairs
                elif str_label in ['2', '3'] and stored_label not in ['2', '3']:
                    neg_samples.append(distances)  # Negative pairs
                elif str_label in ['0', '0.5', '1'] and stored_label not in ['0', '0.5', '1']:
                    neg_samples.append(distances)
                else:
                    pass  

            # Compute losses if we have pairs
            if pos_samples:
                pos_distances = torch.cat(pos_samples)
                print(f'pos_distances = {pos_distances}, shape = {pos_distances.shape}')
                sim_class = torch.zeros_like(pos_distances)
                loss += self.contrastive_loss(dif_class=sim_class, dist=pos_distances, margin=margin)

            if neg_samples:
                neg_distances = torch.cat(neg_samples)
                print(f'neg_distances = {neg_distances}, shape = {neg_distances.shape}')
                dif_class = torch.ones_like(neg_distances)
                loss += self.contrastive_loss(dif_class=dif_class, dist=neg_distances, margin=margin)

        # Update memory bank (keep last N embeddings per label)
        if str_label not in embedding_memory:
            embedding_memory[str_label] = []
        # embedding_memory[str_label].append(embedding.detach().squeeze(0))  # Remove batch dim
        embedding_memory[str_label].append(teacher_embedding.detach().squeeze(0))  # Remove batch dim
        print(f'embedding_memory.keys() = {embedding_memory.keys()}')

        # Limit memory size (e.g., store only the last 5 per class)
        if len(embedding_memory[str_label]) > 1:
            embedding_memory[str_label].pop(0)  

        return loss
    

def margin_loss(preds, targets, margin_scale=0.5):
    """
    preds: (N, 1) predicted scores
    targets: (N, 1) ground truth scores
    margin_scale: fraction of GT difference to enforce in preds
    """
    N = preds.size(0)
    diff_gt = torch.abs(targets.unsqueeze(0) - targets.unsqueeze(1))  # [N, N]
    diff_pred = torch.abs(preds.unsqueeze(0) - preds.unsqueeze(1))    # [N, N]

    # Required margin = scaled GT difference
    required_margin = margin_scale * diff_gt

    # Loss: only penalize if predicted diff < required margin
    loss_matrix = F.relu(required_margin - diff_pred)

    # Avoid counting self-pairs
    mask = ~torch.eye(N, dtype=torch.bool, device=preds.device)
    loss = loss_matrix[mask].mean()
    return loss


def soft_rank(x, regularization_strength=1.0):
    """
    NeuralSort-based soft ranking function.
    x: [batch]
    Returns approximate ranks with smooth gradients.
    """
    x = x.view(-1, 1)
    n = x.size(0)

    # Pairwise differences
    diff = x - x.t()  # shape [n, n]

    # Soft permutation matrix with temperature = regularization_strength
    # P = F.softmax(-diff / regularization_strength, dim=-1)
    P = F.sigmoid(-diff / regularization_strength)

    # Compute expected rank from permutation matrix
    # rank = torch.sum(P * torch.arange(n, device=x.device, dtype=torch.float).view(1, -1), dim=-1)
    rank = n - P.sum(dim=1)
    return rank


class SoftSpearmanLoss(torch.nn.Module):
    def __init__(self, regularization_strength=1.0, eps=1e-8):
        """
        regularization_strength ~ temperature of sorting
        lower = sharper → better ordering but less smooth gradients
        """
        super().__init__()
        self.reg = regularization_strength
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred, y_true = y_pred.view(-1), y_true.view(-1)
        # Soft ranks for prediction
        y_pred_rank = soft_rank(y_pred, self.reg)

        # Hard ranks for ground truth (no need to be differentiable)
        y_true_rank = torch.argsort(torch.argsort(y_true)).float()
        # print(f'y_true_rank = {y_true_rank}')

        # Spearman correlation = Pearson on ranks
        vx = y_pred_rank - torch.mean(y_pred_rank)
        vy = y_true_rank - torch.mean(y_true_rank)
        # print(f'vx = {vx}, vy = {vy}')

        corr = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx ** 2)) *
            torch.sqrt(torch.sum(vy ** 2)) + self.eps
        )
        return 1 - corr
        

def get_regression_losses():
    return {'mse':torch.nn.MSELoss(), 'mae':torch.nn.L1Loss(), 'huber':torch.nn.HuberLoss(delta=0.5), 'logcosh':LogCoshLoss(), 'cox':PartialLogLikelihoodLoss(), 'trunc_mse':Truncated_MSE_Loss(), 
            'weighted_mse': Weighted_MSE_Loss(), 'a_mse': Assymmetric_MSE_Loss(), 'spearmanr': spearmanr, 'pearsonr': pearsonr}


def get_loss_function(args):
    '''Get the loss function based on the task configuration.'''
    task_setting = args.task_config.get('setting', 'multi_class')
    if task_setting == 'multi_label':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif task_setting == 'multi_class' or task_setting == 'binary':
        loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = PenalizedCELoss(args.n_classes)
    elif task_setting == 'continuous':
        losses = get_regression_losses()
        loss_fn = losses[args.loss_fn]
    else:
        raise NotImplementedError
    print(f'Using {loss_fn} as the loss function.')
    return loss_fn


def get_records_array(record_len: int, n_classes, args, save_embed: bool = False) -> dict:
    '''Get the records array based on the task configuration.'''
    if not args.survival:
        record = {
            'prob': np.zeros((record_len, n_classes)), # if n_classes > 2 else np.zeros(record_len, dtype=np.float32),
            'label': np.zeros((record_len, n_classes)), # if n_classes > 2 else np.zeros(record_len, dtype=np.float32),
            'loss': 0.0,
        }
    else:
        record = {
            'prob': np.zeros((record_len, n_classes)), # if n_classes > 2 else np.zeros(record_len, dtype=np.float32),
            'label': np.zeros((record_len, n_classes)), # if n_classes > 2 else np.zeros(record_len, dtype=np.float32),
            'censoreship': np.zeros((record_len, n_classes)), # if n_classes > 2 else np.zeros(record_len, dtype=np.float32),
            'loss': 0.0,
        }
    return record


class Monitor_Score:
    # ------------------------------------------------------------------------------------------
    # References:
    # MCAT: https://github.com/mahmoodlab/MCAT/blob/master/utils/core_utils.py
    # ------------------------------------------------------------------------------------------
    def __init__(self):
        self.best_score = None

    def __call__(self, val_score, model, ckpt_name:str='checkpoint.pt'):

        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def log_writer(log_dict: dict, step: int, report_to: str='tensorboard', writer=None):
    '''Log the dictionary to the writer.'''
    if report_to == 'tensorboard':
        for k, v in log_dict.items():
            writer.add_scalar(k, v, step)
    elif report_to == 'wandb':
        writer.log(log_dict, step=step)
    else:
        raise NotImplementedError


def parse_tile_name(tile_name):
    # Example format: "33024x_12288y.png"
    match = re.match(r"(\d+)x_(\d+)y.png", tile_name)
    if match:
        x_start, y_start = map(int, match.groups())
        x_end, y_end = x_start + SLIDE_PATCH_SIZE, y_start + SLIDE_PATCH_SIZE
        return x_start, x_end, y_start, y_end
    else:
        raise ValueError(f"Invalid tile_name format: {tile_name}")

def correct_coords(x_y_coords: np.array, desired_mpp: float):
    rounded_mpp = round_mpp(SLIDE_MPP)
    mpp_correction_factor = SLIDE_MPP / rounded_mpp
    x_y_coords = x_y_coords * (mpp_correction_factor * desired_mpp) / SLIDE_MPP
    return x_y_coords

def slide_to_thumb_coord(x_y_slide_coords: np.array, slide_dimensions, thumb_size):
    """
    Converts a coordinate from the slide to the corresponding coordinate in the thumbnail.

    Parameters:
    - slide: The OpenSlide object.
    - thumb: The thumbnail image (PIL Image or similar).
    - x_y_coords: An np.array (x, y) representing coordinates on the slide.

    Returns:
    - An np.array (x, y) representing the corresponding coordinates on the thumbnail.
    """
    # Get dimensions of the thumbnail
    thumb_width, thumb_height = thumb_size

    # Get dimensions of the slide
    slide_width, slide_height = slide_dimensions

    # Calculate scale factors
    scale_x = thumb_width / slide_width
    scale_y = thumb_height / slide_height

    # Convert slide coordinates to thumbnail coordinates
    x_y_slide_coords[:, 0] = (x_y_slide_coords[:, 0] * scale_x).astype(int)
    x_y_slide_coords[:, 1] = (x_y_slide_coords[:, 1] * scale_y).astype(int)

    return x_y_slide_coords.astype(int)

def thumb_to_slide_coord(x_y_thumb_coords: np.array, slide_dimensions, thumb_size):
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

    # Convert thumb coordinates to slide coordinates
    x_y_thumb_coords[:, 0] = (x_y_thumb_coords[:, 0] * scale_x).astype(int)
    x_y_thumb_coords[:, 1] = (x_y_thumb_coords[:, 1] * scale_y).astype(int)

    return x_y_thumb_coords.astype(int)

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