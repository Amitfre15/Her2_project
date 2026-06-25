import torch

from torch import nn
import torch.nn.functional as F
from . import slide_encoder
from torchvision import models


class ClassificationHead(nn.Module):
    """
    The classification head for the slide encoder

    Arguments:
    ----------
    input_dim: int
        The input dimension of the slide encoder
    latent_dim: int
        The latent dimension of the slide encoder
    feat_layer: str
        The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers
    n_classes: int
        The number of classes
    model_arch: str
        The architecture of the slide encoder
    pretrained: str
        The path to the pretrained slide encoder
    freeze: bool
        Whether to freeze the pretrained model
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        feat_layer,
        n_classes=2,
        model_arch="gigapath_slide_enc12l768d",
        pretrained="hf_hub:prov-gigapath/prov-gigapath",
        freeze=False,
        mean=0,
        std=0,
        tumor_size_mean=0,
        tumor_size_std=0,
        age_mean=0,
        age_std=0,
        **kwargs,
    ):
        super(ClassificationHead, self).__init__()

        # setup the slide encoder
        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim
        if self.feat_dim != 768:
            model_arch = f"gigapath_slide_enc12l{self.feat_dim}d"
        self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)

        # whether to freeze the pretrained model
        if freeze:
            print("Freezing Pretrained GigaPath model")
            for name, param in self.slide_encoder.named_parameters():
                param.requires_grad = False
            print("Done")
        # setup the classifier
        self.classifier = nn.Sequential(*[nn.Linear(self.feat_dim, n_classes)])
        
        #save normalization parameters for use in inference
        self.mean = nn.Parameter(torch.tensor(mean, dtype = float))
        self.std = nn.Parameter(torch.tensor(std, dtype = float))
        self.mean.requires_grad = False
        self.std.requires_grad = False
        self.tumor_size_mean = nn.Parameter(torch.tensor(tumor_size_mean, dtype = float))
        self.tumor_size_std = nn.Parameter(torch.tensor(tumor_size_std, dtype = float))
        self.tumor_size_mean.requires_grad = False
        self.tumor_size_std.requires_grad = False
        self.age_mean = nn.Parameter(torch.tensor(age_mean, dtype = float))
        self.age_std = nn.Parameter(torch.tensor(age_std, dtype = float))
        self.age_mean.requires_grad = False
        self.age_std.requires_grad = False

    def forward(self, images: torch.Tensor, coords: torch.Tensor, return_embed: bool = False) -> torch.Tensor:
        """
        Arguments:
        ----------
        images: torch.Tensor
            The input images with shape [N, L, D]
        coords: torch.Tensor
            The input coordinates with shape [N, L, 2]
        """
        # inputs: [N, L, D]
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        assert len(images.shape) == 3
        # forward GigaPath slide encoder
        img_enc = self.slide_encoder.forward(images, coords, all_layer_embed=True)
        img_enc = [img_enc[i] for i in self.feat_layer]
        img_enc = torch.cat(img_enc, dim=-1)
        # classifier
        h = img_enc.reshape([-1, img_enc.size(-1)])
        logits = self.classifier(h)
        if return_embed:
            return logits, h
        else:
            return logits


class ModulatedClassificationHead(nn.Module):
    """
    The classification head for the slide encoder

    Arguments:
    ----------
    input_dim: int
        The input dimension of the slide encoder
    latent_dim: int
        The latent dimension of the slide encoder
    feat_layer: str
        The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers
    n_classes: int
        The number of classes
    model_arch: str
        The architecture of the slide encoder
    pretrained: str
        The path to the pretrained slide encoder
    freeze: bool
        Whether to freeze the pretrained model
    """

    def __init__(
        self,
        input_dim,
        tile_input_dim,
        latent_dim,
        feat_layer,
        n_classes=2,
        model_arch="gigapath_slide_enc12l768d",
        pretrained="hf_hub:prov-gigapath/prov-gigapath",
        freeze=False,
        external_model_logits=False,
        **kwargs,
    ):
        super(ModulatedClassificationHead, self).__init__()

        # setup the slide encoder
        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim
        self.external_model_logits = external_model_logits
        if self.feat_dim != 768:
            model_arch = f"gigapath_slide_enc12l{self.feat_dim}d"
        self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)

        # whether to freeze the pretrained model
        if freeze:
            print("Freezing Pretrained GigaPath model")
            for name, param in self.slide_encoder.named_parameters():
                param.requires_grad = False
            print("Done")
        # setup the classifier
        self.classifier = nn.Sequential(*[nn.Linear(self.feat_dim, n_classes)])
        self.tile_classifier = nn.Sequential(
            nn.Linear(tile_input_dim, tile_input_dim // 2), 
            nn.GELU(),
            nn.Linear(tile_input_dim // 2, tile_input_dim // 4), 
            nn.GELU(),
            nn.Linear(tile_input_dim // 4, n_classes), 
        )
        # FiLM layer
        if not external_model_logits:
            self.FiLM = nn.Linear(2, input_dim * 2) # tile logit and cancer prob as bias and scale
        else:
            self.FiLM = nn.Linear(3, input_dim * 2) # tile logit and cancer prob as bias and scale


    def forward(self, images: torch.Tensor, coords: torch.Tensor, tile_scores: torch.Tensor = None, tile_cancer_probs: torch.Tensor = None, score_as_scale: bool = True, 
                film: bool = False, weighted_film: bool = False, external_model_scores: torch.Tensor = None, predict_tile_logits: bool = False,
                tile_classifier: bool = False, return_embed: bool = False) -> torch.Tensor:
        """
        Arguments:
        ----------
        images: torch.Tensor
            The input images with shape [N, L, D]
        coords: torch.Tensor
            The input coordinates with shape [N, L, 2]
        """
        # inputs: [N, L, D]
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        assert len(images.shape) == 3

        if not predict_tile_logits:
            if weighted_film:
                cond = torch.cat([(tile_scores / 3).view(-1, 1), tile_cancer_probs.view(-1, 1)], dim=-1)  # [N, 2]
                print(f'cond.shape = {cond.shape}')
                features, weights = self.FiLM(cond).chunk(2, dim=-1)    # each [N, 1536]
                weights = torch.sigmoid(weights)  # Convert to [0, 1]
                images = (1 - weights) * images + weights * features.view(1, images.shape[1], -1)  # Apply weights to images
            else:
                if not film:
                    scale = tile_scores / 3 if score_as_scale else tile_cancer_probs
                    bias = tile_cancer_probs if score_as_scale else (tile_scores / 3)
                    scale = scale.view(1, images.shape[1], -1)
                    bias = bias.view(1, images.shape[1], -1)
                else:
                    if not self.external_model_logits:
                        cond = torch.cat([(tile_scores / 3).view(-1, 1), tile_cancer_probs.view(-1, 1)], dim=-1)  # [N, 2]
                        # cond = torch.cat([(tile_scores).view(-1, 1), tile_cancer_probs.view(-1, 1)], dim=-1)  # [N, 2]
                        print(f'cond.shape = {cond.shape}')
                        scale, bias = self.FiLM(cond).chunk(2, dim=-1)    # each [N, 1536]
                    else:  # external model logits are also included in the FiLM input
                        cond = torch.cat([(tile_scores / 3).view(-1, 1), tile_cancer_probs.view(-1, 1), external_model_scores.view(-1, 1)], dim=-1)  # [N, 3]
                        print(f'cond.shape = {cond.shape}')
                        scale, bias = self.FiLM(cond).chunk(2, dim=-1)    # each [N, 1536]

                print(f"images.shape = {images.shape}, tile_scores.shape = {tile_scores.shape}, tile_cancer_probs.shape = {tile_cancer_probs.shape}")
                images = images * scale + bias  # Apply scaling and bias to images
        elif tile_classifier:
            tile_logits = self.tile_classifier(images)
            return tile_logits
        
        # forward GigaPath slide encoder
        img_enc = self.slide_encoder.forward(images, coords, all_layer_embed=True)
        img_enc = [img_enc[i] for i in self.feat_layer]
        img_enc = torch.cat(img_enc, dim=-1)
        # classifier
        h = img_enc.reshape([-1, img_enc.size(-1)])
        logits = self.classifier(h)
        if return_embed:
            return logits, h
        else:
            return logits


class PairedClassificationHead(nn.Module):
    """
    The classification head for the slide encoder

    Arguments:
    ----------
    input_dim: int
        The input dimension of the slide encoder
    latent_dim: int
        The latent dimension of the slide encoder
    feat_layer: str
        The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers
    n_classes: int
        The number of classes
    model_arch: str
        The architecture of the slide encoder
    pretrained: str
        The path to the pretrained slide encoder
    freeze: bool
        Whether to freeze the pretrained model
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        feat_layer,
        n_classes=2,
        model_arch="gigapath_slide_enc12l768d",
        pretrained="hf_hub:prov-gigapath/prov-gigapath",
        freeze=False,
        **kwargs,
    ):
        super(PairedClassificationHead, self).__init__()

        # setup the slide encoder
        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim
        if self.feat_dim != 768:
            model_arch = f"gigapath_slide_enc12l{self.feat_dim}d"
        self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)
        self.ihc_slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)

        # whether to freeze the pretrained model
        if freeze:
            print("Freezing Pretrained GigaPath model")
            for name, param in self.slide_encoder.named_parameters():
                param.requires_grad = False
            for name, param in self.ihc_slide_encoder.named_parameters():
                param.requires_grad = False
            print("Done")

        # Learnable vectors to distinguish H&E from IHC
        self.he_type_embed = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.ihc_type_embed = nn.Parameter(torch.zeros(1, 1, input_dim))
        
        # Initialize with small random values
        nn.init.normal_(self.he_type_embed, std=0.02)
        nn.init.normal_(self.ihc_type_embed, std=0.02)

        # setup the classifier
        self.classifier = nn.Sequential(*[nn.Linear(self.feat_dim, n_classes)])
        self.double_classifier = nn.Sequential(*[nn.Linear(self.feat_dim * 2, n_classes)])
        # FiLM layer
        self.FiLM = nn.Linear(input_dim * 2, input_dim) # H&E weight vs IHC weight
        self.cancer_FiLM = nn.Linear(1, input_dim * 2)


    def forward(self, images: torch.Tensor, coords: torch.Tensor, ihc_images: torch.Tensor = None, ihc_coords: torch.Tensor = None, tile_cancer_probs: torch.Tensor = None, 
                return_embed: bool = False, modality_weights: bool = False, modality_bias: bool = False, concat_h: bool = False, cancer_weights: bool = False,
                cancer_scale: bool = False) -> torch.Tensor:
        """
        Arguments:
        ----------
        images: torch.Tensor
            The input images with shape [N, L, D]
        coords: torch.Tensor
            The input coordinates with shape [N, L, 2]
        """
        # inputs: [N, L, D]
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        assert len(images.shape) == 3

        if modality_weights:
            cond = torch.cat([images, ihc_images], dim=-1)  # [N, L, 2 * D]
            print(f'cond.shape = {cond.shape}')
            he_weights = self.FiLM(cond)  # each [N, 1536]
            he_weights = torch.sigmoid(he_weights)  # Convert to [0, 1]

            print(f"he_weights.shape = {he_weights.shape}")
            images = images * he_weights + (1 - he_weights) * ihc_images  # Apply weights to images

            if cancer_weights:
                cancer_cond = tile_cancer_probs.view(-1, 1)  # [N, 1]
                scale, bias = self.cancer_FiLM(cancer_cond).chunk(2, dim=-1)    # each [N, 1536]
                images = images * scale.view(1, images.shape[1], -1) + bias.view(1, images.shape[1], -1)  # Apply scaling and bias to images
            elif cancer_scale:
                images = images * tile_cancer_probs.view(1, -1, 1)  # Scale images by cancer probabilities
        elif modality_bias:
            # 1. Add Modality Embeddings (Broadcasting handles the batch/sequence dims)
            images = images + self.he_type_embed
            ihc_images = ihc_images + self.ihc_type_embed
            
            # 2. Concatenate along the 'num_tiles' dimension (dim=1)
            # Resulting shape: (batch_size, num_tiles_he + num_tiles_ihc, feat_dim)
            images = torch.cat([images, ihc_images], dim=-2)
            coords = torch.cat([coords, ihc_coords], dim=-2)
        
        # forward GigaPath slide encoder
        img_enc = self.slide_encoder.forward(images, coords, all_layer_embed=True)
        img_enc = [img_enc[i] for i in self.feat_layer]
        img_enc = torch.cat(img_enc, dim=-1)
        # classifier
        h = img_enc.reshape([-1, img_enc.size(-1)])

        if concat_h:
            ihc_img_enc = self.ihc_slide_encoder.forward(ihc_images, ihc_coords, all_layer_embed=True)
            ihc_img_enc = [ihc_img_enc[i] for i in self.feat_layer]
            ihc_img_enc = torch.cat(ihc_img_enc, dim=-1)
            ihc_h = ihc_img_enc.reshape([-1, ihc_img_enc.size(-1)])
            h = torch.cat([h, ihc_h], dim=-1)
            logits = self.double_classifier(h)
        else:
            logits = self.classifier(h)

        if return_embed:
            return logits, h
        else:
            return logits



class TileClassificationHead(nn.Module):
    """
    The classification head for the tile encoder

    Arguments:
    ----------
    input_dim: int
        The input dimension of the slide encoder
    latent_dim: int
        The latent dimension of the slide encoder
    feat_layer: str
        The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers
    n_classes: int
        The number of classes
    """

    def __init__(
        self,
        latent_dim,
        feat_layer,
        input_dim,
        tile_input_dim,
        n_classes=2,
        model_arch="gigapath_slide_enc12l768d",
        pretrained="hf_hub:prov-gigapath/prov-gigapath",
        dropout=0.2,
        mean=0,
        std=0,
        **kwargs,
    ):
        super(TileClassificationHead, self).__init__()

        # setup the slide encoder
        # self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        # self.feat_dim = len(self.feat_layer) * latent_dim
        # if self.feat_dim != 768:
        #     model_arch = f"gigapath_slide_enc12l{self.feat_dim}d"
        # self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)

        # setup the tile classifier
        # self.classifier = nn.Sequential(*[nn.Linear(latent_dim, n_classes)])

        self.tile_classifier = nn.Sequential(
            nn.Linear(tile_input_dim, tile_input_dim // 2), # LocalRef3
            # nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(tile_input_dim // 2, tile_input_dim // 4), # LocalRef3
            # nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(tile_input_dim // 4, n_classes), # LocalRef3
            # nn.Linear(input_dim, n_classes),
        )

        self.tile_log_var_regress = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, n_classes),            
        )

        # init_weight_tensor = torch.tensor([[0, 1, 2, 3]], device='cuda:0')  
        # init_bias_tensor = torch.tensor([0.5], device='cuda:0')                      

        self.classifier = nn.Sequential(
            nn.Linear(4, 4),
            nn.GELU(),
            nn.Linear(4, 4),
            nn.GELU(),
            nn.Linear(4, n_classes)
        )
        # with torch.no_grad():
        #     self.classifier[0].weight.copy_(init_weight_tensor)
        #     self.classifier[0].bias.copy_(init_bias_tensor)

        #save normalization parameters for use in inference
        self.mean = nn.Parameter(torch.tensor(mean, dtype = float))
        self.std = nn.Parameter(torch.tensor(std, dtype = float))
        self.mean.requires_grad = False
        self.std.requires_grad = False

    def freeze_regressors(self):
        """Freeze the regressor by setting requires_grad to False."""
        print("Freezing the regressor")
        # for param in self.HE_regressor.parameters():
        #     param.requires_grad = False
        for param in self.IHC_regressor.parameters():
            param.requires_grad = False
        print("Regressors frozen")

    def soft_binning(self, x, bins, sharpness=10.0):
        """
        Differentiable binning using sigmoids.
        Returns a tensor of shape (x.shape[0], len(bins)-1)
        """
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
        widths = [(bins[i+1] - bins[i]) / 2 for i in range(len(bins)-1)]
        
        bin_scores = []
        for c, w in zip(bin_centers, widths):
            score = torch.sigmoid(sharpness * (x - (c - w))) * (1 - torch.sigmoid(sharpness * (x - (c + w))))
            bin_scores.append(score)
        
        return torch.stack(bin_scores, dim=1)

    def hard_binning(self, probs, bins):
        bin_pairs = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
        counts = torch.tensor([
            torch.sum((probs >= lower) & (probs < upper))
            for lower, upper in bin_pairs
        ])
        ratios = counts / len(probs)
        return ratios
        

    def forward(self, images: torch.Tensor, coords: torch.Tensor, return_embed: bool = False, 
                regress_HE: bool = False, regress_IHC: bool = False, return_regress: bool = False, return_conf: bool = False) -> torch.Tensor:
        """
        Arguments:
        ----------
        images: torch.Tensor
            The input images with shape [N, L, D]
        """
        tile_logits = None
        logits = None
        weights = None
        # inputs: [N, L, D]
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        assert len(images.shape) == 3
        
        # if regress_HE:
        #     regressor_output = self.HE_regressor(images)
        # elif regress_IHC:
        #     regressor_output = self.IHC_regressor(images)
        # # images = images + regressor_output  # Add regressor output to images
        # images = regressor_output

        # forward GigaPath slide encoder
        # img_enc = self.slide_encoder.forward(images, coords, all_layer_embed=True)
        # img_enc = [img_enc[i] for i in self.feat_layer]
        # img_enc = torch.cat(img_enc, dim=-1)
        # # classifier
        # h = img_enc.reshape([-1, img_enc.size(-1)])
        # tile_logits = self.tile_classifier(images)
        tile_logits = self.tile_classifier(images)
        # tile_log_vars = self.tile_log_var_regress(images)
        tile_logits = tile_logits.view(images.size(0), -1)
        # tile_log_vars = tile_log_vars.view(images.size(0), -1)

        tile_logits_flat = tile_logits.view(-1)
        bins = [0.0, 0.5, 1.0, 1.5, 2.0] # regression bins
        # soft_bins = self.soft_binning(tile_logits_flat, bins)  # (num_tiles, 4)
        # soft_bins = self.soft_binning(softmax_logits, bins)  # (num_tiles, 4)

        # softmax_logits = F.softmax(tile_logits, dim=1)[:, -1]
        # print(f"softmax_logits = {softmax_logits}")
        # bins = [0.0, 0.25, 0.5, 0.75, 1.01] # classification bins
        

        # # Normalize per slide (aggregate tile info)
        # ratios = soft_bins.sum(dim=0)  # sum across tiles → shape (4,)
        # ratios = ratios / (ratios.sum() + 1e-8)  # normalize

        # ratios = self.hard_binning(softmax_logits, bins).to(images.device)
        ratios = self.hard_binning(tile_logits_flat, bins).to(images.device)
        print(f"ratios = {ratios}, avg_tile_logits = {tile_logits.mean(dim=-1)}")

        logits = self.classifier(ratios)

        
        if return_embed:
            # if not return_regress:
            if not return_conf:
                return logits, tile_logits, images
            else:
                # return logits, tile_logits, regressor_output, images
                return logits, tile_logits, tile_log_vars, images
        else:
            # if not return_regress:
            if not return_conf:
                return logits, tile_logits
            else:
                # return logits, tile_logits, regressor_output
                return logits, tile_logits, tile_log_vars


# Linear regression from slide scores and clinical features to slide label
class LinearClassificationHead(nn.Module):
    def __init__(self, input_dim, n_classes=2, **kwargs):
        super(LinearClassificationHead, self).__init__()
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, slide_score, clinical_features):
        x = torch.cat([slide_score, clinical_features], dim=0).float()
        logits = self.classifier(x)
        return logits



def get_model(**kwargs):
    if kwargs["train_on_y"] and not kwargs["pred_y_baseline"]:
        model = TileClassificationHead(**kwargs)
        print(f'model = TileClassificationHead')
    elif kwargs["score_can_as_sb"] or kwargs["score_can_as_bs"] or kwargs["film"] or kwargs["weighted_film"]:
        model = ModulatedClassificationHead(**kwargs)
        tile_model = TileClassificationHead(**kwargs)
        print(f'model = ModulatedClassificationHead')
        if kwargs["load_tile_logits"] or kwargs["tile_slide_model"]:
            return model
        return tile_model, model
    elif (kwargs["paired_training_mw"] and not kwargs["malig_baseline"]) or kwargs["paired_training_mb"] or kwargs["paired_training_cat_h"]:
        model = PairedClassificationHead(**kwargs)
        print(f'model = PairedClassificationHead')
    elif kwargs["clinical_features"]:
        model = LinearClassificationHead(**kwargs)
        print(f'model = LinearClassificationHead')
    else:
        model = ClassificationHead(**kwargs)
        
    return model
