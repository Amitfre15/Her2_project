import torch

from torch import nn
import torch.nn.functional as F
from . import slide_encoder


def reshape_input(imgs, coords, pad_mask=None):
    if len(imgs.shape) == 4:
        imgs = imgs.squeeze(0)
    if len(coords.shape) == 4:
        coords = coords.squeeze(0)
    if pad_mask is not None:
        if len(pad_mask.shape) != 2:
            pad_mask = pad_mask.squeeze(0)
    return imgs, coords, pad_mask

def create_reducing_sequential(input_dim, num_layers, reduction):
    assert reduction > 1, "Reduction rate must be greater than 1"
    
    layers = []
    current_dim = input_dim
    
    for _ in range(num_layers):
        next_dim = max(1, current_dim // reduction)  # Ensure dimension doesn’t go below 1
        layers.append(nn.Linear(current_dim, next_dim))
        layers.append(nn.LayerNorm(next_dim))
        layers.append(nn.GELU())  # Optional activation
        current_dim = next_dim
    
    return nn.Sequential(*layers)


class TransformerStyleProjector(nn.Module):
    def __init__(self, input_dim=1536, output_dim=24, num_heads=4, ffn_dim=768, hidden_dim=None):
        super().__init__()
        self.num_heads = num_heads
        if hidden_dim is None:
            hidden_dim = input_dim
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # LayerNorms
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        # self.ln3 = nn.LayerNorm(output_dim)
        self.ln3 = nn.BatchNorm1d(1)

        # Q, K, V projections
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)

        # Residual projection if needed
        self.residual_proj1 = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # FFN: two-layer MLP with GELU
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim)
        )

        # Residual projection for FFN (identity here since dims match)
        self.residual_proj2 = nn.Identity()

        # Final output projection
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, E_B):
        """
        E_B: [N, input_dim]
        Returns: [N, output_dim]
        """
        N = E_B.size(-2)

        # ---- Attention block ----
        x = self.ln1(E_B)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # [N, num_heads, head_dim] -> [num_heads, N, head_dim]
        Q = Q.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(N, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [H, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [H, N, N]
        attn_output = torch.matmul(attn_weights, V)  # [H, N, head_dim]
        # print(f"attn_weights = {attn_weights}, attn_output = {attn_output}")

        # Merge heads: [N, hidden_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(N, -1)

        # Residual connection
        x1 = attn_output + self.residual_proj1(E_B).squeeze(0)  # [N, hidden_dim]
        # print(f"x1 = {x1}")

        # ---- FFN block ----
        x2 = self.ln2(x1)
        ffn_out = self.ffn(x2)
        # print(f"ffn_out = {ffn_out}")

        # Final residual
        x3 = ffn_out + x1  # [N, hidden_dim]
        # print(f"x3 = {x3}")

        # ---- Output projection ----
        x4 = self.out_proj(x3).unsqueeze(0).squeeze(-1)  # [N, output_dim]
        # print(f"x4 = {x4}")
        # x4 = self.ln3(x4)  # Apply LayerNorm to the output
        # print(f"x4 = {x4}")
        return x4

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


class CompressedClassificationHead(nn.Module):
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
        reduction=1,
        comp_power=1,
        model_arch="gigapath_slide_enc12l768d",
        pretrained="hf_hub:prov-gigapath/prov-gigapath",
        freeze=False,
        x_dim=0,
        mean=0,
        std=0,
        **kwargs,
    ):
        super(CompressedClassificationHead, self).__init__()

        # setup the slide encoder
        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim
        if self.feat_dim != 768:
            model_arch = f"gigapath_slide_enc12l{self.feat_dim}d"
        in_chans = input_dim // (reduction ** comp_power) if not kwargs.get('x_as_sb', False) else input_dim
        self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=in_chans, **kwargs)

        # whether to freeze the pretrained model
        if freeze:
            print("Freezing Pretrained GigaPath model")
            for name, param in self.slide_encoder.named_parameters():
                param.requires_grad = False
            print("Done")
        # setup the classifier
        self.classifier = nn.Sequential(*[nn.Linear(self.feat_dim, n_classes)])

        # setup the compressor
        # self.compressor = create_reducing_sequential(input_dim=input_dim, num_layers=comp_power, reduction=reduction)
        self.compressor = TransformerStyleProjector(
            input_dim=input_dim,
            output_dim=input_dim // (reduction ** comp_power),
        )

        num_heads = 3 if comp_power == 8 else 4 # x_dim = 6 % 3 = 0
        self.x_scale = TransformerStyleProjector(
            input_dim=input_dim // (reduction ** comp_power),
            output_dim=input_dim,
            num_heads=num_heads,
        )
        self.x_bias = TransformerStyleProjector(
            input_dim=input_dim // (reduction ** comp_power),
            output_dim=input_dim,
            num_heads=num_heads,
        )

        # setup the HE_compressor
        self.HE_compressor = nn.Sequential(*[nn.Linear(input_dim, input_dim - x_dim),
                                             nn.LayerNorm(input_dim - x_dim)])

        self.mean_head = TransformerStyleProjector(input_dim=input_dim, output_dim=input_dim)
        self.std_head = TransformerStyleProjector(input_dim=input_dim, output_dim=input_dim)
        
        #save normalization parameters for use in inference
        self.mean = nn.Parameter(torch.tensor(mean, dtype = float))
        self.std = nn.Parameter(torch.tensor(std, dtype = float))
        self.mean.requires_grad = False
        self.std.requires_grad = False

    def forward(self, images: torch.Tensor, coords: torch.Tensor, x: torch.Tensor = None, y: torch.Tensor = None, 
                return_embed: bool = False, return_images: bool = False, trim_images: bool = False, x_as_sb: bool = False,
                pred_mean_std: bool = False) -> torch.Tensor:
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

        if x is not None:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            
            if x_as_sb:
                scale = self.x_scale(x)
                bias = self.x_bias(x)
                images = images * scale + bias  # Apply scaling and bias to images
            else:
                if not trim_images:
                    images = self.HE_compressor(images)  # Apply HE compression if x is provided
                else:
                    images = images[:, :, :-x.shape[-1]]  # Trim the last x.shape[-1] features if trim_images is True
                print(f'image.shape before cat x: {images.shape}')
                
                images = torch.cat([images, x], dim=-1)
        elif y is not None:
            if len(y.shape) == 2:
                y = y.unsqueeze(0)

            if not trim_images:
                images = self.HE_compressor(images)  # Apply HE compression if y is provided
            else:
                images = images[:, :, :-1]  # Trim the last feature if trim_images is True
            print(f'images.shape before cat y: {images.shape}')

            # Concatenate y to images
            images = torch.cat([images, y], dim=-1)
        elif pred_mean_std:
            mean_HE, std_HE = self.compute_style_stats(images)  # [1, 1536]
            # Apply AdaIN normalization
            mean_pred = self.mean_head(mean_HE)  # [1, 1536]
            std_pred = self.std_head(std_HE)    # [1, 1536]
            if mean_pred.shape != mean_HE.shape:
                mean_pred = mean_pred.expand_as(mean_HE)
                std_pred = std_pred.expand_as(std_HE)
            print(f'mean_HE.shape = {mean_HE.shape}, std_HE.shape = {std_HE.shape}, mean_pred.shape = {mean_pred.shape}, std_pred.shape = {std_pred.shape}')
            # images_adain = std_pred * (images - mean_HE) / std_HE + mean_pred
            # images = images_adain
        else:
            # compress the input images
            images = self.compressor(images)
        # forward GigaPath slide encoder
        img_enc = self.slide_encoder.forward(images, coords, all_layer_embed=True)
        img_enc = [img_enc[i] for i in self.feat_layer]
        img_enc = torch.cat(img_enc, dim=-1)
        # classifier
        h = img_enc.reshape([-1, img_enc.size(-1)])
        logits = self.classifier(h)
        if return_embed:
            return logits, h
        elif return_images:
            return logits, images
        elif pred_mean_std:
            return logits, mean_pred, std_pred
        else:
            return logits

    
    def compute_style_stats(self, emb: torch.Tensor) -> tuple:
        mean = emb.mean(dim=-2, keepdim=True)
        std = emb.std(dim=-2, keepdim=True, unbiased=False)
        return mean, std



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
        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim
        if self.feat_dim != 768:
            model_arch = f"gigapath_slide_enc12l{self.feat_dim}d"
        self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)

        # setup the tile classifier
        # self.classifier = nn.Sequential(*[nn.Linear(latent_dim, n_classes)])
        print(f'dropout = {dropout}')
        # setup the attention network
        # self.attention = nn.Sequential(*[nn.Linear(latent_dim, 1)])
        # self.HE_attention = nn.Sequential(*[nn.Linear(latent_dim + n_classes, 1)])
        # self.IHC_attention = nn.Sequential(*[nn.Linear(latent_dim + n_classes, 1)])

        # self.HE_attention = nn.Sequential(*[nn.Linear(latent_dim // 8 + n_classes, 1)])
        # self.IHC_attention = nn.Sequential(*[nn.Linear(latent_dim // 8 + n_classes, 1)])
        self.HE_attention = TransformerStyleProjector(input_dim=input_dim // (2 ** 7) + 1, output_dim=1, hidden_dim=latent_dim)

        # self.attention = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim // 2),
        #     nn.LayerNorm(latent_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),

        #     nn.Linear(latent_dim // 2, latent_dim // 4),
        #     nn.LayerNorm(latent_dim // 4),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),

        #     nn.Linear(latent_dim // 4, 1)  # Outputs a single attention weight
        # )

        self.tile_classifier = nn.Sequential(
            # nn.Linear(latent_dim, latent_dim // 2),
            nn.Linear(input_dim, input_dim // 2),
            # nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.Linear(input_dim // 2, input_dim // 4),
            # nn.LayerNorm(latent_dim // 4),
            nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(latent_dim // 4, n_classes),
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

        self.HE_regressor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.LayerNorm(latent_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 4, latent_dim // 8),
            # nn.LayerNorm(latent_dim // 8),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(latent_dim // 8, latent_dim // 16),
            # nn.LayerNorm(latent_dim // 16),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(latent_dim // 16, latent_dim // 32),
        )

        self.IHC_regressor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.LayerNorm(latent_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 4, latent_dim // 8),
            # nn.LayerNorm(latent_dim // 8),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(latent_dim // 8, latent_dim // 16),
            # nn.LayerNorm(latent_dim // 16),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(latent_dim // 16, latent_dim // 32),
        )

        # self.regressor = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim * 2),
        #     nn.LayerNorm(latent_dim * 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(latent_dim * 2, latent_dim * 4),
        #     nn.LayerNorm(latent_dim * 4),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(latent_dim * 4, latent_dim * 2),
        #     nn.LayerNorm(latent_dim * 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(latent_dim * 2, latent_dim),
        #     nn.LayerNorm(latent_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(latent_dim, latent_dim),
        # )

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
                regress_HE: bool = False, regress_IHC: bool = False, return_regress: bool = False) -> torch.Tensor:
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
        tile_logits = self.tile_classifier(images)
        tile_logits = tile_logits.view(images.size(0), -1)

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
        print(f"ratios = {ratios}")

        logits = self.classifier(ratios)

        # Compute the weighted sum of tile logits
        # tile_logits = self.classifier(images)  # Shape: [N, L, n_classes]
        # images = images.squeeze(1)
        # images = self.compressor(images)  # Compress the input images
        # # print(f"images.shape after compressor = {images.shape}")
        # images = images.squeeze(0)
        # combined_features = torch.cat([images, tile_logits], dim=-1)  # Shape: [N, L, D + n_classes]
        # # if regress_HE:
        # #     attention_scores = self.HE_attention(combined_features).squeeze(-1)
        # # elif regress_IHC:
        # attention_scores = self.HE_attention(combined_features).squeeze(-1)

        # # # TODO: take into account score when calculating attention
        # # attention_scores = self.attention(images).squeeze(-1)  # Shape: [N, L]

        # # Apply softmax to normalize scores and ensure they sum to 1
        # weights = F.softmax(attention_scores, dim=1)  # Shape: [N, L]

        # if weights is not None and tile_logits is not None:
        #     logits = torch.sum(weights.unsqueeze(-1) * tile_logits, dim=1)  # Shape: [N, n_classes]
        #     print(f"weights = {weights}")
        #     logits = (weights * tile_logits).sum() / weights.sum()
        #     logits = logits.unsqueeze(0).unsqueeze(0)
        #     tile_logits = tile_logits.unsqueeze(0)
        
        if return_embed:
            if not return_regress:
                return logits, tile_logits, images
            else:
                return logits, tile_logits, regressor_output, images
        else:
            if not return_regress:
                return logits, tile_logits
            else:
                return logits, tile_logits, regressor_output


class StudentTileClassificationHead(nn.Module):
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
        **kwargs,
    ):
        super(StudentTileClassificationHead, self).__init__()

        # setup the slide encoder
        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim
        if self.feat_dim != 768:
            model_arch = f"gigapath_slide_enc12l{self.feat_dim}d"
        self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)
        # Include an extra regressor
        self.regressor = nn.Sequential(
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 1536),
        )
        self.merger = nn.Sequential(
            nn.Linear(3072, 1536),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 1536),
        )

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

        regressor_output = self.regressor(images)
        # concat_images = torch.cat((images, regressor_output), dim=2)
        # print(f"images.shape = {images.shape}, concat_images.shape = {concat_images.shape}")
        # merged_images = self.merger(concat_images)

        # forward GigaPath slide encoder
        # img_enc = self.slide_encoder.forward(merged_images, coords, all_layer_embed=True)
        img_enc = self.slide_encoder.forward(regressor_output, coords, all_layer_embed=True)
        img_enc = [img_enc[i] for i in self.feat_layer]
        img_enc = torch.cat(img_enc, dim=-1)
        # classifier
        h = img_enc.reshape([-1, img_enc.size(-1)])
        
        logits = self.classifier(h)
        regressor_output = regressor_output.reshape([-1, images.size(-1)])
        if return_embed:
            return logits, regressor_output, h
        else:
            return logits, regressor_output


class WindowClassificationHead(nn.Module):
    """
    The classification head for the window encoder

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
        **kwargs,
    ):
        super(WindowClassificationHead, self).__init__()

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
        self.classifier = nn.Sequential(*[#nn.Linear(self.feat_dim, 1),
            nn.Linear(self.feat_dim, self.feat_dim * 2),
            nn.LayerNorm(self.feat_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim * 2, self.feat_dim * 4),
            nn.LayerNorm(self.feat_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim * 4, self.feat_dim * 2),
            nn.LayerNorm(self.feat_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim * 2, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.LayerNorm(self.feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim // 2, self.feat_dim // 4),
            nn.LayerNorm(self.feat_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim // 4, n_classes),
        ])
        # setup the attention network
        self.attention = nn.Sequential(*[nn.Linear(latent_dim, 1)])
        
        #save normalization parameters for use in inference
        self.mean = nn.Parameter(torch.tensor(mean, dtype = float))
        self.std = nn.Parameter(torch.tensor(std, dtype = float))
        self.mean.requires_grad = False
        self.std.requires_grad = False
        self.n_classes = n_classes

    def forward(self, windows: list[dict], return_embed: bool = False) -> torch.Tensor:
        """
        Arguments:
        ----------
        images: torch.Tensor
            The input images with shape [N, L, D]
        coords: torch.Tensor
            The input coordinates with shape [N, L, 2]
        """
        window_logits = None
        logits = None
        weights = None
        for imgs_coords_dict in windows:
            images, coords = imgs_coords_dict['imgs'], imgs_coords_dict['coords']
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
            # logits = self.classifier(h)
            if window_logits is None:
                window_logits = self.classifier(h)
            else:
                window_logits = torch.cat([window_logits, self.classifier(h)])
            # TODO: try taking into account position and score when calculating attention
            if weights is None:
                weights = self.attention(h)
            else:
                weights = torch.cat([weights, self.attention(h)])
        # if window_logits is not None:
        #     # TODO: try and learn temperature parameter
        #     weights = F.softmax(window_logits, dim=0)

        if weights is not None and window_logits is not None:
            # logits = (weights * window_logits).sum() / weights.sum()
            logits = (weights * window_logits).sum()
            logits = logits.unsqueeze(0).unsqueeze(0)
            window_logits = window_logits.unsqueeze(0)
            
        if return_embed:
            return logits, window_logits, h
        else:
            return logits, window_logits


class StudentClassificationHead(nn.Module):
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
        **kwargs,
    ):
        super(StudentClassificationHead, self).__init__()

        # setup the slide encoder
        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim
        if self.feat_dim != 768:
            model_arch = f"gigapath_slide_enc12l{self.feat_dim}d"
        self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)
        # Include an extra regressor
        # self.regressor = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.LayerNorm(768),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(768, 768),
        #     nn.LayerNorm(768),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(768, 768),
        #     nn.LayerNorm(768),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(768, 768),
        #     nn.LayerNorm(768),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(768, 768),
        # )
        self.regressor = nn.Sequential(nn.Linear(768, 768))

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
        regressor_output = self.regressor(h)
        # logits = self.classifier(h)
        logits = self.classifier(regressor_output)
        if return_embed:
            return logits, regressor_output, h
        else:
            return logits, regressor_output


class TileFeaturesRegressor(nn.Module):
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
        latent_dim=1536,
        **kwargs,
    ):
        super(TileFeaturesRegressor, self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 1536),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 1536),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 1536),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1536, 1536),
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
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

        regressor_output = self.regressor(images)
        regressor_output = regressor_output.reshape([-1, images.size(-1)])
        return regressor_output


def get_regressor(**kwargs):
    regressor = TileFeaturesRegressor(**kwargs)
        
    return regressor


def get_model(**kwargs):
    if kwargs["use_tile_classification"] or (kwargs["train_on_y"] and not kwargs["pred_y_baseline"]):
        model = TileClassificationHead(**kwargs)
        print(f'model = TileClassificationHead')
    elif kwargs["window_training"]:
        model = WindowClassificationHead(**kwargs)
        print(f'model = WindowClassificationHead')
    elif "student_net" in kwargs:
        model = StudentClassificationHead(**kwargs)
        print(f'model = StudentClassificationHead')
    elif kwargs["matching_tiles_training"]:
        model = StudentTileClassificationHead(**kwargs)
        print(f'model = StudentTileClassificationHead')
    elif kwargs["compress_features"]:
        model = CompressedClassificationHead(**kwargs)
        print(f'model = CompressedClassificationHead')
    else:
        model = ClassificationHead(**kwargs)
        
    return model
