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
        latent_dim=1536,
        n_classes=2,
        dropout=0.2,
        mean=0,
        std=0,
        **kwargs,
    ):
        super(TileClassificationHead, self).__init__()

        # setup the tile classifier
        self.classifier = nn.Sequential(*[nn.Linear(latent_dim, n_classes)])
        # setup the attention network
        self.attention = nn.Sequential(*[nn.Linear(latent_dim, 1)])

        # self.classifier = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim // 2),
        #     nn.LayerNorm(latent_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),

        #     nn.Linear(latent_dim // 2, latent_dim // 4),
        #     nn.LayerNorm(latent_dim // 4),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),

        #     nn.Linear(latent_dim // 4, n_classes)
        # )

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
        """
        tile_logits = None
        logits = None
        weights = None
        # inputs: [N, L, D]
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        assert len(images.shape) == 3
        # tile classifier
        for tile_num in range(images.size(1)):
            h = images[:, tile_num].view(1, -1)
            if tile_logits is None:
                tile_logits = self.classifier(h)
            else:
                tile_logits = torch.cat([tile_logits, self.classifier(h)])
            # TODO: try taking into account position and score when calculating attention
            if weights is None:
                weights = self.attention(h)
            else:
                weights = torch.cat([weights, self.attention(h)])
        if weights is not None and tile_logits is not None:
            # print(f"weights = {weights}")
            logits = (weights * tile_logits).sum() / weights.sum()
            logits = logits.unsqueeze(0).unsqueeze(0)
            tile_logits = tile_logits.unsqueeze(0)
        if return_embed:
            return logits, tile_logits, h
        else:
            return logits, tile_logits


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
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 1536),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(1536, 1536),
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
        self.classifier = nn.Sequential(*[nn.Linear(self.feat_dim, 1)])
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
            # if weights is None:
            #     weights = self.attention(h)
            # else:
            #     weights = torch.cat([weights, self.attention(h)])
        if window_logits is not None:
            # TODO: try and learn temperature parameter
            weights = F.softmax(window_logits, dim=0)

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
        # Include an extra regressor (in our case linear)
        self.regressor = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
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
        # forward GigaPath slide encoder
        img_enc = self.slide_encoder.forward(images, coords, all_layer_embed=True)
        img_enc = [img_enc[i] for i in self.feat_layer]
        img_enc = torch.cat(img_enc, dim=-1)
        # classifier
        h = img_enc.reshape([-1, img_enc.size(-1)])
        regressor_output = self.regressor(h)
        logits = self.classifier(h)
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
    if kwargs["use_tile_classification"]:
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
    else:
        model = ClassificationHead(**kwargs)
        
    return model
