import os
import h5py
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class SlideDatasetForTasks(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 root_path: str,
                 task_config: dict, 
                 slide_key: str='file',
                 label: str='onco_score_26',
                 dataset_name: list= ['TAILORx_1'],
                 folds: list=[2,3,4,5],
                 use_clinical_features: bool=False,
                 test_on_all: bool=False,
                 get_single_slide: str=None,
                 censoreship: str = None,
                 survival = False,
                 **kwargs
                 ):
        '''
        This class is used to set up the slide dataset for different tasks

        Arguments:
        ----------
        data_df: pd.DataFrame
            The dataframe that contains the slide data
        root_path: str
            The root path of the tile embeddings
        task_config: dict
            The task configuration dictionary
        slide_key: str
            The key that contains the slide id
        '''
        self.root_path = root_path
        self.slide_key = slide_key
        self.task_cfg = task_config
        self.label = label
        self.teacher_label = kwargs['teacher_label'] if 'teacher_label' in kwargs else None
        self.comp_power = kwargs['comp_power'] if 'comp_power' in kwargs else None
        self.cycle_gan_train = kwargs['cycle_gan_train'] if 'cycle_gan_train' in kwargs else None
        self.synth_ihc_train = kwargs['synth_ihc_train'] if 'synth_ihc_train' in kwargs else None
        self.val_fold = kwargs['val_fold'] if 'val_fold' in kwargs else None
        self.cnn_on_y = kwargs['cnn_on_y'] if 'cnn_on_y' in kwargs else None
        self.tile_y_pred_file_name = kwargs['tile_y_pred_file_name'] if 'tile_y_pred_file_name' in kwargs else None
        self.ext_tile_y_pred_file_name = kwargs['ext_tile_y_pred_file_name'] if 'ext_tile_y_pred_file_name' in kwargs else None
        self.use_clinical_features = use_clinical_features
        self.test_on_all = test_on_all
        self.censoreship = censoreship
        self.survival = survival
        self.folds = folds
        #TODO: update list of columns of clinical features
        # self.columns = ['er_status', 'pr_status', 'TumorSize', 'age']
        self.columns = ["Age","Grade","label_ER","label_PR","idc","ilc","muc","epc","ic","spc","pc","necb","lcis","dcis","meta","imc","ipc","cpc", "score"]
        # allowed_values = {'er_status': [0,1], 'pr_status':[0,1], 'RecChemo':[0,1]}
        
        self.batches = kwargs['batches'] if 'batches' in kwargs else None
        self.oversample = kwargs['oversample'] if 'oversample' in kwargs else None
        self.slide_path_key = kwargs['slide_path_key'] if 'slide_path_key' in kwargs else None
        self.window_training = kwargs['window_training'] if 'window_training' in kwargs else None
        self.use_tile_classification = kwargs['use_tile_classification'] if 'use_tile_classification' in kwargs else None
        self.y_and_tumor_from_map = kwargs['y_and_tumor_from_map'] if 'y_and_tumor_from_map' in kwargs else None
        try:
            data_df[slide_key] = data_df[slide_key].str.removesuffix('.svs')
            data_df[slide_key] = data_df[slide_key].str.removesuffix('.mrxs')
            data_df[slide_key] = data_df[slide_key].str.removesuffix('.tiff')
            data_df[slide_key] = data_df[slide_key].str.removesuffix('.tif')            
        except:
            pass
        if get_single_slide is None:
            folds_list = data_df['fold']
            folds_list = pd.to_numeric(folds_list, errors='coerce').fillna(folds_list)
            data_df['fold'] = folds_list
            label_list = data_df[label]
            label_list = pd.to_numeric(label_list, errors='coerce').fillna(label_list)
            data_df[label] = label_list
            if survival:
                censoreship_list = data_df[censoreship]
                censoreship_list = pd.to_numeric(censoreship_list, errors='coerce').fillna(censoreship_list)
                data_df[censoreship] = censoreship_list
                #filter by valid label
                data_df = data_df[data_df[censoreship].isin([0,1,0.0,1.0])]

            print(f"data_df.size before id filter: {data_df.shape}")
            #filter based on dataset
            data_df = data_df[data_df['id'].isin(dataset_name)]
            print(f"data_df.size after id filter: {data_df.shape}")
            #filter based on fold
            data_df = data_df[data_df['fold'].isin(folds)]
            print(f"data_df.size after fold filter: {data_df.shape}")
            #filter by valid label
            #data_df = data_df[data_df[label].isin([0,1,0.0,1.0])]
            if self.batches is not None and self.slide_path_key is not None:
                data_df = data_df[data_df[self.slide_path_key].str.contains('|'.join(self.batches), na=False)]
            if self.oversample:
                # print(f'self.oversample: {self.oversample}')
                # data_df = data_df[data_df[self.label] != 0.5]

                label_counts = data_df[self.label].value_counts()
                label_count_dict = label_counts.to_dict()
                norm_label_counts = data_df[self.label].value_counts(normalize=True).sort_index()
                print("Normalized label proportions:\n", norm_label_counts)

                # Step 2: Find the minimum count (for undersampling)
                min_count = label_counts.min()
                print(f'min_count: {min_count}')

                # # Step 3: oversample each class to have min_count samples
                label_count_dict[0] = int(label_count_dict[0] * 0.2)
                label_count_dict[1] = int(label_count_dict[1] * 2)
                # label_count_dict[2] = int(label_count_dict[2] * 0.5) # 0.25
                # label_count_dict[2] = int(label_count_dict[2] * 1) # 3.5
                # label_count_dict[3] = int(label_count_dict[3] * 1) # 3.5

                # Sample accordingly
                data_df = data_df.groupby(self.label, group_keys=False) \
                .apply(lambda x: x.sample(n=label_count_dict.get(x.name, len(x)), replace=True) if x.name in [0, 1, 2, 3] \
                       else x.sample(n=label_count_dict.get(x.name, len(x))))

                norm_label_counts = data_df[self.label].value_counts(normalize=True).sort_index()
                print("Normalized label proportions after oversampling:\n", norm_label_counts)

            if not test_on_all:
                data_df = data_df[pd.to_numeric(data_df[label], errors='coerce').notnull()]
        else:
            data_df = data_df[data_df[slide_key] == get_single_slide]
        print(f"data_df.size before valid slides filter: {data_df.shape}")
        # get slides that have tile encodings
        valid_slides = self.get_valid_slides(root_path, data_df[slide_key].values)
        # filter out slides that do not have tile encodings
        data_df = data_df[data_df[slide_key].isin(valid_slides)]
        print(f"data_df.size after valid slides filter: {data_df.shape}")
        #filter based on clinical features
        if self.use_clinical_features:
            for column in self.columns:
                column_values = data_df[column]
                column_values = pd.to_numeric(column_values, errors='coerce')
                if column.lower() == 'age':
                    column_values /= 100 # scale age to be between 0 and 1, since the age range is between 0 and 100
                data_df[column] = column_values
                data_df = data_df[data_df[column].notna()]
                
                # if column in allowed_values:
                #     data_df = data_df[data_df[column].isin(allowed_values[column])]
            
        print(f'data_df unique labels: {data_df[self.label].unique()}')
        # set up the task
        self.setup_data(data_df, task_config.get('setting', 'multi_class'))

        # TODO: set up the tile labels data
        
        self.max_tiles = task_config.get('max_tiles', 1000)
        self.shuffle_tiles = task_config.get('shuffle_tiles', False)
        print('Dataset has been initialized!')
        
    def get_valid_slides(self, root_path: str, slides: list) -> list:
        '''This function is used to get the slides that have tile encodings stored in the tile directory'''
        valid_slides = []
        for i in range(len(slides)):
            if 'pt_files' in root_path.split('/')[-1]:
                sld = slides[i].replace(".svs", "") + '.pt'
            elif 'gigapath_features' in root_path:
                sld = os.path.join(slides[i], "tile_embeds_"+slides[i]+".npy")
            else:
                sld = slides[i].replace(".svs", "") + '.h5'
            sld_path = os.path.join(root_path, sld)
            if not os.path.exists(sld_path):
                print('Missing: ', sld_path)
            else:
                valid_slides.append(slides[i])
        return valid_slides
    
    def setup_data(self, df: pd.DataFrame, task: str='multi_class'):
        '''Prepare the data for multi-class setting or multi-label setting'''
        # Prepare slide data
        if task == 'multi_class' or task == 'binary':
            prepare_data_func = self.prepare_multi_class_or_binary_data
        elif task == 'multi_label':
            prepare_data_func = self.prepare_multi_label_data
        elif task == 'continuous':
            prepare_data_func = self.prepare_continuous_data
        else:
            raise ValueError('Invalid task: {}'.format(task))
        if self.teacher_label is None: # Delete
            self.slide_data, self.images, self.labels, self.n_classes = prepare_data_func(df)
        else:
            self.slide_data, self.images, self.labels, self.n_classes, self.teacher_labels = prepare_data_func(df)
    
    def prepare_continuous_data(self, df: pd.DataFrame):
        '''Prepare the data for regression'''
        n_classes = 1
        
        images = df[self.slide_key].to_list()
        labels = df[[self.label]].to_numpy().astype(float)
        # add teacher labels # Delete
        teacher_labels = df[[self.teacher_label]].to_numpy().astype(float) if self.teacher_label is not None else None
        
        if self.teacher_label is None: # Delete
            return df, images, labels, n_classes
        else:
            return df, images, labels, n_classes, teacher_labels
    
    def prepare_multi_class_or_binary_data(self, df: pd.DataFrame):
        '''Prepare the data for multi-class classification'''
        # set up the label_dict
        label_dict = self.task_cfg.get('label_dict', {})
        assert  label_dict, 'No label_dict found in the task configuration'
        # set up the mappings
        assert self.label in df.columns, 'No label column found in the dataframe'
        df[self.label] = df[self.label].map(label_dict)
        # n_classes = len(label_dict)
        n_classes = len(set(label_dict.values()))
        
        images = df[self.slide_key].to_list()
        labels = df[[self.label]].to_numpy().astype(int)
        
        return df, images, labels, n_classes
        
    def prepare_multi_label_data(self, df: pd.DataFrame):
        '''Prepare the data for multi-label classification'''
        # set up the label_dict
        label_dict = self.task_cfg.get('label_dict', {})
        assert label_dict, 'No label_dict found in the task configuration'
        # Prepare mutation data
        label_keys = label_dict.keys()
        # sort key using values
        label_keys = sorted(label_keys, key=lambda x: label_dict[x])
        n_classes = len(label_dict)

        images = df[self.slide_key].to_list()
        labels = df[label_keys].to_numpy().astype(int)
            
        return df, images, labels, n_classes
    
    def update_clinical_features_params_and_handle_data(self, model):
        if self.use_clinical_features:
            if 'TumorSize' in self.columns:
                self.tumor_size_mean = model.tumor_size_mean.item()
                self.tumor_size_std = model.tumor_size_std.item()
                self.slide_data['TumorSize'] = (self.slide_data['TumorSize'] - self.tumor_size_mean)/self.tumor_size_std
            if 'age' in self.columns:
                self.age_mean = model.age_mean.item()
                self.age_std = model.age_std.item()
                self.slide_data['age'] = (self.slide_data['age'] - self.age_mean)/self.age_std
    
    
class SlideDataset(SlideDatasetForTasks):
    def __init__(self,
                 data_df: pd.DataFrame,
                 root_path: str,
                 task_config: dict,
                 slide_key: str='file',
                 label: str='onco_score_26',
                 dataset_name: list= ['TAILORx_1'],
                 folds: list=[1,2,3,4,5],
                 use_clinical_features: bool=False,
                 test_on_all: bool=False,
                 get_single_slide: str=None,
                 **kwargs
                 ):
        '''
        The slide dataset class for retrieving the slide data for different tasks

        Arguments:
        ----------
        data_df: pd.DataFrame
            The dataframe that contains the slide data
        root_path: str
            The root path of the tile embeddings
        task_config_path: dict
            The task configuration dictionary
        slide_key: str
            The key that contains the slide id
        '''
        super(SlideDataset, self).__init__(data_df, root_path, task_config, slide_key, label, dataset_name, folds, use_clinical_features, test_on_all, get_single_slide, **kwargs)

    def shuffle_data(self, images: torch.Tensor, coords: torch.Tensor) -> tuple:
        '''Shuffle the serialized images and coordinates'''
        indices = torch.randperm(len(images))
        images_ = images[indices]
        coords_ = coords[indices]
        return images_, coords_

    def read_assets_from_h5(self, h5_path: str) -> tuple:
        '''Read the assets from the h5 file'''
        assets = {}
        attrs = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                assets[key] = f[key][:]
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs
    
    def get_sld_name_from_path(self, sld: str) -> str:
        '''Get the slide name from the slide path'''
        sld_name = os.path.basename(sld).split('.h5')[0]
        return sld_name

    def load_tensor_from_npy(self, npy_path: str) -> torch.Tensor:
        try:
            tsr = np.load(npy_path) # load the numpy file
            tsr = torch.from_numpy(tsr) # convert to tensor
            return tsr
        except BaseException as e:
            # print(e)
            return None
    
    def get_images_from_path(self, img_path: str) -> dict:
        '''Get the images from the path'''
        if '.pt' in img_path:
            images = torch.load(img_path)
            coords = 0
        elif '.h5' in img_path:
            assets, _ = self.read_assets_from_h5(img_path)
            images = torch.from_numpy(assets['features'])
            coords = torch.from_numpy(assets['coords'])

            # if shuffle the data
            if self.shuffle_tiles:
                images, coords = self.shuffle_data(images, coords)

            if images.size(0) > self.max_tiles:
                images = images[:self.max_tiles, :]
            if coords.size(0) > self.max_tiles:
                coords = coords[:self.max_tiles, :]
        elif '.npy' in img_path:
            images = np.load(img_path) # load the numpy file
            images = torch.from_numpy(images) # convert to tensor  
            
            # We get the coordinates from file names in a sister dir
            # First we remove the .npy file name, like doing .. in cmd
            path = os.path.dirname(img_path).rsplit('gigapath_features', 1)
            try:
                coords_path = 'jpg_tiles'.join(path)

                # The file names contain the patch coordinates, so we list the coors_path dir
                coords_files = os.listdir(coords_path)
            except:
                #we want to support tiles being saved in png and in jpg
                coords_path = 'png_tiles'.join(path)
                coords_files = os.listdir(coords_path)
            # The coords files are named in the format <x coord>x_<y coord>y.jpg, so we split by x_ and y and convert to int
            coords = torch.tensor([[int(coord.split('x_')[0]), int(coord.split('x_')[1].split('y')[0])] for coord in coords_files])

            tile_y_file_name = f"tile_y_val{self.val_fold}.npy" if self.batches is not None else f"tile_y_ad_val{self.val_fold}.npy"
            tile_y_path = os.path.join(os.sep, "SSDStorage", "Breast", "Carmel", "Her2", "gigapath_IHC", f"tile_y{path[-1]}", tile_y_file_name)
            tile_y_pred_path = os.path.dirname(tile_y_path.replace('tile_y', 'tile_y_pred'))
            if os.path.exists(tile_y_pred_path) and self.tile_y_pred_file_name is not None:
                if "mpp2" in tile_y_pred_path:  # this file was used to apply film model with tile model applied on all mpp2 tiles 
                    y_pred_file_name = next(filter(lambda x: "all_he_tiles" in x, os.listdir(tile_y_pred_path)))
                else:
                    # tile_y_pred_path = os.path.dirname(tile_y_path.replace('tile_y_mpp', 'tile_y_from_y_map_mpp')) # Delete: for testing - y as tile logits
                    # y_pred_file_name = next(filter(lambda x: x.endswith(f"{self.val_fold}.npy"), os.listdir(tile_y_pred_path))) # Delete

                    # y_pred_file_name = next(filter(lambda x: x.startswith("tile_y_from_map"), os.listdir(tile_y_pred_path))) # revert
                    y_pred_file_name = next(filter(lambda x: x.startswith(f"{self.tile_y_pred_file_name}"), os.listdir(tile_y_pred_path))) # check with thufa/other models
                if self.ext_tile_y_pred_file_name is not None:
                    ext_tile_y_pred_file_name = next(filter(lambda x: x.startswith(f"{self.ext_tile_y_pred_file_name}") and x.endswith(f"{self.val_fold}.npy"), os.listdir(tile_y_pred_path)))
                    ext_tile_y_pred_path = os.path.join(tile_y_pred_path, ext_tile_y_pred_file_name)
                tile_y_pred_path = os.path.join(tile_y_pred_path, y_pred_file_name)
                
            if self.y_and_tumor_from_map:
                tile_y_path = tile_y_path.replace('tile_y_mpp', 'tile_y_from_y_map_mpp')
            
            ihc_images = None
            ihc_coords = None
            tiles = None
            ihc_tiles = None
            matching_tiles = None
            tumor_indices = None
            non_tumor_indices = None
            cancer_prob = None
            tile_y = None
            synth_ihc_imgs = None
            tile_logits = None
            external_tile_logits = None
            virchow2_feats = None
            uni2_feats = None
            conch_feats = None
            if "gigapath_CAT_features" in img_path or "gigapath_HE" in img_path or "gigapath_cancer" in img_path:
                try:
                    he_slide_dir = os.path.dirname(img_path)
                    he_slide = he_slide_dir.split('/')[-1]
                    
                    if "gigapath_CAT_features" in img_path:
                        ihc_imgs_path = he_slide_dir.replace("gigapath_CAT_features", "Carmel/Her2/gigapath_IHC").replace('HE', 'IHC')[:-1]
                        matching_tiles_dir = he_slide_dir.replace("gigapath_CAT_features", "Carmel/Her2/gigapath_IHC").replace('HE', 'IHC').replace('gigapath_features', 'matching_tiles')
                    else:
                        ihc_imgs_path = he_slide_dir.replace("gigapath_HE", "gigapath_IHC")[:-1]
                        matching_tiles_dir = he_slide_dir.replace("gigapath_HE", "gigapath_IHC").replace('gigapath_features', 'matching_tiles')
                    # print(f'matching_tiles_dir: {matching_tiles_dir}')
                    matching_tiles_file = os.path.join(matching_tiles_dir, 'ihc_tiles.npy')

                    block = ihc_imgs_path.split('/')[-1]
                    matching_slide = next(filter(lambda x: x.startswith(block), os.listdir(os.path.dirname(ihc_imgs_path))))
                    ihc_imgs_path = ihc_imgs_path.replace(block, matching_slide)
                    full_path = os.path.join(ihc_imgs_path, f"tile_embeds_{matching_slide}.npy")
                    ihc_coords_path = ihc_imgs_path.replace('gigapath_features', 'png_tiles')
                    ihc_coords_files = os.listdir(ihc_coords_path)
                    # The coords files are named in the format <x coord>x_<y coord>y.jpg, so we split by x_ and y and convert to int
                    ihc_coords = torch.tensor([[int(coord.split('x_')[0]), int(coord.split('x_')[1].split('y')[0])] for coord in ihc_coords_files])
                    if self.cycle_gan_train:
                        ihc_tiles = torch.stack([transform(Image.open(os.path.join(ihc_coords_path, tile)).convert('RGB')) for tile in ihc_coords_files])

                    cancer_prob_dir = he_slide_dir.replace('gigapath_features', 'cancer_probs_from_cancer_map') if self.y_and_tumor_from_map else he_slide_dir.replace('gigapath_features', 'cancer_probs')
                    cancer_prob_file = os.path.join(cancer_prob_dir, f'cancer_prob_val{self.val_fold}.npy')

                    # tumor_indices_dir = matching_tiles_dir.replace('matching_tiles', 'tumor_indices') if not self.y_and_tumor_from_map else matching_tiles_dir.replace('matching_tiles', 'tumor_indices_from_cancer_map')
                    tumor_indices_dir = matching_tiles_dir.replace('matching_tiles', 'tumor_indices_from_cancer_map')
                    if "gigapath_cancer" in img_path:
                        tumor_indices_dir = he_slide_dir.replace("gigapath_features", "tumor_indices")
                        print(f'tumor_indices_dir: {tumor_indices_dir}')
                        tumor_indices_file = os.path.join(tumor_indices_dir, 'tumor_ann_indices.npy')
                        non_tumor_indices_file = os.path.join(tumor_indices_dir, 'non_tumor_indices.npy')
                    else:
                        try:
                            tumor_ind_npy = next(filter(lambda x: 'ensemble' in x, os.listdir(tumor_indices_dir)))
                        except StopIteration:
                            tumor_ind_npy = next(filter(lambda x: f'tumor_indices{self.val_fold}' in x, os.listdir(tumor_indices_dir)))
                            # raise FileNotFoundError("No matching tumor indices file found")
                        tumor_indices_file = os.path.join(tumor_indices_dir, tumor_ind_npy)

                    virchow2_feats_pth = img_path.replace('gigapath_features', 'virchow2_features')
                    uni2_feats_pth = img_path.replace('gigapath_features', 'uni2_features')
                    conch_feats_pth = img_path.replace('gigapath_features', 'conch_features')

                    ihc_images = self.load_tensor_from_npy(full_path)
                    matching_tiles = self.load_tensor_from_npy(matching_tiles_file)
                    tile_y = self.load_tensor_from_npy(tile_y_path)
                    tumor_indices = self.load_tensor_from_npy(tumor_indices_file)
                    cancer_prob = self.load_tensor_from_npy(cancer_prob_file)
                    tile_logits = self.load_tensor_from_npy(tile_y_pred_path) if os.path.exists(tile_y_pred_path) else None
                    external_tile_logits = self.load_tensor_from_npy(ext_tile_y_pred_path) if os.path.exists(ext_tile_y_pred_path) else None
                    virchow2_feats = self.load_tensor_from_npy(virchow2_feats_pth) if os.path.exists(virchow2_feats_pth) else None
                    uni2_feats = self.load_tensor_from_npy(uni2_feats_pth) if os.path.exists(uni2_feats_pth) else None
                    conch_feats = self.load_tensor_from_npy(conch_feats_pth) if os.path.exists(conch_feats_pth) else None

                    valid_tile_y_indices = tile_y != -1
                    valid_matching_tiles = (~torch.isnan(matching_tiles)) & (matching_tiles < ihc_images.shape[0])

                    if self.cycle_gan_train or self.cnn_on_y:
                        # pick only valid tiles
                        tiles = [os.path.join(coords_path, tile) for i, tile in enumerate(coords_files) if valid_matching_tiles[i]]
                        tiles = [tiles[i] for i in range(len(tiles)) if valid_tile_y_indices[i]]
                        tile_y = tile_y[valid_tile_y_indices]
                        # pick random 100 tiles
                        if len(tiles) > 100:
                            indices = torch.randperm(len(tiles))[:100]
                            tiles = [tiles[i] for i in indices]
                            tile_y = tile_y[indices]

                        transform = transforms.Compose([transforms.ToTensor()])
                        tiles = torch.stack([transform(Image.open(tile).convert('RGB')) for tile in tiles])
                        print(f'Loaded {tiles.size(0)} tiles')
                        if tiles.size(0) == 0:
                            tiles = None
                    else:
                        tiles = None

                    if self.synth_ihc_train:
                        synth_ihc_imgs_path = img_path.replace('gigapath_features_mpp2', 'synth_ihc_gigapath_features_mpp2')
                        synth_ihc_imgs = self.load_tensor_from_npy(synth_ihc_imgs_path)
                    
                except BaseException as e:  
                    print(e)
                    pass
        
        # set the input dict
        data_dict = {'imgs': images,
                'ihc_imgs': ihc_images,
                'synth_ihc_imgs': synth_ihc_imgs,
                'img_lens': images.size(0),
                'pad_mask': 0,
                'coords': coords,
                'ihc_coords': ihc_coords,
                'matching_tiles': matching_tiles,
                'tumor_indices': tumor_indices,
                'non_tumor_indices': non_tumor_indices,
                'cancer_prob': cancer_prob,
                'tile_logits': tile_logits,
                'external_tile_logits': external_tile_logits,
                'tiles': tiles,
                'ihc_tiles': ihc_tiles,
                'virchow2_feats': virchow2_feats,
                'uni2_feats': uni2_feats,
                'conch_feats': conch_feats,
                'tile_y': tile_y}
        
        return data_dict
    
    def get_one_sample(self, idx: int) -> dict:
        '''Get one sample from the dataset'''
        # get the slide id
        slide_id = self.images[idx]
        # get the slide path
        if 'pt_files' in self.root_path.split('/')[-1]:
            slide_path = os.path.join(self.root_path, slide_id + '.pt')
        elif 'gigapath_features' in self.root_path.split('/')[-1]:
            slide_path = os.path.join(self.root_path, slide_id, "tile_embeds_"+slide_id+".npy")
        else:
            slide_path = os.path.join(self.root_path, slide_id + '.h5')
        
        # get the slide label
        label = torch.from_numpy(self.labels[idx])
        # add teacher labels
        teacher_label = torch.from_numpy(self.teacher_labels[idx]) if self.teacher_label is not None else None # Delete

        if self.use_clinical_features:
            clinical_features = torch.tensor(self.slide_data[self.slide_data[self.slide_key] == slide_id][self.columns].values).flatten()
            data_dict = {'clinical_feats': clinical_features[:-1], 'score': clinical_features[-1]}
            sample = {'clinical_feats': data_dict['clinical_feats'], 'score': data_dict['score'], 'slide_id': slide_id, 'labels': label}
            return sample
        else:
            # get the slide images
            data_dict = self.get_images_from_path(slide_path)
        
        if self.survival:
            censoreship = torch.tensor(self.slide_data[self.slide_data[self.slide_key] == slide_id][self.censoreship].values)
        else:
            censoreship = None
            

        # if self.use_tile_classification and self.oversample:
        #     self.shuffle_indices(data_dict, slide_id)
        
        # set the sample dict
        sample = {'imgs': data_dict['imgs'],
                  'ihc_imgs': data_dict['ihc_imgs'],
                  'synth_ihc_imgs': data_dict['synth_ihc_imgs'],
                  'img_lens': data_dict['img_lens'],
                  'pad_mask': data_dict['pad_mask'],
                  'coords': data_dict['coords'],
                  'ihc_coords': data_dict['ihc_coords'],
                  'matching_tiles': data_dict['matching_tiles'],
                  'tumor_indices': data_dict['tumor_indices'],
                  'non_tumor_indices': data_dict['non_tumor_indices'],
                  'cancer_prob': data_dict['cancer_prob'],
                  'tile_logits': data_dict['tile_logits'],
                  'external_tile_logits': data_dict['external_tile_logits'],
                  'tiles': data_dict['tiles'],
                  'ihc_tiles': data_dict['ihc_tiles'],
                  'tile_y': data_dict['tile_y'],
                  'virchow2_feats': data_dict['virchow2_feats'],
                  'uni2_feats': data_dict['uni2_feats'],
                  'conch_feats': data_dict['conch_feats'],
                  'slide_id': slide_id,
                  'labels': label,
                  'censoreship': censoreship}
        return sample
    
    def get_sample_with_try(self, idx, n_try=3):
        '''Get the sample with n_try'''
        for _ in range(n_try):
            try:
                sample = self.get_one_sample(idx)
                return sample
            except BaseException as e:
                print(f'Error in getting sample {idx}: {e}')
                raise
                print('Error in getting the sample, try another index')
                idx = np.random.randint(0, len(self.slide_data))
        print('Error in getting the sample, skip the sample')
        return None
    
    def reorder_samples_for_cox(self, batch_size):
        ordered_censorship = (self.slide_data.set_index(self.slide_key).loc[self.images, self.censoreship].values)
        indices_with_1 = [i for i, cens in enumerate(ordered_censorship) if cens == 1]
        indices_with_0 = [i for i, cens in enumerate(ordered_censorship) if cens == 0]

        # Shuffle both groups
        np.random.shuffle(indices_with_1)
        np.random.shuffle(indices_with_0)

        # Calculate the number of batches
        num_batches = (len(self.images) + batch_size - 1) // batch_size  # Ceiling division

        # Ensure at least one censorship == 1 index per batch
        if len(indices_with_1) < num_batches:
            raise ValueError("Not enough slides with censorship == 1 to satisfy every batch.")

        # Distribute `indices_with_1` evenly across all batches
        indices_with_1_per_batch = []
        for _ in range(num_batches):
            indices_with_1_per_batch.append(indices_with_1.pop(0))

        # Construct batches
        reordered_indices = []
        for batch_idx in range(num_batches):
            batch = [indices_with_1_per_batch[batch_idx]]  # Add one `censorship == 1` index to the batch

            # Fill the rest of the batch with randomly chosen indices
            remaining_slots = batch_size - len(batch)
            available_indices = indices_with_1 + indices_with_0  # Remaining pool of indices
            np.random.shuffle(available_indices)
            batch.extend(available_indices[:remaining_slots])

            # Remove used indices
            indices_with_0 = [idx for idx in indices_with_0 if idx not in batch]
            indices_with_1 = [idx for idx in indices_with_1 if idx not in batch]

            # Shuffle the batch to randomize its internal order
            np.random.shuffle(batch)
            reordered_indices.extend(batch)
            
        self.images = [self.images[i] for i in reordered_indices]
        self.labels = self.labels[reordered_indices]
        self.slide_data = self.slide_data.iloc[reordered_indices]

    def shuffle_indices(self, data_dict, slide_id):
        '''Shuffle the indices of the data_dict'''
        if self.use_tile_classification and data_dict['imgs'] is not None:
            indices = torch.randperm(data_dict['imgs'].size(0))
            data_dict['imgs'] = data_dict['imgs'][indices]
            data_dict['coords'] = data_dict['coords'][indices]
            if 'matching_tiles' in data_dict:
                data_dict['matching_tiles'] = data_dict['matching_tiles'][indices]
            print(f'Shuffled {data_dict["imgs"].size(0)} tiles for slide {slide_id}')
        
        
    def __len__(self):
        return len(self.slide_data)
    
    def __getitem__(self, idx):
        sample = self.get_sample_with_try(idx)
        return sample

    
class SlidingWindowDataset(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 root_path: str,
                 task_config: dict,
                 slide_key: str='file',
                 label: str='onco_score_26',
                 dataset_name: list= ['TAILORx_1'],
                 folds: list=[1,2,3,4,5],
                 use_clinical_features: bool=False,
                 test_on_all: bool=False,
                 get_single_slide: str=None,                 
                 window_size: int=10,
                 stride: int=1,
                 **kwargs
                 ):
        self.slide_dataset = SlideDataset(data_df, root_path, task_config, slide_key, label, dataset_name, folds, use_clinical_features, test_on_all, get_single_slide, **kwargs)
        assert self.slide_dataset.__len__() == 1, 'SlidingWindowDataset should be used for visualization of one slide at a time.'
        self.window_size = window_size
        self.stride = stride
        self.only_matching_tiles = kwargs.get('only_matching_tiles', False)
        
        if not use_clinical_features: 
            #if we use clinical features we only want to take the samples after the features have been normalized
            #we assume in this case that update_clinical_features_params_and_handle_data will be called before we try to get samples
            self.generate_windows()
    
    def __len__(self):
        return len(self.window_coords)
    
    def __getitem__(self, idx):
        window_coords = self.window_coords[idx] * 256
        window_indices = self.window_indices[idx]
        window_features = self.generate_window_features(self.images, window_indices)
        sample = {'imgs': window_features,
                  'coords': window_coords,
                  'labels': self.label,
                  'window_y': self.window_y[idx] if self.only_matching_tiles else None,
                  }
        return sample
    
    def update_clinical_features_params_and_handle_data(self, model):
        # self.slide_dataset.update_clinical_features_params_and_handle_data(model)
        self.generate_windows()
    
    def add_slide_bbox(self):
        slide = openslide.OpenSlide(slide_file)
        bbox_x = int(slide.properties['openslide.bounds-x'])
        bbox_y = int(slide.properties['openslide.bounds-y'])
        for tumor_polygon in tumor_polygons:
            tumor_polygon[:, 0] += bbox_y
            tumor_polygon[:, 1] += bbox_x

    
    def generate_windows(self):
        sample = self.slide_dataset.__getitem__(0)
        self.images, self.img_coords, self.slide_id, self.label = sample['imgs'], sample['coords'], sample['slide_id'], sample['labels']
        self.tile_y = sample['tile_y']
        self.valid_matching_indices = torch.nonzero(~torch.isnan(sample['matching_tiles'])).view(-1)
        if self.only_matching_tiles:
            self.images = self.images[self.valid_matching_indices]
            self.img_coords = self.img_coords[self.valid_matching_indices]

        self.img_coords = (self.img_coords/256)
        self.window_coords, self.window_indices = self.group_coords_into_overlapping_windows(self.img_coords.int(), self.window_size, self.stride)
        if self.only_matching_tiles:
            self.window_y = torch.tensor([torch.mean(self.tile_y[self.window_indices[win_num]]) for win_num in range(len(self.window_indices))])
        assert len(self.window_coords) == len(self.window_indices), 'Number of windows in window coords is not the same as the number of windows as shown by window_indices, this is a bug.'
    
    def group_coords_into_overlapping_windows(self, tile_coords, window_size, stride):
        """
        Groups coordinates into overlapping windows of size `window_size x window_size`.

        Args:
            tile_coords (torch.Tensor): A tensor of shape (b, 2) containing grid coordinates.
            window_size (int): The size of the square window.
            stride (int): The stride between windows.

        Returns:
            list: A list of torch tensors, where each tensor contains the coordinates in a window.
            list: A list of lists, where each inner list contains indices of the coordinates in the window.
        """
        # Find the min and max range of coordinates
        x_min, y_min = tile_coords.min(dim=0).values
        x_max, y_max = tile_coords.max(dim=0).values

        # Initialize lists to store results
        window_coords = []
        window_indices = []

        # Iterate over all possible top-left corners of windows with a given stride
        for x_start in range(x_min, x_max + 1, stride):
            for y_start in range(y_min, y_max + 1, stride):
                # Define the window boundaries
                x_end = x_start + window_size
                y_end = y_start + window_size

                # Find the indices of coordinates that fall within this window
                in_window = (
                    (tile_coords[:, 0] >= x_start) & (tile_coords[:, 0] < x_end) &
                    (tile_coords[:, 1] >= y_start) & (tile_coords[:, 1] < y_end)
                )
                indices = torch.nonzero(in_window, as_tuple=False).squeeze(dim=1)

                sq_window_size = window_size ** 2
                # If the window is not empty, store the coordinates and their indices
                # if len(indices) > 0:
                # If at least part of the window is valid, store the coordinates and their indices
                if len(indices) >= 0.3 * sq_window_size:
                    window_coords.append(tile_coords[indices])
                    window_indices.append(indices.tolist())

        return window_coords, window_indices

    def generate_window_features(self, features, window_indices):
        """
        Generates a 2D tensor `window_features` for a given window.

        Args:
            features (torch.Tensor): A tensor of shape (b, 1536) containing feature vectors.
            window_indices (list): A list of indices corresponding to the tiles in the window.

        Returns:
            torch.Tensor: A 2D tensor of shape (len(window_indices), 1536) containing the feature vectors
                          for the tiles in the window.
        """
        # Extract the feature vectors for the tiles in the window using the indices
        window_features = features[torch.tensor(window_indices)]
        return window_features

    def generate_window_y(self, y, window_indices):
        """
        Generates a 1D tensor `window_y` for a given window.

        Args:
            y (torch.Tensor): A tensor of shape (b,) containing labels.
            window_indices (list): A list of indices corresponding to the tiles in the window.

        Returns:
            torch.Tensor: A 1D tensor of shape (len(window_indices),) containing the labels
                          for the tiles in the window.
        """
        # Extract the feature vectors for the tiles in the window using the indices
        window_y = y[torch.tensor(window_indices)]
        return window_y