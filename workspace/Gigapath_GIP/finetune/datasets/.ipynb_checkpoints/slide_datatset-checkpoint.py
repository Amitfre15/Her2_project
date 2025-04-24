import os
import h5py
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


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
        self.use_clinical_features = use_clinical_features
        self.test_on_all = test_on_all
        #TODO: update list of columns of clinical features
        self.columns = ['er_status', 'pr_status', 'TumorSize', 'age']
        #self.columns = ['er_status', 'pr_status', 'TumorSize', 'RecChemo', 'age']
        allowed_values = {'er_status': [0,1], 'pr_status':[0,1], 'RecChemo':[0,1]}
        
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

            #filter based on dataset
            data_df = data_df[data_df['id'].isin(dataset_name)]
            #filter based on fold
            data_df = data_df[data_df['fold'].isin(folds)]
            #filter by valid label
            #data_df = data_df[data_df[label].isin([0,1,0.0,1.0])]
            if not test_on_all:
                data_df = data_df[pd.to_numeric(data_df[label], errors='coerce').notnull()]
        else:
            data_df = data_df[data_df[slide_key] == get_single_slide]
        # get slides that have tile encodings
        valid_slides = self.get_valid_slides(root_path, data_df[slide_key].values)
        # filter out slides that do not have tile encodings
        data_df = data_df[data_df[slide_key].isin(valid_slides)]
        #filter based on clinical features
        if self.use_clinical_features:
            for column in self.columns:
                column_values = data_df[column]
                column_values = pd.to_numeric(column_values, errors='coerce')
                data_df[column] = column_values
                data_df = data_df[data_df[column].notna()]
                if column == 'TumorSize':
                    self.tumor_size_mean = data_df[column].mean()
                    self.tumor_size_std = data_df[column].std()
                elif column == 'age':
                    self.age_mean = data_df[column].mean()
                    self.age_std = data_df[column].std()
                if column in allowed_values:
                    data_df = data_df[data_df[column].isin(allowed_values[column])]
            
            
        # set up the task
        self.setup_data(data_df, task_config.get('setting', 'multi_class'))
        
        self.max_tiles = task_config.get('max_tiles', 1000)
        self.shuffle_tiles = task_config.get('shuffle_tiles', False)
        print('Dataset has been initialized!')
        
    def get_valid_slides(self, root_path: str, slides: list) -> list:
        '''This function is used to get the slides that have tile encodings stored in the tile directory'''
        valid_slides = []
        for i in range(len(slides)):
            if 'pt_files' in root_path.split('/')[-1]:
                sld = slides[i].replace(".svs", "") + '.pt'
            elif 'gigapath_features' in root_path.split('/')[-1]:
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
        self.slide_data, self.images, self.labels, self.n_classes = prepare_data_func(df)
    
    def prepare_continuous_data(self, df: pd.DataFrame):
        '''Prepare the data for regression'''
        n_classes = 1
        
        images = df[self.slide_key].to_list()
        labels = df[[self.label]].to_numpy().astype(int)
        
        return df, images, labels, n_classes
    
    def prepare_multi_class_or_binary_data(self, df: pd.DataFrame):
        '''Prepare the data for multi-class classification'''
        # set up the label_dict
        label_dict = self.task_cfg.get('label_dict', {})
        assert  label_dict, 'No label_dict found in the task configuration'
        # set up the mappings
        assert self.label in df.columns, 'No label column found in the dataframe'
        df[self.label] = df[self.label].map(label_dict)
        n_classes = len(label_dict)
        
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
        

        
        # set the input dict
        data_dict = {'imgs': images,
                'img_lens': images.size(0),
                'pad_mask': 0,
                'coords': coords}
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
        # get the slide images
        data_dict = self.get_images_from_path(slide_path)
        if self.use_clinical_features:
            clinical_features = torch.tensor(self.slide_data[self.slide_data[self.slide_key] == slide_id][self.columns].values)
            #TODO: maybe use some data handling function for the clinical features
            #clinical_features = handle_func(clinical_features)
            clinical_features = clinical_features.expand(data_dict['imgs'].shape[0],-1)
            data_dict['imgs'] = torch.cat((data_dict['imgs'], clinical_features), -1)
            
            
        # get the slide label
        label = torch.from_numpy(self.labels[idx])
        # set the sample dict
        sample = {'imgs': data_dict['imgs'],
                  'img_lens': data_dict['img_lens'],
                  'pad_mask': data_dict['pad_mask'],
                  'coords': data_dict['coords'],
                  'slide_id': slide_id,
                  'labels': label}
        return sample
    
    def get_sample_with_try(self, idx, n_try=3):
        '''Get the sample with n_try'''
        for _ in range(n_try):
            try:
                sample = self.get_one_sample(idx)
                return sample
            except:
                raise
                print('Error in getting the sample, try another index')
                idx = np.random.randint(0, len(self.slide_data))
        print('Error in getting the sample, skip the sample')
        return None
        
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
                 **kwargs
                 ):
        self.slide_dataset = SlideDataset(data_df, root_path, task_config, slide_key, label, dataset_name, folds, use_clinical_features, test_on_all, get_single_slide)
        assert self.slide_dataset.__len__() == 1, 'SlidingWindowDataset should be used for visualization of one slide at a time.'
        if not use_clinical_features: 
            #if we use clinical features we only want to take the samples after the features have been normalized
            #we assume in this case that update_clinical_features_params_and_handle_data will be called before we try to get samples
            self.generate_windows()
    
    def __len__(self):
        return len(self.window_coords)
    
    def __getitem__(self, idx):
        window_coords = self.window_coords[i]
        window_indices = self.window_indices[i]
        window_features = self.generate_window_features(self.images, window_indices)
        sample = {'imgs': window_features,
                  'coords': window_coords}
        return sample
    
    def update_clinical_features_params_and_handle_data(self, model):
        self.slide_dataset.update_clinical_features_params_and_handle_data(model)
        self.generate_windows()
    
    def generate_windows(self):
        sample = self.slide_dataset.__getitem__(0)
        self.images, self.img_coords, self.slide_id = sample['imgs'], sample['coords'], sample['slide_id']
        self.img_coords = (self.img_coords/256)
        self.window_coords, self.window_indices = self.group_coords_into_overlapping_windows(self.img_coords.int(), window_size)
        assert len(self.window_coords) == len(self.window_indices), 'Number of windows in window coords is not the same as the number of windows as shown by window_indices, this is a bug.'
    
    def group_coords_into_overlapping_windows(self, tile_coords, window_size):
        """
        Groups coordinates into overlapping windows of size `window_size x window_size`.

        Args:
            tile_coords (torch.Tensor): A tensor of shape (b, 2) containing grid coordinates.
            window_size (int): The size of the square window.

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

        # Iterate over all possible top-left corners of windows with stride = 1
        for x_start in range(x_min, x_max + 1):
            for y_start in range(y_min, y_max + 1):
                # Define the window boundaries
                x_end = x_start + window_size
                y_end = y_start + window_size

                # Find the indices of coordinates that fall within this window
                in_window = (
                    (tile_coords[:, 0] >= x_start) & (tile_coords[:, 0] < x_end) &
                    (tile_coords[:, 1] >= y_start) & (tile_coords[:, 1] < y_end)
                )
                indices = torch.nonzero(in_window, as_tuple=False).squeeze(dim=1)

                # If the window is not empty, store the coordinates and their indices
                if len(indices) > 0:
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