import os
import torch
import pandas as pd
import numpy as np

from training import train, test, generate_heatmap, run_window_inference
from params import get_finetune_params
from task_configs.utils import load_task_config
from utils import seed_torch, get_exp_code, get_loader, save_obj, get_test_loader
from datasets.slide_datatset import SlideDataset, SlidingWindowDataset
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


if __name__ == '__main__':
    # Set the hf token
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    args = get_finetune_params()
    print(args)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # set the random seed
    seed_torch(device, args.seed)

    # load the task configuration
    print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config = load_task_config(args.task_cfg_path)
    print(args.task_config)
    args.task = args.task_config.get('name', 'task')
    
    # Update model dimensions if using clinical features
    #TODO get number of clinical features
    #num_clinical_features = 2
    #if args.clinical_features:
    #    args.input_dim = args.input_dim + num_clinical_features
    #    args.latent_dim = args.latent_dim + int(num_clinical_features//2)
    
    # set the experiment save directory
    args.save_dir = os.path.join(args.save_dir, args.task, args.exp_name)
    args.model_code, args.task_code, args.exp_code = get_exp_code(args) # get the experiment code
    args.save_dir = os.path.join(args.save_dir, args.exp_code)
    os.makedirs(args.save_dir, exist_ok = True)
    print('Experiment code: {}'.format(args.exp_code))
    print('Setting save directory: {}'.format(args.save_dir))

    # set the learning rate
    if not args.run_inference:
        eff_batch_size = args.batch_size * args.gc
        if args.lr is None or args.lr < 0:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.gc)
        print("effective batch size: %d" % eff_batch_size)

    # set the split key
    if args.pat_strat:
        args.slide_key = 'patient_barcode'
    elif args.student_training and args.model_ckpt != '':
        args.slide_key = 'Matched_HE_SlideName'
    else:
        args.slide_key = 'file'

    # set up the dataset
    dataset = pd.read_csv(args.dataset_csv) # read the dataset csv file

    # use the slide dataset
    if not args.sliding_window_inference:
        DatasetClass = SlideDataset
    else:
        DatasetClass = SlidingWindowDataset
    #DatasetClass = SlideDataset

    # use only annotated data for tile classification
    if args.use_tile_classification or args.window_training or args.only_annotated:
        args.batches = ['Batch_1', 'Batch_2']
        args.slide_path_key = 'Path'
    else:
        args.batches = None
        args.slide_path_key = None
        
    if args.window_training:
        args.dataset = dataset

    # set up the results dictionary
    results = {}

    # start cross validation
    if not args.run_inference and not args.generate_heatmap and not args.sliding_window_inference:
        for fold in range(args.folds):
            # set up the fold directory
            save_dir = os.path.join(args.save_dir, f'fold_{fold}')
            os.makedirs(save_dir, exist_ok=True)
            # instantiate the dataset
            train_data, val_data, test_data = DatasetClass(dataset, args.root_path, args.task_config, slide_key=args.slide_key, label = args.label, \
                                                           dataset_name = args.train_dataset, folds = args.train_fold, use_clinical_features = args.clinical_features, censoreship = args.censoreship, survival = args.survival,
                                                           batches = args.batches, slide_path_key = args.slide_path_key, window_training = args.window_training,
                                                           teacher_label=args.teacher_label) \
                                            , DatasetClass(dataset, args.root_path, args.task_config, slide_key=args.slide_key, label = args.label, \
                                                           dataset_name = args.val_dataset, folds = args.val_fold, use_clinical_features = args.clinical_features, test_on_all = args.test_on_all, censoreship = args.censoreship, survival = args.survival,
                                                           batches = args.batches, slide_path_key = args.slide_path_key, window_training = args.window_training,
                                                           teacher_label=args.teacher_label) if len(args.val_dataset) > 0 else None \
                                            , DatasetClass(dataset, args.root_path, args.task_config, slide_key=args.slide_key, label = args.label, \
                                                           dataset_name = args.test_dataset, folds = args.test_fold, use_clinical_features = args.clinical_features, test_on_all = args.test_on_all, censoreship = args.censoreship, survival = args.survival, 
                                                           batches = args.batches, slide_path_key = args.slide_path_key, window_training = args.window_training,
                                                           teacher_label=args.teacher_label) if len(args.test_dataset) > 0 else None
            #scale the lr in case of regression
            if args.task_config.get('setting', 'multi_class') == 'continuous' and not args.loss_fn == 'cox':
                args.mean = train_data.labels.mean()
                args.std = train_data.labels.std()
                print("labels mean: %.2e" % args.mean)
                print("labels std: %.2e" % args.std)
                print(f'using {args.loss_fn} loss')
            if args.clinical_features:
                # Update model dimensions if using clinical features
                num_clinical_features = len(train_data.columns)
                args.input_dim = args.input_dim + num_clinical_features
                args.latent_dim = args.latent_dim + int(num_clinical_features//2)
                #latent dim needs to be divisible by 4
                args.latent_dim = args.latent_dim - args.latent_dim % 4
                if 'TumorSize' in train_data.columns:
                    args.tumor_size_mean = train_data.tumor_size_mean
                    args.tumor_size_std = train_data.tumor_size_std
                    print("tumor size mean: %.2e" % args.tumor_size_mean)
                    print("tumor size std: %.2e" % args.tumor_size_std)
                if 'age' in train_data.columns:
                    args.age_mean = train_data.age_mean
                    args.age_std = train_data.age_std
                    print("age mean: %.2e" % args.age_mean)
                    print("age std: %.2e" % args.age_std)
            args.n_classes = train_data.n_classes # get the number of classes
            # get the dataloader
            train_loader, val_loader, test_loader = get_loader(train_data, val_data, test_data, **vars(args))
            # start training
            val_records, test_records = train((train_loader, val_loader, test_loader), fold, args)

            # update the results
            records = {}
            if val_records is not None:
                records['val'] = val_records
            if test_records is not None:
                records['test'] = test_records
            for record_ in records:
                for key in records[record_]:
                    if 'prob' in key or 'label' in key or 'censoreship' in key:
                        continue
                    key_ = record_ + '_' + key
                    if key_ not in results:
                        results[key_] = []
                    results[key_].append(records[record_][key])
    elif args.run_inference or args.generate_heatmap:
        save_dir = os.path.join(args.save_dir, f'inference_results')
        os.makedirs(save_dir, exist_ok=True)
        args.save_dir = save_dir
        test_data = DatasetClass(dataset, args.root_path, args.task_config, slide_key=args.slide_key, label = args.label, \
                                 dataset_name = args.test_dataset, folds = args.test_fold, use_clinical_features = args.clinical_features, \
                                 test_on_all = args.test_on_all, get_single_slide = args.get_single_slide, censoreship = args.censoreship, survival = args.survival,
                                 batches = args.batches, slide_path_key = args.slide_path_key, window_training = args.window_training, teacher_label=args.teacher_label)
        if args.clinical_features:
            num_clinical_features = len(test_data.columns)
            args.input_dim = args.input_dim + num_clinical_features
            args.latent_dim = args.latent_dim + int(num_clinical_features//2)
            #latent dim needs to be divisible by 4
            args.latent_dim = args.latent_dim - args.latent_dim % 4
        args.n_classes = test_data.n_classes
        test_loader = get_test_loader(test_data, **vars(args))
        
        if args.run_inference:
            test_records = test(test_loader, args)
        else:
            generate_heatmap(test_loader, args)
            print('Done!')
            exit()
        records = {'test': test_records}
        for record_ in records:
            for key in records[record_]:
                if 'prob' in key or 'label' in key or 'censoreship' in key:
                    continue
                key_ = record_ + '_' + key
                if key_ not in results:
                    results[key_] = []
                results[key_].append(records[record_][key])
    else:
        save_dir = os.path.join(args.save_dir, f'inference_results')
        os.makedirs(save_dir, exist_ok=True)
        args.save_dir = save_dir
        run_window_inference(dataset, DatasetClass, args)
        print('Done!')
        exit()
        '''
        heatmap_dataset = DatasetClass(dataset, args.root_path, args.task_config, slide_key=args.slide_key, label = args.label, \
                                 dataset_name = args.test_dataset, folds = args.test_fold, use_clinical_features = args.clinical_features, \
                                       test_on_all = args.test_on_all, get_single_slide = args.get_single_slide, window_size = args.window_size, stride = args.stride)
        if args.clinical_features:
            num_clinical_features = len(test_data.slide_dataset.columns)
            args.input_dim = args.input_dim + num_clinical_features
            args.latent_dim = args.latent_dim + int(num_clinical_features//2)
            #latent dim needs to be divisible by 4
            args.latent_dim = args.latent_dim - args.latent_dim % 4
        args.n_classes = test_data.slide_dataset.n_classes
        test_loader = get_test_loader(test_data, for_heatmap = True, **vars(args))
        '''

    # save the results into a csv file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.save_dir, 'summary.csv'), index=False)

    # print the results, mean and std
    for key in results_df.columns:
        print('{}: {:.4f} +- {:.4f}'.format(key, np.mean(results_df[key]), np.std(results_df[key])))
    print('Results saved in: {}'.format(os.path.join(args.save_dir, 'summary.csv')))
    print('Done!')
