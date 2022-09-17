cfg = {
    # parameters for dataset and dataloader
    "data":
        {
            "dataset_path": 'D:/Универ/Магистратура, 3 семестр/Advanced CV/archive/',#'D:/datasets/homeworks/DL SOP classification/',
            "nb_train_images": 71940,
            "nb_valid_images": 24045,
            "nb_classes": 12,
            "dataloader": {
                "shuffle": {
                    "train": True,
                    "valid": False
                },
                "batch_size": {
                    "train": 8,
                    "valid": 1
                },
                "drop_last": {
                    "train": True,
                    "valid": False
                },
            },
            "augmentation":
                {
                    "sz_crop": 224,
                    "sz_resize": 256,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "contrast": 0.4,
                    "saturation": 0.4,
                    "brightness": 0.4,
                }
        },

    # parameters for ResNet-50 model parts
    "model":
        {
            'pretrained':
                {
                    'load_pretrained': False,
                    'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                    'progress': True
                },

            'FirstConv':
                {
                    'in_channels': 3,
                    'out_channels': 32,
                    'kernel_size': 3,
                    'stride': 2,
                    'padding': 1,
                    'bias': False
                },
            'MaxPool':
                {
                    'kernel_size': 3,
                    'stride': 2,
                    'padding': 1
                },
            'LayersGroup':
                {
                    # BottleNeck class common parameters
                    'BottleNeck':
                        {
                            'kernel_size': [1, 3, 1],
                            'padding': 1,
                            'bias': False,
                            'avg_pool_output_size': (1, 1),
                            'downsample':
                                {
                                    'avg_pool_kernel_size': 2,
                                    'avg_pool_stride': 2,
                                    'kernel_size': 1,
                                    'bias': False
                                }
                        },
                    # layer group specific parameters
                    'layer1': {
                        'in_channels': 64,
                        'out_channels': 64,
                        'nb_layers': 3,
                        'stride': 1
                    },
                    'layer2': {
                        'in_channels': 256,
                        'out_channels': 128,
                        'nb_layers': 4,
                        'stride': 2
                    },
                    'layer3': {
                        'in_channels': 512,
                        'out_channels': 256,
                        'nb_layers': 6,
                        'stride': 2
                    },
                    'layer4': {
                        'in_channels': 1024,
                        'out_channels': 512,
                        'nb_layers': 3,
                        'stride': 2
                    }
                },
            'AvgPool':
                {
                    'output_size': (1, 1)
                },
            'Linear':
                {
                    'in_features': 2048,
                    'out_features': 12,
                    'bias': True
                }
        },

    # parameters for setting up training parameters
    "train":
        {
            # training stage common parameters
            'epochs': 100,

            # optimizer parameters
            'opt':
                {
                    'optim_type': 'SGD',
                    'learning_rate': 0.1,  # init_lr
                    'resnet_batch_size': 256,  # b_size from paper
                    'momentum': 0.9,
                    'weight_decay': 5e-4,
                    'nesterov': True
                },
            'lr_scheduler':
                {
                    'eta_min': 1e-8,
                    'multiplier': 1,
                    'warmup_iters': 200
                },
            'loss':
                {
                    'smooth_eps': 0.1
                }
        },

    # parameters for model evaluation
    "eval":
        {
            'evaluate_on_train_data': False,
            'evaluate_before_training': True,
        },

    # parameters for logging training process, saving/restoring model
    "logging":
        {
            'log_metrics': True,
            'experiment_name': 'improved_model,init_lr=0.1',
            'checkpoints_dir': 'checkpoints/',
            'save_model': True,
            'load_model': True,
            'epoch_to_load': 92,
            'save_frequency': 1,
        },

    # parameters to debug training and check if everything is ok
    "debug":
        {
            # to check batches before training
            "save_batch":
                {
                    "enable": False,
                    "nrof_batches_to_save": 5,
                    "path_to_save": 'batches_images/',
                },
            "overfit_on_batch":
                {
                    "enable": False,
                    "nb_iters": 1000,
                },
            "save_embeddings":
                {
                    "enable": True,
                    'tensorboard_dir': 'tensorboard/improved/',
                    'num_images': 150,
                }
        }
}
