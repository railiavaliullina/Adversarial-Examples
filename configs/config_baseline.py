cfg = {
    # parameters for dataset and dataloader
    "data":
        {
            "dataset_path": 'D:/datasets/homeworks/DL SOP classification/',
            "nb_train_images": 71940,
            "nb_valid_images": 24045,
            "nb_classes": 12,
            "dataloader": {
                "nb_epochs": 100,
                "shuffle": {
                    "train": True,
                    "valid": False
                },
                "batch_size": {
                    "train": 8,
                    "valid": 8
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
                    'out_channels': 64,
                    'kernel_size': 7,
                    'stride': 2,
                    'padding': 3,
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
                            'downsample':
                                {
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
                    'learning_rate': 0.01,
                    'momentum': 0.9,
                    'weight_decay': 5e-4,
                    'nesterov': True
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
            'experiment_name': 'baseline',
            'checkpoints_dir': 'checkpoints/',
            'save_model': True,
            'load_model': True,
            'epoch_to_load': 89,
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
                    'tensorboard_dir': 'tensorboard/baseline/',
                    'num_images': 150,
                }
        },
}
