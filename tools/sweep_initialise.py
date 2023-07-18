import wandb
wandb.login()

sweep_configuration = {
    'method': 'random',
    'metric':
    {
        'goal': 'minimize',
        'name': 'loss'
    },
    'parameters':
    {
        'num_epochs': {'max': 10, 'min': 5},
        'batch_size': {'values': [16, 32, 64]},
        'learning_rate': {'max': 0.1, 'min': 0.005},
        'l1_lambda': {'max': 0.75, 'min': 0.0},
        'l2_lambda': {'values': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0]},
        'cv_num_filters': {'max': 32, 'min': 2},
        'cv_kernel_size': {'max': 5, 'min': 3},
        'cv_stride': {'max': 2, 'min': 1},
        'pool_kernel_size': {'max': 4, 'min': 2},
        'pool_stride': {'max': 4, 'min': 2},
        'cv_clip_norm': {'max': 100.0, 'min': 1.0},
        'fc_clip_norm': {'max': 100.0, 'min': 1.0},
        'num_hidden_layers': {'max': 2, 'min': 1},
        #'beta1': {'max': 0.9, 'min': 0.5},
        #'beta2': {'max': 0.999, 'min': 0.7},
        'momentum': {'max': 0.9, 'min': 0.0}
    }
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='cnn_mnist_test'
)
