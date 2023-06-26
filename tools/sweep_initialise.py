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
        'num_epochs': {'max': 20, 'min': 10},
        'batch_size': {'max': 40, 'min': 10},
        'learning_rate': {'max': 0.1, 'min': 0.005},
        'momentum': {'max': 0.9, 'min': 0.0},
        'l1_lambda': {'max': 0.75, 'min': 0.0},
        'l2_lambda': {'max': 0.75, 'min': 0.0},
        'cv_num_filters': {'max': 32, 'min': 2},
        'cv_clip_norm': {'max': 50.0, 'min': 1.0},
        'fc_clip_norm': {'max': 50.0, 'min': 1.0},
        'num_hidden_layers': {'max': 4, 'min': 1}
    }
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='cnn_mnist_test'
)
