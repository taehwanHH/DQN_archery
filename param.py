from datetime import datetime

Hyper_Param = {
    'today': datetime.now().strftime('%Y-%m-%d'),
    'discount_factor': 0.9,
    'learning_rate': 0.0005,
    'epsilon': 1,
    'epsilon_decay_rate': 0.9999,
    'epsilon_min': 0.00005,
    'batch_size': 512,
    'train_start': 4000,
    'num_episode': 200000,
    'memory_size': 10**5,
    'target_update_interval': 5,
    'print_every': 1000,
    'num_neurons': [32,64,64,64,32],
    'step_max': 300,
    'vw_max': 5,
    'window_size': 1000,
    'Saved_using': False,
    'MODEL_PATH': "saved_model",
    'MODEL_NAME': "model_(227, 1001.0).h5"
}

