import argparse
import torch

def get_training_config(args=None):

    parser = argparse.ArgumentParser(
        description="Graph Neural Network based reinforcement learning solution for Power network reconfiguration")

    parser.add_argument('--features_dim', type=int, default=128, help="Embedding length")
    parser.add_argument('--total_steps', type=int, default=2000000, help='Total number of steps')
    parser.add_argument('--node_encoder', type=str, default='CAPAM',
                        help='Node embedding type. Available ones are [CAPAM, MLP]')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for training')#25000
    parser.add_argument('--n_steps', type=int, default=50000, help='Number of steps for rollout')#100000
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--ent_coef', type=float, default=0.1, help='Entropy coefficient')
    parser.add_argument('--val_coef', type=float, default=0.5, help='Value coefficient')
    parser.add_argument('--gamma', type=float, default=1.00, help='Discount factor')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs per rollout')

    parser.add_argument('--save_freq', type=int, default=50000, help="Save frequency")

    parser.add_argument('--logger', type=str, default='Tensorboard_logger/', help='Directory for tensorboard logger')
    parser.add_argument('--model_save', type=str, default='Trained_Models/',
                        help='Directory for saving the trained models')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_cpu', type=int, default=10, help="Number of parallel environments for rollout")

    config = parser.parse_args(args)
    config.use_cuda = torch.cuda.is_available() and not config.no_cuda

    return config
