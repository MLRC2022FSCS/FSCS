import numpy as np
import random
import pickle
import argparse
from dataloader import LoadData
from model import TwoLayerNN, Baseline
from train import train, train_base
from eval import test, test_base
import torch
from torch.utils.data import DataLoader
import os


def set_seed(seed):
    """
    Set the seed for standard libraries and cuda.

    Args:
        seed: The seed number that should ber used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def main(args):
    """
    The main function that calls training and evaluation functions.

    Args:
        args:   Namespace object from the argument parser.
    """
    torch.cuda.empty_cache()
    set_seed(args.seed)

    # Get data from selected dataset.
    Data = LoadData(args.dataset_name.lower())

    # Create train-, validiation- if available, and testloaders.
    train_loader = DataLoader(Data.train, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    if Data.val:
        val_loader = DataLoader(Data.val, batch_size=args.batch_size,
                                drop_last=True)
    else:
        val_loader = None
    test_loader = DataLoader(Data.test, batch_size=args.batch_size,
                             drop_last=True)

    # Set up list for upcoming results.
    res = [[], [], []]

    # Coverage starts at 1, when threshold starts at 0.
    curr_coverage = 1.0
    threshold = 0.0
    threshold_step = args.threshold_step

    # Load baseline model if available for dataset, else train one.
    if os.path.isfile('./Models/{}_base.pt'.format(args.dataset_name.lower())) and not args.force_train:
        model = Baseline(args.dataset_name.lower(), args.num_classes,
                         args.model_size)
        model.load_state_dict(
            torch.load(
                './Models/{}_base.pt'.format(args.dataset_name.lower()), map_location=torch.device('cpu')))
    else:
        model = train_base(args.dataset_name.lower(), args.num_classes,
                           args.lr, args.epochs, train_loader, val_loader,
                           args.progress_bar, args.model_size)

    while curr_coverage > 0.19:
        model, acc, coverage, results = test_base(args.dataset_name.lower(),
                                                  model, test_loader, threshold)

        print(f"Model accuracy after {args.epochs} epochs: ", acc, coverage,
              results["pred_correct_0"], results["pred_correct_1"])

        # Save the results.
        res[0].append(acc)
        res[1].append(coverage)
        res[2].append(results)

        curr_coverage = coverage

        threshold += threshold_step

    # Save results to file.
    open_results = open('./Results/{}_base.pkl'.format(args.dataset_name.lower()),
                        'wb')
    pickle.dump(res, open_results)
    open_results.close()

    torch.cuda.empty_cache()

    res = [[], [], []]
    curr_coverage = 1.0
    threshold = 0.0

    # Load sufficiency regularized model if available for dataset, else train one.
    if os.path.isfile('./Models/{}.pt'.format(args.dataset_name.lower())) and not args.force_train:
        model = TwoLayerNN(args.dataset_name.lower(), args.num_classes,
                           Data.ratio, args.model_size)
        model.load_state_dict(
            torch.load('./Models/{}.pt'.format(args.dataset_name.lower())))
    else:
        model = train(args.dataset_name.lower(), args.num_classes, args.lr,
                      args.epochs, train_loader, val_loader, Data.ratio,
                      args.lambd, args.progress_bar, args.model_size)

    while curr_coverage > 0.19:
        model, acc, coverage, results = test(args.dataset_name.lower(),
                                             model, test_loader, threshold)

        print(f"Model accuracy after {args.epochs} epochs: ", acc, coverage,
              results["pred_correct_0"], results["pred_correct_1"])

        res[0].append(acc)
        res[1].append(coverage)
        res[2].append(results)

        curr_coverage = coverage

        threshold += threshold_step

    open_results = open('./Results/{}.pkl'.format(args.dataset_name.lower()),
                        'wb')
    pickle.dump(res, open_results)
    open_results.close()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--dataset_name', default='Adult', type=str,
                        help='Name of the dataset to run the experiment on.')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='The amount of classes, usually you want to keep this at default value.')
    parser.add_argument('--lambd', default=0.7, type=float,
                        help='The lambda value for sufficiency.')
    parser.add_argument('--model_size', default='tiny', type=str,
                        help='The model size of BERT to be used with the Civil comments dataset.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Minibatch set to 8/16 for adult but 96 (max for celeba for 3090).')

    # Other hyperparameters
    parser.add_argument('--epochs', default=20, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--threshold_step', default=0.01, type=float,
                        help='Step size to increase confidence threshold.')

    # Other arguments
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the datasets.')
    parser.add_argument('--progress_bar', action="store_true",
                        help="Turn the progress bar of tqdm off.")
    parser.add_argument('--force_train', default=0, type=int,
                        help="Train a model even if one is available.")

    args = parser.parse_args()

    main(args)
