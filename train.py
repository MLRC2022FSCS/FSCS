from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from model import TwoLayerNN, Baseline
from eval import test, test_base


def train(name, num_classes, lr, epochs, train_loader, val_loader,
          ratio, lambd, progress_bar, model_size):
    """
    Training function that trains the TwolayerNN model with the given data.

    Args:
        name:   Name of the dataset that has to be trained.
        num_classes:    The number of classes to predict.
        lr:     Learning rate used for the Adam optimizer.
        epochs: Number of training epochs to perform.
        train_loader:   The training pytorch Dataloader of the dataset.
        val_loader:     The validation pytorch Dataloader.
        ratio:  The ratio of the protected in contrast to the unprotected
                attribute.
        labmd:  Value that scales the regularisation of the sufficiency losses.
        progress_bar:   The progress_bar flag for tqdm.
        model_size:     The size for the BERT pretrained model used.
    Returns:
        model:  The trained TwolayerNN model.
    """

    # Set the default device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and loss module.
    model = TwoLayerNN(name, num_classes, ratio, model_size)
    loss_module = nn.CrossEntropyLoss()

    # Set the model on the correct device.
    model.to(device)

    # Set the model in training mode.
    model.train()

    # Load the parameters for all layers.
    phi_parameters = model.phi.parameters()
    group_parameters = [parameters for group in model.groups
                        for parameters in group.parameters()]
    joint_parameters = model.joint.parameters()

    # Define a simple Adam optimizer for each set of parameters.
    phi_optimizer = optim.Adam(phi_parameters, lr=lr, weight_decay=0)
    group_optimizer = optim.Adam(group_parameters, lr=lr, weight_decay=0)
    joint_optimizer = optim.Adam(joint_parameters, lr=lr, weight_decay=0)

    # Initialize a variable.
    best_acc = 0

    # Loop over epochs using tqdm to show progress.
    for e in tqdm(range(epochs), leave=True, disable=progress_bar):

        # The group specific loop over the data.
        for x, y, d in train_loader:

            # If there is a BERT model before the network,
            # find the mask and data appropriately.
            if name == 'civil':
                x_mask = x['attention_mask'].to(device)
                x = x['input_ids'].squeeze(1).to(device)
            else:
                x_mask = None
                x = x.to(device)

            # Move the data to device.
            y = y.to(device)
            d = d.to(device)

            # Before calculating the gradients, we need to ensure that they
            # are all zero. The gradients would not be overwritten, but
            # actually added to the previous ones otherwise.
            group_optimizer.zero_grad()

            # Perform the forward of the group layers and calculate the loss.
            group_specific_preds = model.forward_group(x, d, x_mask)
            group_specific_loss = loss_module(group_specific_preds.float(),
                                              y.reshape([-1, 1]))

            # Perform the backward of the loss and optimizer step.
            group_specific_loss.backward()
            group_optimizer.step()

        # The featurizer and joint classifier loop over the data.
        for x, y, d in train_loader:

            # If there is a BERT model before the network,
            # find the mask and data appropriately.
            if name == 'civil':
                x_mask = x['attention_mask'].to(device)
                x = x['input_ids'].squeeze(1).to(device)
            else:
                x_mask = None
                x = x.to(device)

            # Move the data to device.
            y = y.to(device)
            d = d.to(device)

            # Perform the model forward and return the predictions.
            joint_preds, group_specific_preds, group_agnostic_preds = model(x, d,
                                                                            x_mask)

            # Calculate the joint loss l0 and regularizer loss lr.
            l0 = loss_module(joint_preds, y.long())
            lr = lambd * (loss_module(group_specific_preds,  y.reshape([-1, 1])) -
                          loss_module(group_agnostic_preds,  y.reshape([-1, 1])))

            # Before calculating the gradients, we need to ensure that they
            # are all zero. The gradients would not be overwritten, but
            # actually added to the previous ones otherwise.
            phi_optimizer.zero_grad()

            # Already perform the backward propegation for the regularizer loss.
            lr.backward(retain_graph=True)

            # Set the joint gradients to zero.
            joint_optimizer.zero_grad()

            # Add the joint backward to both optimizers.
            l0.backward()

            # Take a step with both optimizers.
            joint_optimizer.step()
            phi_optimizer.step()

        # Check for validation data.
        if val_loader is not None:
            model, accuracy, _, _ = test(name, model, val_loader, 0.0)
            print("acc_val:", accuracy)

            # Save the model with the best validation accuracy as it
            # generalizes best.
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), './Models/{}.pt'.format(name))

            # Put the model in training mode.
            model.train()
        else:
            torch.save(model.state_dict(), './Models/{}.pt'.format(name))

    # Load the best model parameters into the model before returning it.
    model.load_state_dict(torch.load('./Models/{}.pt'.format(name)))

    return model


def train_base(name, num_classes, lr, epochs, train_loader, val_loader,
               progress_bar, model_size):
    """
    Training function that trains the Baseline model with the given data.

    Args:
        name:   Name of the dataset that has to be trained.
        num_classes:    The number of classes to predict.
        lr:     Learning rate used for the Adam optimizer.
        epochs: Number of training epochs to perform.
        train_loader:   The training pytorch Dataloader of the dataset.
        val_loader:     The validation pytorch Dataloader.
        progress_bar:   The progress_bar flag for tqdm.
        model_size:     The size for the BERT pretrained model used.
    Returns:
        model:  The trained Baseline model.
    """

    # Set the default device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and loss module.
    model = Baseline(name, num_classes, model_size)
    loss_module = nn.CrossEntropyLoss()

    # Set the model on the correct device.
    model.to(device)

    # Set the model in training mode.
    model.train()

    # Define a simple Adam optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize a variable.
    best_acc = 0

    # Loop over epochs using tqdm to show progress.
    for e in tqdm(range(epochs), leave=True, disable=progress_bar):

        # Loop over the batches in the Dataloader.
        for x, y, d in train_loader:

            # If there is a BERT model before the network,
            # find the mask and data appropriately.
            if name == 'civil':
                x_mask = x['attention_mask'].to(device)
                x = x['input_ids'].squeeze(1).to(device)
            else:
                x_mask = None
                x = x.to(device)

            # Move the data to device.
            y = y.to(device)
            d = d.to(device)

            # Perform a forward pass on the data with the model.
            joint_preds = model(x, x_mask)

            # Calculate the joint loss of the predictions.
            l0 = loss_module(joint_preds, y.long())

            # Before calculating the gradients, we need to ensure that they
            # are all zero. The gradients would not be overwritten, but
            # actually added to the previous ones otherwise.
            optimizer.zero_grad()

            # Add the joint backward to the optimizer.
            l0.backward()

            # Take an optimization step of the parameters.
            optimizer.step()

        # Check for validation data.
        if val_loader is not None:
            model, accuracy, _, _ = test_base(name, model, val_loader, 0.0)
            print("acc_val:", accuracy)

            # Save the model with the best validation accuracy as it
            # generalizes best.
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), './Models/{}_base.pt'.format(name))

            # Put the model in training mode.
            model.train()
        else:
            torch.save(model.state_dict(), './Models/{}_base.pt'.format(name))

    # Load the best model parameters into the model before returning it.
    model.load_state_dict(torch.load('./Models/{}_base.pt'.format(name)))

    return model
