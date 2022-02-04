import torch
import torch.nn as nn


def test(name, model, test_loader, threshold):
    """
    Evaluation function that is used for the TwolayerNN model to find results.

    Args:
        name:   Name of the model that needs to be evaluated.
        model:  The TwolayerNN on which to evaluate.
        test_loader:    Dataloader given to evaluate.
        threshold:      The current threshold for selective classification.
    Returns:
        model:      The same model that was given as input.
        accuracy:   Accuracy of the model with the given threshold on the data.
        coverage:   The current coverage of the margin distiburion.
        results:    The calculated results of the evaluated data on the model.
    """

    # Set the default device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the model on the correct device.
    model.to(device)

    # Set the model in evaluation mode.
    model.eval()

    # Predefine a softmax layer.
    softmax = nn.Softmax(dim=1)

    # Initialize a dictionary with the calculated results.
    results = {"total_samples": 0,
               "pred_made": 0,
               "pred_correct_0": 0,
               "pred_correct_1": 0,
               "d_correct_0": 0,
               "d_correct_1": 0,
               "d_total_0": 0,
               "d_total_1": 0,
               "total_err": 0,
               "margins_0": [],
               "margins_1": []}

    # Loop over the batches in the dataloader.
    for x, y, d in test_loader:

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
        jointPreds, _, _ = model(x, d, x_mask)

        # Find the highest values and calculate the confidence scores.
        maxes = torch.max(jointPreds, dim=1)
        s = softmax(jointPreds)

        batch_k = 1/2 * torch.log(s/(1-s))

        results["total_samples"] += len(y)

        # Loop over the calculated confidence scores.
        for i, k in enumerate(batch_k):
            max_k = torch.max(k).item()

            # Apply selective classification.
            if max_k >= threshold:
                results["pred_made"] += 1

                # When a correct prediction is made save the results.
                if maxes[1][i] == y[i]:
                    results[f"pred_correct_{maxes[1][i]}"] += 1
                    results[f"d_correct_{int(d[i])}"] += 1
                    results[f"margins_{int(d[i])}"].append(max_k)

                # When an incorrect prediction is made save the error.
                else:
                    results["total_err"] += 1
                    results[f"margins_{int(d[i])}"].append(-max_k)

                results[f"d_total_{int(d[i])}"] += 1

            else:
                # Abstain from choosing
                jointPreds[i] = torch.tensor([-1, -1], device=device)

    # Calculate the accuracy when predictions are made.
    if results["pred_made"] > 0:
        accuracy = (results["pred_correct_0"] +
                    results["pred_correct_1"]) / results["pred_made"]
    else:
        accuracy = -1

    # Calculate the coverage for the given threshold.
    coverage = results["pred_made"] / results["total_samples"]

    return model, accuracy, coverage, results


def test_base(name, model, test_loader, threshold):
    """
    Evaluation function that is used for the Baseline model to find results.

    Args:
        name:   Name of the model that needs to be evaluated.
        model:  The Baseline model on which to evaluate.
        test_loader:    Dataloader given to evaluate.
        threshold:      The current threshold for selective classification.
    Returns:
        model:      The same model that was given as input.
        accuracy:   Accuracy of the model with the given threshold on the data.
        coverage:   The current coverage of the margin distiburion.
        results:    The calculated results of the evaluated data on the model.
    """
    # Set the default device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the model on the correct device.
    model.to(device)

    # Set the model in evaluation mode.
    model.eval()

    # Predefine a softmax layer.
    softmax = nn.Softmax(dim=1)

    # Initialize a dictionary with the calculated results.
    results = {"total_samples": 0,
               "pred_made": 0,
               "pred_correct_0": 0,
               "pred_correct_1": 0,
               "d_correct_0": 0,
               "d_correct_1": 0,
               "d_total_0": 0,
               "d_total_1": 0,
               "total_err": 0,
               "margins_0": [],
               "margins_1": []}

    # Loop over the batches in the dataloader.
    for x, y, d in test_loader:

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
        jointPreds = model(x, x_mask)

        # Find the highest values and calculate the confidence scores.
        maxes = torch.max(jointPreds, dim=1)
        s = softmax(jointPreds)

        batch_k = 1/2 * torch.log(s/(1-s))

        results["total_samples"] += len(y)

        # Loop over the calculated confidence scores.
        for i, k in enumerate(batch_k):
            max_k = torch.max(k).item()

            # Apply selective classification.
            if max_k >= threshold:
                results["pred_made"] += 1

                # When a correct prediction is made save the results.
                if maxes[1][i] == y[i]:
                    results[f"pred_correct_{maxes[1][i]}"] += 1
                    results[f"d_correct_{int(d[i])}"] += 1
                    results[f"margins_{int(d[i])}"].append(max_k)

                # When an incorrect prediction is made save the error.
                else:
                    results["total_err"] += 1
                    results[f"margins_{int(d[i])}"].append(-max_k)

                results[f"d_total_{int(d[i])}"] += 1

            else:
                # Abstain from choosing
                jointPreds[i] = torch.tensor([-1, -1], device=device)

    # Calculate the accuracy when predictions are made.
    if results["pred_made"] > 0:
        accuracy = (results["pred_correct_0"] +
                    results["pred_correct_1"]) / results["pred_made"]
    else:
        accuracy = -1

    # Calculate the coverage for the given threshold.
    coverage = results["pred_made"] / results["total_samples"]

    return model, accuracy, coverage, results
