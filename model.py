import random
import torch
import torchvision.models as torch_models
from transformers import AutoModel
import torch.nn as nn

# Global variable that is altered when loading the data.
INPUT_SIZE_ADULT = 0

# Supported BERT model sizes and its number of output features.
BERT_MODELS = {'tiny': 128,
               'mini': 256,
               'small': 512,
               'medium': 512,
               'base': 768}

# Amount of groups D, among which is the protected group.
NUM_GROUPS = 2


def get_phi(name, model_size='tiny'):
    """
    Function that returns the correct featurizer for the dataset.

    Args:
        name:           The name of the dataset that needs to be trained.
        model_size:     Size of BERT model to use on CivilComments dataset.
    Returns:
        out_features:   The size of the output features of the featurizer.
        pretrained:     A pretrained deep neural network, such as BERT.
        phi:            The featurizer of the dataset that will be trained.
    """

    def remove_classifier(model):
        """
        Function that removes the last layer of a given model.

        Args:
            model:  A pretrained model with layers.
        Returns:
            new_model:  The same model, but with its last layer removed.
        """
        return torch.nn.Sequential(*(list(model.children())[:-1]))

    # Whether a pretrained network is used. Defaults to None.
    pretrained = None

    if name == 'adult':
        # Define the output size of the layers.
        out_features = 80

        # Define the featurizer layers.
        layers = nn.Sequential(
            nn.Linear(INPUT_SIZE_ADULT, out_features),
            nn.SELU())

        # Make the featurizer module.
        phi = FeaturizerPhi(layers)

    elif name == 'celeba':
        out_features = 2048

        # Gather pytorch pretrained resnet50 model.
        layers = torch_models.resnet50(pretrained=True)

        # Remove the classifier, which is the last layer.
        layers = remove_classifier(layers)

        phi = FeaturizerPhi(layers)

    elif name == 'civil':
        # Raise error if unsupported model size is given.
        if model_size not in BERT_MODELS:
            raise ValueError(f"Invalid input {model_size} for argument 'model_size'. \nUse 'tiny', 'mini', 'small', 'medium' or 'base' instead.")

        out_features = 80

        # Gather pytorch pretrained BERT model of specified size.
        pretrained = AutoModel.from_pretrained(f'prajjwal1/bert-{model_size}')

        layers = nn.Sequential(
            nn.Linear(BERT_MODELS[model_size], out_features),
            nn.SELU())

        phi = FeaturizerPhi(layers)

    else:
        # Exception for when a wrong name was given.
        raise Exception('An invalid dataset name was chosen.')

    return out_features, pretrained, phi


class FeaturizerPhi(nn.Module):
    """
    The featurizer Phi module, which extracts the features from the data.
    """
    def __init__(self, layers) -> None:
        """
        The initialization of the module.

        Args:
            layers: The layers needed for this module depending on the dataset.
        """
        super(FeaturizerPhi, self).__init__()
        self.layers = layers

    def forward(self, x):
        """
        The forward step of the module.

        Args:
            x:  The data for which to extract the features.
        Returns:
            out: The extracted features of the data.
        """
        return self.layers(x)


class TwoLayerNN(nn.Module):
    """
    Two layer network that will form the sufficiency regularized model.
    Contains the joint classifier, along with the group- and random splits.
    """
    def __init__(self, name, num_classes, ratio, model_size) -> None:
        """
        The initialization of the two layer network.

        Args:
            name:           The name of the dataset that needs to be trained.
            num_classes:    The amount of target classes in the dataset.
            ratio:          Ratio of protected group datapoints within dataset.
            model_size:     Size of BERT model to use on CivilComments dataset.
        """
        super(TwoLayerNN, self).__init__()
        self.num_classes = num_classes
        self.ratio = ratio

        # Obtain hidden layer size, a pretrained network, if any,
        # and the featurizer.
        hidden_size, self.pretrained, self.phi = get_phi(name, model_size)

        self.groups = nn.ModuleList([
                        nn.Linear(hidden_size, 1) for _ in range(NUM_GROUPS)])
        self.joint = nn.Linear(hidden_size, num_classes)

    def forward_group(self, X, D, X_mask=None):
        """
        The forward for the fully connected group layers.

        Args:
            X:      The input data to the module.
            D:      The group of the input data.
            X_mask: Attention mask used for BERT model only.
        Returns:
            group_preds: Group-specific predictions to be used
                         for group-specific aggregate loss.
        """
        # Check if there is a BERT module before the featurizer.
        if self.pretrained:
            X = self.pretrained(X, X_mask)[1]

        # Receive features from featurizer.
        phi = self.phi(X).squeeze()

        # Prepare a tensor of correct shape for predictions.
        group_preds = torch.zeros(phi.shape[0],
                                  1, device=self.device)

        # Make predictions using the correct layer depending on group d.
        for i, (x, d) in enumerate(zip(phi, D)):
            group_preds[i] = self.groups[int(d.item())](x)

        return group_preds

    def forward(self, X, D, X_mask=None):
        """
        The forward pass of the module. Here X is transformed
        through several layer transformations.

        Args:
            X:      The input data to the module.
            D:      The protected attribute of the input data.
            X_mask: Attention mask used for BERT model only.
        Returns:
            jointPreds:             The output predictions by the
                                    sufficiency regularized joint classifier.
            group_specific_preds:   The group specific input predictions.
            group_agnostic_preds:   The group agnostic input predictions.
        """
        # Check if there is a network before the featurizer.
        if self.pretrained:
            X = self.pretrained(X, X_mask)[1]

        # Receive features from featurizer.
        phi = self.phi(X).squeeze()

        # Prepare tensors of correct shapes for predictions.
        group_specific_preds = torch.zeros(phi.shape[0], 1,
                                           device=self.device)
        group_agnostic_preds = torch.zeros(phi.shape[0], 1,
                                           device=self.device)

        # Make group specific and group agnostic predictions with the
        # fully connected layers. For which the former depends on the
        # protected attribute of the data and the latter on random
        # choise given the ratio of the protected and unprotected attribute.
        for i, (x, d) in enumerate(zip(phi, D)):
            group_specific_preds[i] = self.groups[int(d.item())](x)
            group_agnostic_preds[i] = self.groups[1
                                                  if random.random() < self.ratio
                                                  else 0](x)

        # Classify the features found by the featurizer.
        jointPreds = self.joint(phi)

        return jointPreds, group_specific_preds, group_agnostic_preds

    @property
    def device(self):
        """
        Returns the device on which the model is.
        """
        return next(self.parameters()).device


class Baseline(nn.Module):
    """
    The baseline module for the datasets. Not sufficiency regularized.
    """
    def __init__(self, name, num_classes, X_mask=None) -> None:
        """
        The initialization of the module.

        Args:
            name:           The name of the dataset.
            num_classes:    The amount of classes the dataset has.
            X_mask:         Attention mask used for BERT model only.
        """
        super(Baseline, self).__init__()

        # Get the featurizer, hidden_size and pretrained module.
        hidden_size, self.pretrained, self.phi = get_phi(name, X_mask)

        # Define the classifier layer.
        self.joint = nn.Linear(hidden_size, num_classes)

    def forward(self, X, X_mask=None):
        """
        Performs forward pass of the input. Here an input tensor x is
        transformed through several layer transformations.

        Args:
            X:      The data input to the module.
            X_mask: Attention mask used for BERT model only.
        Returns:
            pred:   The output predictions of the module.
        """
        # Check if there is a network before the featurizer.
        if self.pretrained is not None:
            X = self.pretrained(X, X_mask)[1]

        # Receive features from featurizer.
        phi = self.phi(X).squeeze()

        # Pass the features through the classifier.
        return self.joint(phi)

    @property
    def device(self):
        """
        Returns the device on which the model is.
        """
        return next(self.parameters()).device
