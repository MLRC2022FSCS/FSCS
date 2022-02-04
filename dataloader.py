from utils import MapAdultDataset, MapCelebADataset, MapCivilCommentsDataset
from aif360.datasets import AdultDataset
from transformers import BertTokenizer, logging
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pickle
import os
import shutil
import aif360
import model
import math


# Disable BERT warning about on unused weights. This due to us using BERT
# for a different task than it was originally created for.
logging.set_verbosity_error()

# The continuous columns in the adult dataset.
ADULT_CONTINUOUS = ['age', 'fnlwgt', 'education_num', 'capital_gain',
                    'capital_loss', 'hours_per_week']


def normalization(table, categories):
    """
    The columns of a table are normalized to be between 0 and 1.

    Args:
        table:   The table containing the data that needs to be normalized.
        categories: The list of table columns names to be normalized.
    Returns:
        table:   The table with normalized categories.
    """
    # Loop over the categories to normalize them.
    for column in categories:
        # Find the extreme values of the column.
        min_val = table[column].min()
        max_val = table[column].max()

        # Remap the column values to be between 0 and 1.
        table[column] = (table[column] - min_val) / (max_val - min_val)
    return table


class LoadData(object):
    """
    Gather and process data.
    """
    def __init__(self, data_name) -> None:
        """
        The initialization of the module.

        Args:
            data_name: The name of the dataset that needs to be loaded.
        """
        super(LoadData, self).__init__()

        if data_name == 'adult':

            # Find the aif toolkit path.
            aif_path = os.path.abspath(aif360.__file__)
            aif_path = aif_path.replace('__init__.py', '') + 'data/raw/adult/'

            # Check if train, validation, and test data files are available.
            if not (os.path.isfile(aif_path + 'adult.data') and
                    os.path.isfile(aif_path + 'adult.test') and
                    os.path.isfile(aif_path + 'adult.names')):

                # Find the file directories.
                source = os.path.abspath(os.getcwd()) + "/Data/Adult/"
                get_files = os.listdir(source)

                # Copy the files to the aif toolkit.
                for f in get_files:
                    shutil.move(source + f, aif_path)

            results = self.preprocessAdult()

        elif data_name == 'celeba':
            # Find the path of the correct data.
            path = os.path.abspath(os.getcwd()) + "/Data/CelebA/"

            # Check if the data is still zipped.
            if not (os.path.isdir(path + "img_align_celeba/")):
                zipped = path + "img_align_celeba.zip"

                # Open the zip file in READ mode.
                with ZipFile(zipped, 'r') as zip:
                    # Extract all of the files.
                    print('Extracting all the files for CelebA now.')
                    zip.extractall(path=path)

                print('Done!')

            results = self.preprocessCelebA(path)

        elif data_name == 'civil':
            # Get the path to the data.
            path = os.path.abspath(os.getcwd()) + "/Data/CivilComments/"

            results = self.preprocessCivilComments(path)

        else:
            raise Exception('An invalid dataset name was chosen.')

        self.train = results[0]
        self.val = results[1]
        self.test = results[2]
        self.ratio = results[3]

    def preprocessAdult(self):

        # Load the data from AIF360, as described in the paper.
        ad = AdultDataset(label_name='income-per-year',
                          favorable_classes=['>50K', '>50K.'],
                          protected_attribute_names=['sex'],
                          privileged_classes=[['Male']],
                          categorical_features=['workclass', 'education',
                                                'marital-status', 'occupation',
                                                'relationship',
                                                'native-country', 'race'],
                          features_to_keep=[],
                          features_to_drop=[],
                          na_values=['?'],
                          custom_preprocessing=None,
                          metadata={'label_maps': [
                                        {1.0: '>50K', 0.0: '<=50K'}],
                                    'protected_attribute_maps': [
                                        {1.0: 'Male', 0.0: 'Female'}]})

        # Split the data as stated in the adult.names file.
        train_df_adult, test_df_adult = ad.split([30162])

        # Save the amount of features input.
        model.INPUT_SIZE_ADULT = len(train_df_adult.feature_names) - 1

        # Convert to pandas dataframes.
        train, _ = train_df_adult.convert_to_dataframe()
        test, _ = test_df_adult.convert_to_dataframe()

        # Replace the special chars.
        train.columns = train.columns.str.replace('-', '_')
        test.columns = test.columns.str.replace('-', '_')

        # Drop but the first 50 the female data with high income (d=0 y=1)
        # occurences to introduce bias.
        train.drop(train[(train.sex == 0.0) &
                   (train.income_per_year == 1.0)].index[50:],
                   axis=0, inplace=True)

        # Calculate the protected distribution for the train dataset.
        femaleDistribution = len(train[
                                (train.sex == 0.0)].index) / len(train.index)

        # Normalize the continuous columns in the data.
        train = normalization(train, ADULT_CONTINUOUS)
        test = normalization(test, ADULT_CONTINUOUS)

        # Remove the protected and target data.
        d_train = train.pop('sex')
        y_train = train.pop('income_per_year')
        x_train = train
        d_test = test.pop('sex')
        y_test = test.pop('income_per_year')
        x_test = test

        # Reset the index numbering in pandas.
        d_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        x_train.reset_index(drop=True, inplace=True)
        d_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        x_test.reset_index(drop=True, inplace=True)

        # Load data as a map-style dataset.
        train_data = MapAdultDataset(x_train, y_train, d_train)
        test_data = MapAdultDataset(x_test, y_test, d_test)

        return train_data, None, test_data, femaleDistribution

    def preprocessCelebA(self, path):

        images_folder = path + 'img_align_celeba/'
        eval_file = 'list_eval_partition.txt'
        attr_file = 'list_attr_celeba.txt'

        # Gather partition and attribution data.
        eval_data = pd.read_csv(path + eval_file, sep=" ",
                                header=None, names=["img", "partition"])
        attr_data = pd.read_csv(path + attr_file, sep=r"\s+", header=1)

        # Get partition idices.
        train_partition_idx = eval_data[eval_data.partition == 0].index.values
        val_partition_idx = eval_data[eval_data.partition == 1].index.values
        test_partition_idx = eval_data[eval_data.partition == 2].index.values

        # Get target and protected data.
        train_y = attr_data.iloc[train_partition_idx][['Blond_Hair']]
        train_d = attr_data.iloc[train_partition_idx][['Male']]
        val_y = attr_data.iloc[val_partition_idx][['Blond_Hair']]
        val_d = attr_data.iloc[val_partition_idx][['Male']]
        test_y = attr_data.iloc[test_partition_idx][['Blond_Hair']]
        test_d = attr_data.iloc[test_partition_idx][['Male']]

        # Find ratio of protected group (female) in dataset.
        femaleDistribution = (len(train_d[(train_d.Male == -1)].index) /
                              len(train_d.index))

        # Reset indexing Pandas adds.
        train_d.reset_index(drop=True, inplace=True)
        train_y.reset_index(drop=True, inplace=True)
        val_d.reset_index(drop=True, inplace=True)
        val_y.reset_index(drop=True, inplace=True)
        test_d.reset_index(drop=True, inplace=True)
        test_y.reset_index(drop=True, inplace=True)

        train_d[:] = np.where(train_d < 0, 0, train_d)
        train_y[:] = np.where(train_y < 0, 0, train_y)
        val_d[:] = np.where(val_d < 0, 0, val_d)
        val_y[:] = np.where(val_y < 0, 0, val_y)
        test_d[:] = np.where(test_d < 0, 0, test_d)
        test_y[:] = np.where(test_y < 0, 0, test_y)

        # Load data as map-style dataset.
        train_data = MapCelebADataset(images_folder,
                                      train_y.squeeze(), train_d.squeeze())
        val_data = MapCelebADataset(images_folder,
                                    val_y.squeeze(), val_d.squeeze())
        test_data = MapCelebADataset(images_folder,
                                     test_y.squeeze(), test_d.squeeze())

        return train_data, val_data, test_data, femaleDistribution

    def preprocessCivilComments(self, path):
        # Check if train, validation, and test data files are available.
        if not (os.path.isfile(path + 'pcivil_train.pkl') and
                os.path.isfile(path + 'pcivil_val.pkl') and
                os.path.isfile(path + 'pcivil_test.pkl')):
            self.unpackCivil(path)

        # Open train, validation, and test files respectively.
        with open(path + 'pcivil_train.pkl', 'rb') as f:
            df_train = pickle.load(f)

        with open(path + 'pcivil_val.pkl', 'rb') as g:
            df_val = pickle.load(g)

        with open(path + 'pcivil_test.pkl', 'rb') as h:
            df_test = pickle.load(h)

        # Extract x, y, d.
        train_x = df_train['comment_text']
        train_y = df_train['target']
        train_d = df_train['christian']

        val_x = df_val['comment_text']
        val_y = df_val['target']
        val_d = df_val['christian']

        test_x = df_test['comment_text']
        test_y = df_test['target']
        test_d = df_test['christian']

        # Find ratio of protected group (christian) in dataset.
        ratio = sum(train_d) / len(train_d)

        # Reset indexing Pandas adds.
        train_d.reset_index(drop=True, inplace=True)
        train_y.reset_index(drop=True, inplace=True)
        train_x.reset_index(drop=True, inplace=True)
        val_d.reset_index(drop=True, inplace=True)
        val_y.reset_index(drop=True, inplace=True)
        val_x.reset_index(drop=True, inplace=True)
        test_d.reset_index(drop=True, inplace=True)
        test_y.reset_index(drop=True, inplace=True)
        test_x.reset_index(drop=True, inplace=True)

        # Load data as map-style dataset.
        train_data = MapCivilCommentsDataset(train_x, list(train_y),
                                             list(train_d))
        val_data = MapCivilCommentsDataset(val_x, list(val_y), list(val_d))
        test_data = MapCivilCommentsDataset(test_x, list(test_y), list(test_d))

        return train_data, val_data, test_data, ratio

    def unpackCivil(self, path):
        # Load lower-case BERT tokenizer.
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Check if train.csv exists.
        try:
            df = pd.read_csv(path + 'train.csv')
        except IOError:
            print("File not found: train.csv.")

        print("Preprocessing CivilComments dataset:")
        # Drop lines with NaNs.
        df = df.dropna()

        print("---Starting tokenizing...")
        # Tokenize comments using BERT tokenizer.
        comment_text = [tokenizer(comment,
                                  padding='max_length', max_length=512,
                                  truncation=True, return_tensors="pt")
                        for comment in df['comment_text']]

        # Set target and christian to binary values.
        target = [1 if t >= 0.5 else 0 for t in df['target']]
        christian = [1 if c >= 0.5 else 0 for c in df['christian']]

        df_data = {"comment_text": comment_text,
                   "target": target,
                   "christian": christian}

        # Make new dataframe with only relevant columns.
        cleaned_df = pd.DataFrame(df_data)

        # Partition the data: 80% train, 10% validation, 10% test.
        train_len = math.floor(0.8 * len(cleaned_df))
        val_len = math.floor(0.1 * len(cleaned_df)) + train_len
        df_train = cleaned_df[:train_len]
        df_val = cleaned_df[train_len:val_len]
        df_test = cleaned_df[val_len:]

        print("---Starting dataframe writing...")
        # Save the results as pickles.
        with open(path + 'pcivil_train.pkl', 'wb') as f:
            pickle.dump(df_train, f)

        with open(path + 'pcivil_val.pkl', 'wb') as g:
            pickle.dump(df_val, g)

        with open(path + 'pcivil_test.pkl', 'wb') as h:
            pickle.dump(df_test, h)

        print("---Finished preprocessing!")
