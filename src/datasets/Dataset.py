"""
Defines the classes to manage the datasets and the CSV-based Metadata database.
"""
import os
import json
import string
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Exceptions import NoDatasetFoundError


class MetadataDB:
    """Metadata Database object. """

    META_PATH = "../data"

    def __init__(self, filename):
        """ Initializes the database from/to a csv file."""
        self._filename = filename
        self._datasets = []

    # ############################################################################################################
    def load(self):
        """
        Loads the model from a csv file (as it was initialized in the object).
        Updates the current object.
        :return: None.
        :raise FileNotFoundError.

        """
        if not os.path.exists(self.META_PATH + "/" + self._filename):
            raise FileNotFoundError(f"""The source file {self.META_PATH + "/" + self._filename} does not exist.""")

        self._datasets = []
        data = pd.read_csv(self.META_PATH + "/" + self._filename).fillna("")
        for _, row in data.iterrows():
            ds = Dataset(row["source"], row["id"], row["split"])
            if row["model"] != "":
                ds.synthetic_model = row["model"]
            self._datasets.append(ds)

    # ############################################################################################################
    def save(self):
        """
        Saves the current metadata to the file
        :return:
        """
        rows = []
        ds:Dataset
        for ds in self._datasets:
            row = {"id": ds.ds_id,
                   "source": ds.source_filename,
                   "split": ds.split_filename,
                   "model": ds.synthetic_model}
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.META_PATH + "/" + self._filename)

    # ############################################################################################################
    def add_dataset(self, source, train_pct=0.8):
        """
        Adds a dataset to the metadata. The dataset will be partitioned into train and test splits.
        :param source: dataset source filename. Must exist in Dataset Configuration Path.
        :param train_pct: Default 80-20 split.
        :return: None
        :raises FileExistsError if the source is already in the metadata.
        """
        ds: Dataset
        for ds in self._datasets:
            if ds.source_filename == source:
                raise FileExistsError(f"""The source {source} is already in the metadata.""")

        ds = Dataset(source, train_pct=train_pct, auto_load=True)
        self._datasets.append(ds)
        self.save()

    # ############################################################################################################
    def associate_synthetic_model(self, ds_id, model_filename):
        """
        Associates the model_filename (synthetic generative model) to the dataset.
        :param ds_id: the dataset id
        :param model_filename: the model filename
        :return:
        :raises: NoDatasetFoundError, FileNotFoundError
        """
        ds: Dataset
        for ds in self._datasets:
            if ds.ds_id == ds_id:
                ds.synthetic_model = model_filename
                self.save()
                return

        raise  NoDatasetFoundError(f"""No Dataset with ID ({ds_id}) was found.""")

    # ############################################################################################################
    def __repr__(self):
        s = f"""Metadata File: {self._filename}\n"""
        s += f"""Datasets:\n"""
        s += "-"*30 + "\n"
        for ds in self._datasets:
            s += str(ds) + "\n"
        return s

    # ############################################################################################################
    def get_dataset_by_id(self,ds_id: str):
        ds: Dataset
        for ds in self._datasets:
            if ds.ds_id == ds_id.upper():
                return ds
        return None

    # ############################################################################################################
    @property
    def datasets(self):
        return self._datasets



class Dataset:
    ID_LEN = 5
    config = {"SOURCE_PATH": "../data/source", "SPLIT_PATH": "../data/split", "MODELS_PATH": "../data/models"}

    def __init__(self, source_filename, ds_id=None, split_file=None, train_pct=0.8, auto_load=False):
        """
        Initializes the dataset.
        :param source_filename: source filename. e.g. iris.csv. iris.json must exists with metadata
        :param ds_id: Id. If it's autoload to True, id will be generated automatically.
        :param split_file: the compressed npz file containing train and test np arrays.
        :param train_pct: default 80-20  split/
        :param auto_load: If true will load the dataset.
        """
        self._source_filename = source_filename
        self._metadata_filename = ".".join(source_filename.split(".")[:-1])+".json"
        self._synthetic_model = None
        self._discrete_attributes = []
        with open(self.config["SOURCE_PATH"] + "/" + self._metadata_filename) as json_file:
            self._metadata = json.load(json_file)
            for k, v in self._metadata.items():
                if v == "CATEGORICAL" or v == "DISCRETE":
                    self._discrete_attributes.append(k)

        if auto_load:
            self._ds_id = ''.join(random.choice(string.hexdigits.upper()) for i in range(self.ID_LEN))
            self.generate_splits(train_pct)
        else:
            if (ds_id is None) or (split_file is None):
                raise ValueError("For not autoload, the id, train and test files need to be specified")
            else:
                self._ds_id = ds_id
                self._split_file = split_file

    def generate_splits(self, train_pct):
        """
        Loads the data, and generates a train/test split.
        :param train_pct: the train size. usually 0.8
        :return: None
        :raises: FileNotFoundError if the source filename does not exists in the sources path.
        """
        self._split_file = "split_" + self._ds_id + ".npz"

        if not os.path.exists(self.config["SOURCE_PATH"]+"/"+self._source_filename):
            raise FileNotFoundError(f"""The source file {self.config["SOURCE_PATH"]+"/"+self._source_filename} does not exist.""")
        data = pd.read_csv(self.config["SOURCE_PATH"]+"/"+self._source_filename)#.values()
        train, test = train_test_split(data, train_size=train_pct)
        np.savez(self.config["SPLIT_PATH"]+"/"+self._split_file, train=train, test=test)

    def load_split_files(self):
        """
        Load the split files and return those. Data is not stored in the object.
        :return: (train, test) data
        :raises: FileNotFoundError
        """
        if not os.path.exists(self.config["SPLIT_PATH"] + "/" + self._split_file):
            raise FileNotFoundError(
                f"""The split file file {self.config["SPLIT_PATH"] + "/" + self._split_file} does not exist.""")
        split_data_tmp = np.load(self.config["SPLIT_PATH"] + "/" + self._split_file, allow_pickle=True)
        return split_data_tmp["train"], split_data_tmp["test"]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"""Dataset {self._ds_id}. Source: {self._source_filename}. Split: {self._split_file}. Model:{self._synthetic_model}."""

    @property
    def ds_id(self):
        return self._ds_id

    @property
    def source_filename(self):
        return self._source_filename

    @property
    def split_filename(self):
        return self._split_file

    @property
    def has_synthetic_model(self):
        return self._synthetic_model is not None

    @property
    def synthetic_model(self):
        return self._synthetic_model

    @synthetic_model.setter
    def synthetic_model(self, value):
        if not os.path.exists(self.config["MODELS_PATH"] + "/" + value):
            raise FileNotFoundError(f"""The model file {self.config["MODELS_PATH"] + "/" + value} does not exist.""")

        self._synthetic_model = value

    @property
    def synthetic_model_full_path(self):
        return self.config["MODELS_PATH"] + "/" + self._synthetic_model

    @property
    def discrete_attributes(self):
        return self._discrete_attributes

    @property
    def metadata(self):
        return self._metadata


# ##################################################################################################################
# TESTING
# ##################################################################################################################
if __name__ == "__main__":

    print("Test Menu")
    print("1) Generate Metadata")
    print("2) Load Metadata")
    print("3) Associate Model")
    option = int(input("Option-> "))

    if option == 1:
        meta = MetadataDB("metadata.csv")
        meta.add_dataset("iris.csv")

        print(meta)

        meta.add_dataset("adult.csv")

        print(meta)

        try:
            meta.add_dataset("adult.csv")
        except FileExistsError as e:
            print(e)

    elif option == 2:
        meta = MetadataDB("metadata.csv")
        meta.load()
        print(meta)
        ds: Dataset
        print("Categorical/Discrete Attributes:")
        for ds in meta.datasets:
            print(ds.discrete_attributes)


    elif option == 3:
        meta = MetadataDB("metadata.csv")
        meta.load()
        print(meta)
        try:
            id1 = input("select id")
            val1 = input("non existent filename:")
            meta.associate_synthetic_model(id1,val1)

        except Exception as e:
            print(e)

        print(meta)
        try:
            id2 = input("select id")
            val2 = input("existent filename:")
            meta.associate_synthetic_model(id2,val2)

        except Exception as e:
            print(e)
        print(meta)