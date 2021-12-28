# Data-Blind ML: Building privacy-aware machine learning models without direct data access
This site contains source code and extra information of the paper, published in the IEEE International Conference on Artificial Intelligence &
Knowledge Engineering 2021. [IEEE AIKE 2021](https://www.ieee-aike.org/)

Publication link: [TBA](https://ieeexplore.ieee.org/Xplore/home.jsp)

## Authors
- [Javier Pastorino](https://cse.ucdenver.edu/~pastorij)
- [Ashis K. Biswas](https://cse.ucdenver.edu/~biswasa)

[Machine Learning Lab](http://ml.cse.ucdenver.edu) - University of Colorado, Denver

## Description
The project has the main goal of developing a methodology that allows a ML Practitioner to develop and train an 
ML model without accessing the dataset. This situation may arise due to several circumstances, but usually is 
due to privacy concerns and/or difficulties on sharing sensitive data that will make the process of having a 
third party developer.

In addition, this will enable situations where the data owner either has no knowledge of ML domain but has 
knowledge of his/her own data. Or moreover, the data owner does not possess enough computational resources to
train a model, particularly a DL model (e.g. GPU clusters, or server clusters). 


## Environments
The development was conducted using Python 3.8.
As the methodology uses TensorFlow as backend for development, but CTGAN models to generate synthetic data uses Pytorch, there will be then 
two different environments (as TF has incompatible dependencies with PyTorch).
- `datablind_tf`: is the environment that have TensorFlow. It is used for the `Model_eval` and `Model_devel` 
  programs/API. 
- `datablind`: is the environment that have PyTorch for CTGAN, and is used in the program that generates 
synthetic data (i.e. `data_server`)
  
The specific libraries are listed in `environments/requirements_datablind.txt` and 
`environments/requirements_datablind_tf.txt` for `datablind` and `datablind_tf` respectively.

## Sources Description

- `/src`
    - `/data_analysis`
        - Contains the scripts to complete the data analysis on the datasets. 
    - `/data_server`: is the application the data owner will use to generate a synthetic generator model and to run 
      the service that will provide the ML practitioner with synthetic data.
        - __Dependencies__:
            - socket
            - pickle
            - pandas
            - threading
            - sdv (CTGAN)
            - Dataset (MetadataDB, Dataset)
        - __Packages__:
            - `/Server.py`: implements the communication with the client and process client requests, such as 
          generate synthetic data and return it back to the client. 
        - __Applications__:
            - `/data_server_app.py`: Implements the server side of the application. _Intended for data owners (SiteA)_. 
              **Provides the following functionality:**
                - Load Metadata database, add new dataset to the database and display current database information.
                - Learn generative synthetic model for generating synthetic data for a specific dataset. 
                - Start the server to provide synthetic data generation to the client (intended to be the ML Practitioner)
            - `/data_client.py`: Implements the client side of the application. _Intended for ML Practitioners  (SiteB)_. 
              The application connects to the server, ask for the Dataset id and quantity of samples to retrieve. 
              If the input is correct request the data from the server and save it to a file. Otherwise, presents the user
              with the error. 
    - `/datasets`: package to manage the datasets
        - __Dependencies__:
            - string
            - random
            - numpy
            - pandas
            - sklearn
        - __Packages__:
            - `/Dataset.py`: contains the `MetadataDB` and `Dataset`
                - _**Classes**_:
                    - `MetadataDB` : manages the metadata of the datasets, including source file, and train-test 
                      split files.
                    - `Dataset`: manages one single dataset. Stores the source file, and the split for training/testing. 
    - `/devel_framework`: it is the package (API) the ML Practitioner will use to connect to the synthetic data server to 
      get synthetic data, and to connect to the model eval server to evaluate the current model to get some
      performance metrics in order to determine the performance of the model.   
        - __Dependencies__:
            - ABC, abstractmethod
            - socket
            - pickle
            - numpy 
            - inspect
            - Preprocessing
            - Tensorflow
        - __Classes__:
            - `/preprocessing.py/Preprocessing`: is the class that depicts how to implement the preprocessing. 
              currently implemented as a class with a single method `prepare(data)` that takes the data in and 
              outputs `x,y` arrays. 
              Has `numpy` imported as **np** and `scikit-learn` imported as `scikit` that can be used within the 
              prepare method. *more to come...*
            - `/evaluation.py/EvaluationClient`: is the class the practitioner should use to evaluate his/her model 
              to the real data. 
              Creates the object with the Evaluation server ip and port. 
              Use the `evaluate_model` method that takes the dataset id (provided by the data owner), a model id for
              reference, the model object (instance of keras.models.Sequential) and the preprocessing object 
              (instance of Preprocessing).
              Prints the Loss and Accuracy or an error in case of something happens while evaluating/communicating with
              the evaluation server.
    - `/exceptions`:
        - Contains exceptions using during the programs, such as NoDatasetFoundError and NoMetadataError.
    - `/model_eval`: is the application the data owner (SiteA) will use to provide and interface to the ML practitioner
      to evaluate the performance of the trained model without sharing the real data with him/her.
        - __Dependencies__:
            - abc
            - sys
            - os
            - socket
            - pickle
            - numpy
            - tensorflow
            - threading
            - Preprocessing
            - Dataset
            - Exceptions
        - __Packages__:
            - `/Server.py`: implements the evaluation server. Server and ClientWorker: process evaluation requests from clients, and evaluate models.
        - __Applications__:
            - `/model_eval_server_app.py`: implements the frontend for the data owner to load the metadata and start the evaluation server. It is the app for the data owner to start the evaluation server.
    - `/practitioner`: Practitioner development environment. Includes sample modeling for the six provided datasets.
    - `/real_data_evaluation`: prediction tasks using the six provided datasets in the paper using the real data to compare with the model that was trained with synthetic data. 
        
### Running on Ubuntu
- the path to find the packages needs to be added to the python path:
    -  `export PYTHONPATH=$PYTHONPATH:src/datasets:src/exceptions`

## Datasets Description
- `/data`
    - `/source`: contains the source data, i.e., the datasets to be used as input in the Data-Blind ML Pipeline development. This contains a `.csv` file with the data and a `.json` file with the description of files (name and type).
        1. Adult Dataset ([source](http://archive.ics.uci.edu/ml/datasets/Adult))
        2. AI4I 2020 Predictive Maintenance Dataset Data Set ([soruce](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset))
            - This is a synthetic, real-world maintenance dataset published at AI4I 2020.
        3. Iris Dataset ([source](http://archive.ics.uci.edu/ml/datasets/Adult))
        4. MNist Dataset
            - The MNist dataset was tabulated, meaning that the images were converted to a linear array of features. The goal 
            of adding this dataset is to have a large number of dimensions to evaluate the performance of the framework.
        5. Myocardial Infarction Complications ([source](https://archive.ics.uci.edu/ml/datasets/Myocardial+infarction+complications#))
        6. In-vehicle coupon recommendation DataSet ([source](https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation))
    - `/models`
        - stores the generated CTGAN models once trained.
    - `/split`
        - stores the train/test splits used for training CTGAN and for evaluating models respectively. 
    - `metadata.csv`   
        - stores the metadata of datasets, splits, and models available in the system.
    
## Data Analysis

The results of the skewness and curtosis analysis for the datasets is available in the folder `/data_analysis`.