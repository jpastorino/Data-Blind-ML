import sys
import pandas as pd
from time import time
from os import system, name
from sdv.tabular import CTGAN
from Dataset import MetadataDB, Dataset
from sklearn.utils.random import sample_without_replacement
from Exceptions import NoMetadataError, NoDatasetFoundError
from Server import Server

CTGAN_TRAIN_THRESHOLD_SIZE = 2000  # Max size to train CTGAN without asking the the user.
CTGAN_EPOCHS = 30
CTGAN_BATCH_SIZE = 100

def cls():
    if name == 'nt':# for windows
        _ = system('cls')
    else:# for mac and linux(here, os.name is 'posix')
        _ = system('clear')


def display_menu() -> int:
    cls()
    opt = -1
    opt_range = [0, 5, 99]
    while not opt_range[0] <= opt <= opt_range[1] and opt != opt_range[2]:
        print(f"""{"=" * 40}""")
        print(f"""{"Main Menu":^40}""")
        print(f"""{"=" * 40}""")
        print(f"""{" " * 5}{"0) Create New Metadata"}""")
        print(f"""{" " * 5}{"1) Load Current Metadata"}""")
        print(f"""{" " * 5}{"2) Add Dataset"}""")
        print(f"""{" " * 5}{"3) Print Current Metadata"}""")
        print(f"""{" " * 5}{"4) Learn Synthetic Model"}""")
        print(f"""{" " * 5}{"5) Start Server"}""")
        print(f"""{"99) Exit":>40}""")
        print(f"""{"-" * 40}""")
        opt = int(input(f"""{" " * 5}{"Select Option -> "}"""))

    return opt


def create_metadata():
    cont = input("Warning! this will delete all pre-existing metadata. Continue [yes/NO]?:")
    if cont.upper() == "YES":
        dataset_source = input("Metadata filename (do not include path) [metadata.csv]:")
        if dataset_source == "":
            dataset_source = "metadata.csv"
        m = MetadataDB(dataset_source)
        return m
    else:
        print("Aborted by user.")
        input("Press any key to continue...")
        return None


def load_metadata():
    dataset_source = input("Metadata filename (do not include path) [metadata.csv]:")
    if dataset_source == "":
        dataset_source = "metadata.csv"
    start_time = time()
    m = MetadataDB(dataset_source)
    m.load()
    end_time = time()
    print(f"""Loaded in {end_time-start_time:.2f} sec.""")
    return m


def print_metadata(meta):
    if meta is None:
        raise NoMetadataError("Invalid metadata. Load metadata first.")
    else:
        print(meta)
        input("Press any key to continue...")


def add_dataset(meta: MetadataDB):
    if meta is None:
        raise NoMetadataError("Invalid metadata. Load metadata first.")
    else:
        dataset_source = input("Dataset Source filename (do not include path):")

        start_time = time()
        meta.add_dataset(dataset_source)
        print("Dataset added successfully.")
        end_time = time()
        print(f"""Loaded in {end_time - start_time:.2f} sec.""")
        input("Press any key to continue...")


def learn_ctgan_model(meta: MetadataDB):
    if meta is None:
        raise NoMetadataError("Invalid metadata. Load metadata first.")

    cls()
    print("=" * 80)
    print(f"={'Select Dataset to learn Model':^78}=")
    print("=" * 80)
    ds: Dataset
    print(f"""{"ID":^15}|{"Source":^50}|{"Has SynMod?":^15}""")
    print("-" * 80)
    for ds in meta.datasets:
        if ds.has_synthetic_model:
            print(f"""{ds.ds_id:^15}|{ds.source_filename:^50}|{"Yes":^15}""")
        else:
            print(f"""{ds.ds_id:^15}|{ds.source_filename:^50}|{"NO":^15}""")
    print("-" * 80)
    input_ds_id = input(f"""{" " * 5}Input Dataset ID [empty to abort]>""")
    if input_ds_id.strip() == "":
        print("Aborted by User...")
        input("Press any key to continue...")
    else:
        selected_ds: Dataset = None
        for ds in meta.datasets:
            if ds.ds_id == input_ds_id.upper():
                selected_ds = ds
        if selected_ds is None:
            raise NoDatasetFoundError(f"""Can't find dataset with that id ({input_ds_id.upper()}). Try again.""")

        print("-" * 80)
        print(f"""={f"Learning Synthetic Model for {selected_ds.ds_id}":^78}=""")
        print("-" * 80)

        train, _ = selected_ds.load_split_files()

        if train.shape[0] > CTGAN_TRAIN_THRESHOLD_SIZE:
            print(f"""{" " * 5}Input Dataset Size is large ({train.shape[0]}).""")
            input_train_size = input(f"""{" " * 10}Select Max size for learning [{CTGAN_TRAIN_THRESHOLD_SIZE}]>""")
            if input_train_size == "":
                input_train_size = CTGAN_TRAIN_THRESHOLD_SIZE
            else:
                input_train_size = min(int(input_train_size),train.shape[0])
            print(f"""{" " * 5}Setting learning data size to {input_train_size}.""")
            if input_train_size > CTGAN_TRAIN_THRESHOLD_SIZE:
                print(f"""{" " * 5}NOTICE: This process may take some time.""")
            ids = sample_without_replacement(n_population=train.shape[0], n_samples=input_train_size)
            train = train[ids]
        print(f"""{" " * 5}Learning Synthetic model...""")
        start_time = time()
        df_train = pd.DataFrame(train, columns=["col" + str(i) for i in range(train.shape[1])])
        model_filename = "ctgan_" + selected_ds.ds_id + ".pkl"
        gen_model = CTGAN(batch_size=CTGAN_BATCH_SIZE, epochs=CTGAN_EPOCHS)
        gen_model.fit(df_train)
        gen_model.save(Dataset.config["MODELS_PATH"] + "/" + model_filename)
        meta.associate_synthetic_model(selected_ds.ds_id, model_filename)
        print("-" * 80)
        print(f"""={f"Model for {selected_ds.ds_id} Learned and Saved.":^78}=""")
        print("-" * 80)
        end_time = time()
        print(f"""Learnt in {end_time - start_time:.2f} sec.""")
        input("Press any key to continue...")


def run_server(meta: MetadataDB):
    if meta is None:
        raise NoMetadataError("Invalid metadata. Load metadata first.")
    else:
        print(meta)
        # server = Server("127.0.0.1", 32000, 10, metadata)
        server = Server("0.0.0.0", 32000, 10, metadata)
        server.run()
    pass


# #####################################################################################################################
#    MAIN PROGRAM
# #####################################################################################################################
if __name__ == "__main__":
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter(action='ignore')  # , category=FutureWarning)  #hide warnings.

    option: int = -1
    metadata: MetadataDB = None

    while True:
        try:
            option = display_menu()
            if option == 0: #Creates new metadata
                metadata = create_metadata()
            elif option == 1:  # Load Metadata
                metadata = load_metadata()
            elif option == 2:  # Add Dataset
                add_dataset(metadata)
            elif option == 3:  # Print Metadata
                print_metadata(metadata)
            elif option == 4:  # Learn CTGAN Model
                learn_ctgan_model(metadata)
            elif option == 5:  # Start Server
                run_server(metadata)
                pass
            elif option == 99:  # Terminate server and exit
                print("Terminating Server and Exiting...")
                exit(0)
        except Exception as e:
            print(f"""Error --> {e}""")
            input("Press any key to continue...")
