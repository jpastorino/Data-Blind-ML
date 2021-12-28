import sys
import pandas as pd
from os import system, name
from Dataset import MetadataDB, Dataset
from Exceptions import NoMetadataError, NoDatasetFoundError
from Server import Server

def cls():
    if name == 'nt':# for windows
        _ = system('cls')
    else:# for mac and linux(here, os.name is 'posix')
        _ = system('clear')


def display_menu() -> int:
    cls()
    opt = -1
    opt_range = [1,3, 99]
    while not opt_range[0] <= opt <= opt_range[1] and opt != opt_range[2]:
        print(f"""{"=" * 40}""")
        print(f"""{"Main Menu":^40}""")
        print(f"""{"=" * 40}""")
        print(f"""{" " * 5}{"1) Load Current Metadata"}""")
        print(f"""{" " * 5}{"2) Print Current Metadata"}""")
        print(f"""{" " * 5}{"3) Start Server"}""")
        print(f"""{"99) Exit":>40}""")
        print(f"""{"-" * 40}""")
        opt = int(input(f"""{" " * 5}{"Select Option -> "}"""))

    return opt

def load_metadata():
    dataset_source = input("Metadata filename (do not include path) [metadata.csv]:")
    if dataset_source == "":
        dataset_source = "metadata.csv"
    m = MetadataDB(dataset_source)
    m.load()
    return m


def print_metadata(meta):
    if meta is None:
        raise NoMetadataError("Invalid metadata. Load metadata first.")
    else:
        print(meta)
        input("Press any key to continue...")


def run_server(meta: MetadataDB):
    if meta is None:
        raise NoMetadataError("Invalid metadata. Load metadata first.")
    else:
        print(meta)
        # server = Server("127.0.0.1", 35000, 10, metadata)
        server = Server("0.0.0.0", 35000, 10, metadata)
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
            if option == 1:  # Load Metadata
                metadata = load_metadata()
            elif option == 2:  # Print Metadata
                print_metadata(metadata)
            elif option == 3:  # Start Server
                run_server(metadata)
                pass
            elif option == 99:  # Terminate server and exit
                print("Terminating Server and Exiting...")
                exit(0)
        except Exception as e:
            print(f"""Error --> {e}""")
            input("Press any key to continue...")
