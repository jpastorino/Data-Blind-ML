import socket
import pickle

import numpy as np


# ======================================================================================================================
# ======================================================================================================================
class Client:
    #
    # **************************************************************************************************************
    def __init__(self, ip:str, port:int):
        self.__ip = ip
        self.__port = port
        self.__is_connected = False
        self.__client_socket = None

    #
    # **************************************************************************************************************
    def connect(self):
        self.__client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__client_socket.connect((self.__ip, self.__port))
        self.__is_connected = True

    #
    # **************************************************************************************************************
    def receive_message(self):
        """
        Receives bytes from the server.
        :return: the whole message in bytes
        """
        msg = self.__client_socket.recvmsg(4096)[0].decode("UTF-8")
        return msg

    #
    # **************************************************************************************************************
    def send_message(self, msg:str):
        self.__client_socket.send(msg.encode("UTF-8"))

    #
    # **************************************************************************************************************
    def receive_bytes(self, len_msg:int):
        chunks = []
        received_bytes = 0

        while received_bytes < len_msg:
            buff = self.__client_socket.recvmsg(4096)[0]
            if not buff == b"":
                chunks.append(buff)

            received_bytes += len(buff)

        msg = b"".join(chunks)
        # print(f'<<<<<<<<<< Received {len(msg)} bytes')
        return msg

    #
    # **************************************************************************************************************
    def disconnect(self):
        self.__client_socket.close()
        self.__is_connected = False

    #
    # **************************************************************************************************************
    @property
    def is_connected(self):
        return self.__is_connected


# ======================================================================================================================
# ======================================================================================================================
def disconnect_and_exit(a_client: Client):
    print()
    print(f"""{" " * 5}Disconnecting...""")
    a_client.send_message(f"""D|""")
    response = a_client.receive_message()
    print(f"""{" " * 5}Disconnected from server. [{response}]""")
    print()
    print(f"""{" " * 5}Application Terminated.""")


# ======================================================================================================================
# ======================================================================================================================
if __name__ == "__main__":
    DATA_PATH = "../data/generated/"
    print(f"""{"=" * 60}""")
    print(f"""{"Synthetic Data Retrieval Application":^60}""")
    print(f"""{"=" * 60}""")
    print()
    server_ip = input(f"""{" " * 5}Input Data Server IP Address [127.0.0.1]:""")
    server_port = input(f"""{" " * 5}Input Data Server Port [32000]:""")
    server_ip = server_ip if server_ip != "" else "127.0.0.1"
    server_port = int(server_port) if server_port != "" else 32000

    print()
    print(f"""{" " * 5}Connecting to Data Server {server_ip}:{server_port}""")
    client = Client(server_ip, server_port)
    client.connect()
    response = client.receive_message()
    print(f"""{" " * 5}{response}""")

    ds_id = input(f"""{" " * 5}Input Dataset ID [E.g. FBA6D. Empty to abort]:""").upper()
    if ds_id == "":
        print(f"""{" " * 5}Aborted by user.""")
        disconnect_and_exit(client)
        exit(1)
    else:
        qty = input(f"""{" " * 5}Input Qty of records to request [100]:""")
        qty = int(qty) if qty != "" else 100

        print()
        print(f"""{" " * 5}Requesting {qty} synthetic records for dataset {ds_id}...""")
        msg = f"""G|{ds_id}|{qty}"""
        client.send_message(msg)
        response = client.receive_message()
        arguments = response.split("|")
        if arguments[0] == "OK":
            response_bytes = client.receive_bytes(int(arguments[2])) # receive payload
            data_array = pickle.loads(response_bytes)
            print(f"""{" " * 5}Retrieved data with shape:{data_array.shape}""")
            print()
            filename = input(f"""{" " * 5}Input Filename to Save Numpy Array [data.npz]:""")
            filename = filename if filename != "" else "data.npz"
            np.savez(DATA_PATH+filename, data=data_array)
            print(f"""{" " * 5}Data saved to:{DATA_PATH+filename}""")
            print(f"""{" " * 5}Compressed file has a dimension called "data" with the array""")
        else:
            print(f"""{" " * 5}An Error received from server:{arguments[1]}""")

        disconnect_and_exit(client)
