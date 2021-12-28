import socket
import pickle
import pandas as pd
from threading import Thread
from sdv.tabular import CTGAN
from Dataset import MetadataDB, Dataset

class Server:
    def __init__(self, ip: str, port: int, backlog: int, meta: MetadataDB):
        self.__ip = ip
        self.__port = port
        self.__backlog = backlog
        self.__server_socket = None
        self.__keep_running = True
        self.__metadata = meta
        self.__connection_count = 0
        self.__list_cw = []

    def terminate_server(self):
        self.__keep_running = False
        self.__server_socket.close()

    def run(self):
        self.__server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__server_socket.bind((self.__ip, self.__port))
        self.__server_socket.listen(self.__backlog)

        while self.__keep_running:
            print(f"""[SRV] Waiting for Client""")
            try:
                client_socket, client_address = self.__server_socket.accept()
                print(f"""[SRV] Got a connection from {client_address}""")
                self.__connection_count += 1
                cw = ClientWorker(self.__connection_count, client_socket, self.__metadata, self)
                self.__list_cw.append(cw)
                cw.start()
            except Exception as e:
                print(e)

        cw: ClientWorker
        for cw in self.__list_cw:
            cw.terminate_connection()
            cw.join()


# *******************************************************************************************************
class ClientWorker(Thread):
    def __init__(self, client_id: int, client_socket: socket, meta: MetadataDB, server: Server):
        super().__init__()
        self.__client_socket: socket = client_socket
        self.__keep_running_client = True
        self.__metadata = meta
        self.__server = server
        self.__id = client_id

    def run(self):
        self._send_message("Connected to Data-Server Application\n")

        while self.__keep_running_client:
            self._process_client_request()

        self.__client_socket.close()

    def terminate_connection(self):
        self.__keep_running_client = False
        self.__client_socket.close()

    def _display_message(self, message: str):
        print(f"MSG >> {message}")

    def _send_bytes(self, msg: bytes):
        # self._display_message(f"""SEND>> {msg}""")
        # self.__client_socket.send(msg.encode("UTF-8"))
        print(f'sending {len(msg)} bytes')
        sent = self.__client_socket.send(msg)
        print(f'Sent {sent} bytes')
        # self.__client_socket.
        pass

    def _send_message(self, msg: str):
        self.__client_socket.send(msg.encode("UTF-8"))

    def _receive_message(self, max_length: int = 4096):
        msg = self.__client_socket.recvmsg(max_length)[0].decode("UTF-8")
        return msg

    def _process_client_request(self):
        client_message = self._receive_message()
        self._display_message(f"CLIENT SAID>>>{client_message}")

        arguments = client_message.split("|")
        try:
            if arguments[0] == "G":  # G|FBA6D|100
                ds_id = str(arguments[1]).upper()
                qty = int(arguments[2])
                dataset: Dataset = self.__metadata.get_dataset_by_id(ds_id)
                if dataset is None:
                    # self._send_message(b"""ERR|No such dataset""")
                    self._send_message(f"""ERR|No such dataset""")
                else:
                    ctgan_model: CTGAN = CTGAN.load(dataset.synthetic_model_full_path)
                    new_data: pd.DataFrame = ctgan_model.sample(qty)
                    # print(new_data.values)
                    bytes_to_send=pickle.dumps(new_data.values)
                    # self._send_message(b"""OK|Request OK|""")
                    self._send_message(f"""OK|Request OK|{len(bytes_to_send)}""")
                    # self._send_message( pickle.dumps(new_data.values) )
                    self._send_bytes(bytes_to_send)
            elif arguments[0] == "D":
                response = "OK|Disconnect.\n"
                self._send_message(response)
                self.__keep_running_client = False
            else:
                response = "ERR|Unknown Command.\n"
                self._send_message(response)
        except ValueError as ve:
            response = "ERR|" + str(ve)
            self._send_message(response)

