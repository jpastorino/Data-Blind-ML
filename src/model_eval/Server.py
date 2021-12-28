import os
import socket
import pickle
import numpy as np
import tensorflow as tf
from threading import Thread
from abc import ABC, abstractmethod
from preprocessing import Preprocessing
from Dataset import MetadataDB, Dataset
from tensorflow.keras.models import Sequential, load_model


# ======================================================================================================================
# ======================================================================================================================
class Server:
    #
    # **************************************************************************************************************
    def __init__(self, ip: str, port: int, backlog: int, meta: MetadataDB):
        self.__ip = ip
        self.__port = port
        self.__backlog = backlog
        self.__server_socket = None
        self.__keep_running = True
        self.__metadata = meta
        self.__connection_count = 0
        self.__list_cw = []

    #
    # **************************************************************************************************************
    def terminate_server(self):
        self.__keep_running = False
        self.__server_socket.close()

    #
    # **************************************************************************************************************
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


# ======================================================================================================================
# ======================================================================================================================
class ClientWorker(Thread):
    #
    # **************************************************************************************************************
    def __init__(self, client_id: int, client_socket: socket, meta: MetadataDB, server: Server):
        super().__init__()
        self.__client_socket: socket = client_socket
        self.__keep_running_client = True
        self.__metadata = meta
        self.__server = server
        self.__id = client_id

    #
    # **************************************************************************************************************
    def run(self):
        self._send_message("Connected to Model Evaluation Server")

        while self.__keep_running_client:
            self._process_client_request()

        self.__client_socket.close()

    #
    # **************************************************************************************************************
    def terminate_connection(self):
        self.__keep_running_client = False
        self.__client_socket.close()

    #
    # **************************************************************************************************************
    def _display_message(self, message: str):
        print(f"MSG >> {message}")

    #
    # **************************************************************************************************************
    def _send_message(self, msg: str):
        self.__client_socket.send(msg.encode("UTF-8"))

    #
    # **************************************************************************************************************
    def _receive_message(self, max_length: int = 16_384) -> str:
        msg = self.__client_socket.recvmsg(max_length)[0]
        print(f'<<<<<[{msg}] len:{len(msg)}')
        msg = msg.decode("UTF-8")
        return msg

    #
    # **************************************************************************************************************
    def _send_bytes(self, msg: bytes):
        # self._display_message(f"""SEND>> {msg}""")
        print(f'sending {len(msg)} bytes')
        sent = self.__client_socket.send(msg)
        print(f'Sent {sent} bytes')

    #
    # **************************************************************************************************************
    def _receive_bytes(self, len_msg: int, max_length: int = 16_384) -> bytes:
        chunks = []
        received_bytes = 0

        while received_bytes < len_msg:
            buff = self.__client_socket.recvmsg(max_length)[0]
            if not buff == b"":
                chunks.append(buff)

            received_bytes += len(buff)

        msg = b"".join(chunks)

        print(f'<<<<<<<<<< Received {len(msg)} bytes')
        return msg

    # **************************************************************************************************************
    def _receive_file(self, filename, filesize: int, max_length: int = 16_384):
        """
        Receives a file.
        :param filename:
        :param max_length:
        :return:
        """
        print(f"Receiving file of {filesize} bytes....")
        received_bytes = 0
        with open(filename, 'wb') as file:
            while received_bytes< filesize:
                buff = self.__client_socket.recv(max_length)
                if not buff == b"":
                    file.write(buff)
                received_bytes += len(buff)
                    #
                    # if len(buff) >= max_length:
                    #     buff = self.__client_socket.recv(max_length)
                    # else:
                    #     buff = False
        print(f"Completed. Received {received_bytes} bytes")

    # **************************************************************************************************************
    def _process_client_request(self):
        client_message = self._receive_message()
        self._display_message(f"CLIENT SAID>>>{client_message}")

        arguments = client_message.split("|")
        try:
            if arguments[0] == "E":  # E|dataset_id|model_id|
                ds_id = str(arguments[1]).upper()
                model_id = arguments[2]

                dataset: Dataset = self.__metadata.get_dataset_by_id(ds_id)
                if dataset is None:
                    self._send_message("""ERR|No such dataset""")
                else:
                    self._send_message("""OK|Request OK. Receiving Payloads""") #payload in next message. maybe.

                    # 1st receive Preprocessing class
                    print("Receiving Class Name and CODE")
                    tmp = self._receive_message()  # Class name|code byte len
                    preproc_class_name = tmp.split("|")[0]
                    preproc_class_code_size = int(tmp.split("|")[1])
                    preproc_class_code = self._receive_bytes(preproc_class_code_size)

                    print("Loading Preprocessing class")
                    with open(f"""./{model_id}_preproc_class_temp.py""", mode="wb") as file:
                        file.write(b"""import numpy as np\n""")
                        file.write(b"""import tensorflow as tf\n""")
                        file.write(b"""import sklearn as scikit\n""")
                        file.write(b"""from preprocessing import Preprocessing\n""")
                        file.write(b"""\n""")
                        file.write(preproc_class_code)

                    preproc_class_module = __import__(f"""{model_id}_preproc_class_temp""")
                    preproc_class_class_ = getattr(preproc_class_module, preproc_class_name)
                    preproc: Preprocessing = preproc_class_class_()

                    # 2nd Receive model file.
                    print("Receive h5 model.")
                    tmp = self._receive_message()  # Class name|code byte len
                    model_file_size = int(tmp.split("|")[1])

                    self._receive_file(f"""./{model_id}.h5""", model_file_size)  # Model Payload -- should receive the TF model.

                    print("Loading model")
                    model: Sequential = load_model(f"""./{model_id}.h5""")

                    print("Loading test data")
                    _, test_data = dataset.load_split_files()

                    try:
                        print("preprocessing data")
                        x, y = preproc.prepare(test_data)
                        print(f"""Data shapes are x:{x.shape}, y:{y.shape}""")

                        print("evaluating model")
                        evaluation = model.evaluate(x, y)

                        print("Sending Response")
                        self._send_message(f"""OK|{evaluation[0]}|{evaluation[1]}""")

                    except Exception as e:
                        print(f"""Exception: {e}""")
                        self._send_message(f"""ERR|{str(e)}""")

                    try:
                        os.remove(f"""{model_id}_preproc_class_temp.py""")
                        os.remove(f"""./{model_id}.h5""")
                    except Exception as e:
                        print(f"""Exception when removing files: {e}""")

            elif arguments[0] == "D": # Disconnect
                response = "OK|Disconnect."
                self._send_message(response)
                self.__keep_running_client = False

            else:
                response = "ERR|Unknown Command.\n"
                self._send_message(response)
        except ValueError as ve:
            response = "ERR|" + str(ve)
            self._send_message(response)

