import os
import time
import socket
import pickle
import inspect
import numpy as np
from preprocessing import Preprocessing
from tensorflow.keras.models import Sequential


# ======================================================================================================================
# ======================================================================================================================
class EvaluationClient:
    def __init__(self, ip: str, port: int):
        """
        Connects to the evaluation server to evaluate the performance of the model.
        :param ip: evaluation server ip
        :param port: evaluation server port.
        """
        self.__ip = ip
        self.__port = port
        self.__client_socket = None

    def __connect(self):
        self.__client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__client_socket.connect((self.__ip, self.__port))

    #
    # **************************************************************************************************************
    def _receive_message(self, max_length: int = 16_384) -> str:
        """
        Receives bytes from the server.
        :return: the whole message in bytes
        """
        msg = self.__client_socket.recvmsg(max_length)[0]
        print(f'<<<<<[{msg}] len:{len(msg)}')
        msg = msg.decode("UTF-8")
        return msg

    #
    # **************************************************************************************************************
    def _send_message(self, msg: str):
        print(f">>>>[{msg}]")
        self.__client_socket.send(msg.encode("UTF-8"))
        # self.__client_socket.sendall(msg.encode("UTF-8"))
        time.sleep(1)

    #
    # **************************************************************************************************************
    def _receive_bytes(self, len_msg:int, max_length:int = 16_384) -> bytes:
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

    #
    # **************************************************************************************************************
    def _send_bytes(self, msg: bytes, sleep_time:float = 1):
        # self._display_message(f"""SEND>> {msg}""")
        # self.__client_socket.send(msg.encode("UTF-8"))
        print(f'sending {len(msg)} bytes')
        sent = self.__client_socket.send(msg)
        # sent = self.__client_socket.sendall(msg)
        print(f'Sent {sent} bytes')
        # self.__client_socket.
        time.sleep(sleep_time)

    #
    # **************************************************************************************************************
    def __disconnect(self):
        self.__client_socket.close()

    #
    # **************************************************************************************************************
    def evaluate_model(self, ds_id: str, model_id: str, model: Sequential, preprocessing: Preprocessing):
        """
        Evaluates a model remotely using the real data.
        :param ds_id: dataset id. should match the ds_id provided by the data owner.
        :param model_id: model id/name, for reference.
        :param model: model object
        :param preprocessing: preprocessing object. will be used to transform real test data to evaluate the model.
        """
        print(f"""{"=" * 80}""")
        print(f"""{"MODEL EVALUATION with REAL DATA.":^80}""")
        print(f"""{"=" * 80}""")

        # ### Connection
        print(f"""{" " * 5}Connecting to Evaluation server [{self.__ip}:{self.__port}]...""")
        self.__connect()
        response = self._receive_message()
        print(f"""{" " * 5}{response}""")

        # ### Evaluation
        print(f"""{" " * 5}Evaluating model [{model_id}] with real data on dataset [{ds_id}]...""")
        msg = f"""E|{ds_id}|{model_id}|"""
        self._send_message(msg)

        # Reading Response. May take a while
        response = self._receive_message()
        arguments = response.split("|")
        if arguments[0] == "OK":
            # Send Payload:
            # 1st preprocessing
            #  # Send Class name
            #  # Send Class code

            preproc_class_code_b = inspect.getsource(type(preprocessing)).encode("UTF-8")
            print("Sending Preproc Class name.")
            self._send_message(f'''{preprocessing.__class__.__name__}|{len(preproc_class_code_b)}''')
            print("Sending Preproc Class Code.")
            self._send_bytes(preproc_class_code_b)

            # 2nd Model
            print("Sending model.")
            model.save("./temp.h5")
            file_size = os.stat("./temp.h5").st_size
            self._send_message(f'''MODEL_SIZE|{file_size}''')
            sent_so_far = 0
            with open("./temp.h5",mode="rb") as model_file:
                buffer = model_file.read(16_384)
                while (buffer):
                    print(f'{sent_so_far}...',end="")
                    self._send_bytes(buffer,sleep_time=0.1)
                    sent_so_far += len(buffer)
                    buffer = model_file.read(16_384)
            print()
            # Reading Response. May take a while
            response = self._receive_message()
            arguments = response.split("|")
            if arguments[0] == "OK":
                print()
                print(f"""{" " * 5}Evaluation completed successfully.""")
                print(f"""{" " * 15}Loss:{float(arguments[1]):.8f} Accuracy:{float(arguments[2]):.8f}""")
            else:
                print(f"""{" " * 5}An Error received from while evaluating server: {arguments[1]}""")
        else:
            print(f"""{" " * 5}An Error received from server: {arguments[1]}""")

        print()
        print(f"""{" " * 5}Disconnecting...""")
        msg = f"""D|"""
        self._send_message(msg)
        response = self._receive_message()
        print(f"""{" " * 5}Disconnected from server.""")

        print()
        print(f"""{" " * 5}Evaluation concluded.""")
        print(f"""{"-" * 80}""")

        try:
            if os.path.exists("./temp.h5"):
                os.remove("./temp.h5")
        except:
            pass