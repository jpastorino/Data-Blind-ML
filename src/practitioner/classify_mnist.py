import numpy as np
import sklearn as scikit
import tensorflow as tf
from preprocessing import Preprocessing
from evaluation import EvaluationClient
from sklearn.model_selection import train_test_split


# #####################################################################################################################
#  Implementation of Pre-Processing
# #####################################################################################################################
class MyPreprocess(Preprocessing):

    def prepare(self, data):
        x = data[: , 1:]
        y = np.abs(data[:, 0])
        # Fixing labels.
        y = np.where(y < 0, 0, y)
        y = np.where(y > 9, 9, y)

        return x, y


if __name__ == "__main__":
    MODEL_NAME = "MNIST_Synthetic"
    input_shape = [784]

    print (f"""Using Tensorflow version {tf.__version__}""")

    # ################################################################################
    # LOADING DATA
    mnist_synthetic = np.load("../../data/generated/mnist_synt.npz", allow_pickle=True)
    mnist_data = mnist_synthetic["data"]

    print(f"""Adult DS Synthetic data shape:{mnist_data.shape}""")


    # ################################################################################
    # Preprocessing
    pre_proc = MyPreprocess()
    x, y = pre_proc.prepare(mnist_data)

    print(f"""Preprocessed data: x:{x.shape}, y:{y.shape}""")

    x_train, x_test, y_train, y_test =  train_test_split(x, y)
    print(f"""Train: x:{x_train.shape}, y:{y_train.shape}. Test: x:{x_test.shape}, y:{y_test.shape}""")

    # ################################################################################
    # DEFINING THE MODEL AND TRAINING
    model = tf.keras.models.Sequential(name=MODEL_NAME)
    model.add( tf.keras.layers.LayerNormalization( input_shape=input_shape,
                                                   axis=-1, center=True, scale=True,
                                                   trainable=True, name='input_normalized'))
    model.add(tf.keras.layers.Dense(units=150, name="dense1"))
    model.add(tf.keras.layers.Dropout(0.8, name="dropout_1"))
    model.add(tf.keras.layers.Dense(units=150, name="dense2"))
    model.add(tf.keras.layers.Dropout(0.8, name="dropout_2"))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="dense3_softmax"))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    # ################################################################################
    # Training
    model.fit(x_train, y_train, batch_size=8, epochs=100)

    # ################################################################################
    # Local Evaluation
    print()
    print(f"={'Evaluating using synthetic data':^78}=")
    print(model.evaluate(x_test, y_test))

    # ################################################################################
    # Remote Evaluation
    eval = EvaluationClient("127.0.0.1", 35000)
    eval.evaluate_model("F8DBC",MODEL_NAME, model, pre_proc)
