import numpy as np
import pandas as pd
import sklearn as scikit
import tensorflow as tf
from preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# #####################################################################################################################
#  Implementation of Pre-Processing
# #####################################################################################################################
class MyPreprocess(Preprocessing):

    def prepare(self, in_data):
        # Data has a shape of (-1, 124) where the last 12 correspond to different events that can be predicted.
        # In this script we will evaluate the performance of predicting Atrial Fib, #113, and first target class.

        # Discarding ID and other targets.
        x = in_data[:, 1:112]  # First 112 features expect for the very first which is record id.

        y = in_data[:, 112].astype('float32')  # Using atrial fib targer with label 0/1

        return x, y


if __name__ == "__main__":
    print(f"""Using Tensorflow version {tf.__version__}""")
    print("*" * 80)
    print("""---- THIS IS THE EVALUATION OF THE MODEL TRAINED DIRECTLY WITH REAL DATA""")
    print("*" * 80)

    # ------------------------------------------------------------------------------------------------------------------
    # LOADING DATA
    data_df = pd.read_csv("../../data/source/miocardial.csv")
    data = data_df.values

    print(f"""MIOCARDIAL Real DS shape:{data.shape}""")

    # ------------------------------------------------------------------------------------------------------------------
    # Preprocessing
    #
    pre_proc = MyPreprocess()
    x, y = pre_proc.prepare(data)

    print(f"""Preprocessed data: x:{x.shape}, y:{y.shape}""")

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(f"""Train: x:{x_train.shape}, y:{y_train.shape}. Test: x:{x_test.shape}, y:{y_test.shape}""")

    # ------------------------------------------------------------------------------------------------------------------
    # DEFINING THE MODEL AND TRAINING
    model = tf.keras.models.Sequential(name="Miocardial_Real")
    model.add(tf.keras.layers.Dense(units=150, name="dense1", input_shape=[111]))
    model.add(tf.keras.layers.Dropout(0.8, name="dropout_1"))
    model.add(tf.keras.layers.Dense(units=150, name="dense2"))
    model.add(tf.keras.layers.Dropout(0.8, name="dropout_2"))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="dense3_softmax"))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    # ------------------------------------------------------------------------------------------------------------------
    # Training
    model.fit(x_train, y_train, batch_size=8, epochs=15)

    # ------------------------------------------------------------------------------------------------------------------
    # Local Evaluation
    print()
    print(f"={'Evaluating using Real data':^78}=")
    print(model.evaluate(x_test, y_test))
