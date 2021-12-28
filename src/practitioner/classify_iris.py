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
        x = data[: , 0:4]
        x = np.asarray(x).astype('float32')
        x = scikit.preprocessing.normalize(x)

        # labels encoding.
        y = data[: , 4]
        le = scikit.preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        return x, y


if __name__ == "__main__":
    print (f"""Using Tensorflow version {tf.__version__}""")

    # ################################################################################
    # LOADING DATA
    iris_synthetic = np.load("../../data/generated/iris_synt.npz", allow_pickle=True)
    iris_data = iris_synthetic["data"]

    print(f"""Iris Synthetic data shape:{iris_data.shape}""")

    # ################################################################################
    # Preprocessing
    pre_proc = MyPreprocess()
    x, y = pre_proc.prepare(iris_data)

    print(f"""Preprocessed data: x:{x.shape}, y:{y.shape}""")

    x_train, x_test, y_train, y_test =  train_test_split(x, y)
    print(f"""Train: x:{x_train.shape}, y:{y_train.shape}. Test: x:{x_test.shape}, y:{y_test.shape}""")

    # ################################################################################
    # DEFINING THE MODEL AND TRAINING
    model = tf.keras.models.Sequential(name="Iris_Synthetic")
    # model.add( tf.keras.layers.LayerNormalization( input_shape=[4],
    #                                                axis=-1, center=True, scale=True,
    #                                                trainable=True, name='input_normalized'))
    model.add(tf.keras.layers.Dense(units=150, name="dense1", input_shape=[4]))
    model.add(tf.keras.layers.Dropout(0.8, name="dropout_1"))
    model.add(tf.keras.layers.Dense(units=150, name="dense2"))
    model.add(tf.keras.layers.Dropout(0.8, name="dropout_2"))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="dense3_softmax"))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    # ################################################################################
    # Training
    model.fit(x_train, y_train, batch_size=8, epochs=15)

    # ################################################################################
    # Local Evaluation
    print()
    print(f"={'Evaluating using synthetic data':^78}=")
    print(model.evaluate(x_test, y_test))

    # ################################################################################
    # Remote Evaluation
    eval = EvaluationClient("goliath.ucdenver.pvt", 35000)
    eval.evaluate_model("79e3b","Iris_Synthetic", model, pre_proc)