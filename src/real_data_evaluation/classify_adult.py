import numpy as np
import pandas as pd
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
        metadata = {
            "age": "CONTINUOUS",
            "workclass": "CATEGORICAL",
            "fnlwgt": "CONTINUOUS",
            "education": "CATEGORICAL",
            "educ-num": "CONTINUOUS",
            "marital-status": "CATEGORICAL",
            "occupation": "CATEGORICAL",
            "relationship": "CATEGORICAL",
            "race": "CATEGORICAL",
            "sex": "CATEGORICAL",
            "capital-gain": "CONTINUOUS",
            "capital-loss": "CONTINUOUS",
            "hours-per-week": "CONTINUOUS",
            "native-country": "CATEGORICAL",
            "income": "CATEGORICAL"
        }

        x = data[: , 0:-1]
        # Encoding categorical attributes on the inputs.
        keys = list (metadata.keys())
        for i in range(x.shape[1]):
            key = keys[i]
            if metadata[key] == "CATEGORICAL":
                x_ = x[:, i]
                le = scikit.preprocessing.LabelEncoder()
                le.fit(x_)
                x_ = le.transform(x_)
                x[:, i] = x_

        x = np.asarray(x).astype('float32')
        # labels encoding.
        y = data[: , -1]
        le = scikit.preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        return x, y


if __name__ == "__main__":
    MODEL_NAME = "Adult_Real"
    input_shape = [14]

    print (f"""Using Tensorflow version {tf.__version__}""")
    print("*" * 80)
    print("""---- THIS IS THE EVALUATION OF THE MODEL TRAINED DIRECTLY WITH REAL DATA""")
    print("*" * 80)

    # ################################################################################
    # LOADING DATA
    adult_data = pd.read_csv("../../data/source/adult.csv").values
    print(f"""Adult Real DS data shape:{adult_data.shape}""")


    # ################################################################################
    # Preprocessing
    pre_proc = MyPreprocess()
    x, y = pre_proc.prepare(adult_data)

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
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="dense3_softmax"))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    # ################################################################################
    # Training
    model.fit(x_train, y_train, batch_size=8, epochs=100)

    # ################################################################################
    # Local Evaluation
    print()
    print(f"={'Evaluating using Real data':^78}=")
    print(model.evaluate(x_test, y_test))
