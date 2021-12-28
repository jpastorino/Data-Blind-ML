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

    def prepare(self, in_data):
        from sklearn.preprocessing import OneHotEncoder
        x = in_data[:, 2:8]

        ohe = OneHotEncoder()
        sample_type = ohe.fit_transform(x[:, 0].reshape(-1, 1)).toarray()
        x = np.append(x, sample_type, axis=1)
        x = np.delete(x, 0, axis=1)

        x = np.asarray(x).astype('float32')
        x = scikit.preprocessing.normalize(x)

        y = in_data[:, 8].astype('float32')

        return x, y


if __name__ == "__main__":
    print(f"""Using Tensorflow version {tf.__version__}""")

    # ------------------------------------------------------------------------------------------------------------------
    # LOADING DATA
    data_synthetic = np.load("../../data/generated/ai4i2020_synt.npz", allow_pickle=True)
    data_synthetic = data_synthetic["data"]

    print(f"""Ai4i Synthetic data shape:{data_synthetic.shape}""")

    # ------------------------------------------------------------------------------------------------------------------
    # Preprocessing
    pre_proc = MyPreprocess()
    x, y = pre_proc.prepare(data_synthetic)

    print(f"""Preprocessed data: x:{x.shape}, y:{y.shape}""")

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(f"""Train: x:{x_train.shape}, y:{y_train.shape}. Test: x:{x_test.shape}, y:{y_test.shape}""")

    # ------------------------------------------------------------------------------------------------------------------
    # DEFINING THE MODEL AND TRAINING
    model = tf.keras.models.Sequential(name="AI4I_Synthetic")
    model.add(tf.keras.layers.Dense(units=150, name="dense1", input_shape=[8]))
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
    print(f"={'Evaluating using synthetic data':^78}=")
    print(model.evaluate(x_test, y_test))

    # ------------------------------------------------------------------------------------------------------------------
    # Remote Evaluation
    eval = EvaluationClient("goliath.ucdenver.pvt", 35000)
    eval.evaluate_model("079A7", "AI4ISynthetic", model, pre_proc)
