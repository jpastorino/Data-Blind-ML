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
        x = data[: , 1:].reshape((-1, 28, 28, 1)) / 255.
        y = np.abs(data[:, 0])
        # Fixing labels.
        y = np.where(y < 0, 0, y)
        y = np.where(y > 9, 9, y)
        y = tf.keras.utils.to_categorical(y)
        return x, y


if __name__ == "__main__":
    MODEL_NAME = "MNIST_Synthetic"
    input_shape = (28, 28, 1)

    print (f"""Using Tensorflow version {tf.__version__}""")

    # ################################################################################
    # LOADING DATA
    mnist_synthetic = np.load("../../data/generated/mnist_synt.npz", allow_pickle=True)
    mnist_data = mnist_synthetic["data"]

    mnist_synthetic = np.load("../../data/generated/mnist_synt_2.npz", allow_pickle=True)
    mnist_data = np.append(mnist_data, mnist_synthetic["data"], axis=0)

    print(f"""MNIST DS Synthetic data shape:{mnist_data.shape}""")

    train, test = train_test_split(mnist_data, train_size=0.8)

    # ################################################################################
    # Preprocessing
    pre_proc = MyPreprocess()
    # x, y = pre_proc.prepare(mnist_data)
    x_train, y_train = pre_proc.prepare(train)
    x_test, y_test = pre_proc.prepare(test)

    # print(f"""Preprocessed data: x:{x.shape}, y:{y.shape}""")
    #
    # x_train, x_test, y_train, y_test =  train_test_split(x, y)
    print(f"""Train: x:{x_train.shape}, y:{y_train.shape}. Test: x:{x_test.shape}, y:{y_test.shape}""")

    # ################################################################################
    # DEFINING THE MODEL AND TRAINING
    model = tf.keras.models.Sequential(name=MODEL_NAME)
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  loss=tf.keras.losses.CategoricalCrossentropy()  ,
                  metrics=['accuracy'])

    model.summary()
    # ################################################################################
    # Training
    model.fit(x_train, y_train, batch_size=32, epochs=10)

    # ################################################################################
    # Local Evaluation
    print()
    print(f"={'Evaluating using synthetic data':^78}=")
    print(model.evaluate(x_test, y_test))

    # ################################################################################
    # Remote Evaluation
    eval = EvaluationClient("127.0.0.1", 35000)
    eval.evaluate_model("F8DBC", MODEL_NAME, model, pre_proc)
