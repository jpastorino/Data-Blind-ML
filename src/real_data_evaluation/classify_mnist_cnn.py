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
        x = data[: , 1:].reshape((-1, 28, 28, 1))
        x = x/255.
        y = np.abs(data[:, 0])
        y = tf.keras.utils.to_categorical(y)
        return x, y


if __name__ == "__main__":
    MODEL_NAME = "MNIST_Synthetic"
    input_shape = (28, 28, 1)

    print (f"""Using Tensorflow version {tf.__version__}""")
    print ("*" * 80)
    print ("""---- THIS IS THE EVALUATION OF THE MODEL TRAINED DIRECTLY WITH REAL DATA""")
    print("*" * 80)

    # ################################################################################
    # LOADING  REAL DATA
    mnist_data = pd.read_csv("../../data/source/mnist.csv")
    print(f"""MNIST DS shape:{mnist_data.shape}""")

    train, test = train_test_split(mnist_data.values[:7000], train_size=0.8)


    # ################################################################################
    # Preprocessing
    pre_proc = MyPreprocess()
    x_train, y_train = pre_proc.prepare(train)
    x_test, y_test = pre_proc.prepare(test)

    print(f"""TRAIN Preprocessed data: x:{x_train.shape}, y:{y_train.shape}""")
    print(f"""TEST Preprocessed data: x:{x_test.shape}, y:{y_test.shape}""")
    #
    # x_train, x_test, y_train, y_test =  train_test_split(x, y)
    # print(f"""Train: x:{x_train.shape}, y:{y_train.shape}. Test: x:{x_test.shape}, y:{y_test.shape}""")

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
    print(f"={'Evaluating using Real data':^78}=")
    print(model.evaluate(x_test, y_test))

