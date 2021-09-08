"""
Fashion MNIST

Create and train a classifier for the Fashion MNIST dataset.

The neural network should output 10 classifications (other words - the last layer should have 10 neurons).
and that the input shape should be the normal size of the Fashion MNIST images (28x28).

Info: https://keras.io/api/datasets/fashion_mnist/

"""

import tensorflow as tf


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(units=128, activation="sigmoid"),
        tf.keras.layers.Dense(units=10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

    return model


# save model in the .h5 format.

if __name__ == "__main__":
    model = solution_model()
    model.save("fashion_mnist.h5")
