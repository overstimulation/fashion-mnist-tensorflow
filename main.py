import matplotlib.pyplot as plt
import tensorflow as tf


def main():
    try:
        # Load the fashion_mnist dataset from Keras datasets
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Class names for Fashion MNIST
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    try:
        # Normalise the pixel values to the range [0, 1] for better training performance
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        # Define a simple neural network model using Keras Sequential API
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into a 1D array
                tf.keras.layers.Dense(128, activation="relu"),  # Hidden layer with 128 neurons and ReLU activation
                tf.keras.layers.Dense(
                    10, activation="softmax"
                ),  # Output layer with 10 neurons (one per class), softmax activation
            ]
        )

        # Compile the model with Adam optimiser, sparse categorical cross-entropy loss, and accuracy metric
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # Train the model on the training data for 5 epochs
        model.fit(train_images, train_labels, epochs=5)

        # Evaluate the model on the test data and print the test accuracy
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print("\nTest accuracy:", test_acc)

        # Make predictions on the test images
        predictions = model.predict(test_images)
        print("First prediction:", predictions[0])  # Print the probability distribution for the first test image
        print(
            "Predicted label:", tf.argmax(predictions[0]).numpy(), f"({class_names[tf.argmax(predictions[0]).numpy()]})"
        )  # Print the predicted class for the first test image

        # Visualise a few test images with predictions
        num_images = 5
        plt.figure(figsize=(10, 2))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(test_images[i], cmap=plt.cm.binary)
            pred_label = tf.argmax(predictions[i]).numpy()
            true_label = test_labels[i]
            color = "blue" if pred_label == true_label else "red"
            plt.xlabel(f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}", color=color)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during training or evaluation: {e}")


if __name__ == "__main__":
    main()
