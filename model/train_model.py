import tensorflow as tf
from tensorflow.keras import layers, models
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("MNIST dataset loaded successfully!")


x_train = x_train / 255.0
x_test = x_test / 255.0

model = models.Sequential([

    layers.Flatten(input_shape=(28, 28)),

    layers.Dense(128, activation="relu"),

    layers.Dense(64, activation="relu"),

    layers.Dense(10, activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStarting model training...\n")

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n==========================================")
print("Model Evaluation")
print("==========================================")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save("model/mnist_ann.keras")

print("\nModel saved successfully!")
print("Location: model/mnist_ann.keras")