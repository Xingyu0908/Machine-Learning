import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# CNN requires a channel dimension (grayscale channel is 1)
x_train = x_train[..., tf.newaxis]  # Shape: (samples, 28, 28, 1)
x_test = x_test[..., tf.newaxis]

# Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Evaluate model
(loss, accuracy) = model.evaluate(x_test, y_test, verbose=2)
print(f"Loss on test set: {loss}") 
print(f"Accuracy on test set: {accuracy}")
model.save('my_model.keras')


# Test model
from PIL import Image
img = Image.open('data.Number.png')
img = img.convert('L')  # Convert to grayscale
img = img.resize((28, 28))  # Resize
img = tf.keras.preprocessing.image.img_to_array(img)
img = img / 255.0  # Normalize
img = img[..., tf.newaxis]  # Add channel dimension
predictions = model.predict(tf.expand_dims(img, axis=0))
print(predictions)
