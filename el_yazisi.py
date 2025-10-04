import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='linear')
])

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.1,
    batch_size=32
)

random_image = random.randint(0, len(X_test)-1)
sample_image = X_test[random_image]
plt.imshow(sample_image, cmap='gray')
plt.show()

prediction = model.predict(sample_image.reshape(1,28,28))
predicted_label = np.argmax(prediction)
print("Tahmin edilen rakam:", predicted_label)
print("Olasılıklar:", prediction)
