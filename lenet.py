import keras
import keras.utils
from keras import datasets
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
)

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential()

model.add(
    Conv2D(
        6,
        kernel_size=(5, 5),
        activation="relu",
        input_shape=(32, 32, 3),
    )
)
# 15 Max Pool Layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# 13 Conv Layer
model.add(Conv2D(16, kernel_size=(5, 5), activation="relu"))
# 6 Max Pool Layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten the Layer for transitioning to the Fully Connected Layers
model.add(Flatten())
# 120 Fully Connected Layer
model.add(Dense(120, activation="relu"))
# 84 Fully Connected Layer
model.add(Dense(84, activation="relu"))
# 10 Output
model.add(Dense(10, activation="softmax"))


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# train
model.fit(x_train, keras.utils.to_categorical(y_train), epochs=10, batch_size=16)

# test
test_loss, test_acc = model.evaluate(x_test, keras.utils.to_categorical(y_test))
print("Test accuracy:", test_acc)
