import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import SGD


# define cnn model
def define_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Load dataset
def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # Chuyển dạng của đầu vào dưới dạng [6000,28,28]-> [6000,28,28,1]
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    # Chuyển dạng labels dạng [6000,1] thành [6000,10]
    # Ví dụ 0=> [1,0,0,0,0,0,0,0,0,0] có kích thước 10x1
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels


# Chuẩn hóa dữ liệu
def normalize_images(image):
    image = image / 255.0
    return image


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_dataset()
    model = define_model()
    model.summary()
    model.fit(train_images, train_labels, epochs=5)
    model.evaluate(test_images, test_labels)
    model.save("model.tf")
    exit(0)
