import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("hello")
# Function to load images and labels from a folder
def load_images_from_folder(folder):
    images = [] # Initialize empty list to store images
    labels = [] # Initialize empty list to store labels

    class_folders = [class_folder for class_folder in os.listdir(folder) if os.path.isdir(os.path.join(folder, class_folder))]

    for class_folder in class_folders:
        class_path = os.path.join(folder, class_folder)

        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)

            if img_path.endswith(('.jpg', '.jpeg', '.png')):
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(28, 28))  # Resize images to 28x28
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(class_folder)

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    #print("heelo3")
    train_folder_path = "/Users/abhudaysingh/Downloads/A,B,CNNS_with_Tim/veggie_heap_training"
    test_folder_path= "/Users/abhudaysingh/Downloads/A,B,CNNS_with_Tim/veggie_heap_testing"

    # Load training images and labels
    train_images, train_labels = load_images_from_folder(train_folder_path)

    # Load testing images and labels
    test_images, test_labels = load_images_from_folder(test_folder_path)

    # Encode labels using scikit-learn LabelEncoder
    label_encoder = LabelEncoder()
    encoded_train_labels = label_encoder.fit_transform(train_labels)
    encoded_test_labels = label_encoder.transform(test_labels)

    # Normalize pixel values to [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the CNN architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(train_labels)), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callback to print train accuracy after each epoch
    class PrintTrainAccuracy(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f'\nTrain accuracy for epoch {epoch+1}: {logs["accuracy"]:.4f}')

    # Train the model
    model.fit(train_images, encoded_train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[PrintTrainAccuracy()])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, encoded_test_labels)
    print('\nTest accuracy:', test_acc)

    # Evaluate train accuracy
    train_loss, train_acc = model.evaluate(train_images, encoded_train_labels)
    print('Train accuracy:', train_acc)
