import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess images
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Design CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate accuracy on test set
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Test Accuracy: {accuracy:.2f}")

# Generate dummy data and save to Excel
def generate_dummy_image_data(num_samples):
    np.random.seed(42)
    images = np.random.randint(0, 256, size=(num_samples, 28, 28), dtype=np.uint8)
    labels = np.random.randint(0, 10, size=num_samples)
    return images, labels

num_samples = 100
dummy_images, dummy_labels = generate_dummy_image_data(num_samples)

# Save dummy image data to Excel file
excel_file = 'imagedata.xlsx'
dummy_image_data = {'Image': [img.tolist() for img in dummy_images], 'Label': dummy_labels}
dummy_image_df = pd.DataFrame(dummy_image_data)
dummy_image_df.to_excel(excel_file, index=False)

# Load dummy image data from Excel
loaded_dummy_image_df = pd.read_excel(excel_file)
loaded_dummy_images = np.array([np.array(img) for img in loaded_dummy_image_df['Image']])
loaded_dummy_labels = loaded_dummy_image_df['Label'].values

# Perform predictions on loaded dummy images
loaded_predictions = model.predict(loaded_dummy_images.reshape(-1, 28, 28, 1))
loaded_predicted_labels = np.argmax(loaded_predictions, axis=1)

# Calculate accuracy on loaded dummy data
loaded_accuracy = accuracy_score(loaded_dummy_labels, loaded_predicted_labels)
print(f"Accuracy on Loaded Dummy Data: {loaded_accuracy:.2f}")
