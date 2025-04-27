import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

def convert_rgb_to_thermal(input_folder, output_folder, colormap=cv2.COLORMAP_INFERNO):
    """
    Converts RGB images to synthetic thermal-like images using a colormap.

    Args:
        input_folder (str): Path to the folder containing RGB images.
        output_folder (str): Path to the folder where thermal images will be saved.
        colormap (int): OpenCV colormap for thermal effect (default: cv2.COLORMAP_INFERNO).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for root, dirs, files in os.walk(input_folder):  # Recursively go through subfolders
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # Case-insensitive check for images
                img_path = os.path.join(root, filename)

                try:
                    img = cv2.imread(img_path)

                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                        thermal = cv2.applyColorMap(normalized, colormap)

                        # Maintain subfolder structure in the output folder
                        relative_path = os.path.relpath(root, input_folder)
                        output_subfolder = os.path.join(output_folder, relative_path)

                        if not os.path.exists(output_subfolder):
                            os.makedirs(output_subfolder)
                            print(f"Created subfolder: {output_subfolder}")

                        output_image_path = os.path.join(output_subfolder, filename)
                        success = cv2.imwrite(output_image_path, thermal)

                        # Check if the image was saved successfully
                        if success:
                            print(f"Processed and saved: {output_image_path}")
                        else:
                            print(f"Failed to save: {output_image_path}")

                    else:
                        print(f"Failed to read image: {img_path}")

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print(f"Conversion complete! Thermal images saved in: {output_folder}")

def preprocess_thermal_images(data_dir, output_dir, img_size=(224, 224)):
    image_data = []
    labels = []

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each folder (label) in the dataset
    for label, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        output_folder = os.path.join(output_dir, folder)

        # Check if the item is indeed a folder (e.g., animal class folder)
        if not os.path.isdir(folder_path):
            continue

        # Create output folder for each animal class if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(f"Processing folder: {folder} ({folder_path})")

        # Iterate through each image in the current folder
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            # Skip non-file items (subdirectories or anything else)
            if not os.path.isfile(img_path):
                print(f"Skipping non-file: {img_path}")
                continue

            try:
                # Check if it's a valid image file using Pillow
                with Image.open(img_path) as img_check:
                    img_check.verify()  # Verify if it's an actual image
            except (IOError, SyntaxError):
                print(f"Skipping non-image file: {img_path}")
                continue

            # Read the image as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # If image is None, skip it
            if img is None:
                print(f"Warning: Unable to load image {img_path}")
                continue

            try:
                # Resize the image and normalize it
                img = cv2.resize(img, img_size)  # Resize image to target size
                img = img / 255.0  # Normalize to [0, 1] range

                # Add to image data and labels
                image_data.append(img)
                labels.append(label)

                # Save the preprocessed image to the output directory (greyscale)
                save_path = os.path.join(output_folder, img_name)
                cv2.imwrite(save_path, (img * 255).astype(np.uint8))  # Save back to [0, 255]

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    # Return the data and labels as numpy arrays
    return np.array(image_data), np.array(labels)

# Path to your dataset folder
data_dir = '/Users/resindunavoda/PycharmProjects/Wild_Animal_Detection/Dataset/Thermal/animals'
output_dir = '/Users/resindunavoda/PycharmProjects/Wild_Animal_Detection/Dataset/Output/animals'
image_data, labels = preprocess_thermal_images(data_dir, output_dir)
X = np.array(image_data) # Example of random data for X (replace with actual data)
y = np.array(labels)


# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Build a simple CNN model
def build_model(input_shape=(224, 224, 1)):  # Single channel for thermal images
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Build the model
model = build_model(input_shape=(224, 224, 1))

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


def predict_image(img_path, model, img_size=(224, 224)):
    # Load and preprocess the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.resize(img, img_size)  # Resize to match input size
    img = img / 255.0  # Normalize image
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (1 channel for grayscale)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the image
    prediction = model.predict(img)

    if prediction >= 0.5:
        return 'Animal'
    else:
        return 'No Animal'


# Test prediction on a new thermal image
img_path = '/Users/resindunavoda/PycharmProjects/Wild_Animal_Detection/dee.jpg'
result = predict_image(img_path, model)
print(f"Prediction: {result}")

if __name__ == '__main__':
    # convert_rgb_to_thermal(
    #     "/Users/resindunavoda/Documents/MSE/ITS/Dataset/RGB/animals",
    #     "/Users/resindunavoda/Documents/MSE/ITS/Dataset/Thermal/animals"
    # )
    # convert_rgb_to_thermal(
    #     "/Users/resindunavoda/Documents/MSE/ITS/Dataset/RGB/non-animals",
    #     "/Users/resindunavoda/Documents/MSE/ITS/Dataset/Thermal/non-animals"
    # )

    # Preprocess the images
    X, y = preprocess_thermal_images(data_dir,output_dir)
    predict_image(img_path,model)
