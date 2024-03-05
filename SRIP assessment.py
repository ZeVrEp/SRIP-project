import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scikeras.wrappers import KerasClassifier
from sklearn.multiclass import  OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold

#splitting the data into training,validation and test

data_dir = "C:/archive(1)/animals/Dataset/animals/animals"   #path of dataset
image_size = (224, 224)


def load_and_preprocess_data(data_dir, image_size):
    images = []
    labels = []

    class_names = sorted(os.listdir(data_dir))

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)

        # Get a list of all image filenames in the class_path
        image_filenames = os.listdir(class_path)

        # Shuffle the list of image filenames to randomly select images
        random.shuffle(image_filenames)

        # Choose only the specified number of images per class
        selected_images = image_filenames[:10]

        for image_filename in selected_images:
            image_path = os.path.join(class_path, image_filename)

            # Load and preprocess the image
            img = load_img(image_path, target_size=image_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    print(f"Number of images loaded: {len(X)}")  # Print the total number of images loaded

    return X, y


# Load and preprocess the data
X, y = load_and_preprocess_data(data_dir, image_size)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# define the custom CNN model
def custom_CNNmodel(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

#ONEVSREST CLASSIFICATION
model_ivr=custom_CNNmodel((224,224,3),1)

#here we use onevsrest classifier from sklearn and use kerasclassifier for training
cnn_classifier = KerasClassifier(model_ivr, epochs=1, batch_size=32, verbose=1)
ovr = OneVsRestClassifier(cnn_classifier)

#fit model to training data
ovr.fit(X_train, y_train)

y_pred_ovr = ovr.predict(X_test)

accuracy_ovr = accuracy_score(y_test, y_pred_ovr)

print("Overall Accuracy (One-vs-Rest CNN):", accuracy_ovr)

# Confusion matrix
conf_matrix_1vsrest = confusion_matrix(y_test, y_pred_ovr)
print("Confusion Matrix (One-vs-Rest):\n", conf_matrix_1vsrest)

# Plot the confusion matrix
plt.imshow(conf_matrix_1vsrest, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('One-vs-Rest Confusion Matrix')
plt.colorbar()
plt.show()


# we assess the model using kfold cross validation

# Number of folds for cross-validation
num_folds = 3

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterate over the folds
for fold, (train_index, test_index) in enumerate(stratified_kfold.split(X_train, y_train)):
    print(f"\nFold {fold + 1}/{num_folds}")

    # Create train and validation sets for this fold
    X_fold_train, X_fold_val = X_train[train_index], X_train[test_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[test_index]

    
    model = custom_CNNmodel((224,224,3),1)  

    # Train the model on the current fold's training set
    model.fit(X_fold_train, y_fold_train, epochs=5, batch_size=32, validation_data=(X_fold_val, y_fold_val))

    # Evaluate the model on the current fold's test set
    accuracy = model.evaluate(X_test, y_test)[1]
    print(f"Accuracy on Fold {fold + 1}: {accuracy}")