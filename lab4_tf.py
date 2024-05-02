import numpy as np 
import matplotlib.pyplot as plt  
from tensorflow.keras.datasets.mnist import load_data 
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 

(x_train, y_train), (x_test, y_test) = load_data()  # Loading MNIST dataset

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))  # Reshaping training data for CNN
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))  # Reshaping testing data for CNN

x_train = x_train.astype('float32') / 255.0  # Normalizing training data
x_test = x_test.astype('float32') / 255.0  # Normalizing testing data

model = Sequential([  # Creating a Sequential model
    Conv2D(32, (3,3), activation='relu', input_shape=x_train.shape[1:]),  # Adding a convolutional layer with ReLU activation
    MaxPooling2D((2, 2)),  # Adding a max pooling layer
    Conv2D(64, (3,3), activation='relu'),  # Adding another convolutional layer with ReLU activation
    MaxPooling2D((2, 2)),  # Adding another max pooling layer
    Dropout(0.25),  # Adding dropout regularization
    Flatten(),  # Flattening the output
    Dense(128, activation='relu'),  # Adding a dense layer with ReLU activation
    Dropout(0.5),  # Adding dropout regularization
    Dense(10, activation='softmax')  # Adding a dense layer with softmax activation for classification
])

model.summary()  # Displaying model summary

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compiling the model

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20,  # Setting rotation range for data augmentation
                             width_shift_range=0.1,  # Setting width shift range for data augmentation
                             height_shift_range=0.1,  # Setting height shift range for data augmentation
                             zoom_range=0.1)  # Setting zoom range for data augmentation

image = x_train[30]  # Selecting an image for visualization
image = image.reshape((1,) + image.shape)  # Reshaping the image to (1, height, width, channels)

plt.figure(figsize=(10, 10))  # Creating a figure for visualization
plt.suptitle('Customized Data Augmentation Example', fontsize=16)  # Adding a title to the figure
i = 0
for batch in datagen.flow(image, batch_size=1):  # Generating augmented images
    plt.subplot(3, 3, i+1)  # Creating subplots
    plt.grid(False)  # Turning off grid
    plt.imshow(batch.reshape(image.shape[1:3]), cmap='gray')  # Displaying augmented images
    if i == 8:  # Breaking after 9 images
        break
    i += 1
plt.show()

# Callbacks
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')  # Adding model checkpoint to save the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Adding early stopping to prevent overfitting

# Fit the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=128, shuffle=True),  # Training the model with data augmentation
                    epochs = 15,  # Setting number of epochs
                    steps_per_epoch = len(x_train) // 128,  # Setting steps per epoch
                    validation_data = (x_test, y_test),  # Providing validation data
                    callbacks = [checkpoint, early_stopping],  # Adding callbacks
                    verbose=2)  # Setting verbosity level

model.load_weights("best_model.keras")  # Loading the best model

plt.figure(figsize=(12, 5))  # Creating a figure for plotting

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)  
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('Customized Model Accuracy')  
plt.ylabel('Accuracy')  
plt.xlabel('Epoch')  
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2) 
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('Customized Model Loss')  
plt.ylabel('Loss')  
plt.xlabel('Epoch')  
plt.legend(['Train', 'Validation'], loc='upper left') 

plt.show() 

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)  # Evaluating model on test data
print(f'Customized Model Accuracy: {accuracy*100}%') 

images = x_test[:20]  # Selecting first 20 test images
predicted_labels = []  # Initializing list to store predicted labels

plt.figure(figsize=(10, 4)) 
plt.suptitle('Customized Model Predictions', fontsize=16)  
for i, image in enumerate(images):
    plt.subplot(2, 10, i + 1)
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.axis('off')

    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    p = model.predict([image])
    predicted_label = np.argmax(p)
    predicted_labels.append(predicted_label)

    plt.title(f'Pred: {predicted_label}, Actual: {y_test[i]}')

plt.show()
