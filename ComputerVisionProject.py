#!/usr/bin/env python
# coding: utf-8

# Computer Vision for COVID-19 Detection in X-ray Images

# Import necessary libraries
import os
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionV3
from sklearn.metrics import classification_report, confusion_matrix

# Define paths for the datasets
data_directory = '/content/drive/My Drive/ProcessedData'
train_dir = os.path.join(data_directory, 'train')
val_dir = os.path.join(data_directory, 'val')
test_dir = os.path.join(data_directory, 'test')

# Image data augmentation configuration for training dataset
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Image data generator for validation and test datasets
val_data_gen = ImageDataGenerator(rescale=1./255)

# Model training using VGG-16 as the base model
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model_vgg16.layers:
    layer.trainable = False

# Add new layers on top of VGG16
vgg16_top_layers = Flatten()(base_model_vgg16.output)
vgg16_top_layers = Dense(256, activation='relu')(vgg16_top_layers)
vgg16_top_layers = Dropout(0.5)(vgg16_top_layers)
output_layer_vgg16 = Dense(1, activation='sigmoid')(vgg16_top_layers)
model_vgg16 = Model(inputs=base_model_vgg16.input, outputs=output_layer_vgg16)

# Model compilation
model_vgg16.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model_vgg16.summary()

# Prepare the data generators
train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_data_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = val_data_gen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Fit the model
history_vgg16 = model_vgg16.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10
)

# Evaluate the model on test data
predictions_vgg16 = model_vgg16.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
predictions_vgg16 = (predictions_vgg16 > 0.5).astype(int)
true_classes_vgg16 = test_generator.classes
conf_matrix_vgg16 = confusion_matrix(true_classes_vgg16, predictions_vgg16)
print("Confusion Matrix for VGG-16:")
print(conf_matrix_vgg16)
classification_report_vgg16 = classification_report(true_classes_vgg16, predictions_vgg16, target_names=['Normal', 'COVID-19'])
print("Classification Report for VGG-16:")
print(classification_report_vgg16)

# Inception V3 Model Setup
# Initialize Inception V3 with pre-trained ImageNet weights
base_model_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
for layer in base_model_inception.layers:
    layer.trainable = False

# Build the top layers for our Inception V3 model
inception_top_layers = GlobalAveragePooling2D()(base_model_inception.output)
inception_top_layers = Dense(1024, activation='relu')(inception_top_layers)
inception_top_layers = Dropout(0.5)(inception_top_layers)
output_layer_inception = Dense(1, activation='sigmoid')(inception_top_layers)
model_inception = Model(inputs=base_model_inception.input, outputs=output_layer_inception)

# Compile the Inception model
model_inception.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model_inception.summary()

# Fit the Inception model
history_inception = model_inception.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10
)

# Evaluate the Inception model on test data
predictions_inception = model_inception.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
predictions_inception = (predictions_inception > 0.5).astype(int)
true_classes_inception = test_generator.classes
conf_matrix_inception = confusion_matrix(true_classes_inception, predictions_inception)
print("Confusion Matrix for Inception V3:")
print(conf_matrix_inception)
classification_report_inception = classification_report(true_classes_inception, predictions_inception, target_names=['Normal', 'COVID-19'])
print("Classification Report for Inception V3:")
print(classification_report_inception)

# AlexNet Model Setup
# Define the AlexNet architecture
model_alexnet = Sequential([
    Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)),
    MaxPooling2D(3, strides=2),
    Conv2D(256, (5, 5), padding='same', activation='relu'),
    MaxPooling2D(3, strides=2),
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(384, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(3, strides=2),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the AlexNet model
model_alexnet.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model_alexnet.summary()

# Fit the AlexNet model
history_alexnet = model_alexnet.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10
)

# Evaluate the AlexNet model on test data
predictions_alexnet = model_alexnet.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
predictions_alexnet = (predictions_alexnet > 0.5).astype(int)
true_classes_alexnet = test_generator.classes
conf_matrix_alexnet = confusion_matrix(true_classes_alexnet, predictions_alexnet)
print("Confusion Matrix for AlexNet:")
print(conf_matrix_alexnet)
classification_report_alexnet = classification_report(true_classes_alexnet, predictions_alexnet, target_names=['Normal', 'COVID-19'])
print("Classification Report for AlexNet:")
print(classification_report_alexnet)
