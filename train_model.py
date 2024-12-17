from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers

# Step 1: Set up ImageDataGenerator for image loading and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)  # Same rescaling for validation data

# Step 2: Load training images from the 'train' folder
train_generator = train_datagen.flow_from_directory(
    'data/train',  # Path to the 'train' directory
    target_size=(224, 224),  # Resize images to (224, 224)
    batch_size=32,  # Number of images to process per batch
    class_mode='categorical'  # Multi-class classification
)

# Step 3: Load validation images from the 'val' folder
validation_generator = val_datagen.flow_from_directory(
    'data/val',  # Path to the 'val' directory
    target_size=(224, 224),  # Resize images to (224, 224)
    batch_size=32,  # Number of images per batch
    class_mode='categorical'  # Multi-class classification
)

# Step 4: Use Transfer Learning with VGG16 (pre-trained)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base_model layers to prevent training on them
base_model.trainable = False

# Step 5: Create a custom model on top of VGG16
model = Sequential([
    base_model,  # Add VGG16 base model
    Flatten(),  # Flatten the 3D output to 1D
    Dense(512, activation='relu'),  # Fully connected layer with 512 units
    Dropout(0.5),  # Dropout for regularization
    Dense(3, activation='softmax')  # Output layer with 3 units (one for each class)
])

# Step 6: Compile the model
model.compile(optimizer=Adam(lr=0.0001),  # Using Adam optimizer with a low learning rate
              loss='categorical_crossentropy',  # Multi-class classification
              metrics=['accuracy'])

# Step 7: Set up EarlyStopping to monitor validation accuracy and avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 8: Train the model
history = model.fit(
    train_generator,  # Training data
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Number of steps per epoch
    epochs=20,  # Number of epochs
    validation_data=validation_generator,  # Validation data
    validation_steps=validation_generator.samples // validation_generator.batch_size,  # Steps per validation epoch
    callbacks=[early_stopping]  # Apply early stopping
)

# Save the trained model to a file after training
model.save(r"D:\Dev\Deep Learning\Food-Recognition-Using-Deep-Learning\models\food_recognition_model.h5")  # Save the model for later use in app.py
