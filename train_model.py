from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Set up ImageDataGenerator for image loading and augmentation
train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]
val_datagen = ImageDataGenerator(rescale=1./255)    # Same rescaling for validation data

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

# Step 4: Define the model architecture (simple CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Output layer with 3 units for 3 classes
])

# Step 5: Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model using the data generators
model.fit(
    train_generator,  # Training data
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Number of batches per epoch
    epochs=10,  # Number of epochs to train for
    validation_data=validation_generator,  # Validation data
    validation_steps=validation_generator.samples // validation_generator.batch_size  # Number of validation batches per epoch
)
