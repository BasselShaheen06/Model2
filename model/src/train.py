import os
import sys
import json
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Add the directory containing 'model.py' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from model import OrganClassifierModel  # Ensure correct import

# Paths to the train, validation, and test directories
train_dir = 'Dataset/train'
val_dir = 'Dataset/validate'
test_dir = 'Dataset/Test'

def load_and_validate_dataset(directory, batch_size=16):  # Reduced batch size
    try:
        # Get total number of files for cardinality
        num_files = sum([len(files) for _, _, files in os.walk(directory)])
        print(f"Found {num_files} files in {directory}")
        
        # Adjust batch size if necessary
        adjusted_batch_size = min(batch_size, num_files)
        if adjusted_batch_size != batch_size:
            print(f"Adjusted batch size to {adjusted_batch_size} due to dataset size")
        
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=(224, 224),
            batch_size=adjusted_batch_size,
            label_mode='categorical',
            shuffle=True,
            seed=5
        )
        
        # Calculate number of batches
        num_batches = num_files // adjusted_batch_size
        print(f"Number of batches per epoch: {num_batches}")
        
        # Filter out corrupted images
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        # Cache and prefetch for better performance
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset, num_batches
    except Exception as e:
        print(f"Error loading dataset from {directory}: {str(e)}")
        raise

# Load and validate the datasets
train_dataset, train_steps = load_and_validate_dataset(train_dir, batch_size=16)
val_dataset, val_steps = load_and_validate_dataset(val_dir, batch_size=16)
test_dataset, test_steps = load_and_validate_dataset(test_dir, batch_size=16)

# Print class names
class_names = train_dataset.class_names
print(f"Class names: {class_names}")

# Enhanced preprocessing layers
preprocessing_layer = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# Apply preprocessing with error handling
def preprocess_with_error_handling(x, y):
    try:
        return preprocessing_layer(x, training=True), y
    except:
        return x, y

train_dataset = train_dataset.map(preprocess_with_error_handling).repeat()
val_dataset = val_dataset.map(preprocess_with_error_handling).repeat()
test_dataset = test_dataset.map(preprocess_with_error_handling).repeat()

# Initialize and compile the model with modified optimizer
model = OrganClassifierModel()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Create save directories if they don't exist
os.makedirs('savedmodels', exist_ok=True)
os.makedirs('savedmodels/checkpoints', exist_ok=True)

# Enhanced callbacks
callbacks = [
    ModelCheckpoint(
        filepath='savedmodels/checkpoints/best_weights.h5',
        save_best_only=True,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Train with modified parameters
history = model.fit(
    train_dataset,
    epochs=25,  # Reduced number of epochs
    validation_data=val_dataset,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1
)

# Save the final model in TensorFlow format
print("\nSaving final model...")
model_path = model.save_model('savedmodels')

# Save class names
with open('savedmodels/class_names.json', 'w') as f:
    json.dump(class_names, f)

print("Training completed!")
print(f"Final model saved to: {model_path}")

# Print final metrics
final_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"\nFinal training accuracy: {final_accuracy:.4f}")
print(f"Final validation accuracy: {final_val_accuracy:.4f}")
