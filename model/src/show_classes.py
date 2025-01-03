import os
import tensorflow as tf

def load_and_show_classes(directory, batch_size=16):
    try:
        # Load the dataset
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=(224, 224),
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=True,
            seed=5
        )
        
        # Print class names
        class_names = dataset.class_names
        print(f"Class names: {class_names}")
        
    except Exception as e:
        print(f"Error loading dataset from {directory}: {str(e)}")

if __name__ == "__main__":
    # Define the dataset directory
    dataset_dir = 'Dataset/train'
    
    # Show classes
    load_and_show_classes(dataset_dir)
