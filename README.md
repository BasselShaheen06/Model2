# Organ Classification AI Model

This project focuses on creating a deep learning-based AI model to classify medical images of organs into one of four categories: **Eye**, **Breast**, **Brain**, and **Limbs**. The model uses TensorFlow and is paired with a user-friendly PyQt5-based GUI to allow seamless interaction.

## Features

- **Model Architecture**: A custom CNN built using TensorFlow to classify images into four organ categories.
- **User Interface**: A dynamic GUI created with PyQt5 to:
  - Upload an image for classification.
  - Display the predicted organ class with confidence levels.
  - Provide notifications for incorrect inputs.
- **Training Pipeline**: Organized code structure for training the model on your dataset.
- **Dataset Management**: Structured dataset folder for training, validation, and testing.

## Project Structure

```
organ-classifier/
├── app/
│   ├── src/
│   │   ├── main.py                  # Main entry point for the app, runs the PyQt5 GUI
│   │   ├── model.py                 # Contains the model architecture and loading code
│   │   ├── dataset_utils.py         # Functions related to loading or processing the dataset
│   │   ├── predict.py               # Functionality to handle predictions and model inference
│   ├── assets/                      # Images, icons, and other UI assets
│   │   └── logo.png                 # Example logo for the UI (optional)
│   ├── saved_models/                # Folder to store saved models (e.g., trained model weights)
│   │   └── trained_model.h5         # Trained TensorFlow model file
│   ├── requirements.txt             # List of dependencies needed to run the project
│   └── README.md                    # Project documentation (this file)
└── dataset/                         # Dataset folder containing images for training and testing
    ├── train/                       # Training data
    │   ├── Eye/
    │   ├── Breast/
    │   ├── Brain/
    │   └── Limbs/
    ├── validate/                    # Validation data
    │   ├── Eye/
    │   ├── Breast/
    │   ├── Brain/
    │   └── Limbs/
    └── test/                        # Test data
        ├── Eye/
        ├── Breast/
        ├── Brain/
        └── Limbs/
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/organ-classifier.git
   cd organ-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```

3. Ensure you have Python 3.8 or later installed.

## Usage

### Training the Model

1. Place your dataset into the `dataset/` folder, ensuring it follows the structure outlined above.
2. Modify any hyperparameters or paths in `train.py` as needed.
3. Run the training script:
   ```bash
   python app/src/train.py
   ```

### Running the Application

1. Ensure the trained model (`trained_model.h5`) is in the `saved_models/` directory.
2. Launch the GUI:
   ```bash
   python app/src/main.py
   ```
3. Use the GUI to upload images and view predictions.

## Dataset Preparation

- The dataset should be organized into `train`, `validate`, and `test` directories.
- Each category (Eye, Breast, Brain, Limbs) should have its own subdirectory containing images.
- Example:
  ```
  dataset/
  ├── train/
  │   ├── Eye/
  │   │   ├── image1.jpg
  │   │   └── image2.jpg
  │   ├── Breast/
  │   └── ...
  └── test/
      ├── Eye/
      └── ...
  ```

## Notes

- Ensure all images are preprocessed to the required input size for the model (default is 224x224).
- The model's output includes the predicted class and the associated confidence score.

## Future Improvements

- Incorporate additional organ classes.
- Enhance the GUI for better user experience.
- Add real-time image preprocessing and augmentation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to contribute to the project by submitting pull requests or reporting issues!

