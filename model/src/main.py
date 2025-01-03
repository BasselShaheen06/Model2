import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QGroupBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
import tensorflow as tf
from PIL import Image
import numpy as np
from model import OrganClassifierModel
import traceback
import os

class OrganClassifierApp(QMainWindow):
    def __init__(self, model_path, weights_path, dataset_path):
        super().__init__()
        # Instantiate the model
        self.model = OrganClassifierModel(num_classes=4)
        # Call the model with dummy input to create variables
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        self.model(dummy_input)

        # Check if weights file exists
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Unable to find weights file at {weights_path}")

        # Load the best weights
        self.model.load_weights(weights_path)
        self.dataset_path = dataset_path
        self.init_ui()

        # Define a dictionary to map class indices to organ names
        self.class_names = {
            0: "Eye",
            1: "Brain",
            2: "Breast",
            3: "Limb"
            # Add more mappings as needed
        }

    def init_ui(self):
        self.setWindowTitle("Organ Classifier")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        main_layout = QVBoxLayout()

        # Group box
        group_box = QGroupBox("Organ Classifier")
        group_box_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Organ Classifier")
        title_label.setFont(QFont("Arial", 24))
        title_label.setAlignment(Qt.AlignCenter)
        group_box_layout.addWidget(title_label)

        # Image display
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(500, 500)
        self.image_label.setScaledContents(True)
        group_box_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Result display
        self.result_label = QLabel("Prediction: N/A", self)
        self.result_label.setFont(QFont("Arial", 16))
        self.result_label.setAlignment(Qt.AlignCenter)
        group_box_layout.addWidget(self.result_label)

        self.confidence_label = QLabel("Confidence: N/A", self)
        self.confidence_label.setFont(QFont("Arial", 16))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        group_box_layout.addWidget(self.confidence_label)

        # Buttons layout
        buttons_layout = QHBoxLayout()

        self.select_button = QPushButton("Select Image", self)
        self.select_button.setFont(QFont("Arial", 14))
        self.select_button.clicked.connect(self.select_image)
        buttons_layout.addWidget(self.select_button)

        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setFont(QFont("Arial", 14))
        self.predict_button.clicked.connect(self.predict_image)
        buttons_layout.addWidget(self.predict_button)

        group_box_layout.addLayout(buttons_layout)
        group_box.setLayout(group_box_layout)

        # Add group box to main layout
        main_layout.addWidget(group_box, alignment=Qt.AlignCenter)

        # Set main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def select_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap)
            self.result_label.setText("Prediction: N/A")
            self.confidence_label.setText("Confidence: N/A")

    def predict_image(self):
        if not hasattr(self, 'image_path'):
            self.result_label.setText("No image selected!")
            return

        try:
            # Preprocess the image
            image = Image.open(self.image_path).convert("RGB")
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array.astype(np.float32)  # Ensure the input tensor is of type float32

            # Perform prediction
            predictions = self.model(image_array, training=False)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions, axis=1)[0]

            # Get the organ name from the class index
            organ_name = self.class_names.get(predicted_class, "Unknown")

            # Update the UI with the prediction
            self.result_label.setText(f"Prediction: {organ_name}")
            self.confidence_label.setText(f"Confidence: {confidence * 100:.2f}%")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            traceback.print_exc()
            self.result_label.setText("Prediction: Error")
            self.confidence_label.setText("Confidence: N/A")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Define paths
    model_path = "D:/besooo/SBME/model/model/src/savedmodels/model/saved_model.pb"
    weights_path = "D:/besooo/SBME/model/model/src/savedmodels/checkpoints/best_weights.h5"
    dataset_path = "D:/besooo/SBME/model/Dataset"

    # Start the app
    main_window = OrganClassifierApp(model_path, weights_path, dataset_path)
    main_window.show()
    sys.exit(app.exec_())
