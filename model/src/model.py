import tensorflow as tf
from keras import layers, models, regularizers
import os

class OrganClassifierModel(tf.keras.Model):
    def __init__(self, num_classes=4):
        super(OrganClassifierModel, self).__init__()
        # Use a pre-trained ResNet50 model for feature extraction
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Fine-tune the last few layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        self.resnet = base_model
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dropout1 = layers.Dropout(0.5)
        self.dense1 = layers.Dense(
            512, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )
        self.dropout2 = layers.Dropout(0.3)
        self.dense = layers.Dense(
            num_classes, 
            activation='softmax',
            kernel_regularizer=regularizers.l2(0.01)
        )

    def call(self, inputs, training=False):
        x = self.resnet(inputs)
        x = self.global_avg_pool(x)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.dense(x)
    
    def save_model(self, save_dir='savedmodels'):
        """Save the full model in TensorFlow format."""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'model')
        tf.saved_model.save(self, model_path)
        return model_path
    
    @classmethod
    def load_saved_model(cls, model_path='savedmodels/model'):
        """Load a saved model from directory."""
        return tf.saved_model.load(model_path)
