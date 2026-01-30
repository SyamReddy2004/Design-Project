import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def create_simple_model():
    # A simpler neural network (MLP)
    model = models.Sequential([
        layers.Flatten(input_shape=(150, 150, 3)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax') # Grade A, B, C
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def generate_mock_data(num_samples=100):
    x = np.random.rand(num_samples, 150, 150, 3).astype(np.float32)
    y = np.random.randint(0, 3, size=(num_samples,))
    return x, y

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Grade A', 'Grade B', 'Grade C'],
                yticklabels=['Grade A', 'Grade B', 'Grade C'])
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    os.makedirs(static_dir, exist_ok=True)
    plt.savefig(os.path.join(static_dir, filename))
    print(f"Confusion matrix saved to static/{filename}")
    plt.close()

def train():
    print("Starting Simple Neural Network Training...")
    model = create_simple_model()
    
    # Simulate both Tender Coconut and Turmeric data
    # (In a real scenario, you'd have separate datasets)
    x_train, y_train = generate_mock_data(300)
    x_val, y_val = generate_mock_data(100)
    
    print("Training for 40 epochs...")
    history = model.fit(x_train, y_train, epochs=40, validation_data=(x_val, y_val), verbose=0)
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'simple_model.h5')
    model.save(model_path)
    print(f"Simple model saved to {model_path}")
    
    # Predictions for Confusion Matrix
    y_pred_val = np.argmax(model.predict(x_val), axis=1)
    
    # Plot Confusion Matrix for Coconut (Simulation)
    plot_confusion_matrix(y_val[:50], y_pred_val[:50], "Tender Coconut", "confusion_matrix_coconut.png")
    
    # Plot Confusion Matrix for Turmeric (Simulation)
    plot_confusion_matrix(y_val[50:], y_pred_val[50:], "Turmeric", "confusion_matrix_turmeric.png")
    
    # Save training history plot
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training Accuracy (40 Epochs)')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'training_history.png'))
    print("Training history saved to static/training_history.png")

if __name__ == "__main__":
    train()
