#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf



def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Change to sigmoid for binary classification
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy',  # Recommended for binary classification
                  optimizer='adam',
                  metrics=['accuracy'])  # Change to accuracy for classification tasks
    return model

# Create the model instance
neural_network = create_model()

# Train the model
history = neural_network.fit(
    X_train, y_train,
    epochs=100,  # Number of epochs, can be adjusted
    batch_size=32,  # Size of batches, can be adjusted
    validation_data=(X_test, y_test)
)

# After training, you can evaluate your model's performance
loss, accuracy = neural_network.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# For predictions, the model will output a probability between 0 and 1
# You can convert these to class labels if needed
predictions = neural_network.predict(X_test)

# To convert probabilities to class labels based on a threshold (default 0.5)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)