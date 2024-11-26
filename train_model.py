import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data
data = pd.read_csv('C:\\Users\\ANSHUMAN\\EEG_Emotion_Recognition\\EEG_Emotion_Recognition\\EEG_EMOTION_RECOGNITION\\eeg_data.csv')

# Preprocess the data
X = data.drop('label', axis=1)  # Features
y = data['label']  # Labels

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Reshape data for time series input (if necessary)
# ... (Reshaping logic)

# Create the model
model = tf.keras.Sequential([
    # Input layer
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),

    # Hidden layers (adjust as needed)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    # Output layer
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model, label encoder, and scaler
model.save('models/emotion_recognition_model.h5')

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)