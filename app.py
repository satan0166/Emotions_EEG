import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the model and scaler
model = tf.keras.models.load_model('models/emotion_recognition_model.h5')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

emotion_labels = label_encoder.classes_

st.set_page_config(page_title="EEG-Based Emotion Recognition", layout="wide")
st.title("EEG-Based Emotion Recognition")

st.subheader("Input EEG Data Values")
st.write("Please enter EEG feature values corresponding to specific brain regions (positions marked below).")

num_features = model.input_shape[1]

input_data = []
st.write("### EEG Feature Inputs")
st.write("The following inputs correspond to various EEG features extracted from different brain regions:")

for i in range(num_features):
    label = f"EEG Feature {i+1} (Position {i+1})"  # Label format: EEG Feature <number> with placeholder position
    value = st.number_input(label, min_value=-10.0, max_value=10.0, step=0.1)
    input_data.append(value)

if st.button("Predict Emotion"):
    # Preprocess and make prediction
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    predicted_emotion_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_emotion_index]
    predicted_probability = np.max(prediction)

    # Display emotion prediction with explanation
    st.write(f"**Predicted Emotion:** {predicted_emotion}")
    st.write(f"**Confidence:** {predicted_probability:.2f}")

    # Provide more detailed explanation (e.g., using a language model or rule-based system)
    explanation = "Based on the EEG patterns, the model predicts that the person is likely feeling " + predicted_emotion + ". This prediction is associated with specific brain activity patterns related to " + predicted_emotion + "."

    st.write(f"**Interpretation:** {explanation}")

    # Visualize the input EEG data (e.g., as a line plot)
    fig, ax = plt.subplots(figsize=(5, 3))  # Adjust figure size
    ax.plot(input_data[0])
    ax.set_xlabel("Time")
    ax.set_ylabel("EEG Signal")
    st.pyplot(fig)

    # Visualize the predicted emotion as a pie chart
    fig = go.Figure(data=[go.Pie(labels=emotion_labels, values=prediction[0])])
    fig.update_layout(title='Predicted Emotion Probabilities', width=500, height=400)  # Adjust width and height
    st.plotly_chart(fig)

st.markdown("---")
st.write("### Understanding EEG and Emotional Analysis")
st.write("""
EEG (electroencephalography) detects brain activity that can be associated with emotional states. This tool predicts emotions by 
analyzing EEG feature values associated with certain brain regions, making it easier to interpret neural activity in terms of emotions.
""")