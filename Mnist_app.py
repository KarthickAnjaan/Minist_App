import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from skorch import NeuralNetClassifier

st.title("Fun Learning with Neural Networks")

# Overview of Neural Networks
st.markdown("Neural networks are like magical detectives that can learn to recognize numbers. They work by mimicking how our brains learn things.")


st.markdown("In the magical world of neural networks, we have two key detectives: the **Input Detective** and the **Output Detective**.")
st.markdown("The **Input Detective** learns to spot clues in numbers, and the **Output Detective** makes the final guess. Our goal is to train these detectives to become experts at recognizing numbers.")

# Sidebar Explanation
st.sidebar.title("Teachable Points")
st.sidebar.markdown("Let's learn the spells to train our detectives!")

st.sidebar.title("Training Parameters:")
st.sidebar.title("Activation Function")

# Activation Functions
st.sidebar.markdown("Choose Activation Functions to give our detectives special skills:")
hidden_activation = st.sidebar.radio("Hidden Layer Activation", ["ReLU", "Sigmoid", "Linear", "Tanh"], index=0)
output_activation = st.sidebar.radio("Output Layer Activation", ["Sigmoid", "Softmax"], index=0)
st.sidebar.markdown("The **Hidden Layer Activation** is like a secret spell that adds power to the detectives. 'ReLU' is a simple and effective spell.")
st.sidebar.markdown("The **Output Layer Activation** depends on the problem. 'Sigmoid' is like a binary spell for two choices, and 'Softmax' is a multi-choice spell for many options.")

st.sidebar.title("Hidden layers")

# Hidden Layer Size
st.sidebar.markdown("The **Hidden Layer Size** determines how many assistants our detectives have.")
hidden_layer_size = st.sidebar.slider("Hidden Layer Size", 16, 128, 32)

st.sidebar.title("Learning Rate")

# Learning Rate
st.sidebar.markdown("Learning Rate is like the speed at which our detectives learn. It's crucial to choose it wisely!")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
st.sidebar.markdown("A high learning rate (e.g., 0.1) makes learning faster but might lead to mistakes. A low learning rate (e.g., 0.01) makes learning slower but more precise. We need to find the right balance.")

st.sidebar.title("Epochs")

# Number of Epochs
st.sidebar.markdown("Number of Epochs represents how many times our detectives go through their training. What's the right number?")
num_epochs = st.sidebar.selectbox("Number of Epochs", [5, 10, 15, 20, 25, 30])
st.sidebar.markdown("More epochs give our detectives more chances to learn but might make them remember too much. Fewer epochs mean less learning. We aim for the right amount.")


# Image Upload
uploaded_image = st.file_uploader("Upload your mysterious number", type=["jpg", "png", "jpeg"])

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
X = mnist.data.astype('float32') / 255.0
y = mnist.target.astype('int64')

mnist_dim = X.shape[1]
output_dim = len(np.unique(mnist.target))

# Define the neural network module
class ClassifierModule(torch.nn.Module):
    def __init__(self, input_dim=mnist_dim, hidden_dim=hidden_layer_size, output_dim=output_dim):
        super(ClassifierModule, self).__init__()
        self.hidden_activation = self.get_activation(hidden_activation)
        self.output_activation = self.get_activation(output_activation)
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = self.hidden_activation(self.hidden(X))
        X = self.output_activation(self.output(X))
        return X

    def get_activation(self, name):
        if name == "ReLU":
            return F.relu
        elif name == "Sigmoid":
            return torch.sigmoid
        elif name == "Softmax":
            return F.softmax
        elif name == "Linear":
            return lambda x: x
        elif name == "Tanh":
            return torch.tanh

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# Initialize the model
net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=num_epochs,
    lr=learning_rate,
    device=device,
)

# Display the model parameters and explanations
st.markdown("Our detectives' tools and training:")
st.write(f"Learning Rate: {learning_rate}")
st.write(f"Number of Epochs: {num_epochs}")
st.write(f"Hidden Layer Activation: {hidden_activation}")
st.write(f"Output Layer Activation: {output_activation}")
st.write(f"Hidden Layer Size: {hidden_layer_size}")

st.sidebar.markdown("Our detectives are now training to become experts. Let's see their magic!")

# Train the model
net.fit(X, y)

# Predict and Display Results
if uploaded_image is not None:
    st.image(uploaded_image, caption="Your Mysterious Number", use_column_width=True)
    
    if st.button("Predict"):
        image = Image.open(uploaded_image)
        image = image.convert("L")
        image = image.resize((28, 28))
        image = np.array(image)
        image = 255 - image
        image = image.astype('float32') / 255.0
        image = image.reshape(1, -1)
        
        prediction = net.predict(image)
        st.write(f"Our detectives predict: {prediction[0]}")

# Add a Reset Button
if st.button("Reset and Start Over"):
    st.caching.clear_cache()
    st.experimental_rerun()

# Explanation for Kids
st.markdown("Our computer detectives are like friendly magicians. You can teach them how to recognize numbers!")
st.markdown("1. Choose the magical tools for our detectives. These tools help them become experts at recognizing numbers!")
st.markdown("2. Our detectives will learn some magic tricks (epochs) to become better at recognizing!")
