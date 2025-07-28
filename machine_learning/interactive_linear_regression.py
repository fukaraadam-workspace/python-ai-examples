import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# Cache data generation
@st.cache
def create_data(n_samples=100):
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1)
    return X, y


X, y = create_data()

st.title("Linear Regression with Gradient Descent")

# Controls
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)
max_iters = st.sidebar.slider("Max Iterations", 10, 200, 50, step=10)
show_iter = st.sidebar.slider("Show Iteration", 1, max_iters, 1)


# Compute history
@st.cache
def train_history(X, y, lr, iters):
    m, b = 0.0, 0.0
    history = []
    for i in range(iters):
        y_pred = m * X + b
        error = y_pred - y
        m_grad = (2 / len(X)) * (error * X).sum()
        b_grad = (2 / len(X)) * error.sum()
        m -= lr * m_grad
        b -= lr * b_grad
        history.append((m, b))
    return history


history = train_history(X, y, learning_rate, max_iters)
m_i, b_i = history[show_iter - 1]

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X, y, color="blue", label="Data")
ax.plot(X, m_i * X + b_i, color="red", label=f"Iter {show_iter}")
ax.set_xlabel("Image Quality Score")
ax.set_ylabel("Fraud Risk Score")
ax.legend()
ax.grid(True)

st.pyplot(fig)
st.write(f"Iteration {show_iter}: slope = {m_i:.3f}, intercept = {b_i:.3f}")
