import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# NAND truth table
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([1, 1, 1, 0])  # NAND outputs

# Initialize weights and bias
w = np.random.uniform(-1, 1, 2)
b = np.random.uniform(-1, 1)
lr = 0.1
epochs = 50

history = []

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# Train and record weight updates
for epoch in range(epochs):
    for i in range(len(X)):
        xi = X[i]
        target = y[i]

        z = np.dot(w, xi) + b
        out = sigmoid(z)
        error = target - out
        delta = error * sigmoid_deriv(z)

        w += lr * delta * xi
        b += lr * delta

    # Save weights/bias for animation
    history.append((w.copy(), b))

# Setup the plot
fig, ax = plt.subplots()
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.set_title("NAND Perceptron Training")
scat = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=100)
line, = ax.plot([], [], 'g--', linewidth=2)

# Animation function
def update(frame):
    w, b = history[frame]
    x_vals = np.array([-0.2, 1.2])
    # Decision boundary: w1*x + w2*y + b = 0 => y = -(w1*x + b)/w2
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
    else:
        y_vals = np.array([0, 0])
    line.set_data(x_vals, y_vals)
    ax.set_title(f"Epoch {frame}")
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=200, blit=True)
plt.show()

