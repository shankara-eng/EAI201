import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 


np.random.seed(0)
X = np.random.rand(100, 1) 
y = 2 * X + 1 + np.random.normal(0, 0.1, (100, 1))  


model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss='mse'  # Mean squared error loss
)

# Train the model
history = model.fit(X, y, epochs=100, verbose=0)

# Get the learned parameters
slope = model.layers[0].weights[0].numpy()[0][0]
intercept = model.layers[0].weights[1].numpy()[0]

# Print the results
print(f"Learned equation: y = {slope:.2f}x + {intercept:.2f}")

# Plotting
plt.scatter(X, y, label='Data points')
plt.plot(X, model.predict(X), color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using TensorFlow')
plt.legend()
plt.show()

# Plot the loss over training
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()