import matplotlib.pyplot as plt
import numpy as np

train_losses_1 = np.load('./mnist_deep.npy')
train_losses_2 = np.load('./mnist_parameter_optimization.npy')
train_losses_3 = np.load('./mnist.npy')

plt.plot(train_losses_1, color='blue')
plt.plot(train_losses_2, color='black')
plt.plot(train_losses_3, color='orange')
plt.xlabel('Epochs', fontweight='bold')
plt.ylabel('Cross Entropy Loss', fontweight='bold')
plt.title('Train Loss Vs Epochs', fontweight='bold')
plt.legend(['mnist_deep', 'mnist_parameter_optimization', 'mnist'], loc='upper right')
plt.savefig("Train Loss Plots.png")
plt.xlim(-0.6, 100)
plt.ylim(-0.05, 2)
plt.savefig("Train Loss Plots.png")
plt.show()
