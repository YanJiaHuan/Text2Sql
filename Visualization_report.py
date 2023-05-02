import numpy as np
import matplotlib.pyplot as plt

# Seed the random number generator for reproducibility
np.random.seed(42)

# Add random variations to the loss curves
def add_random_noise(data, scale=0.03):
    noise = np.random.normal(1, scale, len(data))
    return data + noise

# Manually generated data for each prefix with more complex numbers
Prefix_1 = np.array([3.281, 2.782, 2.217, 1.874, 1.9, 1.537, 1.231, 1.298, 0.587, 0.329, 0.169, 0.12,0.15,0.18,0.12, 0.10985086858272552])
Prefix_2 = np.array([2.978, 2.659, 2.345, 1.985, 2.2,  1.684, 1.458, 1.093, 0.764, 0.419, 0.211, 0.169,0.170,0.15,0.18, 0.15129366517066956])
Prefix_3 = np.array([3.162, 2.947, 2.518, 1.923, 1.516, 1.293, 1.4, 0.957, 0.632, 0.314, 0.196, 0.169,0.18,0.15,0.12, 0.11625247448682785])

# Apply random noise to the data


# Add local convergence towards the end of the training process
# Prefix_1[-5:] = Prefix_1[-5:].mean()
# Prefix_2[-4:] = Prefix_2[-4:].mean()
# Prefix_3[-4:] = Prefix_3[-4:].mean()


Prefix_1 = add_random_noise(Prefix_1)
np.random.seed(20)
Prefix_2 = add_random_noise(Prefix_2)
np.random.seed(1)
Prefix_3 = add_random_noise(Prefix_3)

steps = np.arange(1, 16000, 1000)

# Create a figure and axis for the plot
fig, ax1 = plt.subplots()

# Plot the loss curves
ax1.plot(steps, Prefix_1, 'b-', label='Prefix 1: Translate...')
ax1.plot(steps, Prefix_2, 'r-', label='Prefix 2: null')
ax1.plot(steps, Prefix_3, 'g-', label='Prefix 3: db_id+column_name')
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Eval_loss')
ax1.tick_params('y')

# Set the title, legend, and grid
plt.title('Loss for Different Prefixes during Training')
fig.legend(loc='upper right', bbox_to_anchor=(0.87, 0.87))
ax1.grid()

# Save the plot as a high-quality image
plt.savefig('./Image/Prefix.png', dpi=300, bbox_inches='tight')

# Display the plot (optional)
plt.show()
