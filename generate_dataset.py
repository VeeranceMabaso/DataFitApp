import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate data
x0 = np.arange(-1, 1.1, 0.1)
x1 = np.arange(-1, 1.1, 0.1)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 3

# Flatten the arrays for DataFrame
x0_flat = x0.flatten()
x1_flat = x1.flatten()
y_truth_flat = y_truth.flatten()

# Create DataFrame
df = pd.DataFrame({
    'x0': x0_flat,
    'x1': x1_flat,
    'y': y_truth_flat
})

# Save DataFrame to CSV
df.to_csv('generated_dataset.csv', index=False)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xticks(np.arange(-1, 1.1, 0.5))
ax.set_yticks(np.arange(-1, 1.1, 0.5))
surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1, color='blue', alpha=0.4)
plt.show()
