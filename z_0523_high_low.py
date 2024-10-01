import numpy as np
import matplotlib.pyplot as plt

# Define Q_0 and Q_L ranges
Q_0_values = np.linspace(1000, 2000, 100)
Q_L_values = [100, 300, 500, 700, 900]

# Calculate Q_ext values for different Q_L values
Q_ext_values = {}
for Q_L in Q_L_values:
    Q_ext_values[Q_L] = (Q_0_values * Q_L) / (Q_0_values - Q_L)

# Plotting the results
plt.figure(figsize=(10, 6))
for Q_L in Q_L_values:
    plt.plot(Q_0_values, Q_ext_values[Q_L], label=f'Q_L = {Q_L}')

plt.xlabel('Q_0')
plt.ylabel('Q_ext')
# plt.title('Q_ext as a function of Q_0 for different Q_L values')
plt.legend()
plt.grid(True)
plt.show()
