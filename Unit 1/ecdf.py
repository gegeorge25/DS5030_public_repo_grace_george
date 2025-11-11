import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data: daily coffee cups for 30 people
coffee = np.array([1, 2, 3, 0, 2, 1, 4, 3, 2, 1,
                   2, 3, 2, 4, 5, 3, 1, 0, 2, 3,
                   4, 2, 1, 3, 2, 2, 3, 4, 1, 2])

# Bootstrap resampling for mean (optional)
n_boot = 5000
boot_means = []
for _ in range(n_boot):
    sample = np.random.choice(coffee, size=len(coffee), replace=True)
    boot_means.append(np.mean(sample))
boot_means = np.array(boot_means)

# 95% confidence interval
ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
mean_est = np.mean(coffee)

# Set up side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left: KDE (data concentration) ---
sns.kdeplot(boot_means, fill=True, color='sandybrown', ax=axes[0])
axes[0].axvline(ci_lower, color='red', linestyle='--', label='95% CI lower')
axes[0].axvline(ci_upper, color='red', linestyle='--', label='95% CI upper')
axes[0].axvline(mean_est, color='black', linestyle='-', label='Sample mean')
axes[0].set_title('KDE: Where data is most frequent')
axes[0].set_xlabel('Mean Cups per Day')
axes[0].set_ylabel('Density')
axes[0].legend()

# --- Right: ECDF (cumulative probability) ---
sorted_means = np.sort(boot_means)
y = np.arange(1, len(sorted_means)+1) / len(sorted_means)
axes[1].plot(sorted_means, y, color='peru')
axes[1].axvline(ci_lower, color='red', linestyle='--', label='2.5%')
axes[1].axvline(ci_upper, color='red', linestyle='--', label='97.5%')
axes[1].set_title('ECDF: Cumulative probability')
axes[1].set_xlabel('Mean Cups per Day')
axes[1].set_ylabel('Cumulative Probability')
axes[1].legend()

plt.tight_layout()
plt.show()

# Summary
print(f"Sample mean = {mean_est:.2f} cups/day")
print(f"95% Bootstrap CI = ({ci_lower:.2f}, {ci_upper:.2f})")
#%%
import numpy as np
from scipy.stats import gaussian_kde

# Sample data
data = np.array([1, 2, 3, 4, 5])

# Create KDE
kde = gaussian_kde(data)

# Evaluate KDE at 100 evenly spaced points from min-1 to max+1
x = np.linspace(data.min()-1, data.max()+1, 100)
kde_values = kde(x)

# Calculate sum of KDE values
sum_values = np.sum(kde_values)

# Calculate approximate integral (sum * interval width)
dx = x[1] - x[0]
approx_integral = np.sum(kde_values * dx)

print("Sum of KDE values:", sum_values)
print("Approximate integral (area under curve):", approx_integral)


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example data (small set so bumps are clear)
np.random.seed(0)
data = np.random.normal(0, 1, 5)

# Grid for KDE evaluation
x_grid = np.linspace(-3, 3, 200)
bandwidth = 0.5

# Gaussian kernel function
def gaussian_kernel(x, xi, bandwidth):
    return (1 / (np.sqrt(2 * np.pi) * bandwidth)) * np.exp(-0.5 * ((x - xi) / bandwidth)**2)

# Precompute bumps for each data point
bumps = np.array([gaussian_kernel(x_grid, xi, bandwidth) for xi in data])

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, color='blue')
ax.set_xlim(-3, 3)
ax.set_ylim(0, bumps.max() * 1.5)
ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('KDE Bumps Adding Up')

# Animation function
def animate(i):
    if i == 0:
        line.set_data([], [])
    else:
        # Sum the bumps of the first i data points
        density = bumps[:i].sum(axis=0) / len(data)
        line.set_data(x_grid, density)
    return line,

# Create animation
anim = FuncAnimation(fig, animate, frames=len(data)+1, interval=1000, blit=True)

plt.show()

# %%

