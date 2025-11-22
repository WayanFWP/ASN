import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data
df = pd.read_csv('data/nia.csv')

# Create the plot
plt.figure(figsize=(15, 8))
plt.plot(df['Index'], df['Amplitude'], linewidth=0.8, color='blue')
plt.title('Gisel Signal Data - Amplitude vs Index', fontsize=16, fontweight='bold')
plt.xlabel('Index', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True, alpha=0.3)

# Add some statistics to the plot
mean_amp = df['Amplitude'].mean()
std_amp = df['Amplitude'].std()
plt.axhline(y=mean_amp, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_amp:.1f}')
plt.axhline(y=mean_amp + std_amp, color='orange', linestyle='--', alpha=0.7, label=f'Mean + σ: {mean_amp + std_amp:.1f}')
plt.axhline(y=mean_amp - std_amp, color='orange', linestyle='--', alpha=0.7, label=f'Mean - σ: {mean_amp - std_amp:.1f}')

plt.legend()
plt.tight_layout()

# Show basic statistics
print(f"Data Summary:")
print(f"Total samples: {len(df)}")
print(f"Index range: {df['Index'].min()} to {df['Index'].max()}")
print(f"Amplitude range: {df['Amplitude'].min()} to {df['Amplitude'].max()}")
print(f"Mean amplitude: {mean_amp:.2f}")
print(f"Standard deviation: {std_amp:.2f}")

plt.show()

# Optional: Create a zoomed-in view of the first 1000 samples
plt.figure(figsize=(15, 6))
subset = df.head(1000)
plt.plot(subset['Index'], subset['Amplitude'], linewidth=1, color='green')
plt.title('Gisel Signal Data - First 1000 Samples (Zoomed View)', fontsize=16, fontweight='bold')
plt.xlabel('Index', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()