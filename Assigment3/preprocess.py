import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

namefile = "S01"
args = "data/raw/"+namefile
data = wfdb.rdrecord(args)

# Extract signal names and data
signal_names = data.sig_name
signals = data.p_signal

# Create DataFrame with all signals
df = pd.DataFrame(signals, columns=signal_names)

# Extract specific signals you need:
# Foot Switch (baso) - pressure sensors on feet
foot_switch = df['baso RT FOOT']

# Gastrocnemius Lateralis (GL) - lateral gastrocnemius muscle
gl = df['semg RT LAT.G']  # Right Lateral Gastrocnemius

# Vastus Lateralis (VL) - lateral vastus muscle  
vl = df['semg RT LAT.V']  # Right Lateral Vastus

# Create a new DataFrame with only the signals you need
extracted_data = pd.DataFrame({
    'foot_switch': foot_switch,
    'gl': gl,
    'vl': vl
})

# Save to CSV
extracted_data.to_csv(f'data/{namefile}_extracted.csv', index=False)

# Create time vector
time = np.arange(len(extracted_data)) / data.fs

# Create 6-subplot figure
fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
fig.suptitle(f'Signal Analysis - {namefile}', fontsize=16, fontweight='bold')

# Plot 1-2: Foot Switch signals
axes[0, 0].plot(time, foot_switch, 'r-', linewidth=0.8)
axes[0, 0].set_title('Foot Switch - Right')
axes[0, 0].set_ylabel('Amplitude (mV)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 3-4: Gastrocnemius Lateralis (GL) signals
axes[1, 0].plot(time, gl, 'orange', linewidth=0.8)
axes[1, 0].set_title('Gastrocnemius Lateralis - Right')
axes[1, 0].set_ylabel('EMG Amplitude (mV)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5-6: Vastus Lateralis (VL) signals
axes[2, 0].plot(time, vl, 'brown', linewidth=0.8)
axes[2, 0].set_title('Vastus Lateralis - Right')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('EMG Amplitude (mV)')
axes[2, 0].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Optional: Display signal information
print("Available signals:")
for i, name in enumerate(signal_names):
    print(f"{i}: {name}")

print("\nExtracted signals shape:", extracted_data.shape)
print("Sampling frequency:", data.fs, "Hz")
print(f"Recording duration: {len(extracted_data)/data.fs:.2f} seconds")