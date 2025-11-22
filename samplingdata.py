import serial as sr
import csv
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import os

# port
espport = "/dev/ttyACM0"
baud = 115200
filename = str(input("input nama : ")) + ".csv"
header = ['Index', 'Amplitude']

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

plt.ion()
fig, ax = plt.subplots()
ax.set_title("plot (real-time)")

# Use deque for better performance
currentdata = deque(maxlen=500)
line_plot, = ax.plot([], [])  # Create line object once

# var
recordingtime = 30  # seconds
setuptime = 5 # seconds
delay = 0.02 # seconds
plot_counter = 0
plot_frequency = 5  # Update plot every 5 samples instead of every sample

def is_valid_number(input_str):
    try:
        float(input_str)
        return True
    except:
        return False

try:
    with sr.Serial(port=espport, baudrate=baud, timeout=1) as serialread, \
        open("data/"+filename,'w',newline='') as file:
        print("Serial connection established")
        print("Press enter button on device if needed")
        
        var = csv.writer(file)
        var.writerow(header)
        i = 0
        l = 0
        
        # skip setup time
        print("Setup phase starting...")
        while l < setuptime/delay:
            try:
                line = serialread.readline().decode('utf-8').strip()
                print(f"setup: {line}")
                print(f"setup time elapsed: {l*delay:.2f}s")
                l += 1
            except Exception as e:
                print(f"Setup read error: {e}")
                l += 1
        
        print("Recording phase starting...")
        while i < (recordingtime/delay):
            try:
                # reading and decode bytes to string
                line = serialread.readline().decode('utf-8').strip()
                print(f"line: '{line}'")
                print(f"recording time elapsed: {i*delay:.2f}s")
                
                # writing
                if line and is_valid_number(line):  # Check if line is not empty and is a number
                    value = int(float(line))  # Convert to float first, then int
                    currentdata.append(value)
                    
                    # Only update plot every few samples
                    plot_counter += 1
                    if plot_counter >= plot_frequency:
                        # Update existing line instead of clearing
                        line_plot.set_data(range(len(currentdata)), list(currentdata))
                        ax.set_xlim(0, len(currentdata))
                        
                        # Adaptive Y-axis with some padding
                        if len(currentdata) > 0:
                            data_min = min(currentdata)
                            data_max = max(currentdata)
                            padding = (data_max - data_min) * 0.1  # 10% padding
                            if padding == 0:  # If all values are the same
                                padding = max(data_max * 0.1, 10)  # At least 10 units padding
                            ax.set_ylim(data_min - padding, data_max + padding)
                        
                        plt.draw()
                        plt.pause(0.001)
                        plot_counter = 0
                    
                    data = [i, value]
                    print(f"Valid data: {data}")
                    var.writerow(data)
                else:
                    print(f"Invalid data received: '{line}'")
                    # Still write invalid data but mark it
                    data = [i, line if line else "NO_DATA"]
                    var.writerow(data)
                
                i += 1
                
            except Exception as e:
                print(f"Recording error: {e}")
                i += 1

        print("Recording completed!")

except sr.SerialException as e:
    print(f"Serial connection error: {e}")
    print("Please check:")
    print("1. Device is connected")
    print("2. Port is correct (/dev/ttyACM0)")
    print("3. No other program is using the port")
    exit()
except FileNotFoundError:
    print("Could not create data directory or file")
    exit()
except Exception as e:
    print(f"Unexpected error: {e}")
    exit()

# Plot the final data
try:
    print("Creating final plot...")
    df = pd.read_csv('data/' + filename)
    
    # Filter out non-numeric data
    df['Amplitude'] = pd.to_numeric(df['Amplitude'], errors='coerce')
    df = df.dropna()
    
    if len(df) == 0:
        print("No valid data to plot!")
        exit()
    
    # Create the main plot
    plt.figure(figsize=(15, 8))
    plt.plot(df['Index'], df['Amplitude'], linewidth=0.8, color='blue')
    plt.title('Signal Data - Amplitude vs Index', fontsize=16, fontweight='bold')
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
    print(f"Amplitude range: {df['Amplitude'].min():.2f} to {df['Amplitude'].max():.2f}")
    print(f"Mean amplitude: {mean_amp:.2f}")
    print(f"Standard deviation: {std_amp:.2f}")
    
    plt.show(block=True)  # This forces the plot to stay open
    
    # Optional: Create a zoomed-in view of the first 1000 samples
    if len(df) > 100:
        plt.figure(figsize=(15, 6))
        subset = df.head(1000)
        plt.plot(subset['Index'], subset['Amplitude'], linewidth=1, color='green')
        plt.title('Signal Data - First 1000 Samples (Zoomed View)', fontsize=16, fontweight='bold')
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)

except Exception as e:
    print(f"Error creating final plot: {e}")