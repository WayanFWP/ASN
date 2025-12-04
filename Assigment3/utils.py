import numpy as np

# ----------------------------
# helpers
# ----------------------------
def dirac(x):
    return 1 if x == 0 else 0

def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

def debounce(events, min_distance=30):
    """Remove falsely repeated heel/toe events that are too close."""
    cleaned = []
    prev = -9999
    for e in events:
        if e - prev >= min_distance:
            cleaned.append(e)
            prev = e
    return np.array(cleaned)


# ----------------------------
# Footswitch detection
# ----------------------------
def detect_cycle(data, threshold=0.2, debounce_len=30):
    """
    Convert analog foot-switch signal → binary → detect HS / TO.
    """
    data = normalize(data)
    binary = (data > threshold).astype(int)
    diff = np.diff(binary)

    heel = np.where(diff == 1)[0]
    toe  = np.where(diff == -1)[0]

    heel = debounce(heel, debounce_len)
    toe  = debounce(toe, debounce_len)

    cycles = []
    for i in range(len(heel) - 1):
        hs = heel[i]
        hs_next = heel[i+1]

        # choose the toe-off inside this interval
        toe_inside = toe[(toe > hs) & (toe < hs_next)]
        to = toe_inside[0] if len(toe_inside) else None

        cycles.append((hs, to, hs_next))

    return cycles


# ----------------------------
# Extract toe-off events for segmentation
# ----------------------------
def extract_toe_off_events(data, threshold=0.2, debounce_len=30):
    data = normalize(data)
    binary = (data > threshold).astype(int)
    diff = np.diff(binary)
    
    # Toe-off events (falling edge: 1 -> 0)
    toe = np.where(diff == -1)[0]
    toe = debounce(toe, debounce_len)
    
    return toe


# ----------------------------
# Segment gait cycles
# ----------------------------
def segment_gait(signals, toe_off_events):
    """
    Segment gait cycles based on toe-off events.
    
    Parameters:
    - signals: dict or DataFrame with EMG signals (e.g., {'gl': array, 'vl': array})
    - toe_off_events: array of toe-off indices
    
    Returns:
    - segments: list of dictionaries, each containing segmented data for one gait cycle
    """
    segments = []
    
    # Convert DataFrame to dict if needed
    if hasattr(signals, 'to_dict'):
        signals_dict = signals.to_dict('series')
    else:
        signals_dict = signals
    
    # Create segments from toe-off to toe-off
    for i in range(len(toe_off_events) - 1):
        start_idx = toe_off_events[i]
        end_idx = toe_off_events[i + 1]
        
        # Create segment dictionary
        segment = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'length': end_idx - start_idx,
            'cycle_number': i + 1
        }
        
        # Add segmented data for each signal
        for signal_name, signal_data in signals_dict.items():
            segment[signal_name] = signal_data.iloc[start_idx:end_idx] if hasattr(signal_data, 'iloc') else signal_data[start_idx:end_idx]
        
        segments.append(segment)
    
    return segments