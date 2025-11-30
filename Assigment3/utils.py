# utils.py
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
# Extract segment per gait cycle
# ----------------------------
def segment_gait(signal, cycles):
    segments = []
    for hs, to, hs_next in cycles:
        segments.append({
            "data": signal[hs:hs_next],
            "hs_idx": 0,
            "to_idx": None if to is None else (to - hs),
            "hs_next_idx": hs_next - hs
        })
    return segments


def extract_cycles(signal, cycles):
    segments = []
    for hs, to, hs_next in cycles:
        seg = signal[hs:hs_next]
        segments.append({
            "segment": seg,
            "hs": 0,
            "to": None if to is None else to - hs,
            "hs_next": hs_next - hs
        })
    return segments

def cwt_power(matrix):
    power = np.sum(np.abs(matrix) ** 2, axis=0)
    return power

def detect_onset_offset(power, fs, threshold_ratio=0.01):
    Pmax = np.max(power)
    threshold = threshold_ratio * Pmax

    onset = None
    offset = None

    # Temukan ONSET
    for i in range(len(power)):
        if power[i] > threshold:
            onset = i
            break

    if onset is None:
        return None, None  # tidak ada aktivasi

    # Temukan OFFSET
    for j in range(onset + 1, len(power)):
        if power[j] < threshold:
            offset = j
            break

    return onset, offset
