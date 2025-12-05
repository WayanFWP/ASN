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


# ----------------------------
# EMG Onset/Offset Detection using CWT - IMPROVED VERSION
# ----------------------------
def compute_cwt_energy(cwt_coeffs, frequencies, freq_range=(20, 200)):
    """
    Compute energy envelope from CWT coefficients within frequency range.
    """
    freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    filtered_coeffs = cwt_coeffs[freq_mask, :]
        
    magnitude = np.abs(filtered_coeffs)
    
    energy_envelope = np.sum(magnitude, axis=0)
        
    return energy_envelope, freq_mask


def adaptive_threshold(signal, method='teager', factor=2.0, baseline_percentage=0.2):
    """
    Compute adaptive threshold for onset/offset detection.
    """
    baseline_samples = int(baseline_percentage * len(signal))
    
    if method == 'teager':
        # Teager-Kaiser Energy Operator
        teo = np.zeros_like(signal)
        for i in range(1, len(signal) - 1):
            teo[i] = signal[i]**2 - signal[i-1] * signal[i+1]
        baseline = np.mean(teo[:baseline_samples])
        baseline_std = np.std(teo[:baseline_samples])
        threshold = baseline + factor * baseline_std
    
    elif method == 'rms':
        # RMS-based threshold
        window_size = max(10, int(0.05 * len(signal)))
        rms_values = []
        for i in range(0, len(signal) - window_size, window_size//2):
            window = signal[i:i + window_size]
            rms_values.append(np.sqrt(np.mean(window**2)))
        
        rms_values = np.array(rms_values)
        baseline_windows = min(3, len(rms_values)//4)
        baseline = np.mean(rms_values[:baseline_windows])
        baseline_std = np.std(rms_values[:baseline_windows])
        threshold = baseline + factor * baseline_std
    
    elif method == 'mean':
        # Simple mean + std threshold
        baseline = np.mean(signal[:baseline_samples])
        baseline_std = np.std(signal[:baseline_samples])
        threshold = baseline + factor * baseline_std
    
    elif method == 'percentile':
        # Percentile-based threshold
        baseline = np.median(signal)
        percentile_value = 75 + factor * 5  # Dynamic percentile
        threshold = np.percentile(signal, min(percentile_value, 95))
    
    print(f"  {method.upper()} - Baseline: {baseline:.2f}, Threshold: {threshold:.2f}, Factor: {factor}")
    
    return threshold, baseline


def merge_close_activations(activations, merge_window_seconds, fs, debug=True):
    """
    Merge activations that occur within a specified time window.
    
    Parameters:
    - activations: list of (onset, offset) tuples
    - merge_window_seconds: time window in seconds for merging (e.g., 0.2)
    - fs: sampling frequency
    - debug: print debug information
    
    Returns:
    - merged_activations: list of merged (onset, offset) tuples
    """
    if len(activations) <= 1:
        return activations
    
    # Sort activations by onset time
    activations = sorted(activations, key=lambda x: x[0])
    
    merge_window_samples = int(merge_window_seconds * fs)
    merged = []
    
    if debug:
        print(f"  Merging activations within {merge_window_seconds}s ({merge_window_samples} samples)")
        print(f"  Input activations: {len(activations)}")
        for i, (onset, offset) in enumerate(activations):
            print(f"    {i+1}: {onset/fs:.3f}s - {offset/fs:.3f}s")
    
    current_onset, current_offset = activations[0]
    
    for i in range(1, len(activations)):
        next_onset, next_offset = activations[i]
        
        # Calculate gap between current offset and next onset
        gap = next_onset - current_offset
        
        if gap <= merge_window_samples:
            # Merge: extend current activation to include next one
            current_offset = max(current_offset, next_offset)  # Take the later offset
            if debug:
                print(f"    Merging: gap={gap/fs:.3f}s <= {merge_window_seconds}s")
        else:
            # Gap is too large, save current activation and start new one
            merged.append((current_onset, current_offset))
            current_onset, current_offset = next_onset, next_offset
            if debug:
                print(f"    Keeping separate: gap={gap/fs:.3f}s > {merge_window_seconds}s")
    
    # Add the last activation
    merged.append((current_onset, current_offset))
    
    if debug:
        print(f"  Output activations: {len(merged)}")
        for i, (onset, offset) in enumerate(merged):
            print(f"    {i+1}: {onset/fs:.3f}s - {offset/fs:.3f}s ({(offset-onset)/fs:.3f}s duration)")
    
    return merged


def detect_onset_offset_cwt(cwt_coeffs, frequencies, fs, method='teager', 
                           threshold_factor=2.0, min_duration=0.05, 
                           freq_range=(20, 200), merge_window=0.2, debug=True):
    """
    Detect EMG onset and offset using CWT analysis with merging of close activations.
    """
    
    if debug:
        print(f"  CWT matrix shape: {cwt_coeffs.shape}")
        print(f"  Frequency range available: {frequencies.min():.1f} - {frequencies.max():.1f} Hz")
    
    # Compute energy envelope
    energy_envelope, freq_mask = compute_cwt_energy(cwt_coeffs, frequencies, freq_range)
    
    # Smooth the envelope
    try:
        from scipy.ndimage import gaussian_filter1d
        sigma = max(1, fs*0.005)  # 5ms smoothing
        energy_smooth = gaussian_filter1d(energy_envelope, sigma=sigma)
    except ImportError:
        window = max(1, int(fs*0.01))
        energy_smooth = np.convolve(energy_envelope, np.ones(window)/window, mode='same')
    
    # Compute adaptive threshold
    threshold, baseline = adaptive_threshold(energy_smooth, method=method, factor=threshold_factor)
    
    # Find crossings
    above_threshold = energy_smooth > threshold
    crossings = np.diff(above_threshold.astype(int))
    
    # Find onset (crossings above threshold)
    onset_candidates = np.where(crossings == 1)[0]
    
    # Find offset (crossings below threshold) 
    offset_candidates = np.where(crossings == -1)[0]
    
    if debug:
        print(f"  Found {len(onset_candidates)} onset candidates: {onset_candidates}")
        print(f"  Found {len(offset_candidates)} offset candidates: {offset_candidates}")
        print(f"  Signal above threshold: {np.sum(above_threshold)} samples ({np.sum(above_threshold)/len(above_threshold)*100:.1f}%)")
    
    if len(onset_candidates) == 0:
        if debug:
            print("  No onset detected - trying lower threshold")
        # Try with lower threshold
        threshold_low = baseline + threshold_factor * 0.5 * (threshold - baseline)
        above_threshold_low = energy_smooth > threshold_low
        crossings_low = np.diff(above_threshold_low.astype(int))
        onset_candidates = np.where(crossings_low == 1)[0]
        offset_candidates = np.where(crossings_low == -1)[0]
        
        if len(onset_candidates) == 0:
            return None, None, energy_smooth, threshold
    
    if len(offset_candidates) == 0:
        if debug:
            print("  No offset detected - using signal end")
        offset_candidates = [len(energy_smooth) - 1]
    
    # Pair onsets with offsets to create initial activations
    raw_activations = []
    min_duration_samples = int(min_duration * fs)
    
    for onset_idx in onset_candidates:
        # Find next offset after this onset
        valid_offsets = offset_candidates[offset_candidates > onset_idx]
        if len(valid_offsets) == 0:
            offset_idx = len(energy_smooth) - 1
        else:
            offset_idx = valid_offsets[0]  # Take first offset after onset
        
        # Check minimum duration
        duration = (offset_idx - onset_idx) / fs
        if duration >= min_duration:
            raw_activations.append((onset_idx, offset_idx))
    
    if debug:
        print(f"  Raw activations (before merging): {len(raw_activations)}")
    
    # Merge close activations
    merged_activations = merge_close_activations(raw_activations, merge_window, fs, debug=debug)
    
    # Return the first activation (or None if no activations found)
    if len(merged_activations) >= 1:
        onset_idx, offset_idx = merged_activations[0]
        if debug:
            print(f"  Selected first activation: {onset_idx/fs:.3f}s - {offset_idx/fs:.3f}s")
        return onset_idx, offset_idx, energy_smooth, threshold
    else:
        return None, None, energy_smooth, threshold


def detect_multiple_activations_cwt(cwt_coeffs, frequencies, fs, method='teager',
                                   threshold_factor=2.0, min_duration=0.03,
                                   min_gap=0.02, freq_range=(20, 200), 
                                   merge_window=0.2, debug=True):
    """
    Detect multiple EMG activations with merging of close activations.
    """
    
    if debug:
        print(f"  Multiple activation detection with {method} method")
    
    # Compute energy envelope
    energy_envelope, freq_mask = compute_cwt_energy(cwt_coeffs, frequencies, freq_range)
    
    # Smooth the envelope
    try:
        from scipy.ndimage import gaussian_filter1d
        sigma = max(1, fs*0.005)
        energy_smooth = gaussian_filter1d(energy_envelope, sigma=sigma)
    except ImportError:
        window = max(1, int(fs*0.01))
        energy_smooth = np.convolve(energy_envelope, np.ones(window)/window, mode='same')
    
    # Compute adaptive threshold
    threshold, baseline = adaptive_threshold(energy_smooth, method=method, factor=threshold_factor)
    
    # Find all crossings
    above_threshold = energy_smooth > threshold
    crossings = np.diff(above_threshold.astype(int))
    
    onsets = np.where(crossings == 1)[0]
    offsets = np.where(crossings == -1)[0]
    
    if debug:
        print(f"  Raw onsets: {len(onsets)}, Raw offsets: {len(offsets)}")
    
    # Pair onsets with offsets
    raw_activations = []
    min_duration_samples = int(min_duration * fs)
    min_gap_samples = int(min_gap * fs)
    
    for onset in onsets:
        # Find next offset after this onset
        valid_offsets = offsets[offsets > onset]
        if len(valid_offsets) == 0:
            continue
            
        offset = valid_offsets[0]
        
        # Check minimum duration
        if (offset - onset) >= min_duration_samples:
            # Check minimum gap from previous activation (before merging)
            if len(raw_activations) == 0 or (onset - raw_activations[-1][1]) >= min_gap_samples:
                raw_activations.append((onset, offset))
            else:
                # If gap is too small, we'll let the merging function handle it
                raw_activations.append((onset, offset))
    
    if debug:
        print(f"  Raw activations (before merging): {len(raw_activations)}")
    
    # Merge close activations
    merged_activations = merge_close_activations(raw_activations, merge_window, fs, debug=debug)
    
    return merged_activations, energy_smooth, threshold


def refine_onset_offset(signal, onset_idx, offset_idx, fs, refinement_window=0.02):
    """
    Refine onset/offset detection using local signal characteristics.
    """
    window_samples = int(refinement_window * fs)
    
    # Refine onset - look for actual signal increase
    start_search = max(0, onset_idx - window_samples)
    end_search = min(len(signal), onset_idx + window_samples)
    
    search_window = signal[start_search:end_search]
    
    # Find steepest positive gradient for onset
    if len(search_window) > 1:
        gradient = np.gradient(search_window)
        max_grad_idx = np.argmax(gradient)
        refined_onset = start_search + max_grad_idx
    else:
        refined_onset = onset_idx
    
    # Refine offset - look for signal return to baseline
    start_search = max(0, offset_idx - window_samples)
    end_search = min(len(signal), offset_idx + window_samples)
    
    search_window = signal[start_search:end_search]
    
    # Find steepest negative gradient for offset
    if len(search_window) > 1:
        gradient = np.gradient(search_window)
        min_grad_idx = np.argmin(gradient)
        refined_offset = start_search + min_grad_idx
    else:
        refined_offset = offset_idx
    
    return refined_onset, refined_offset