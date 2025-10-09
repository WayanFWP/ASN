import numpy as np

def find_peaks_custom(x, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5):
    """
    Find peaks in a 1-D array.
    
    Parameters:
    -----------
    x : array_like
        A 1-D array in which to find the peaks.
    height : number or ndarray or sequence, optional
        Required height of peaks. Either a number, None, an array matching x or a 2-element sequence of the former.
    threshold : number or ndarray or sequence, optional  
        Required threshold of peaks, the vertical distance to its neighboring samples.
    distance : number, optional
        Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    prominence : number or ndarray or sequence, optional
        Required prominence of peaks.
    width : number or ndarray or sequence, optional
        Required width of peaks in samples.
    wlen : int, optional
        Used for calculation of the peaks prominences, thus it is only used if prominence is given.
    rel_height : float, optional
        Used for calculation of the peaks width, thus it is only used if width is given.
        
    Returns:
    --------
    peaks : ndarray
        Indices of peaks in x that satisfy all given conditions.
    properties : dict
        Dictionary containing properties of the returned peaks.
    """
    
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array")
    
    # Find local maxima
    peaks = _find_local_maxima(x)
    
    if len(peaks) == 0:
        return np.array([], dtype=int), {}
    
    properties = {}
    
    # Apply height filter
    if height is not None:
        height_mask = _apply_height_filter(x, peaks, height)
        peaks = peaks[height_mask]
        if len(peaks) == 0:
            return peaks, properties
    
    # Apply threshold filter
    if threshold is not None:
        threshold_mask = _apply_threshold_filter(x, peaks, threshold)
        peaks = peaks[threshold_mask]
        if len(peaks) == 0:
            return peaks, properties
    
    # Apply distance filter
    if distance is not None and distance > 1:
        peaks = _apply_distance_filter(x, peaks, distance)
        if len(peaks) == 0:
            return peaks, properties
    
    # Calculate prominence if needed
    if prominence is not None or wlen is not None:
        prominences = _calculate_prominence(x, peaks, wlen)
        properties['prominences'] = prominences
        
        if prominence is not None:
            prominence_mask = _apply_prominence_filter(prominences, prominence)
            peaks = peaks[prominence_mask]
            properties['prominences'] = prominences[prominence_mask]
            if len(peaks) == 0:
                return peaks, properties
    
    # Calculate width if needed
    if width is not None:
        if 'prominences' not in properties:
            properties['prominences'] = _calculate_prominence(x, peaks, wlen)
        
        widths, width_heights, left_ips, right_ips = _calculate_width(x, peaks, properties['prominences'], rel_height)
        properties['widths'] = widths
        properties['width_heights'] = width_heights
        properties['left_ips'] = left_ips
        properties['right_ips'] = right_ips
        
        width_mask = _apply_width_filter(widths, width)
        peaks = peaks[width_mask]
        for key in properties:
            properties[key] = properties[key][width_mask]
    
    return peaks, properties

def _find_local_maxima(x):
    """Find local maxima in array x."""
    peaks = []
    n = len(x)
    
    for i in range(1, n - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(i)
    
    # Check boundaries
    if n >= 2:
        if x[0] > x[1]:
            peaks.insert(0, 0)
        if x[-1] > x[-2]:
            peaks.append(n-1)
    elif n == 1:
        peaks.append(0)
    
    return np.array(peaks, dtype=int)

def _apply_height_filter(x, peaks, height):
    """Apply height filter to peaks."""
    if np.isscalar(height):
        return x[peaks] >= height
    elif hasattr(height, '__len__') and len(height) == 2:
        min_height, max_height = height
        if min_height is None:
            min_height = -np.inf
        if max_height is None:
            max_height = np.inf
        return (x[peaks] >= min_height) & (x[peaks] <= max_height)
    else:
        height = np.asarray(height)
        if len(height) == len(peaks):
            return x[peaks] >= height
        else:
            raise ValueError("height array must have same length as peaks")

def _apply_threshold_filter(x, peaks, threshold):
    """Apply threshold filter to peaks."""
    if np.isscalar(threshold):
        mask = np.ones(len(peaks), dtype=bool)
        for i, peak in enumerate(peaks):
            left_val = x[peak-1] if peak > 0 else x[peak]
            right_val = x[peak+1] if peak < len(x)-1 else x[peak]
            min_threshold = min(x[peak] - left_val, x[peak] - right_val)
            mask[i] = min_threshold >= threshold
        return mask
    else:
        raise ValueError("threshold must be a scalar")

def _apply_distance_filter(x, peaks, distance):
    """Apply distance filter to peaks."""
    if len(peaks) <= 1:
        return peaks
    
    # Sort peaks by height (descending)
    peak_heights = x[peaks]
    sorted_indices = np.argsort(-peak_heights)
    sorted_peaks = peaks[sorted_indices]
    
    keep = np.ones(len(sorted_peaks), dtype=bool)
    
    for i, peak in enumerate(sorted_peaks):
        if not keep[i]:
            continue
        
        # Mark nearby peaks for removal
        for j in range(i + 1, len(sorted_peaks)):
            if abs(sorted_peaks[j] - peak) < distance:
                keep[j] = False
    
    # Get back the kept peaks and sort by position
    kept_peaks = sorted_peaks[keep]
    return np.sort(kept_peaks)

def _calculate_prominence(x, peaks, wlen=None):
    """Calculate prominence of peaks."""
    prominences = np.zeros(len(peaks))
    
    for i, peak in enumerate(peaks):
        # Determine window for prominence calculation
        if wlen is None:
            left_bound = 0
            right_bound = len(x) - 1
        else:
            half_window = wlen // 2
            left_bound = max(0, peak - half_window)
            right_bound = min(len(x) - 1, peak + half_window)
        
        # Find the lowest point to the left
        left_min = np.min(x[left_bound:peak+1])
        
        # Find the lowest point to the right  
        right_min = np.min(x[peak:right_bound+1])
        
        # Prominence is the smaller of the two depths
        prominences[i] = x[peak] - max(left_min, right_min)
    
    return prominences

def _apply_prominence_filter(prominences, prominence):
    """Apply prominence filter."""
    if np.isscalar(prominence):
        return prominences >= prominence
    elif hasattr(prominence, '__len__') and len(prominence) == 2:
        min_prom, max_prom = prominence
        if min_prom is None:
            min_prom = -np.inf
        if max_prom is None:
            max_prom = np.inf
        return (prominences >= min_prom) & (prominences <= max_prom)
    else:
        prominence = np.asarray(prominence)
        return prominences >= prominence

def _calculate_width(x, peaks, prominences, rel_height=0.5):
    """Calculate width of peaks at relative height."""
    widths = np.zeros(len(peaks))
    width_heights = np.zeros(len(peaks))
    left_ips = np.zeros(len(peaks))
    right_ips = np.zeros(len(peaks))
    
    for i, peak in enumerate(peaks):
        # Height at which to measure width
        width_height = x[peak] - prominences[i] * rel_height
        width_heights[i] = width_height
        
        # Find intersection points
        left_ip = peak
        right_ip = peak
        
        # Search left
        for j in range(peak - 1, -1, -1):
            if x[j] <= width_height:
                # Linear interpolation between j and j+1
                if j < peak - 1:
                    t = (width_height - x[j]) / (x[j+1] - x[j])
                    left_ip = j + t
                else:
                    left_ip = j
                break
        
        # Search right
        for j in range(peak + 1, len(x)):
            if x[j] <= width_height:
                # Linear interpolation between j-1 and j
                if j > peak + 1:
                    t = (width_height - x[j-1]) / (x[j] - x[j-1])
                    right_ip = j - 1 + t
                else:
                    right_ip = j
                break
        
        left_ips[i] = left_ip
        right_ips[i] = right_ip
        widths[i] = right_ip - left_ip
    
    return widths, width_heights, left_ips, right_ips

def _apply_width_filter(widths, width):
    """Apply width filter."""
    if np.isscalar(width):
        return widths >= width
    elif hasattr(width, '__len__') and len(width) == 2:
        min_width, max_width = width
        if min_width is None:
            min_width = -np.inf
        if max_width is None:
            max_width = np.inf
        return (widths >= min_width) & (widths <= max_width)
    else:
        width = np.asarray(width)
        return widths >= width

# Simple wrapper function
def find_peaks(x, height=None, distance=None):
    """    
    Parameters:
    -----------
    x : array_like
        Input signal
    height : float, optional
        Minimum height of peaks
    distance : float, optional
        Minimum distance between peaks
        
    Returns:
    --------
    peaks : ndarray
        Indices of peaks
    """
    peaks, _ = find_peaks_custom(x, height=height, distance=distance)
    return peaks