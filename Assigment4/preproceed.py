import mne

data_raw_file = './data/'
raw = mne.io.read_raw_gdf(data_raw_file, preload=True)
raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
raw.plot(duration=5, n_channels=30)
