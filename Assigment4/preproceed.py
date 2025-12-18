import os
import tempfile
import mne
import numpy as np
import matplotlib.pyplot as plt

feature_data = ["EEG:C3", "EEG:Cz", "EEG:C4"]

class dataLoader:
    def __init__(self, data):
        self.fs = 250

        if data is None:
            raise ValueError("No EEG data provided")

        if hasattr(data, "read"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gdf") as tmp:
                tmp.write(data.read())
                gdf_path = tmp.name

        elif isinstance(data, str):
            if data.endswith(".gdf"):
                # full or relative path
                gdf_path = data
            else:
                # dataset name → map to folder
                gdf_path = f"./data/raw/{data}.gdf"

            if not os.path.exists(gdf_path):
                raise FileNotFoundError(f"{gdf_path} not found")

        else:
            raise TypeError("Unsupported input type for dataLoader")

        # Load EEG
        self.raw = mne.io.read_raw_gdf(gdf_path, preload=True)
        self.X, self.y = self.load_data()

        # self.debug()

    def load_data(self):
        self.raw.pick_channels(feature_data)
        self.raw.compute_psd()
        
        events, event_id = self.labeling()
        
        epochs = mne.Epochs(
        self.raw,
        events,
        event_id=event_id,
        tmin=-4.0,
        tmax=3.0,
        baseline=(-1.0, 0.0),
        preload=True,
        reject_by_annotation=True
        )

        data = epochs.get_data()
        y = epochs.events[:, -1]

        label = np.where(y == event_id["LH"], 1, 2)
        
        return data, label
    
    def debug(self):
        print("====================================")
        print(f"Data Info: {self.raw.info}")
        print("Used Classes is LH and RH\nWhere 769=LH and 770=RH")
        print(f"Annotations: {self.raw.annotations}")
        print("====================================")
        
        # X Values is EEG Signal Data 
        # Shape = (n_trials, n_channels, n_times)
        print("====================================")
        print("X shape:", self.X.shape)
        
        # Y Values is Class Labels
        # Shape = (n_trials,)
        print("Class 2 trials:", np.sum(self.y == 2))
        print("Class 1 trials:", np.sum(self.y == 1))
        print("y shape:", self.y.shape)
        print("====================================")
        
    def labeling(self):
        events, event_id = mne.events_from_annotations(self.raw)
        
        event_id_new = {
        "LH": event_id[np.str_('769')],  # LH
        "RH": event_id[np.str_('770')]   # RH
        }
        return events, event_id_new
        
    
    def trial(self):
        pass