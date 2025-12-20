import os
import tempfile
import mne
import numpy as np
import matplotlib.pyplot as plt

feature_data = ["EEG:C3", "EEG:Cz", "EEG:C4"]

class dataLoader:
    def __init__(self, data, data_type="train"):
        self.fs = 250
        self.data_type = data_type

        if data is None:
            raise ValueError("No EEG data provided")

        if hasattr(data, "read"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gdf") as tmp:
                tmp.write(data.read())
                gdf_path = tmp.name

        elif isinstance(data, str):
            if data.endswith(".gdf"):
                gdf_path = data
            else:
                gdf_path = f"./data/raw/{data}.gdf"

            if not os.path.exists(gdf_path):
                raise FileNotFoundError(f"{gdf_path} not found")

        else:
            raise TypeError("Unsupported input type for dataLoader")

        self.raw = mne.io.read_raw_gdf(gdf_path, preload=True)
        self.info = self.raw.info
        self.X, self.y = self.load_data()

        # self.debug()

    def load_data(self):
        self.raw.pick_channels(feature_data)
        
        channel_mapping = {
            'EEG:C3': 'C3',
            'EEG:Cz': 'Cz',
            'EEG:C4': 'C4'
        }
        self.raw.rename_channels(channel_mapping)
        
        montage = mne.channels.make_standard_montage('standard_1005')
        self.raw.set_montage(montage, on_missing='ignore')
                
        self.raw.compute_psd()
        
        events, event_id = self.labeling(self.data_type)
        
        if self.data_type == "train":
            epochs = mne.Epochs(
                self.raw,
                events,
                event_id=event_id,
                tmin=0.5,
                tmax=3.0,
                baseline=None,
                preload=True,
                reject_by_annotation=True
            )

            data = epochs.get_data()
            y = epochs.events[:, -1]
            label = np.where(y == event_id["LH"], 0, 1)
        else:
            # For evaluation data without labels
            epochs = mne.Epochs(
                self.raw,
                events,
                event_id=None,
                tmin=0.5,
                tmax=3.0,
                baseline=None,
                preload=True,
                reject_by_annotation=True,
                event_repeated='drop'
            )
            data = epochs.get_data()
            label = None  # No labels for evaluation
        
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
            
    def labeling(self, data_type="train"):
            events, event_id = mne.events_from_annotations(self.raw)
            if data_type == "train":
                event_id_new = {
                    "LH": event_id[np.str_('769')],  # LH
                    "RH": event_id[np.str_('770')]   # RH
                }
                return events, event_id_new
            else:
                # For evaluation, return all events without filtering
                return events, None