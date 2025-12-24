import streamlit as st
import numpy as np
from preproceed import dataLoader
from pages import CSPPipeline, ERPPipeline, MLPipeline

class GUI:
    def __init__(self):
        st.set_page_config(page_title="EEG Signal Analysis GUI")

        # ---------- Session State ----------
        defaults = {
            "train_data": None,
            "train_labels": None,
            "test_data": None,
            "test_labels": None,
            "eval_data": None,
            "eval_labels": None,
            "fs": None
        }

        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        # ---------- Sidebar ----------
        self.page = st.sidebar.selectbox(
            "Select Analysis Pipeline",
            ("ERD/ERS Pipeline", "CSP Pipeline", "MLPipeline")
        )

        self.trial_idx = 0

        self.loader = st.sidebar.file_uploader(
            "Upload Train EEG Data (1T,2T,3T)",
            type=["gdf"],
            accept_multiple_files=True
        )

        # ---------- TRAIN / TEST SPLIT ----------
        if self.loader and st.session_state.train_data is None:
            train_X, train_y = [], []
            test_X, test_y = [], []

            fs = None

            for file in self.loader:
                try:
                    pre = dataLoader(file)
                    fs = pre.fs

                    if "3T" in file.name:
                        test_X.append(pre.X)
                        test_y.append(pre.y)
                    elif "1T" in file.name or "2T" in file.name:
                        train_X.append(pre.X)
                        train_y.append(pre.y)
                    else:
                        st.warning(f"Unknown file ignored: {file.name}")

                except Exception as e:
                    st.error(f"Failed loading {file.name}: {e}")

            if train_X:
                st.session_state.train_data = np.concatenate(train_X, axis=0)
                st.session_state.train_labels = np.concatenate(train_y, axis=0)

            if test_X:
                st.session_state.test_data = np.concatenate(test_X, axis=0)
                st.session_state.test_labels = np.concatenate(test_y, axis=0)

            st.session_state.fs = fs

        # ---------- Bind instance variables ----------
        self.data = st.session_state.train_data
        self.labels = st.session_state.train_labels
        self.fs = st.session_state.fs

        # ---------- UI Controls ----------
        if self.data is not None:
            self.trial_idx = st.sidebar.number_input(
                "Trial Index",
                0, self.data.shape[0] - 1, 0
            )
        else:
            st.info("Upload EEG data to begin")

    def run(self):
        if self.data is None:
            st.write("Waiting for EEG data...")
            return

        if self.page == "ERD/ERS Pipeline":
            ERPPipeline.show(
                self.data, self.labels, self.fs,
                st.session_state.test_data, st.session_state.test_labels,
                self.trial_idx
            )

        elif self.page == "CSP Pipeline":
            CSPPipeline.show(
                self.data, self.labels, st.session_state.test_data, st.session_state.test_labels, self.fs,
                self.trial_idx
            )

        elif self.page == "MLPipeline":
            MLPipeline.show(
                X_train=st.session_state.train_data,
                Y_train=st.session_state.train_labels,
                X_test=st.session_state.test_data,
                Y_test=st.session_state.test_labels,
                fs=self.fs,
                trial_idx=self.trial_idx
            )


def main():
    GUI().run()

if __name__ == "__main__":
    main()
