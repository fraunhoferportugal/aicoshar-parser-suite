from __future__ import annotations

import gzip
import io
import logging
import math
import os
import re
import zipfile
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.constants import g as standard_gravity_acceleration
from scipy.io import loadmat

from util import random_train_test_split_ids, save_to_json

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def load_dataset_config(dataset_name: str, config_file: str | Path) -> dict[str, Any]:
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with config_path.open("r") as file:
        configs = yaml.safe_load(file)
    if dataset_name not in configs:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.")
    return configs[dataset_name]


class DatasetParser(ABC):
    """Abstract base class for dataset parsers."""

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        # Ensure input directory exists
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_data(self) -> Any:
        """Load raw data into memory."""
        ...

    @abstractmethod
    def restructure(self, *args, **kwargs) -> None:
        """Process loaded data and save to output_dir."""
        ...


class DatasetFactory:
    """Factory for creating dataset parsers.

    Parsers must register via @DatasetFactory.register("Name").
    """

    _registry: dict[str, type[DatasetParser]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(parser_cls: type[DatasetParser]):
            cls._registry[name] = parser_cls
            return parser_cls

        return decorator

    @classmethod
    def create(cls, name: str, input_dir: Path, output_dir: Path) -> DatasetParser:
        if name not in cls._registry:
            raise ValueError(f"Dataset parser '{name}' is not registered.")
        return cls._registry[name](input_dir, output_dir)


@DatasetFactory.register("CHARM")
class CHARMParser(DatasetParser):
    """Parser for the CHARM dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("CHARM", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]

        self.timestamp_column = self.config["timestamp_column"]
        self.activity_column = self.config["activity_column"]
        self.participant_column = self.config["participant_column"]
        self.repetition_column = self.config["repetition_column"]
        self.acc_columns = self.config["acc_columns"]
        self.gyr_columns = self.config["gyr_columns"]

        self.acc_csv_filename = self.config["acc_csv_filename"]
        self.gyr_csv_filename = self.config["gyr_csv_filename"]

        self.acc_filename = self.config["acc_filename"]
        self.gyr_filename = self.config["gyr_filename"]

        self.device_name = self.config["device_name"]
        self.position = self.config["positions"][0]
        self.min_samples_per_session = self.config["min_samples_per_session"]

        # Partition
        self.test_ratio = float(self.config["test_ratio"])
        self.random_seed = int(self.config["random_seed"])

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load accelerometer and gyroscope CSV files."""
        acc = pd.read_csv(os.path.join(self.dataset_dir, self.acc_csv_filename))
        gyro = pd.read_csv(os.path.join(self.dataset_dir, self.gyr_csv_filename))

        return acc, gyro

    def restructure(self) -> None:
        """Restructure the dataset into standardized format."""

        # Load raw sensor data
        acc, gyro = self.load_data()

        # Process each user independently
        users = list(set(acc[self.participant_column]))
        for user in set(users):

            # Process each activity independently
            activities = acc[acc[self.participant_column] == user][self.activity_column]
            for activity in set(activities):

                # Skip activities not included in the harmonization map
                if activity not in self.activity_map:
                    continue

                # Process each trial independently
                trials = acc[(acc[self.participant_column] == user) & (acc[self.activity_column] == activity)][
                    self.repetition_column
                ]
                for trial in set(trials):
                    # Select accelerometer data for this user/activity/trial
                    acc_save = acc[
                        (acc[self.participant_column] == user)
                        & (acc[self.activity_column] == activity)
                        & (acc[self.repetition_column] == trial)
                    ]

                    # Keep only timestamp and accelerometer axes
                    acc_save = acc_save[[self.timestamp_column] + self.acc_columns]

                    # Build accelerometer output directory
                    user_activity_path = os.path.join(
                        self.destination_dir,
                        user,
                        self.activity_map[activity] + "_" + str(trial),
                        f"{self.device_name}_{self.position}",
                    )
                    os.makedirs(user_activity_path, exist_ok=True)

                    # Save accelerometer signal
                    acc_save.to_csv(
                        os.path.join(user_activity_path, self.acc_filename),
                        header=False,
                        index=False,
                    )

                    # Select gyroscope data for this user/activity/trial
                    gyro_save = gyro[
                        (gyro[self.participant_column] == user)
                        & (gyro[self.activity_column] == activity)
                        & (gyro[self.repetition_column] == trial)
                    ]

                    # Skip if stream is too short
                    if len(gyro_save) <= self.min_samples_per_session:
                        continue

                    # Keep only timestamp and gyroscope axes
                    gyro_save = gyro_save[[self.timestamp_column] + self.gyr_columns]

                    # Save gyroscope signal
                    gyro_save.to_csv(
                        os.path.join(user_activity_path, self.gyr_filename),
                        header=False,
                        index=False,
                    )

        # Train/test split at user level
        train_ids, test_ids = random_train_test_split_ids(users, test_ratio=self.test_ratio, seed=self.random_seed)

        # Save the split to the output directory
        save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)


@DatasetFactory.register("DailySportsActivities")
class DailySportsActivitiesParser(DatasetParser):
    """Parser for the DailySportsActivities dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config(
            "DailySportsActivities",
            Path(__file__).parent / "external_datasets_config.yaml",
        )

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]
        self.position_map = self.config["position_map"]

        self.rate = self.config["rate"]
        self.timestamp_column = self.config["timestamp_column"]

        self.wearable_prefix = self.config["wearable_prefix"]
        self.subject_prefix = self.config["subject_prefix"]
        self.one_digit_subject_prefix = self.config["one_digit_subject_prefix"]

        self.header = self.config["header"]

        self.acc_torso_columns = self.config["acc_torso_columns"]
        self.gyro_torso_columns = self.config["gyro_torso_columns"]
        self.mag_torso_columns = self.config["mag_torso_columns"]

        self.acc_ra_columns = self.config["acc_ra_columns"]
        self.gyro_ra_columns = self.config["gyro_ra_columns"]
        self.mag_ra_columns = self.config["mag_ra_columns"]

        self.acc_la_columns = self.config["acc_la_columns"]
        self.gyro_la_columns = self.config["gyro_la_columns"]
        self.mag_la_columns = self.config["mag_la_columns"]

        self.acc_rl_columns = self.config["acc_rl_columns"]
        self.gyro_rl_columns = self.config["gyro_rl_columns"]
        self.mag_rl_columns = self.config["mag_rl_columns"]

        self.acc_ll_columns = self.config["acc_ll_columns"]
        self.gyro_ll_columns = self.config["gyro_ll_columns"]
        self.mag_ll_columns = self.config["mag_ll_columns"]

        self.acc_filename = self.config["acc_filename"]
        self.gyr_filename = self.config["gyr_filename"]
        self.mag_filename = self.config["mag_filename"]

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self, act: str, user: str) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
        """Loads all sensor data from the input activity and user txt file into
        sensor based DataFrames."""
        data = pd.DataFrame()

        for fl in np.arange(1, 61):
            name = str(fl)
            name = (
                self.one_digit_subject_prefix + name + ".txt" if len(name) == 1 else self.subject_prefix + name + ".txt"
            )
            filepath = os.path.join(self.dataset_dir, act, user, name)

            try:
                df = pd.read_csv(filepath, header=None)
            except Exception:
                logger.error(f"Failed to load data for activity {act} of participant {user}.")
                raise

            data = pd.concat([data, df])

        data.columns = self.header

        data[self.timestamp_column] = np.arange(0, len(data) / self.rate, 1 / self.rate) * 1e9

        acc_torso = data[[self.timestamp_column] + self.acc_torso_columns]
        gyro_torso = data[[self.timestamp_column] + self.gyro_torso_columns]
        mag_torso = data[[self.timestamp_column] + self.mag_torso_columns]

        acc_ra = data[[self.timestamp_column] + self.acc_ra_columns]
        gyro_ra = data[[self.timestamp_column] + self.gyro_ra_columns]
        mag_ra = data[[self.timestamp_column] + self.mag_ra_columns]

        acc_la = data[[self.timestamp_column] + self.acc_la_columns]
        gyro_la = data[[self.timestamp_column] + self.gyro_la_columns]
        mag_la = data[[self.timestamp_column] + self.mag_la_columns]

        acc_rl = data[[self.timestamp_column] + self.acc_rl_columns]
        gyro_rl = data[[self.timestamp_column] + self.gyro_rl_columns]
        mag_rl = data[[self.timestamp_column] + self.mag_rl_columns]

        acc_ll = data[[self.timestamp_column] + self.acc_ll_columns]
        gyro_ll = data[[self.timestamp_column] + self.gyro_ll_columns]
        mag_ll = data[[self.timestamp_column] + self.mag_ll_columns]

        accs = [acc_torso, acc_ra, acc_la, acc_rl, acc_ll]
        gyrs = [gyro_torso, gyro_ra, gyro_la, gyro_rl, gyro_ll]
        mags = [mag_torso, mag_ra, mag_la, mag_rl, mag_ll]

        return accs, gyrs, mags

    def restructure(self) -> None:
        """Restructures raw acquisition txt files into organized standard
        structure."""

        try:
            trial_id = defaultdict(lambda: defaultdict(int))
            activities = os.listdir(self.dataset_dir)

            for act in activities:
                if act not in self.activity_map:
                    continue
                users = os.listdir(os.path.join(self.dataset_dir, act))

                for user in users:
                    trial_id[self.activity_map[act]][user] += 1
                    accs, gyros, mags = self.load_data(act, user)

                    for i, (acc, gyro, mag, pos) in enumerate(zip(accs, gyros, mags, self.position_map.values())):
                        # Build destination path
                        user_activity_path = os.path.join(
                            self.destination_dir,
                            user,
                            self.activity_map[act] + "_" + str(trial_id[self.activity_map[act]][user]),
                            self.wearable_prefix + str(i + 1) + "_" + pos,
                        )

                        os.makedirs(user_activity_path, exist_ok=True)

                        # Save accelerometer data
                        acc.to_csv(
                            os.path.join(user_activity_path, self.acc_filename),
                            header=False,
                            index=False,
                        )

                        # Save gyroscope data
                        gyro.to_csv(
                            os.path.join(user_activity_path, self.gyr_filename),
                            header=False,
                            index=False,
                        )

                        # Save magnetometer data
                        mag.to_csv(
                            os.path.join(user_activity_path, self.mag_filename),
                            header=False,
                            index=False,
                        )

            # Split users into train/test sets (user-level split)
            train_ids, test_ids = random_train_test_split_ids(users, test_ratio=0.2, seed=42)
            save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise


@DatasetFactory.register("ExtraSensory")
class ExtraSensoryParser(DatasetParser):
    """Parser for the ExtraSensory dataset."""

    def __init__(self, input_dir: str, output_dir: str) -> None:
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("ExtraSensory", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]
        self.position_map = self.config["position_map"]

        self.user_labels_folder = self.config["sensor_folder"]["users"]
        self.acc_zip_path = self.config["sensor_folder"]["accelerometer"]
        self.gyro_zip_path = self.config["sensor_folder"]["gyroscope"]
        self.mag_zip_path = self.config["sensor_folder"]["magnetometer"]

        self.acc_sensor_folder = self.config["zip_sensor_folders"]["accelerometer"]
        self.gyro_sensor_folder = self.config["zip_sensor_folders"]["gyroscope"]
        self.mag_sensor_folder = self.config["zip_sensor_folders"]["magnetometer"]

        self.user_file_suffix = self.config["user_file_suffix"]
        self.sensor_file_suffix = self.config["sensor_file_suffix"]

        self.acc_filename = self.config["acc_filename"]
        self.gyro_filename = self.config["gyro_filename"]
        self.mag_filename = self.config["mag_filename"]

        self.device_prefix = self.config["device_prefix"]
        self.unknown_position_label = self.config["unknown_position_label"]
        self.unknown_position_folder = self.config["unknown_position_folder"]

        self.timestamp_column_name = self.config["timestamp_column_name"]
        self.label_source_column = self.config["label_source_column"]
        self.label_prefix = self.config["label_prefix"]

        self.ns_conversion = float(self.config["ns_conversion"])
        self.acc_ms2_conversion = float(self.config["acc_ms2_conversion"])

        # Partition
        self.test_ratio = float(self.config["test_ratio"])
        self.random_seed = int(self.config["random_seed"])

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self, zip_file: zipfile.ZipFile, file_dict: dict[int, str], ts: int) -> np.ndarray | None:
        """Load one sensor file from a zip archive for a given timestamp."""

        # Get the file corresponding to the given timestamp
        filename = file_dict.get(ts)
        if filename:
            try:
                with zip_file.open(filename) as file:
                    wrapper = io.TextIOWrapper(file, encoding="utf-8")
                    data = np.loadtxt(wrapper)
                    if data.ndim == 1:
                        data = data[np.newaxis, :]
                    return data
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")
        return None

    def restructure(self) -> None:
        """Restructure the dataset into the harmonized structure."""

        # Collect all user identifiers from the label files
        uuids = [
            fname.split(".")[0]
            for fname in os.listdir(os.path.join(self.dataset_dir, self.user_labels_folder))
            if fname.endswith(self.user_file_suffix)
        ]

        # Train/test split at user level
        train_ids, test_ids = random_train_test_split_ids(uuids, test_ratio=self.test_ratio, seed=self.random_seed)

        # Save the split to the output directory
        save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        # Open all sensor archives
        with (
            zipfile.ZipFile(self.dataset_dir / self.acc_zip_path, "r") as acc_zip,
            zipfile.ZipFile(self.dataset_dir / self.gyro_zip_path, "r") as gyro_zip,
            zipfile.ZipFile(self.dataset_dir / self.mag_zip_path, "r") as mag_zip,
        ):
            # Process each user independently
            for uuid in uuids:
                logger.info(f"Processing {uuid}...")

                try:
                    X, Y, M, timestamps, feature_names, label_names = self._read_user_data(uuid)
                except Exception as e:
                    logger.info(f"Skipping {uuid} due to error: {e}")
                    continue

                # Identify the label indices for activities and phone positions
                label_indices = {
                    label: label_names.index(label) for label in self.activity_map.keys() if label in label_names
                }
                positions_indices = {
                    label: label_names.index(label) for label in self.position_map.keys() if label in label_names
                }

                # Infer the phone position for each timestamp
                phone_positions_per_timestamp = self._infer_phone_position(Y, positions_indices)

                # Group consecutive timestamps for each activity
                label_to_timestamp_groups = {
                    label: self._group_consecutive_timestamps(
                        Y[:, idx],
                        timestamps,
                        phone_positions_per_timestamp,
                    )
                    for label, idx in label_indices.items()
                }

                # Build the base output folder for this user
                base_folder = self.destination_dir / uuid
                base_folder.mkdir(parents=True, exist_ok=True)

                # Build dictionaries mapping timestamps to sensor files
                acc_files = self._build_file_dict(acc_zip, self.acc_sensor_folder, uuid)
                gyro_files = self._build_file_dict(gyro_zip, self.gyro_sensor_folder, uuid)
                mag_files = self._build_file_dict(mag_zip, self.mag_sensor_folder, uuid)

                # Process each activity independently
                for label, groups in label_to_timestamp_groups.items():
                    for group, positions_in_group in groups:

                        # Determine the dominant phone position for the activity segment
                        most_common_position = Counter(positions_in_group).most_common(1)[0][0]

                        folder_name = f"{self.activity_map[label]}_{group[0]}"
                        pos_name = (
                            self.position_map[most_common_position]
                            if most_common_position != self.unknown_position_label
                            else self.unknown_position_folder
                        )

                        # Build the output path for this segment
                        output_path = base_folder / folder_name / f"{self.device_prefix}{pos_name}"
                        output_path.mkdir(parents=True, exist_ok=True)

                        acc_data_list, gyro_data_list, mag_data_list = [], [], []

                        # Load all sensor segments belonging to this activity segment
                        for ts in group:
                            acc_data = self.load_data(acc_zip, acc_files, ts)
                            gyro_data = self.load_data(gyro_zip, gyro_files, ts)
                            mag_data = self.load_data(mag_zip, mag_files, ts)

                            if acc_data is not None:
                                acc_data_list.append(acc_data)
                            if gyro_data is not None:
                                gyro_data_list.append(gyro_data)
                            if mag_data is not None:
                                mag_data_list.append(mag_data)

                        if acc_data_list:
                            acc_df = pd.DataFrame(np.vstack(acc_data_list))

                            # Convert accelerometer axes from g to m/s2
                            acc_df.loc[:, acc_df.columns[1:]] *= self.acc_ms2_conversion

                            # Convert timestamps to nanoseconds
                            acc_df.iloc[:, 0] = (acc_df.iloc[:, 0].astype(float) * self.ns_conversion).astype("int64")

                            # Save accelerometer data
                            acc_df.to_csv(
                                output_path / self.acc_filename,
                                header=None,
                                index=None,
                            )

                        if gyro_data_list:
                            gyro_df = pd.DataFrame(np.vstack(gyro_data_list))

                            # Convert timestamps to nanoseconds
                            gyro_df.iloc[:, 0] = (gyro_df.iloc[:, 0].astype(float) * self.ns_conversion).astype("int64")

                            # Save gyroscope data
                            gyro_df.to_csv(
                                output_path / self.gyro_filename,
                                header=None,
                                index=None,
                            )

                        if mag_data_list:
                            mag_df = pd.DataFrame(np.vstack(mag_data_list))

                            # Convert timestamps to nanoseconds
                            mag_df.iloc[:, 0] = (mag_df.iloc[:, 0].astype(float) * self.ns_conversion).astype("int64")

                            # Save magnetometer data
                            mag_df.to_csv(
                                output_path / self.mag_filename,
                                header=None,
                                index=None,
                            )

    def _build_file_dict(self, zip_ref: zipfile.ZipFile, sensor_folder: str, uuid: str) -> dict[int, str]:
        """Build a mapping from timestamp to sensor file path inside a zip
        archive."""

        # Map each timestamp to its corresponding file inside the zip archive
        return {
            int(file_name.split("/")[-1].split(".")[0]): file_name
            for file_name in zip_ref.namelist()
            if file_name.startswith(f"{sensor_folder}/{uuid}/") and file_name.endswith(self.sensor_file_suffix)
        }

    def _parse_header_of_csv(self, csv_str: str) -> tuple[list[str], list[str]]:
        """Parse the CSV header and separate feature names from label names."""

        # Isolate the header line
        headline = csv_str[: csv_str.index("\n")]
        columns = headline.split(",")

        # Validate the first and last columns
        assert columns[0] == self.timestamp_column_name
        assert columns[-1] == self.label_source_column

        # Find the first label column
        for ci, col in enumerate(columns):
            if col.startswith(self.label_prefix):
                first_label_ind = ci
                break

        # Feature columns come after timestamp and before the labels
        feature_names = columns[1:first_label_ind]

        # Label columns extend to the one-before-last column
        label_names = columns[first_label_ind:-1]
        for li, label in enumerate(label_names):
            assert label.startswith(self.label_prefix)
            label_names[li] = label.replace(self.label_prefix, "")

        return feature_names, label_names

    def _parse_body_of_csv(
        self,
        csv_str: str,
        n_features: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Parse the CSV body into features, labels, missingness and
        timestamps."""

        # Read the full numeric table
        full_table = np.loadtxt(io.StringIO(csv_str), delimiter=",", skiprows=1)

        # Extract timestamps
        timestamps = full_table[:, 0].astype(int)

        # Extract sensor features
        X = full_table[:, 1 : (n_features + 1)]

        # Extract trinary labels and missing-label indicators
        trinary_labels_mat = full_table[:, (n_features + 1) : -1]
        M = np.isnan(trinary_labels_mat)
        Y = np.where(M, 0, trinary_labels_mat) > 0.0

        return X, Y, M, timestamps

    def _read_user_data(self, uuid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
        """Read the data (precomputed sensor-features and labels) for a
        user."""

        # Build the path to the compressed user label file
        user_data_file = self.dataset_dir / self.user_labels_folder / f"{uuid}{self.user_file_suffix}"

        # Read the full compressed CSV file for the user
        with gzip.open(user_data_file, "rb") as fid:
            csv_str = fid.read().decode("utf-8")

        # Parse header and body separately
        feature_names, label_names = self._parse_header_of_csv(csv_str)
        n_features = len(feature_names)
        X, Y, M, timestamps = self._parse_body_of_csv(csv_str, n_features)

        return X, Y, M, timestamps, feature_names, label_names

    def _infer_phone_position(self, Y: np.ndarray, positions_indices: dict[str, int]) -> list[str]:
        """Infer the phone position at each timestamp from the available
        position labels."""

        phone_positions = []

        # Determine the active phone position label for each timestamp
        for t in range(Y.shape[0]):
            found_position = None
            for pos in self.position_map.keys():
                idx = positions_indices[pos]
                if Y[t, idx]:
                    found_position = pos
                    break

            if not found_position:
                found_position = self.unknown_position_label

            phone_positions.append(found_position)

        return phone_positions

    def _group_consecutive_timestamps(
        self,
        mask: np.ndarray,
        timestamps_array: np.ndarray,
        positions_array: list[str],
    ) -> list[tuple[list[int], list[str]]]:
        """Group consecutive timestamps for which an activity is present."""

        groups = []
        current_group = []
        current_positions = []

        # Build groups of consecutive timestamps with the same active activity label
        for present, ts, pos in zip(mask, timestamps_array, positions_array):
            if present:
                current_group.append(ts)
                current_positions.append(pos)
            else:
                if current_group:
                    groups.append((current_group, current_positions))
                    current_group = []
                    current_positions = []

        if current_group:
            groups.append((current_group, current_positions))

        return groups


@DatasetFactory.register("FLAAP")
class FLAAPParser(DatasetParser):
    """Parser for the FLAAP dataset."""

    def __init__(self, input_dir: str, output_dir: str) -> None:
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("FLAAP", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]
        self.subjects_folder = self.config["subjects_folder"]
        self.master_folder = self.config["master_folder"]
        self.acc_keyword = self.config["acc_keyword"]
        self.gyro_keyword = self.config["gyro_keyword"]
        self.trimmed_keyword = self.config["trimmed_keyword"]
        self.acc_filename = self.config["acc_filename"]
        self.gyro_filename = self.config["gyro_filename"]

        self.activity_column = self.config["activity_column"]
        self.excel_columns = self.config["excel_columns"]
        self.time_column_index = self.config["time_column_index"]
        self.n_output_columns = self.config["n_output_columns"]
        self.device_position = self.config["device_position"]

        self.ns_conversion = int(self.config["ns_conversion"])

        # Partition
        self.test_ratio = float(self.config["test_ratio"])
        self.random_seed = int(self.config["random_seed"])

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self, subject: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load accelerometer and gyroscope data for a subject."""

        # Build path to the subject master folder
        master_dir = self.dataset_dir / self.subjects_folder / subject / self.master_folder

        # Find the trimmed accelerometer and gyroscope files
        acc_file = self._find_sensor_file(master_dir, self.acc_keyword)
        gyro_file = self._find_sensor_file(master_dir, self.gyro_keyword)

        # Load only the relevant columns from each Excel file
        acc = pd.read_excel(acc_file, usecols=self.excel_columns)
        gyro = pd.read_excel(gyro_file, usecols=self.excel_columns)

        return acc, gyro

    def restructure(self) -> None:
        """Restructure the dataset into the intended structure."""

        # Collect users
        users = os.listdir(os.path.join(self.dataset_dir, self.subjects_folder))

        # Process each user independently
        for user in users:
            acc, gyro = self.load_data(user)

            # Process each activity independently
            acts = set(acc[self.activity_column])
            for act in acts:

                # Skip activities not included in the activity map
                if act not in self.activity_map:
                    continue

                # Select accelerometer data for this activity
                acc_save = acc[acc[self.activity_column] == act].copy()

                # Keep only the timestamp and sensor axes columns
                acc_save = acc_save.iloc[:, : self.n_output_columns]

                # Reset timestamps to start at zero and convert to nanoseconds
                offset = acc_save.iloc[0, self.time_column_index]
                acc_save.iloc[:, self.time_column_index] = (
                    acc_save.iloc[:, self.time_column_index] - offset
                ) * self.ns_conversion

                # Build the output directory for this user and activity
                user_activity_path = self.destination_dir / user / f"{self.activity_map[act]}_1" / self.device_position
                user_activity_path.mkdir(parents=True, exist_ok=True)

                # Save accelerometer data
                acc_save.to_csv(
                    user_activity_path / self.acc_filename,
                    header=False,
                    index=False,
                )

                # Select gyroscope data for this activity
                gyro_save = gyro[gyro[self.activity_column] == act].copy()

                # Keep only the timestamp and sensor axes columns
                gyro_save = gyro_save.iloc[:, : self.n_output_columns]

                # Align gyroscope timestamps using the same activity offset
                gyro_save.iloc[:, self.time_column_index] = (
                    gyro_save.iloc[:, self.time_column_index] - offset
                ) * self.ns_conversion

                # Save gyroscope data
                gyro_save.to_csv(
                    user_activity_path / self.gyro_filename,
                    header=False,
                    index=False,
                )

        # Train/test split at user level
        train_ids, test_ids = random_train_test_split_ids(users, test_ratio=self.test_ratio, seed=self.random_seed)

        # Save the split to the output directory
        save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

    def _find_sensor_file(self, directory: Path, sensor_keyword: str) -> Path:
        """Find the trimmed file for one sensor in a subject folder."""

        # Select files matching both the trimmed keyword and the sensor
        matching_files = [
            file
            for file in directory.iterdir()
            if file.is_file() and self.trimmed_keyword in file.name.lower() and sensor_keyword in file.name.lower()
        ]

        return matching_files[0]


@DatasetFactory.register("HARSense")
class HARSenseParser(DatasetParser):
    """Parser for the HARSense dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("HARSense", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.rate = int(self.config["rate"])
        self.activity_map = self.config["activity_map"]

        self.activity_column = self.config["activity_column"]
        self.timestamp_column = self.config["timestamp_column"]

        self.acc_filename = self.config["acc_filename"]
        self.gyr_filename = self.config["gyr_filename"]

        self.acc_columns = self.config["acc_columns"]
        self.gyr_columns = self.config["gyr_columns"]

        self.trial_id = self.config["trial_id"]
        self.smartphone_position_id = self.config["smartphone_position_id"]

        self.ns_conversion = int(float(self.config["ns_conversion"]))

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self, subject: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load sensor data from a CSV file into accelerometer, gyroscope, and
        activity dataframes."""
        # Load raw CSV file
        df = pd.read_csv(subject)

        # Compute timestamp spacing in nanoseconds
        dt_ns = int(self.ns_conversion / self.rate)

        # Create synthetic timestamp column
        df[self.timestamp_column] = np.arange(len(df), dtype="int64") * dt_ns

        columns = df.columns

        # Build accelerometer DataFrame
        acc = pd.DataFrame(df[columns[3:6]].values + df[columns[6:9]].values, columns=[self.acc_columns])
        acc[self.timestamp_column] = df[self.timestamp_column]
        acc = acc[[self.timestamp_column] + self.acc_columns]

        # Build gyroscope DataFrame
        gyro = df[[self.timestamp_column] + self.gyr_columns]

        # Extract activity labels
        activity = df[self.activity_column]

        return acc, gyro, activity

    def restructure(self) -> None:
        """Restructure the dataset into standardized format.

        For each CSV file (user):
            - Load accelerometer, gyroscope, and activity data
            - Split samples by activity label
            - Save each activity segment into the required folder structure

        After processing all users:
            - Perform user-level train/test split
            - Save split information as JSON
        """
        try:
            # Collect all user CSV files in the dataset directory
            users = glob(os.path.join(self.dataset_dir, "*.csv"))

            if not users:
                logger.warning("No CSV files found in dataset directory.")
                return

            # Extract user IDs from filenames (prefix before first "_")
            users_names = [user.split(os.sep)[-1].split("_")[0] for user in users]

            # Process each user file independently
            for user in users:

                # Load accelerometer, gyroscope, and activity labels
                try:
                    acc, gyro, activity = self.load_data(user)
                except Exception:
                    logger.exception(f"Failed to load data for {user}. Skipping file.")
                    continue

                # Iterate over unique activity labels in the file
                for act in set(activity):

                    # Skip activities not defined in the mapping
                    if act not in self.activity_map:
                        continue

                    # Get indices corresponding to the current activity
                    idx = activity[activity == act].index

                    # Select only samples belonging to the activity
                    acc_save = acc.loc[idx]
                    gyro_save = gyro.loc[idx]

                    # Build destination path
                    user_activity_path = os.path.join(
                        self.destination_dir,
                        user.split(os.sep)[-1].split("_")[0],
                        self.activity_map[act] + self.trial_id,
                        self.smartphone_position_id,
                    )

                    os.makedirs(user_activity_path, exist_ok=True)

                    # Save accelerometer data
                    acc_save.to_csv(
                        os.path.join(user_activity_path, self.acc_filename),
                        header=False,
                        index=False,
                    )

                    # Save gyroscope data
                    gyro_save.to_csv(
                        os.path.join(user_activity_path, self.gyr_filename),
                        header=False,
                        index=False,
                    )

            # Split users into train/test sets (user-level split)
            train_ids, test_ids = random_train_test_split_ids(users_names, test_ratio=0.2, seed=42)
            save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise


@DatasetFactory.register("HHAR")
class HHARParser(DatasetParser):
    """Parser for the HHAR dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("HHAR", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]
        self.keep_activities = set(self.config.get("keep_activities", []))
        self.device_map = self.config["device_map"]

        self.timestamp_column = self.config["timestamp_column"]
        self.activity_column = self.config["activity_column"]
        self.sensor_column = self.config["sensor_column"]
        self.device_column = self.config["device_column"]
        self.participant_column = self.config["participant_column"]

        self.filenames = self.config["filenames"]
        self.min_samples_per_session = self.config["min_samples_per_session"]
        self.max_gap_ns = int(self.config["max_gap_ns"])

        # Partition
        self.test_ratio = float(self.config["test_ratio"])
        self.random_seed = int(self.config["random_seed"])

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> pd.DataFrame:
        """Load all activity sensor data from a CSV file into a single
        DataFrame."""
        all_data = []

        # Load each sensor CSV file and associate it with its sensor type
        for filename, sensor_type in self.filenames.items():
            path = self.dataset_dir / filename
            if path.exists():
                df = pd.read_csv(path)
                df[self.sensor_column] = sensor_type
                all_data.append(df)

        # Merge sensor tables into a single dataframe
        df_activity = pd.concat(all_data, ignore_index=True)

        # Remove rows with missing or 'null' labels
        activity = self.activity_column
        df_activity = df_activity[df_activity[activity].notna() & (df_activity[activity] != "null")]

        # Convert raw labels into the harmonized labels
        df_activity[activity] = df_activity[activity].map(self.activity_map).fillna(df_activity[activity])

        # Retain only activities in the configuration YAML
        if self.keep_activities:
            df_activity = df_activity[df_activity[activity].isin(self.keep_activities)]

        return df_activity

    def restructure(self) -> None:
        """Restructure the dataset into standardized format."""
        # Load and preprocess the raw HHAR data
        df = self.load_data()

        # Counter of trials per participant and activity
        global_label_counter = defaultdict(lambda: defaultdict(int))

        # Process each user idenpendently
        for user, user_df in df.groupby(self.participant_column):
            user_df = user_df.copy()

            # Convert the timestamp to numeric
            ts = pd.to_numeric(user_df[self.timestamp_column], errors="coerce")

            # Remove rows with invalid timestamps
            user_df = user_df.loc[ts.notna()].copy()

            # Store and sort the timestamps (already in nanoseconds)
            user_df[self.timestamp_column] = ts.loc[ts.notna()].astype("int64")
            user_df = user_df.sort_values(self.timestamp_column).reset_index(drop=True)

            if user_df.empty:
                continue

            # Split activity segments into trials
            # Split when activity changes or when there is a large global gap
            activity_column = self.activity_column
            label_change = user_df[activity_column] != user_df[activity_column].shift()
            time_gap = user_df[self.timestamp_column].diff().gt(self.max_gap_ns)
            user_df["segment_id"] = (label_change | time_gap).cumsum()

            # Process each activity segment independently
            for _, segment_df in user_df.groupby("segment_id"):
                if segment_df.empty:
                    continue

                activity = segment_df[activity_column].iloc[0]

                # Ignore segments too short to be considered valid trials
                if len(segment_df) < self.min_samples_per_session:
                    continue

                # Increment the trial counter for this participant and activity
                global_label_counter[user][activity] += 1
                trial_idx = global_label_counter[user][activity]

                # Within the trial, separate rows by device and sensor type
                for (device, sensor), device_df in segment_df.groupby([self.device_column, self.sensor_column]):
                    device_info = self.device_map.get(device, (None, None))

                    # Ignore devices not define in the configuration YAML
                    if device_info == (None, None):
                        continue

                    device_name, position = device_info

                    # Sort the device stream chronologically
                    device_df = device_df.sort_values(self.timestamp_column).copy()

                    # Keep only the required output columns
                    df_out = device_df[[self.timestamp_column, "x", "y", "z"]].copy()

                    # Build output path
                    filepath = (
                        self.destination_dir
                        / f"{user}"
                        / f"{activity}_{trial_idx}"
                        / f"{device_name}_{position}"
                        / f"{sensor}.txt"
                    )

                    # Save the output .txt file to the output directory
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    df_out.to_csv(filepath, index=False, header=False, sep=",")

        # Train/test split at user level
        users = df[self.participant_column].unique().tolist()
        train_ids, test_ids = random_train_test_split_ids(users, test_ratio=self.test_ratio, seed=self.random_seed)

        # Save the split to the output directory
        save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)


@DatasetFactory.register("HuGaDB")
class HuGaDBParser(DatasetParser):
    """Parser for the HuGaDB dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("HuGaDB", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]
        self.activity_map_various = self.config["activity_map_various"]

        self.rate = self.config["rate"]
        self.minimum_duration = self.config["minimum_duration"]
        self.minimum_lenght = self.config["minimum_lenght"]

        self.timestamp_column = self.config["timestamp_column"]
        self.activity_column = self.config["activity_column"]

        self.positions = self.config["positions"]

        self.acc_right_foot_columns = self.config["acc_right_foot_columns"]
        self.acc_right_shin_columns = self.config["acc_right_shin_columns"]
        self.acc_right_tight_columns = self.config["acc_right_tight_columns"]
        self.acc_left_foot_columns = self.config["acc_left_foot_columns"]
        self.acc_left_shin_columns = self.config["acc_left_shin_columns"]
        self.acc_left_tight_columns = self.config["acc_left_tight_columns"]

        self.gyr_right_foot_columns = self.config["gyr_right_foot_columns"]
        self.gyr_right_shin_columns = self.config["gyr_right_shin_columns"]
        self.gyr_right_tight_columns = self.config["gyr_right_tight_columns"]
        self.gyr_left_foot_columns = self.config["gyr_left_foot_columns"]
        self.gyr_left_shin_columns = self.config["gyr_left_shin_columns"]
        self.gyr_left_tight_columns = self.config["gyr_left_tight_columns"]

        self.various_tag = self.config["various_tag"]
        self.wearable_prefix = self.config["wearable_prefix"]
        self.accelerometer_indicator = self.config["accelerometer_indicator"]
        self.gyroscope_indicator = self.config["gyroscope_indicator"]

        self.ns_conversion = int(float(self.config["ns_conversion"]))
        self.int16_conversion = int(float(self.config["int16_conversion"]))
        self.ms2_conversion = int(float(self.config["ms2_conversion"]))
        self.degrees_conversion = int(float(self.config["degrees_conversion"]))
        self.rads_conversion = int(float(self.config["rads_conversion"]))

        self.acc_filename = self.config["acc_filename"]
        self.gyr_filename = self.config["gyr_filename"]

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self, input_file, tag=None):
        """Load txt and csv files."""

        # Data loading
        try:
            data = pd.read_csv(
                input_file,
                sep="\t",
                comment="#",
            )
        except Exception:
            logger.error("Failed to load data.")
            raise

        # Case where multiple activities ("various") are handled
        if tag == self.various_tag:
            act_keep = [i for i in set(data[self.activity_column]) if i in self.activity_map_various]
            if len(act_keep) == 0:
                return None, None, None

            all_seg_accs, all_seg_gyrs, all_seg_pos, all_seg_act = [], [], [], []

            # Process each activity separately
            for act_int in act_keep:
                df = data[data[self.activity_column] == act_int]

                idx_start = list(df.index[df.index.diff() != 1])
                idx_start += [int(df.index[-1])]

                for idx in range(len(idx_start) - 1):
                    df_seg = df.loc[idx_start[idx] : idx_start[idx + 1] - 1]

                    # Discard samples with less than minimum duration
                    if len(df_seg) < self.rate * self.minimum_duration:
                        continue
                    accs, gyrs = self._get_data(df_seg)

                    all_seg_accs += [accs]
                    all_seg_gyrs += [gyrs]
                    all_seg_pos += [self.positions]
                    all_seg_act += [[self.activity_map_various[act_int]] * 6]

            return all_seg_accs, all_seg_gyrs, all_seg_pos, all_seg_act

        else:
            accs, gyrs = self._get_data(data)

            return [accs], [gyrs], [self.positions], [[tag] * 6]

    def restructure(self) -> None:
        """Restructures raw acquisition txt files into organized standard
        structure."""
        try:
            users = []
            trial_id = defaultdict(lambda: defaultdict(int))
            fls = os.listdir(self.dataset_dir)

            for fl in fls:
                fl_split = fl.split("_")
                if len(fl_split) < self.minimum_lenght:
                    continue
                act = fl.split("_")[-3]

                # non-considered activities
                if act != self.various_tag and act not in self.activity_map:
                    continue

                user = fl.split("_")[-2]
                users += [user]

                filepath = (
                    os.path.join(self.dataset_dir, fl)
                    if act != self.various_tag
                    else os.path.join(self.dataset_dir, fl)
                )

                tag = self.activity_map[act] if act != self.various_tag else self.various_tag
                seg_accs, seg_gyrs, seg_positions, seg_act_map = self.load_data(filepath, tag=tag)

                for accs, gyrs, self.positions, act_map in zip(seg_accs, seg_gyrs, seg_positions, seg_act_map):
                    trial_id[act_map[0]][user] += 1
                    for i_sensor, (acc, gyr, pos, act_m) in enumerate(zip(accs, gyrs, self.positions, act_map)):

                        user_activity_path = os.path.join(
                            self.destination_dir,
                            user,
                            act_m + "_" + str(trial_id[act_m][user]),
                            self.wearable_prefix + str(i_sensor + 1) + "_" + pos,
                        )

                        os.makedirs(user_activity_path, exist_ok=True)

                        # Save accelerometer data
                        acc.to_csv(
                            os.path.join(user_activity_path, self.acc_filename),
                            header=False,
                            index=False,
                        )

                        # Save gyroscope data
                        gyr.to_csv(
                            os.path.join(user_activity_path, self.gyr_filename),
                            header=False,
                            index=False,
                        )

            # Split users into train/test sets (user-level split)
            train_ids, test_ids = random_train_test_split_ids(users, test_ratio=0.2, seed=42)
            save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise

    def _get_data(self, data):

        # Select accelerometer and gyroscope columns based on prefixes
        acc_columns = [col for col in data.columns if self.accelerometer_indicator in col]
        gyr_columns = [col for col in data.columns if self.gyroscope_indicator in col]

        # Copy relevant data to avoid modifying original dataframe
        acc_data = data[acc_columns].copy()
        gyr_data = data[gyr_columns].copy()

        # Conversion accelerometer data to m/s^2
        acc_data = acc_data / self.int16_conversion * self.ms2_conversion * standard_gravity_acceleration

        # Conversion gyroscope data to rad/s
        gyr_data = gyr_data * self.degrees_conversion / self.int16_conversion * math.pi / self.rads_conversion

        # Compute timestamp step in nanoseconds based on sampling rate
        step_ns = int(round(self.ns_conversion / self.rate))
        acc_data[self.timestamp_column] = np.arange(len(data), dtype=np.int64) * step_ns
        gyr_data[self.timestamp_column] = np.arange(len(data), dtype=np.int64) * step_ns

        acc_rf = acc_data[[self.timestamp_column] + self.acc_right_foot_columns]
        acc_rs = acc_data[[self.timestamp_column] + self.acc_right_shin_columns]
        acc_rt = acc_data[[self.timestamp_column] + self.acc_right_tight_columns]
        acc_lf = acc_data[[self.timestamp_column] + self.acc_left_foot_columns]
        acc_ls = acc_data[[self.timestamp_column] + self.acc_left_shin_columns]
        acc_lt = acc_data[[self.timestamp_column] + self.acc_left_tight_columns]

        gyr_rf = gyr_data[[self.timestamp_column] + self.gyr_right_foot_columns]
        gyr_rs = gyr_data[[self.timestamp_column] + self.gyr_right_shin_columns]
        gyr_rt = gyr_data[[self.timestamp_column] + self.gyr_right_tight_columns]
        gyr_lf = gyr_data[[self.timestamp_column] + self.gyr_left_foot_columns]
        gyr_ls = gyr_data[[self.timestamp_column] + self.gyr_left_shin_columns]
        gyr_lt = gyr_data[[self.timestamp_column] + self.gyr_left_tight_columns]

        accs = [acc_rf, acc_rs, acc_rt, acc_lf, acc_ls, acc_lt]
        gyrs = [gyr_rf, gyr_rs, gyr_rt, gyr_lf, gyr_ls, gyr_lt]

        return accs, gyrs


@DatasetFactory.register("KuHAR")
class KuHARParser(DatasetParser):
    """Parser for the KuHAR dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("KuHAR", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]

        self.acc_filename = self.config["acc_filename"]
        self.gyr_filename = self.config["gyr_filename"]

        self.smartphone_position_id = self.config["smartphone_position_id"]

        self.ns_conversion = int(float(self.config["ns_conversion"]))

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> None:
        """Not implemented for KuHAR dataset.

        Data loading is performed directly inside `restructure`.
        """
        return None

    def restructure(self) -> None:
        """Restructure the dataset into standardized format.

        For each activity folder:
            - Iterate over user trial files
            - Load data
            - Split accelerometer and gyroscope data
            - Convert timestamps to nanoseconds
            - Sort chronologically
            - Save to structured directory layout

        After processing all users:
            - Perform user-level train/test split
            - Save split information as JSON
        """
        try:
            user_trials = {}
            users = []

            if not list(os.listdir(self.dataset_dir)):
                logger.warning("No folders found in dataset directory.")
                return

            # Iterate over activity folders
            for act in os.listdir(self.dataset_dir):
                act_name = act.split(".")[1]

                # Skip activities not defined in the mapping
                if act_name not in self.activity_map:
                    continue

                # Iterate over trial files within activity folder
                for users_code in os.listdir(os.path.join(self.dataset_dir, act)):
                    user = users_code.split("_")[0]

                    # Track trial count per user
                    if user not in user_trials:
                        user_trials[user] = 0
                    else:
                        user_trials[user] += 1

                    # Load raw CSV
                    try:
                        df = pd.read_csv(os.path.join(self.dataset_dir, act, users_code), header=None)
                    except Exception:
                        logger.exception(f"Failed to load data for {user}. Skipping file.")
                        continue

                    # Build accelerometer DataFrame
                    acc = df.iloc[:, :4]
                    acc = acc.dropna()
                    acc.loc[:, 0] = acc.iloc[:, 0] * self.ns_conversion
                    acc.sort_values(by=0, inplace=True)

                    # Build gyroscope DataFrame
                    gyro = df.iloc[:, 4:]
                    gyro = gyro.dropna()
                    gyro.loc[:, 4] = gyro.iloc[:, 0] * self.ns_conversion
                    gyro.sort_values(by=4, inplace=True)

                    # Build destination path
                    user_activity_path = os.path.join(
                        self.destination_dir,
                        user,
                        self.activity_map[act_name] + "_" + str(user_trials[user]),
                        self.smartphone_position_id,
                    )

                    os.makedirs(user_activity_path, exist_ok=True)

                    # Save accelerometer data
                    acc.to_csv(
                        os.path.join(user_activity_path, self.acc_filename),
                        mode="a",
                        header=False,
                        index=False,
                    )

                    # Save gyroscope data
                    gyro.to_csv(
                        os.path.join(user_activity_path, self.gyr_filename),
                        mode="a",
                        header=False,
                        index=False,
                    )

                    users += [user]

            users = list(set(users))

            # Split users into train/test sets (user-level split)
            train_ids, test_ids = random_train_test_split_ids(users, test_ratio=0.2, seed=42)
            save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise


@DatasetFactory.register("MHEALTH")
class MHEALTHParser(DatasetParser):
    """Parser for the MHEALTH (Mobile HEALTH) dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("MHEALTH", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.rate = self.config["rate"]
        self.activity_map = self.config["activity_map"]

        self.device_id = self.config["device_id"]
        self.ns_conversion = float(self.config["ns_conversion"])

        self.position_sensor_map = self.config["position_sensor_map"]

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Load all subject .log files and return as {subject_id: dataframe}"""

        data = {}
        for file_path in self.dataset_dir.glob("*.log"):
            subject_id = file_path.stem.replace("mHealth_subject", "S")

            df = pd.read_csv(file_path, sep=r"\s+", header=None)
            data[subject_id] = df

        return data

    def restructure(self) -> None:
        """Restructure the dataset into standardized format.

        For each subject:
            - Load dataframe containing all sensor data
            - Segment the data into contiguous activity blocks
            - Generate timestamps for each segment
            - Organize data by activity and body position
            - Extract sensor-specific columns and apply necessary conversions

        After processing all users:
            - Perform subject-level train/test split
            - Save split information as JSON
        """
        try:
            subject_dfs = self.load_data()
            activity_counter = defaultdict(int)
            all_subjects = []

            for subj_id, df in subject_dfs.items():

                # Add column to detect changes in activity
                df["activity_change"] = (df.iloc[:, 23] != df.iloc[:, 23].shift()).cumsum()
                all_subjects.append(subj_id)

                # Iterate over contiguous activity segments
                for _, segment in df.groupby("activity_change"):
                    activity_id = int(segment.iloc[0, 23])

                    # Skip invalid activities
                    if activity_id == 0 or activity_id not in self.activity_map:
                        continue

                    # Add timestamps (restart at 0 for each segment)
                    segment = self._add_timestamp(segment)

                    act_name = self.activity_map[activity_id]

                    # Count occurences per subject/activity to create unique trials
                    activity_counter[(subj_id, act_name)] += 1
                    activity_folder_name = f"{act_name}_{activity_counter[(subj_id, act_name)]}"

                    # Iterate over body positions and their associated sensors
                    for position, sensors in self.position_sensor_map.items():
                        position_folder_name = f"{self.device_id}_{position}"

                        out_path = self.destination_dir / subj_id / activity_folder_name / position_folder_name
                        out_path.mkdir(parents=True, exist_ok=True)

                        # Extract and save each sensor stream
                        for sensor, (s_start, s_end) in sensors.items():
                            sensor_df = segment.iloc[:, [0] + list(range(s_start + 1, s_end + 1))]

                            # Convert deg/s to rad/s
                            if sensor == "Gyroscope":
                                sensor_df.iloc[:, 1:] = np.deg2rad(sensor_df.iloc[:, 1:])

                            # Check if not overwriting
                            out_file = out_path / f"{sensor}.txt"
                            if out_file.exists():
                                logger.warning(f"Overwriting existing file: {out_file}")

                            sensor_df.to_csv(out_file, sep=",", index=False, header=False)

                # Train-test split at subject-level
                train_ids, test_ids = random_train_test_split_ids(all_subjects, test_ratio=0.2, seed=42)
                save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise

    def _add_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a timestamp column starting at 0 based on the sampling rate.

        The timestamps are generated based on the dataset sampling rate
        and are converted to nanoseconds.
        """
        df = df.copy()
        n = len(df)
        interval = 1 / self.rate
        times = np.arange(n) * interval
        df.insert(0, "timestamp", (times * self.ns_conversion).astype("int64"))
        return df


@DatasetFactory.register("MotionSense")
class MotionSenseParser(DatasetParser):
    """Parser for the MotionSense dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("MotionSense", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]
        self.rate = self.config["rate"]

        self.device_id = self.config["device_id"]
        self.sensor_position = self.config["sensor_position"]

        self.output_columns = self.config["output_columns"]

        self.acc_filename = self.config["acc_filename"]
        self.gyr_filename = self.config["gyr_filename"]

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> None:
        """Not implemented for MotionSense dataset.

        Data loading is performed directly inside `restructure`.
        """
        return None

    def restructure(self) -> None:
        """Restructure the dataset into standardized format.

        For each subject and activity trial:
            - Load accelerometer (B) and gyroscope (C) data
            - Convert accelerometer units
            - Generate timestamps based on sampling rate
            - Organize data into target folder structure

        After processing all users:
            - Perform subject-level train/test split
            - Save split information as JSON
        """
        try:
            users = []
            period_ns = int(1e9 / self.rate)

            for csv_file in Path(self.dataset_dir).glob(os.path.join("B_Accelerometer_data", "*", "*", "*.csv")):
                csv_file = str(csv_file)

                # Extract metadata from folder/file structure
                dir_split = csv_file.split(os.sep)
                activity = dir_split[-2].split("_")[0]
                user = dir_split[-1].split("_")[1][: -len(".csv")]
                trial = dir_split[-2].split("_")[1]

                # Gyroscope file has identical structure in parallel directory
                csv_file_gyro = csv_file.replace("B_Accelerometer_data", "C_Gyroscope_data")

                acc = pd.read_csv(csv_file, index_col=0)
                acc *= standard_gravity_acceleration
                acc["timestamp"] = pd.Series(range(len(acc))) * period_ns
                acc = acc[self.output_columns]

                gyro = pd.read_csv(csv_file_gyro, index_col=0)
                gyro["timestamp"] = pd.Series(range(len(gyro))) * period_ns
                gyro = gyro[self.output_columns]

                user_activity_path = (
                    self.destination_dir
                    / f"S{user}"
                    / f"{self.activity_map[activity]}_{trial}"
                    / f"{self.device_id}_{self.sensor_position}"
                )

                os.makedirs(user_activity_path, exist_ok=True)

                acc.to_csv(os.path.join(user_activity_path, self.acc_filename), mode="a", header=False, index=False)
                gyro.to_csv(os.path.join(user_activity_path, self.gyr_filename), mode="a", header=False, index=False)

                users += [user]

            users = list(set(users))

            # Train-test split at subject-level
            train_ids, test_ids = random_train_test_split_ids(users, test_ratio=0.2, seed=42)
            save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise


@DatasetFactory.register("OPPORTUNITY")
class OPPORTUNITYParser(DatasetParser):
    """Parser for the OPPORTUNITY dataset."""

    def __init__(self, input_dir, output_dir):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("OPPORTUNITY", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]
        self.position_map = self.config["position_map"]

        self.granularity = self.config["granularity"]
        self.min_samples_per_session = self.config["min_samples_per_session"]
        self.ms2_conversion = float(self.config["ms2_conversion"])
        self.ns_conversion = float(self.config["ns_conversion"])
        self.prefix_ranges = self.config["prefix_ranges"]

        self.label_columns = {int(k): v for k, v in self.config["label_columns"].items()}
        self.locomotion_column = self.config["locomotion_column"]
        self.activity_column = self.config["activity_column"]
        self.timestamp_column = self.config["timestamp_column"]

        self.acc_filename = self.config["acc_filename"]
        self.gyr_filename = self.config["gyr_filename"]
        self.mag_filename = self.config["mag_filename"]

        # Partition
        self.test_ratio = float(self.config["test_ratio"])
        self.random_seed = int(self.config["random_seed"])

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]
        self.column_names_file = self.dataset_dir / self.config["column_names_file"]
        self.label_legend_file = self.dataset_dir / self.config["label_legend_file"]

        # Build header and label mappings
        self.index_to_header = self._build_column_headers()
        self.label_map = self._build_label_mapping()

    def load_data(self) -> list[tuple[Path, pd.DataFrame]]:
        """Load all .dat files and parse them into DataFrames."""
        files = sorted(self.dataset_dir.glob("*.dat"))
        return [(file, self._load_file(file)) for file in files]

    def restructure(self) -> None:
        """Restructure the dataset into standardized format."""
        # Load raw files
        files_df = self.load_data()

        users = set()

        # Counter of trials per participant and activity
        activity_counter = defaultdict(lambda: defaultdict(int))

        # Process each raw file independently
        for filepath, df in files_df:

            # Obtain the user id
            user_id, _ = filepath.stem.split("-")
            users.add(user_id)

            # Ensure activity labels are strings
            df[self.activity_column] = df[self.activity_column].astype(str)

            # In macro mode remove rows with missing labels or class "0"
            if self.granularity == "macro":
                df = df[~df[self.activity_column].isin(["nan", "0"])].copy()

            if df.empty:
                continue

            # Separate segments by activity changes
            df["segment_id"] = (df[self.activity_column] != df[self.activity_column].shift()).cumsum()

            # Process each continuous activity segment independently
            for _, segment in df.groupby("segment_id"):

                # Ignore segments that are too short
                if len(segment) < self.min_samples_per_session:
                    continue

                # Identify the segment's activity
                activity = segment[self.activity_column].iloc[0]

                # Increment the counter for this participant and activity
                activity_counter[user_id][activity] += 1
                trial_id = activity_counter[user_id][activity]

                # Counter for unique device position pairs
                device_counter = {}
                sensor_map = defaultdict(lambda: defaultdict(list))

                # Process every column independently
                for col in segment.columns:

                    # Skip metadata columns
                    if col in [self.activity_column, self.timestamp_column, "segment_id"]:
                        continue

                    # Column names in the format: W.device.position.measurement
                    try:
                        prefix, device, position, measurement = col.split(".", 3)
                    except ValueError:
                        continue

                    # Keep only body worn sensors
                    if prefix != "W":
                        continue

                    # Identify intended sensors
                    if measurement.startswith("acc"):
                        sensor_file = self.acc_filename
                    elif measurement.startswith("gyro"):
                        sensor_file = self.gyr_filename
                    elif measurement.startswith("magnetic"):
                        sensor_file = self.mag_filename
                    else:
                        continue

                    # Convert raw position into a more readable position
                    position = self.position_map.get(position, position)

                    # Increment counter for device poisition pair
                    key = f"{device}.{position}"
                    if key not in device_counter:
                        device_counter[key] = len(device_counter) + 1
                    device_id = device_counter[key]

                    # Define the device position folder name
                    device_position_folder = f"W{device_id}_{position}"

                    # Store column
                    sensor_map[device_position_folder][sensor_file].append(col)

                # Save each grouped sensor stream
                for device_position_folder, sensors in sensor_map.items():
                    for sensor_file, columns in sensors.items():

                        # Keep timestamp and measurement axes
                        sensor_df = segment[[self.timestamp_column] + columns].copy()

                        # Convert accelerometer from g to m/s2
                        if sensor_file == self.acc_filename:
                            sensor_df[columns] = sensor_df[columns].astype(float) * self.ms2_conversion

                        # Build final output filepath
                        output_path = (
                            self.destination_dir
                            / f"{user_id}"
                            / f"{activity}_{trial_id}"
                            / device_position_folder
                            / f"{sensor_file}.txt"
                        )

                        # Save the output .txt file to the output directory
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        sensor_df.to_csv(output_path, index=False, header=False, sep=",")

        # Train/test split at user level
        users = sorted(users)
        train_ids, test_ids = random_train_test_split_ids(users, test_ratio=self.test_ratio, seed=self.random_seed)

        # Save the split to the output directory
        save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

    def _load_file(self, filepath: Path) -> pd.DataFrame:
        """Load and process a raw .dat file."""

        # Load .dat files
        df = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")

        # Rename columns
        df.columns = [self.index_to_header.get(i, f"col_{i}") for i in range(df.shape[1])]

        # Normalize timestamps if present
        if self.timestamp_column in df.columns:
            ts = pd.to_numeric(df[self.timestamp_column], errors="coerce")
            df = df.loc[ts.notna()].copy()

            # Convert timestamps to nanoseconds
            df[self.timestamp_column] = (ts.loc[ts.notna()] * self.ns_conversion).astype("int64")

        # Add the harmonized activity column
        return self._apply_label_strategy(df)

    def _apply_label_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the harmonized activity column from raw labels."""

        # In macro mode select one activity from the locomotion labels
        if self.granularity == "macro":

            def map_label(x):
                if pd.isna(x):
                    return x
                try:
                    return self.label_map.get(int(float(x)), x)
                except (ValueError, TypeError):
                    return x

            df[self.activity_column] = df[self.locomotion_column].apply(map_label)

            # Map activity names
            df[self.activity_column] = df[self.activity_column].replace(self.activity_map)

        # In micro mode concatenate all labels columns
        elif self.granularity == "micro":

            label_cols = [col for col in df.columns if col.startswith(f"{self.activity_column}.")]
            if label_cols:

                # Remove rows where all labels are class '0'
                df = df[~df[label_cols].apply(lambda row: all(str(val).strip() == "0" for val in row), axis=1)].copy()

                for col in label_cols:
                    df[col] = df[col].apply(map_label)

                df[self.activity_column] = df[label_cols].astype(str).agg("-".join, axis=1)

        return df

    def _build_column_headers(self) -> dict[int, str]:
        """Build mapping from raw column index to standardized header names."""

        headers = {}

        with open(self.column_names_file) as file:
            for row in file:
                # Only process rows that describe a raw column
                if not row.startswith("Column:"):
                    continue

                match = re.match(r"Column:\s+(\d+)\s+(.+)", row)
                if not match:
                    continue

                index = int(match.group(1)) - 1
                descriptor = match.group(2).strip()

                # Remove anything after ";" to keep the descriptor compact
                name = descriptor.split(";")[0].strip()

                # First raw column corresponds to the timestamp
                if index == 0:
                    headers[index] = self.timestamp_column

                # Some columns correspond to activity labels
                elif index in self.label_columns:
                    headers[index] = self.label_columns[index]

                else:
                    tokens = name.split()
                    if len(tokens) >= 3:
                        measurement = tokens[-1]
                        position = tokens[-2]
                        device = "_".join(tokens[:-2])

                        prefix = None
                        for candidate_prefix, ranges in self.prefix_ranges.items():
                            if any(start <= index <= end for start, end in ranges):
                                prefix = candidate_prefix
                                break

                        if prefix is not None:
                            headers[index] = f"{prefix}.{device}.{position}.{measurement}"
                        else:
                            headers[index] = name
                    else:
                        headers[index] = name

        return headers

    def _build_label_mapping(self) -> dict[int, str]:
        mapping = {}

        with open(self.label_legend_file) as file:
            for row in file:
                row = row.strip()

                if not row or "-" not in row:
                    continue

                parts = [p.strip() for p in row.split("-", 2)]

                if len(parts) != 3:
                    continue

                code, _, label = parts

                if not code.isdigit():
                    continue

                mapping[int(code)] = label

        return mapping


@DatasetFactory.register("PAMAP2")
class PAMAP2Parser(DatasetParser):
    """Parser for the PAMAP2 dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("PAMAP2", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]

        self.device_id = self.config["device_id"]
        self.ns_conversion = float(self.config["ns_conversion"])

        self.position_map = self.config["position_map"]
        self.sensor_map = self.config["sensor_map"]

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Load all subject .dat files and return as {subject_id: dataframe}"""
        protocol_folder = self.dataset_dir / "Protocol"
        data = {}

        for file_path in protocol_folder.glob("*.dat"):
            subject_id = file_path.stem.replace("subject10", "S")

            df = pd.read_csv(file_path, sep=r"\s+", header=None)
            data[subject_id] = df

        return data

    def restructure(self) -> None:
        """Restructure the dataset into standardized format.

        For each subject:
            - Load the full dataframe containing all sensor streams
            - Segment the data into contiguous activity blocks based on activity ID
            - Filter out non-target activities and short segments (noise)
            - Map activity IDs to standardized activity names and enumerate trials
            - Convert timestamps from seconds to nanoseconds
            - Split data by body position and sensor type
            - Save each stream into the target folder

        After processing all users:
            - Perform subject-level train/test split
            - Save split information as JSON
        """
        try:
            subject_dfs = self.load_data()
            activity_counter = defaultdict(int)
            all_subjects = []

            for subj_id, df in subject_dfs.items():
                all_subjects.append(subj_id)

                # Add column to detect changes in activity
                df["activity_change"] = (df.iloc[:, 1] != df.iloc[:, 1].shift()).cumsum()

                # Iterate through contiguous activity segments
                for _, segment in df.groupby("activity_change"):
                    activity_id = int(segment.iloc[0, 1])

                    # Skip invalid or non-target activities
                    if activity_id == 0 or activity_id not in self.activity_map:
                        logger.debug(f"Skipping activity {activity_id} for subject {subj_id}")
                        continue

                    # Skip micro-segments
                    if len(segment) < 20:
                        logger.debug(f"Skipping short segment (len={len(segment)}) for subject {subj_id}")
                        continue

                    act_name = self.activity_map[activity_id]

                    # Track trial count
                    activity_counter[(subj_id, act_name)] += 1
                    activity_folder_name = f"{act_name}_{activity_counter[(subj_id, act_name)]}"

                    # Convert timestamp from s to ns
                    timestamp = segment.iloc[:, 0]
                    timestamp = (timestamp * int(self.ns_conversion)).astype("int64")

                    # Split by body position
                    for position, (start, end) in self.position_map.items():
                        imu_block = segment.iloc[:, start:end]

                        position_folder_name = f"{self.device_id}_{position}"
                        out_path = self.destination_dir / subj_id / activity_folder_name / position_folder_name
                        out_path.mkdir(parents=True, exist_ok=True)

                        # Extract individual sensor streams
                        for sensor, (s_start, s_end) in self.sensor_map.items():
                            sensor_df = pd.concat([timestamp, imu_block.iloc[:, s_start:s_end]], axis=1)
                            sensor_df = sensor_df.dropna()

                            out_file = out_path / f"{sensor}.txt"
                            if out_file.exists():
                                logger.warning(f"Overwriting existing file: {out_file}")
                            sensor_df.to_csv(out_file, sep=",", index=False, header=False)

                # Train-test split at subject-level
                train_ids, test_ids = random_train_test_split_ids(all_subjects, test_ratio=0.2, seed=42)
                save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise


@DatasetFactory.register("RealWorld")
class RealWorldParser(DatasetParser):
    """Parser for the RealWorld dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("RealWorldHAR", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]
        self.column_names = self.config["column_names"]

        self.position_map = self.config["position_map"]
        self.position_to_device = self.config["position_to_device"]

        self.sensor_map = self.config["sensor_map"]
        self.folder_sensor = self.config["folder_sensor"]

        self.ns_conversion = float(self.config["ns_conversion"])

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> None:
        """Not implemented for RealWorld dataset.

        Data loading is performed directly inside `restructure`.
        """
        return None

    def restructure(self) -> None:
        """Restructure the dataset into standardized format.

        For each subject:
            - Iterate through the subject's data directory
            - Handle heterogeneous folder structures:
                - Flat structure (e.g., acc_activity_csv/)
                - Nested structure with trials (e.g., acc_activity_csv/acc_activity_2_csv/)
            - Automatically detect the level at which CSV files are stored
            - Parse folder names to extract sensor type, activity label and trial number
            - Filter out non-target sensors and activities
            - Map sensor names and body positions
            - Load CSV files and convert timestamps to nanoseconds

        After processing all users:
            - Perform subject-level train/test split
            - Save split information as JSON
        """
        try:
            all_subjects = []

            for subj_folder in sorted(self.dataset_dir.iterdir()):
                if not subj_folder.is_dir() or not subj_folder.name.startswith("proband"):
                    continue

                # Convert folder name to standardized subject ID (proband1 -> S1)
                subj_id = subj_folder.name.replace("proband", "S")
                all_subjects.append(subj_id)

                data_folder = subj_folder / "data"
                for sensor_act_folder in data_folder.iterdir():

                    # Skip zip files and any non-directory entries
                    if not sensor_act_folder.is_dir():
                        continue

                    # Locate csv files
                    direct_csv = list(sensor_act_folder.glob("*.csv"))

                    if direct_csv:
                        folders_to_process = [sensor_act_folder]
                    else:
                        folders_to_process = [f for f in sensor_act_folder.iterdir() if f.is_dir()]

                    # Parse folder to extract activity and trial
                    for folder in folders_to_process:

                        folder_parts = folder.name.split("_")
                        valid_format = len(folder_parts) in (3, 4) and folder_parts[-1].lower() == "csv"

                        if not valid_format:
                            logger.debug(f"Skipping non-CSV folder: {sensor_act_folder.name}")
                            continue

                        sensor_prefix, activity_raw, *rest = folder_parts
                        trial = rest[0] if len(rest) == 2 else "1"

                        if sensor_prefix not in self.folder_sensor or activity_raw not in self.activity_map:
                            continue

                        act_name = self.activity_map[activity_raw]
                        activity_folder_name = f"{act_name}_{trial}"

                        # Process csv files
                        for csv_file in folder.glob("*.csv"):

                            if not csv_file.is_file():
                                continue

                            file_parts = csv_file.stem.split("_")

                            # Extract sensor
                            sensor_name = file_parts[0]
                            sensor_out_name = self.sensor_map.get(sensor_name, sensor_name)

                            # Extract body position from filename (last token)
                            position = file_parts[-1]
                            if position in self.position_map:
                                position = self.position_map[position]

                            device_id = self.position_to_device.get(position)
                            if device_id is None:
                                logger.debug(f"Unknown position mapping: {position}")
                            position_folder_name = f"{device_id}_{position}"

                            # Consistency check
                            if len(file_parts) == 4:
                                file_trial = file_parts[2]
                                if file_trial != trial:
                                    logger.debug(
                                        f"Trial mismatch: folder={trial}, file={file_trial}, file={csv_file.name}",
                                    )

                            # Load CSV
                            df = pd.read_csv(csv_file, index_col=False)

                            # Convert timestamp to nanoseconds
                            df = df[list(self.column_names)].copy()
                            df["attr_time"] = (df["attr_time"] * int(self.ns_conversion)).astype("int64")

                            # Save data in txt file
                            out_path = self.destination_dir / subj_id / activity_folder_name / position_folder_name
                            out_path.mkdir(parents=True, exist_ok=True)

                            # Check if not overwriting
                            out_file = out_path / f"{sensor_out_name}.txt"
                            if out_file.exists():
                                logger.warning(f"Overwriting existing file: {out_file}")

                            df.to_csv(out_file, sep=",", index=False, header=False)

            # Train-test split at subject-level
            train_ids, test_ids = random_train_test_split_ids(all_subjects, test_ratio=0.2, seed=42)
            save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise


@DatasetFactory.register("Shoaib13")
class Shoaib13Parser(DatasetParser):
    """Parser for the Shoaib13 dataset.

    DISCLAIMER:
    This dataset does not provide explicit subject identifiers. To enable subject
    splitting, heuristic thresholds were applied based on signal characteristics.
    As a consequence, subject segmentation may not be fully reliable.

    Additionally, the 'pocket' recording is not synchronized with the remaining
    sensor files. Due to this lack of temporal alignment, it was not possible to
    confidently assign a subject to the pocket file,
    """

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)
        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("Shoaib13", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.rate = self.config["rate"]
        self.activity_map = self.config["activity_map"]

        self.user_prefix = self.config["user_prefix"]
        self.device_prefix = self.config["device_prefix"]

        self.time_gap_threshold = float(self.config["time_gap_threshold"])
        self.time_gap_threshold_trial = float(self.config["time_gap_threshold_trial"])

        self.ms_constant = int(float(self.config["ms_constant"]))
        self.ns_conversion = int(float(self.config["ns_conversion"]))

        self.timestamp_column = self.config["timestamp_column"]
        self.activity_column = self.config["activity_column"]

        self.user_change = self.config["user_change"]
        self.trial_change = self.config["trial_change"]

        self.acc_filename = self.config["acc_filename"]
        self.gyr_filename = self.config["gyr_filename"]
        self.mag_filename = self.config["mag_filename"]

        self.acc_columns = self.config["acc_columns"]
        self.gyr_columns = self.config["gyr_columns"]
        self.mag_columns = self.config["mag_columns"]

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> None:
        """Not implemented for Shoaib13 dataset.

        Data loading is performed directly inside `restructure`.
        """
        return None

    def restructure(self) -> None:
        """Restructure the dataset into standardized format.

        For each device file:
            - Normalize timestamps
            - Group data by activity

        For each activity:
            - Detect users based on large timestamp gaps
            - Within each user, detect trials using a smaller time gap threshold

        For each trial:
            - Ensure valid timestamps
            - Extract accelerometer, gyroscope, and magnetometer data
            - Save to structured directory layout

        After processing all users:
            - Perform user-level train/test split
            - Save split information as JSON
        """

        try:
            # Ensure base output directory
            os.makedirs(self.destination_dir, exist_ok=True)

            # Process each Excel file (each = one phone position)
            for xlsx_file in Path(self.dataset_dir).glob("*.xlsx"):
                position = xlsx_file.stem
                try:
                    df = pd.read_excel(xlsx_file)
                except Exception as e:
                    logging.error(f"Failed to load data from {xlsx_file}: {e}")
                    continue

                df[self.timestamp_column] -= df[self.timestamp_column].min()
                df[self.timestamp_column] *= self.ms_constant

                for activity, activity_segment in df.groupby(self.activity_column):

                    # Detect users within the activity using Time_Stamp.diff()
                    time_diffs = activity_segment[self.timestamp_column].diff().fillna(0).abs()
                    activity_segment[self.user_change] = (time_diffs > self.time_gap_threshold).cumsum()

                    users = []
                    for user_id, user_segment in activity_segment.groupby(self.user_change):
                        user_name = f"{self.user_prefix}{user_id + 1}"  # 1-based
                        users += [user_name]

                        # Detect trials within the user using timestamp_column.diff()
                        time_diffs = user_segment[self.timestamp_column].diff().fillna(0).abs()
                        user_segment[self.trial_change] = (time_diffs > self.time_gap_threshold_trial).cumsum()

                        for trial_id, trial_segment in user_segment.groupby(self.trial_change):

                            user_activity_path = os.path.join(
                                self.destination_dir,
                                user_name,
                                activity + "_" + str(trial_id),
                                self.device_prefix + position,
                            )
                            os.makedirs(user_activity_path, exist_ok=True)

                            if trial_segment[self.timestamp_column].diff().sum() == 0:
                                trial_segment[self.timestamp_column] = np.arange(len(trial_segment), dtype=np.int64) * (
                                    self.ns_conversion / self.rate
                                )

                            # Extract and save sensor data
                            acc = trial_segment[[self.timestamp_column] + self.acc_columns]
                            gyro = trial_segment[[self.timestamp_column] + self.gyr_columns]
                            mag = trial_segment[[self.timestamp_column] + self.mag_columns]

                            # Save accelerometer data
                            acc.to_csv(
                                os.path.join(user_activity_path, self.acc_filename),
                                mode="a",
                                header=False,
                                index=False,
                            )

                            # Save gyroscope data
                            gyro.to_csv(
                                os.path.join(user_activity_path, self.gyr_filename),
                                mode="a",
                                header=False,
                                index=False,
                            )

                            # Save magnetometer data
                            mag.to_csv(
                                os.path.join(user_activity_path, self.mag_filename),
                                mode="a",
                                header=False,
                                index=False,
                            )

                    # Split users into train/test sets (user-level split)
                    train_ids, test_ids = random_train_test_split_ids(users, test_ratio=0.2, seed=42)
                    save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise


@DatasetFactory.register("Shoaib14")
class Shoaib14Parser(DatasetParser):
    """Parser for the Shoaib14 dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("Shoaib14", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.rate = self.config["rate"]
        self.sensors = self.config["sensors"]

        self.activity_map = self.config["activity_map"]
        self.position_map = self.config["position_map"]
        self.sensors_map = self.config["sensors_map"]

        self.original_part = self.config["original_part"]
        self.standardized_part = self.config["standardized_part"]
        self.sp_id = self.config["sp_id"]

        self.activity_column = self.config["activity_column"]
        self.timestamp_column = self.config["timestamp_column"]
        self.timestamp_indicator = self.config["timestamp_indicator"]
        self.ns_conversion = float(self.config["ns_conversion"])

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Loads all CSV files from the input directory into a dictionary of
        DataFrames."""
        raw_dfs = {}

        for file_name in os.listdir(self.dataset_dir):
            full_path = self.dataset_dir / file_name

            if full_path.is_file() and file_name.endswith(".csv"):
                part_name = os.path.splitext(file_name)[0]
                df = pd.read_csv(full_path, dtype=str, header=None, low_memory=False)
                raw_dfs[part_name] = df

        return raw_dfs

    def restructure(self) -> None:
        """Restructures raw participant CSV files into organized standard
        structure."""
        try:
            # Data loading
            try:
                dfs_dict = self.load_data()
            except Exception:
                logger.error("Failed to load data.")
                raise

            # Split users into train/test sets (user-level split)
            users = sorted(
                part_name.replace(self.original_part, self.standardized_part) for part_name in dfs_dict.keys()
            )
            train_ids, test_ids = random_train_test_split_ids(users, test_ratio=0.2, seed=42)
            save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

            for part_name, df in dfs_dict.items():
                part_path = self.destination_dir / part_name.replace(self.original_part, self.standardized_part)
                df = df.dropna(axis=1, how="all")
                df = self._fix_columns(df)

                # Activity separation
                unique_activities = df[self.activity_column].unique()

                for act_name in unique_activities:
                    activity_df = df[df.iloc[:, -1] == act_name]

                    if act_name in self.activity_map.keys():
                        act_name = self.activity_map[act_name]

                    else:
                        continue  # Retain only activities that are common with the AICOS dataset

                    act_path = part_path / f"{act_name}_1"

                    # Position separation
                    position_to_cols = {}

                    for col in activity_df.columns:

                        if self.timestamp_indicator in col:
                            position = col.split(self.timestamp_indicator)[0]
                            position_to_cols.setdefault(position, [])

                    for position in position_to_cols.keys():
                        matching_cols = [col for col in activity_df.columns if col.startswith(position)]
                        position_to_cols[position] = matching_cols

                    position_dfs = {}

                    for position, cols in position_to_cols.items():
                        position_dfs[position] = activity_df[cols]

                    for position, position_df in position_dfs.items():
                        position_df.columns = position_df.columns.str.replace(position + "_", "", regex=False)
                        position = self.position_map[position]
                        position_path = act_path / f"{self.sp_id}{position}"

                        timestamp_column = position_df[self.timestamp_column]
                        position_df = position_df.drop(self.timestamp_column, axis=1)

                        os.makedirs(position_path, exist_ok=True)

                        # Sensor separation
                        for i in range(0, 12, 3):
                            group = position_df.iloc[:, i : i + 3]
                            sensor_id = group.columns[0][0]

                            if sensor_id in self.sensors_map.keys():
                                sensor = self.sensors_map[sensor_id]

                                if (
                                    sensor not in self.sensors
                                ):  # Retain only sensors that are common with the AICOS dataset
                                    continue

                                sensor_path = position_path / f"{sensor}.txt"
                                group.insert(0, self.timestamp_column, timestamp_column)
                                group = self._fix_timestamp(group)
                                group.to_csv(sensor_path, index=False, header=False)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise

    def _fix_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replaces the timestamp column in the DataFrame with a uniformly
        spaced sequence of timestamps based on the sampling rate and a
        conversion factor."""
        timestamp_column = df[self.timestamp_column]
        n_samples = len(timestamp_column)
        time_interval = (1 / self.rate) * self.ns_conversion
        new_timestamp_column = [int(round(i * time_interval)) for i in range(n_samples)]
        df.loc[:, self.timestamp_column] = new_timestamp_column
        return df

    def _fix_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and renames the columns of the DataFrames."""
        new_columns = []
        current_position = None
        df_copy = df.copy()
        df_copy.iloc[1] = df_copy.iloc[1].fillna(self.activity_column)

        for col_idx in range(len(df_copy.columns)):
            position_candidate = df_copy.iloc[0, col_idx]
            suffix = df_copy.iloc[1, col_idx]

            if pd.notna(position_candidate):
                current_position = position_candidate

            new_col_name = f"{current_position}_{suffix}"

            if suffix == self.activity_column:
                new_col_name = self.activity_column

            new_columns.append(new_col_name)

        df_copy.columns = new_columns
        df_copy = df_copy.drop([0, 1]).reset_index(drop=True)
        return df_copy


@DatasetFactory.register("Shoaib16")
class Shoaib16Parser(DatasetParser):
    """Parser for the Shoaib16 dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("Shoaib16", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.rate = self.config["rate"]
        self.sensors = self.config["sensors"]

        self.activity_map = self.config["activity_map"]
        self.position_map = self.config["position_map"]

        self.position_to_model = self.config["position_to_model"]

        self.n_participants = self.config["n_participants"]

        self.columns = self.config["columns"]
        self.columns_sensors = self.config["columns_sensors"]

        self.filenames = self.config["filenames"]
        self.file_prefix = self.config["file_prefix"]

        self.ns_conversion = float(self.config["ns_conversion"])

        self.timestamp_column = self.config["timestamp_column"]
        self.activity_column = self.config["activity_column"]

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Load sensor data into a dictionary of dataframes indexed by
        position."""

        raw_dfs = {}

        for file_name in os.listdir(self.dataset_dir):
            full_path = os.path.join(self.dataset_dir, file_name)

            if os.path.isfile(full_path) and file_name.endswith(".csv"):
                position = file_name.removeprefix(self.file_prefix).removesuffix(".csv")

                # Load raw CSV
                try:
                    df = pd.read_csv(self.dataset_dir / file_name, dtype=str, header=None, low_memory=False)
                except Exception:
                    logger.exception(f"Failed to load data for position {position}. Skipping file.")
                    continue

                raw_dfs[position] = df

        return raw_dfs

    def restructure(self) -> None:
        try:
            # Data loading
            dfs_dict = self.load_data()
            users = set()

            # Position separation
            for position, df in dfs_dict.items():

                participant_id_by_act = 0
                participant_id = 0
                initial_part_id = 0
                n_parts = next(iter(self.n_participants.values()))
                model = self.position_to_model.get(position)
                position = self.position_map[position]
                df.columns = self.columns

                # Activity separation
                for activity, activity_id in self.activity_map.items():
                    activity_df = df[df.iloc[:, -1] == activity_id].copy()
                    activity_df[self.activity_column] = len(activity_df) * [activity]

                    # Participant separation
                    if activity in self.n_participants.keys():
                        n_parts = self.n_participants[activity]
                        samples_per_part = len(activity_df) // n_parts

                    for participant_id_by_act in range(n_parts):
                        participant_activity_df = activity_df.iloc[
                            participant_id_by_act * samples_per_part : participant_id_by_act * samples_per_part
                            + samples_per_part
                        ]
                        participant_id = initial_part_id + participant_id_by_act + 1

                        if participant_id_by_act == n_parts - 1:
                            participant_activity_df = activity_df.iloc[
                                participant_id_by_act * samples_per_part : len(activity_df)
                            ]
                        participant_activity_df = self._fix_timestamp(participant_activity_df)

                        # Df separation by sensors
                        dfs = {key: participant_activity_df[self.columns_sensors[key]] for key in self.columns_sensors}

                        users.add(str(participant_id))

                        for df_sensor, filename in zip(dfs.values(), self.filenames):

                            if (
                                filename not in self.sensors
                            ):  # Retain only sensors that are common with the AICOS dataset
                                continue

                            filepath = self._create_path(
                                self.destination_dir,
                                participant_id,
                                activity,
                                model,
                                position,
                                filename,
                            )

                            self._save_data(filepath, df_sensor)

            users = sorted(users)

            # Split users into train/test sets (user-level split)
            train_ids, test_ids = random_train_test_split_ids(users, test_ratio=0.2, seed=42)
            save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise

    def _fix_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replaces the timestamp column in the DataFrame with a uniformly
        spaced sequence of timestamps based on the sampling rate and a
        conversion factor."""
        timestamp_column = df[self.timestamp_column]
        n_samples = len(timestamp_column)
        time_interval = (1 / self.rate) * self.ns_conversion
        new_timestamp_column = [int(round(i * time_interval)) for i in range(n_samples)]
        df.loc[:, self.timestamp_column] = new_timestamp_column
        return df

    @staticmethod
    def _create_path(
        destination_dir: str,
        participant_id: str,
        activity: str,
        model: str,
        position: str,
        filename: str,
    ) -> str:
        return f"{destination_dir}/{participant_id}/{activity}_1/{model}_{position}/{filename}.txt"

    @staticmethod
    def _save_data(filepath: str, df: pd.DataFrame) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False, header=False)


@DatasetFactory.register("UCIHAR")
class UCIHARParser(DatasetParser):
    """Parser for the UCI HAR Smartphone dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("UCIHAR", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]

        self.ns_conversion = float(self.config["ns_conversion"])
        self.rate = self.config["rate"]

        self.sensor_columns = self.config["sensor_columns"]
        self.sensor_id = self.config["sensor_id"]
        self.sensor_position = self.config["sensor_position"]

        self.acc_filename = self.config["acc_filename"]
        self.gyr_filename = self.config["gyr_filename"]

        self.splits = self.config["splits"]

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

        # Data holders
        self.X = []  # list of DataFrames per window
        self.y = None  # labels array
        self.s = None  # subjects array
        self.reconstructed = None
        self.subjects = None
        self.labels = None

    def load_data(self, split: str) -> None:
        """Load windows, labels, and subject IDs for the given split."""
        raw_dir = self.dataset_dir / split / "Inertial Signals"
        arrays = {axis: np.loadtxt(raw_dir / f"{axis}_{split}.txt") for axis in self.sensor_columns}
        n_windows = arrays[self.sensor_columns[0]].shape[0]
        self.X = [
            pd.DataFrame({axis: arrays[axis][i] for axis in self.sensor_columns}) * standard_gravity_acceleration
            for i in range(n_windows)
        ]
        self.y = np.loadtxt(
            self.dataset_dir / split / f"y_{split}.txt",
            dtype=int,
        )
        self.s = np.loadtxt(
            self.dataset_dir / split / f"subject_{split}.txt",
            dtype=int,
        )

    def reconstruct(self) -> None:
        """Concatenate overlapping windows into a continuous signal and rebuild
        subject/label arrays."""

        # Windows have 50% overlap (skip first half except for first window)
        half = self.X[0].shape[0] // 2

        # Concatenate windows while removing overlapping samples
        segments = (
            df[self.sensor_columns] if idx == 0 else df[self.sensor_columns].iloc[half:]
            for idx, df in enumerate(self.X)
        )
        self.reconstructed = pd.concat(segments, ignore_index=True)

        first_len = self.X[0].shape[0]
        self.subjects = np.hstack(
            (
                np.repeat(self.s[0], first_len),
                np.repeat(self.s[1:], half),
            ),
        )
        self.labels = np.hstack(
            (
                np.repeat(self.y[0], first_len),
                np.repeat(self.y[1:], half),
            ),
        )

        assert len(self.reconstructed) == len(self.subjects) == len(self.labels)

    def restructure(self) -> None:
        """Restructure the dataset into standardized format.

        For each split (train/test):
            - Load windowed sensor data and metadata
            - Reconstruct continuous signals from overlapping windows
            - Group data by subject
            - Segment signals into contiguous activity runs
            - Export each run into the target folder structure

        After processing all splits:
            - Save subject-level train/test split information as JSON
        """
        train_test_splits = {}

        for split in self.splits:
            self.load_data(split)
            self.reconstruct()

            train_test_splits[split] = np.unique(self.s)
            for subject in np.unique(self.s):
                mask = self.subjects == subject
                curr_data = self.reconstructed[mask]
                curr_labels = self.labels[mask]
                self._export_runs(curr_data, curr_labels, subject)

        save_to_json(train_test_splits, self.destination_dir, logger)

    def _export_runs(self, df: pd.DataFrame, y: np.ndarray, subject: int) -> None:
        """Split continuous signal into runs and write each to disk."""

        # Assign a new run ID whenever activity label changes
        runs = (y != pd.Series(y).shift(fill_value=y[0])).cumsum().to_numpy()
        counters = defaultdict(int)

        for run_id in np.unique(runs):
            mask = runs == run_id
            label = int(y[mask][0])
            counters[label] += 1
            activity = self.activity_map[label]
            out_dir = (
                self.destination_dir
                / f"S{subject}"
                / f"{activity}_{counters[label]}"
                / f"{self.sensor_id}_{self.sensor_position}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            n = mask.sum()

            # Generate timestamps with fixed sampling interval (ns)
            ts = np.arange(n, dtype=np.int64) * (self.ns_conversion // self.rate)
            run_df = df.loc[mask, self.sensor_columns].copy()
            run_df.insert(0, "timestamp", ts)

            groups = {
                self.acc_filename: run_df[["timestamp", *[f"total_acc_{axis}" for axis in "xyz"]]],
                self.gyr_filename: run_df[["timestamp", *[f"body_gyro_{axis}" for axis in "xyz"]]],
            }

            for name, subset in groups.items():
                out_file = out_dir / name

                # Check if not overwriting
                if out_file.exists():
                    logger.warning(f"Overwriting existing file: {out_file}")

                subset.to_csv(
                    out_file,
                    sep=",",
                    index=False,
                    header=False,
                )


@DatasetFactory.register("UniMiB-SHAR")
class UnimibSharParser(DatasetParser):
    """Parser for the UniMiB-SHAR dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)
        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("UniMiB-SHAR", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.rate = int(self.config["rate"])

        self.activity_map = self.config["activity_map"]
        self.position_map = self.config["position_map"]

        self.timestamp_column = self.config["timestamp_column"]
        self.activity_column = self.config["activity_column"]
        self.subject_coloumn = self.config["subject_coloumn"]
        self.repetition_column = self.config["repetition_column"]
        self.sample_id_column = self.config["sample_id_column"]

        self.data_filename = self.config["data_filename"]
        self.labels_filename = self.config["labels_filename"]
        self.names_filename = self.config["names_filename"]
        self.acc_filename = self.config["acc_filename"]

        self.mat_variables = self.config["mat_variables"]
        self.axes = self.config["axes"]

        self.device_id = self.config["device_id"]
        self.seg_len = self.config["seg_len"]
        self.ns_conversion = float(self.config["ns_conversion"])

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw MATLAB files (data, labels, names), validate their
        existence, and return a combined dataframe with signals + labels, along
        with the activity names dataframe."""

        # Ensure dataset directory exists
        if not self.dataset_dir.is_dir():
            logging.info(f"Expected data folder at {self.dataset_dir}")
            return None, None

        # Ensure dataset directory is not empty
        if not any(self.dataset_dir.iterdir()):
            logging.info(f"Dataset folder {self.dataset_dir} is empty, skipping.")
            return None, None

        # Build full paths for required .mat files
        paths = {
            "data": self.dataset_dir / self.data_filename,
            "labels": self.dataset_dir / self.labels_filename,
            "names": self.dataset_dir / self.names_filename,
        }

        # Validate existence of each required file
        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {key} file at {path}")

        # Load specific variables from each .mat file using configured variable names
        data_vals = loadmat(paths["data"])[self.mat_variables["data"]]
        label_vals = loadmat(paths["labels"])[self.mat_variables["labels"]]
        name_vals = loadmat(paths["names"])[self.mat_variables["names"]]

        # Convert raw arrays to pandas DataFrames
        df_data = pd.DataFrame(data_vals)
        df_labels = pd.DataFrame(
            label_vals,
            columns=[self.activity_column, self.subject_coloumn, self.repetition_column],
        )
        df_names = pd.DataFrame(name_vals)

        # Combine signal data and labels horizontally. Sort by subject, activity and repetition for consistency
        df_combined = (
            pd.concat([df_data, df_labels], axis=1)
            .sort_values([self.subject_coloumn, self.activity_column, self.repetition_column])
            .reset_index(drop=True)
        )

        return df_combined, df_names

    def restructure(self) -> None:
        """Restructure the dataset into standardized format."""

        try:

            # Load data
            try:
                df, names_df = self.load_data()
            except Exception:
                logger.exception("Failed to load data.")
                return

            # Empty data
            if df is None or names_df is None:
                logging.info(f"No data loaded, skipping {self.__class__.__name__}.")
                return

            df = df.copy()

            # Create a sample ID within each (subject, activity) group with sequential numbering
            df[self.sample_id_column] = df.groupby([self.subject_coloumn, self.activity_column]).cumcount() + 1

            # Extract unique user IDs
            users = sorted(df[self.subject_coloumn].astype(str).unique().tolist())

            # Split users into train/test sets (user-level split)
            train_ids, test_ids = random_train_test_split_ids(users, test_ratio=0.2, seed=42)
            save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

            for _, row in df.iterrows():

                # Extract raw signal values. First 3 * seg_len values correspond to X,Y,Z
                signals = np.array(row.iloc[: 3 * self.seg_len].values).reshape(3, self.seg_len).T

                # Create dataframe with axis names
                seg_df = pd.DataFrame(signals, columns=self.axes)

                # Add timestamp column
                seg_df = self._add_timestamp(seg_df)

                # Extract metadata
                act_code = int(row.ActivityType)
                subject_id = int(row.Subject)
                repetition = int(row.Repetition)
                sample_id = int(row.SampleID)

                # Map activity code to activity name
                act_name = names_df.iat[act_code - 1, 0].item()

                if act_name in self.activity_map.keys():
                    act_name = self.activity_map[act_name]

                else:
                    continue  # Retain only activities that are common with the AICOS dataset

                # Define device position id
                position = self.device_id + str(self.position_map[repetition])

                # Build destination path
                part_dir = self.destination_dir / f"{subject_id}"
                activity_dir = part_dir / f"{act_name}_{sample_id}"
                pos_dir = activity_dir / position
                pos_dir.mkdir(parents=True, exist_ok=True)

                # Save accelerometer data
                file_path = pos_dir / self.acc_filename
                try:
                    seg_df.to_csv(file_path, index=False, header=False)
                except Exception as e:
                    logger.error("Failed to write %s: %s", file_path, e)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise

    def _add_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add timestamp column in the DataFrame with a uniformly spaced
        sequence of timestamps based on the sampling rate and a conversion
        factor."""
        df = df.copy()
        n = len(df)
        interval = 1 / self.rate
        times = np.arange(n) * interval
        df.insert(0, self.timestamp_column, (times * self.ns_conversion).astype(int))

        return df


@DatasetFactory.register("USC-HAD")
class USCHADParser(DatasetParser):
    """Parser for the USC-HAD dataset."""

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("USC-HAD", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.rate = int(self.config["rate"])

        self.activity_map = dict(self.config["activity_map"])
        self.keep_activities = set(self.config["keep_activities"])

        self.ns_conversion = float(self.config["ns_conversion"])
        self.output_columns = self.config["output_columns"]

        self.sensor_id = str(self.config["sensor_id"])
        self.sensor_position = str(self.config["sensor_position"])

        self.acc_filename = self.config["acc_filename"]
        self.gyr_filename = self.config["gyr_filename"]

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> list[str, pd.DataFrame]:
        """Load .mat files for all subjects and return a list of data
        entries."""
        raw_data = []
        for subject_name in os.listdir(self.dataset_dir):

            if "Subject" not in subject_name:
                continue

            for file_name in os.listdir(self.dataset_dir / subject_name):
                full_path = os.path.join(self.dataset_dir / subject_name, file_name)

                try:
                    # Load .mat file as clean Python dict (no MATLAB struct quirks)
                    raw_data.append(loadmat(full_path, simplify_cells=True, squeeze_me=True, struct_as_record=False))

                except Exception as e:
                    logger.warning("Failed to load %s: %s", full_path, e)
                    continue

        return raw_data

    def restructure(self) -> None:
        """Restructure the dataset into standardized format.

        For each subject:
            - Load raw MATLAB recordings
            - Extract activity labels and filter target activities
            - Convert sensor data to standard units
            - Generate timestamps based on sampling rate
            - Organize data into the target folder structure

        After processing all users:
            - Perform subject-level train/test split
            - Save split information as JSON
        """
        try:
            # Create the timestamp column
            step_ns = int(self.ns_conversion / self.rate)

            # Ensure base output directory
            os.makedirs(self.destination_dir, exist_ok=True)

            raw_data = self.load_data()

            subjects = set()
            for data in raw_data:

                # Handle inconsistent key naming in dataset
                activity = data.get("activity_number", data.get("activity_numbr"))
                activity_name = self.activity_map[int(activity)]

                # Skip non-target activities
                if activity_name not in self.keep_activities:
                    continue

                subject = data["subject"]
                subject_id = f"S{subject}"
                subjects.add(subject_id)

                # Track trial count per subject/activity
                search_dir = os.path.join(self.destination_dir, subject_id)
                os.makedirs(search_dir, exist_ok=True)
                existing_trials = [p for p in Path(search_dir).iterdir() if p.is_dir() and activity_name in p.name]
                trial = str(len(existing_trials) + 1)

                sensors = data["sensor_readings"]

                # Convert to m/s²
                acc = sensors[:, :3] * standard_gravity_acceleration

                # Convert to rad/s
                gyro = np.deg2rad(sensors[:, 3:6])

                # Generate timestamps
                timestamps = np.arange(sensors.shape[0], dtype=np.int64) * step_ns

                df = pd.DataFrame(
                    np.hstack((timestamps[:, np.newaxis], acc, gyro)),
                    columns=["timestamp", "Ax", "Ay", "Az", "Gx", "Gy", "Gz"],
                )
                df["timestamp"] = df["timestamp"].astype("int64")

                user_activity_path = os.path.join(
                    self.destination_dir,
                    subject_id,
                    activity_name + "_" + trial,
                    self.sensor_id + "_" + self.sensor_position,
                )
                os.makedirs(user_activity_path, exist_ok=True)

                # Split sensors
                acc = df[self.output_columns["acc"]]
                gyro = df[self.output_columns["gyro"]]

                acc.to_csv(os.path.join(user_activity_path, self.acc_filename), mode="w", header=False, index=False)
                gyro.to_csv(os.path.join(user_activity_path, self.gyr_filename), mode="w", header=False, index=False)

            # Train-test split at subject-level
            subjects = sorted(subjects)
            train_ids, test_ids = random_train_test_split_ids(subjects, test_ratio=0.2, seed=42)
            save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

        except Exception:
            logger.exception("Fatal error during dataset restructuring.")
            raise


@DatasetFactory.register("WISDM")
class WISDMDatasetParser(DatasetParser):
    """Parser for the WISDM dataset."""

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        super().__init__(input_dir, output_dir)

        # Load dataset-specific configuration from YAML
        self.config = load_dataset_config("WISDM", Path(__file__).parent / "external_datasets_config.yaml")

        # Initialize variables from configuration file
        self.activity_map = self.config["activity_map"]
        self.columns = self.config["columns"]
        self.equipments = self.config["equipments"]
        self.position_tags = self.config["position_tags"]
        self.sensor_to_filename = self.config["sensor_to_filename"]
        self.sensors = self.config["sensors"]
        self.expected_n_columns = self.config["expected_n_columns"]
        self.converter_column_index = self.config["converter_column_index"]

        self.activity_key_file = self.config["activity_key_file"]
        self.raw_folder = self.config["raw_folder"]
        self.activity_column = self.config["activity_column"]
        self.time_column = self.config["time_column"]
        self.run_id_column = self.config["run_id_column"]
        self.output_columns = self.config["output_columns"]

        # Partition
        self.test_ratio = float(self.config["test_ratio"])
        self.random_seed = int(self.config["random_seed"])

        # Data holders
        self.activity_letter_map: dict[str, str] = {}
        self.files: dict[tuple[str, str], list[Path]] = {}

        # Paths
        self.dataset_dir = Path(self.input_dir) / self.config["dataset_folder"]
        self.destination_dir = Path(self.output_dir) / self.config["output_folder"]

    def load_data(self) -> tuple[dict[str, str], dict[tuple[str, str], list[Path]]]:
        """Load activity mapping and enumerate files for all equipment/sensor
        combinations."""

        # Load the mapping from activity letters to raw activity names
        self.activity_letter_map = self._parse_activity_file(self.dataset_dir / self.activity_key_file)

        self.user_ids = []

        # Enumerate all raw files for each equipment/sensor combination
        self.files = {
            (eq, se): list(self._iter_data_files(self.dataset_dir, eq, se))
            for eq in self.equipments
            for se in self.sensors
        }
        if not any(self.files.values()):
            raise FileNotFoundError(
                f"No input files found under {self.dataset_dir}/{self.raw_folder}",
            )
        return self.activity_letter_map, self.files

    def restructure(self) -> None:
        """Split raw streams into contiguous activity runs and export
        harmonized sensor files."""

        # Load activity mapping and raw files if not already loaded
        if not self.activity_letter_map or not self.files:
            self.load_data()

        # Process each equipment/sensor combination independently
        for (equipment, sensor), file_list in self.files.items():
            if not file_list:
                continue  # Skip missing modality/equipment

            # Process each raw file independently
            for file_path in file_list:
                subject_id = self._extract_subject_id_from_path(file_path)
                self.user_ids.append(subject_id)

                # Load raw signal file
                df = self._read_wisdm_file(file_path)

                # Identify contiguous runs of the same activity
                df[self.run_id_column] = self._contiguous_run_ids(df[self.activity_column])

                # Count acquisitions for each activity letter
                per_activity_counter: dict[str, int] = {}

                # Process each contiguous activity segment independently
                for _, run_df in df.groupby(self.run_id_column, sort=True):
                    activity_letter = run_df[self.activity_column].iloc[0]
                    per_activity_counter[activity_letter] = per_activity_counter.get(activity_letter, 0) + 1
                    acq_idx = per_activity_counter[activity_letter]

                    # Keep only the harmonized output columns and reset timestamps
                    prepared = self._prepare_single_activity_data(run_df)

                    # Build ouput path
                    out_path = self._build_output_path(
                        root=self.destination_dir,
                        subject_id=subject_id,
                        activity_letter=activity_letter,
                        acquisition_idx=acq_idx,
                        equipment=equipment,
                        sensor=sensor,
                    )

                    # Skip activities not included in the harmonization map
                    if out_path is not None:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        prepared.to_csv(out_path, index=False, header=False)

        # Train/test split at user level
        train_ids, test_ids = random_train_test_split_ids(
            sorted(set(self.user_ids)),
            test_ratio=self.test_ratio,
            seed=self.random_seed,
        )

        # Save the split to the output directory
        save_to_json({"train": train_ids, "test": test_ids}, self.destination_dir, logger)

    def _read_wisdm_file(self, file_path: Path) -> pd.DataFrame:
        """Read WISDM .txt into DataFrame with canonical column names."""

        # Load the raw text file and clean the trailing ';' from the last column
        df = pd.read_csv(
            file_path,
            header=None,
            sep=",",
            engine="python",
            converters={self.converter_column_index: lambda x: float(str(x).rstrip(";\r\n "))},  # strip trailing ';'
            on_bad_lines="error",
        )

        # Validate the expected number of columns
        if df.shape[1] != self.expected_n_columns:
            raise ValueError(f"Unexpected column count in {file_path}: {df.shape[1]}")

        # Assign canonical column names
        df.columns = self.columns

        return df

    def _iter_data_files(self, dataset_dir: Path, equipment: str, sensor: str) -> Iterable[Path]:
        """Iterate over all raw files for one equipment/sensor combination."""

        # Build the glob pattern for the raw files
        pattern = dataset_dir / self.raw_folder / equipment / sensor / "*.txt"
        return (Path(p) for p in sorted(glob(str(pattern))))

    def _prepare_single_activity_data(self, single_activity_df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the selected output columns and reeset timestamps."""

        # Select the harmonized output columns
        out = single_activity_df.loc[:, self.output_columns].copy()

        # Reset timestamps relative to the start of the activity segment
        out[self.time_column] = out[self.time_column] - out[self.time_column].iloc[0]

        # Reset the row index
        out.reset_index(drop=True, inplace=True)
        return out

    def _build_output_path(
        self,
        *,
        root: Path,
        subject_id: int,
        activity_letter: str,
        acquisition_idx: int,
        equipment: str,
        sensor: str,
    ) -> Path | None:
        """Build output path for subject/activity acquisition and sensor
        modality."""

        # Mao the activity letter to the raw activity name
        raw_activity_name = self.activity_letter_map.get(activity_letter)
        if raw_activity_name is None:
            raise KeyError(f"Activity letter '{activity_letter}' not in activity map.")

        # Map the raw activity name to the harmonized activity name
        activity_name = self.activity_map.get(raw_activity_name)
        if activity_name is None:
            return None

        # Identify the harmonized position tag and ouput filename
        position_tag = self.position_tags[equipment]
        modality = self.sensor_to_filename[sensor]

        return root / str(subject_id) / f"{activity_name}_{acquisition_idx}" / position_tag / modality

    @staticmethod
    def _parse_activity_file(filepath: Path) -> dict[str, str]:
        """Parse 'activity = LETTER' lines into {LETTER: activity_name}."""

        # Build the mapping between activity letters and raw activity names
        with filepath.open() as f:
            return {
                letter.strip(): activity.strip()
                for line in f
                if "=" in line
                for activity, letter in [line.split("=", 1)]
            }

    @staticmethod
    def _extract_subject_id_from_path(path: Path) -> int:
        """Extract subject id from filenames like
        'data_<id>_<sensor>_<equipment>.txt'."""

        # Extract the numeric subject id from the filename
        try:
            return int(path.name.split("_")[1])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Cannot extract subject id from filename: {path.name}") from e

    @staticmethod
    def _contiguous_run_ids(labels: pd.Series) -> pd.Series:
        """Return run ids for contiguous blocks of identical labels."""

        # Assign one run id to each contiguous activity block
        return labels.ne(labels.shift()).cumsum()


@DatasetFactory.register("DatasetB")
class DatasetBParser(DatasetParser):
    def load_data(self) -> Any:
        raise NotImplementedError("DatasetBParser.load_data not implemented")

    def restructure(self, *args, **kwargs) -> None:
        raise NotImplementedError("DatasetBParser.restructure not implemented")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and restructure datasets via factory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DatasetFactory._registry.keys(),
        required=True,
        help="Name of the dataset (e.g. 'CHARM', 'ExtraSensory', 'FLAAP', 'HARSense', 'HHAR', 'KuHAR', MHEALTH', 'MotionSense', 'OPPORTUNITY', 'PAMAP2', 'RealWorld', 'Shoaib13', 'Shoaib14', 'Shoaib16', 'UCIHAR', 'UniMiB-SHAR', 'USC-HAD', 'WISDM')",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Raw data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save restructured data",
    )
    args = parser.parse_args()
    parser_inst = DatasetFactory.create(args.dataset, args.input_dir, args.output_dir)
    parser_inst.restructure()
