import logging
from pathlib import Path

from download_public_datasets import DATASETS, download_and_extract_dataset, remove_raw_data, setup_logging
from parser import DatasetFactory

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download, parse and restructure public datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--select_datasets",
        type=str,
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Select one or more datasets from 'CHARM', 'DailySportsActivities','ExtraSensory', 'FLAAP', 'HARSense', 'HHAR', 'HuGaDB', 'KuHAR', MHEALTH', 'MotionSense', 'Opportunity', 'PAMAP2', 'RealWorldHAR', 'Shoaib13', 'Shoaib14', 'Shoaib16', 'UCIHAR', 'UniMiB-SHAR', 'USC-HAD', 'WISDM'.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to directory containing raw datasets or where datasets are downloaded to",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to directory where processed datasets will be saved",
    )
    parser.add_argument(
        "--remove_raw",
        action="store_true",
        help="If set, removes raw datasets after processing",
    )
    args = parser.parse_args()

    for dataset in args.select_datasets:
        setup_logging()

        # Download
        download_and_extract_dataset(dataset, DATASETS[dataset], args.input_dir)

        # Restructure
        logging.info(f"Restructuring {dataset}")
        parser_inst = DatasetFactory.create(dataset, args.input_dir, args.output_dir)
        parser_inst.restructure()

        # Optional removal of raw data
        if args.remove_raw:
            remove_raw_data(dataset, args.input_dir)
