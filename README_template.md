***aicoshar-parser-suite***
*********************************************************************

**Authors:** Marília Barandas, Ana Cravidão Pereira, Carolina Carvalho, Duarte Folgado, Ricardo Santos, Maria Russo, Vânia Guimarães, Hugo Gamboa, Inês Sousa
Date: April 2026
Institution: Fraunhofer Portugal AICOS
Contact: marilia.barandas@aicos.fraunhofer.pt

---------------------------------------------------------------------

**Citation:** If you use the aicoshar-parser-suite tool, please cite the following work, which provides information about the AICOS-HAR dataset and the results of baseline experiments including some of the public datasets supported by aicoshar-parser-suite.

{CITATION}

---------------------------------------------------------------------

The aicoshar-parser-suite is a repository that provides a tool for automatic download and structure harmonization of 20 public human activity recognition datasets. The public datasets were harmonized according the AICOS-HAR dataset structure and content, meaning only activities and sensors present in AICOS-HAR dataset were kept. However, most parsers are easily adaptable.

The harmonized structure is 
`ParticipantID/Activity_TrialID/Device_Position/Sensor.txt`.

- ParticipantID: Identificator of the participant. When existing, the original dataset ID's were kept.
- Activity: Activity performed by the participant. When the protocol features more then one activity, the acquisition was segmented into single-activity acquisitions. 
- TrialID: Repetition of the activity by the same participant.
- Device: Model of the device. If the model was not provided, SP identifies smartphones and W identifies wearables.
- Position: Body position in which the device was placed.

The inertial data is structured in the format:
Timestamp  |  Sensor_x  |  Sensor_y  | Sensor_z

The barometric data is structured in the format:
Timestamp  |  Sensor

Physical units:
- Accelerometer: meters per second squared (m/s²) 
- Barometer: millibar (mbar) 
- Gyroscope: radians per second (rad/s) 
- Magnetometer: microtesla (µT) 
- Timestamp: nanosecond (ns)

The public human activity recognition datasets supported by aicoshar-parser-suite are: AICOS-HAR, CHARM, DailySportsActivities, ExtraSensory, FLAAP, HARSense, HHAR, HuGaDB, KuHAR, MHEALTH, MotionSense, Opportunity, PAMAP2, RealWorldHAR, Shoaib13, Shoaib14, Shoaib16, UCIHAR, UniMiB-SHAR, USC-HAD and WISDM. Only datasets that are publicly accessible without requiring login support automatic download.

---------------------------------------------------------------------

**Usage Notes:** 

To run the aicoshar-parser-suite, execute the script `main.py`. 

*Optional parameters:*

`-- input_dir`: Path to the directory where your dataset is stored locally, or where it should be automatically downloaded. Default: data/raw in the project repository.

`-- outpu_dir`: Path to the directory where the reorganized dataset will be saved. Default: data/processed in the project repository.

`-- select_datasets`: List of datasets you want to download and reorganize. Default: All supported datasets.

`-- remove_raw`: If set, removes the original raw dataset before reorganization.