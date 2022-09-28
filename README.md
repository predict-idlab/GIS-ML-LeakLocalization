Leak Localization in Water Distribution Networks using GIS-Enhanced Autoencoders
==============================

Leak localization experiments using data-driven approaches based on GIS-enhanced autoencoders. Part of the implementation is based on https://github.com/stellargraph/stellargraph.


Running
-------

1. Clone this project: `git clone...`.
2. Preferably create a virtual environment (`conda create --name aeleakloc`) and activate it (`conda activate aeleakloc`).
3. `cd` to the project's root folder and install all required packages: `pip install -r requirements.txt`.
4. Run `python train_and_test.py` to run a basic test. (You will first need to unzip the files in the pressures directory, however.)

Project Organization
------------

    |-- LICENSE
    |-- README.md
    |-- analysis                            <-- Repository for results used for analysis.
    |   |-- 13_14_july_events
    |   |   |-- lstm
    |   |   `-- tgcn
    |   |-- MC1
    |   |   |-- lstm
    |   |   `-- tgcn
    |   `-- MC2
    |       |-- lstm
    |       `-- tgcn
    |-- analysis.py                         <-- Functionality used to perform analysis on results.
    |-- data                                <-- GIS data, inp files, pickle files, pressure time series, evaluation results, trained model files.
    |   |-- BKTown
    |   |   |-- SWG_BK_customerpoints.CSV
    |   |   |-- SWG_BK_fixedheads.CSV
    |   |   |-- SWG_BK_hydrants.CSV
    |   |   |-- SWG_BK_meters.CSV
    |   |   |-- SWG_BK_nodes.CSV
    |   |   |-- SWG_BK_nonreturnvalves.CSV
    |   |   |-- SWG_BK_pipes.CSV
    |   |   |-- SWG_BK_transfernode.CSV
    |   |   `-- SWG_BK_valves.CSV
    |   |-- inp
    |   |   |-- 20210212-BK_01Jul2020_31Aug2020_v1.0.inp
    |   |   |-- 20210511-BK_8Dec2020_17Dec2020.inp
    |   |   `-- 20210806-BK_1Jul2021_15Jul2021.inp
    |   |-- resources
    |   |   |-- 13_14_july_events
    |   |   |   `-- distances.pkl
    |   |   |-- MC1
    |   |   |   |-- distances.pkl
    |   |   |   `-- simulation_data.pkl
    |   |   |-- MC2
    |   |   |   `-- distances.pkl
    |   |   |-- checkpoints
    |   |   |-- coordinates.pkl
    |   |   |-- elevations.pkl
    |   |   |-- leak_partitions.pkl
    |   |   |-- nx_topology_d.pkl
    |   |   |-- nx_topology_ud.pkl
    |   |   |-- pipe_header.pkl
    |   |   |-- pipe_lengths.pkl
    |   |   |-- pressures
    |   |   |   |-- dump_pressure_data.json.zip
    |   |   |   |-- mobile_pressure_data-08072020_17112020-UTCp01h00.csv.zip
    |   |   |   `-- mobile_pressure_data-08122020_15022021-UTCp01h00.csv.zip
    |   |   |-- reverse_translation.pkl
    |   |   |-- translation.pkl
    |   |   `-- wn.pkl
    |   |-- results
    |   |   |-- 13_14_july_events
    |   |   |-- MC1
    |   |   `-- MC2
    |   `-- trained_models
    |-- data_prep.py                            <-- Functionality used to prepare data for training and evaluation.
    |-- helper_functions.py                     <-- Utilities
    |-- models                                  <-- Everything pertaining to machine learning models.
    |   |-- ae_lstm.py
    |   |-- ae_model.py
    |   |-- ae_tgcn.py
    |   `-- gcn_lstm.py
    |-- partition.py                            <-- Functionality used to partition the GIS.
    |-- postprocessing.py                       <-- Functionality used to postprocess results.
    |-- requirements.txt                        
    |-- static.py                               <-- Static variables
    |-- train_and_test.py                       <-- Top-Level functionality to run all tests.
    |-- utils.py                                <-- Utility functions pertaining to pickling.
    `-- wntr_helper_functions.py                <-- Utilities pertaining to wntr simulations.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
