import os

DATA_ROOT = os.path.join(os.path.abspath("."), "data")

INP = "inp"
RESOURCES = "resources"
MODELS = "trained_models"
PLOTS = "plots"
RESULTS = "results"

ts_format = "%Y-%m-%d %H:%M:%S"
bar_unit_to_mwc_unit_factor = 10.1974
nr_of_seconds_in_one_day = 24 * 60 * 60
timestep_in_mins = 5