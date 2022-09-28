import json
import os

import numpy as np

from static import *
from wntr_helper_functions import *


def make_selection(original, selection, data):
    """
    Return the data with only the correct loggers.
    """
    selected_idx = [i for (i, s) in enumerate(original) if s in selection]
    selected_names = [s for (i, s) in enumerate(original) if s in selection]
    original_positions = [list(selection).index(s) for s in selected_names]

    permutation = list(range(len(original_positions)))
    for i in range(len(permutation)):
        permutation[i] = original_positions.index(i)

    # perform selection
    selected_data = data[:, selected_idx]
    selected_data[:] = selected_data[:, permutation]
    return selected_data


def clean_data_dump(data, header=None):
    # ts_string = "b'Timestamp"
    ts_string = "Timestamp"
    if data is None:
        with open(os.path.join(DATA_ROOT, RESOURCES, "pressures", "dump_pressure_data.json")) as data_dump:
            data = json.loads(data_dump.read())
            ts_string = "Timestamp"
    pressure_data_df = pd.DataFrame(data)
    timestamps = pd.to_datetime(pressure_data_df[ts_string]).dt.strftime(ts_format).sort_values().unique()
    pressure_data_df['Pressure'] = pd.to_numeric(pressure_data_df['Pressure'])
    pressure_data_df = pressure_data_df.pivot_table(index=ts_string,
                                                    columns='brandkraan_meetpunt',
                                                    values='Pressure')
    final_data = pressure_data_df.to_numpy()
    new_header = pressure_data_df.columns.tolist()
    if header is not None:
        final_data = make_selection(new_header, header, final_data)
        new_header = header

    return np.array(new_header), np.array(timestamps), final_data
