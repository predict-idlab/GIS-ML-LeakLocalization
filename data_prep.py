from datetime import datetime

from helper_functions import *
from postprocessing import get_distances
from static import *
from utils import *

inp_files = {
    'MC1': '20210212-BK_01Jul2020_31Aug2020_v1.0.inp',
    'MC2': '20210511-BK_8Dec2020_17Dec2020.inp',
    '13_14_july_events': "20210806-BK_1Jul2021_15Jul2021.inp",
}

mobile_pressure_data = {
    'MC1': 'mobile_pressure_data-08072020_17112020-UTCp01h00.csv',
    'MC2': 'mobile_pressure_data-08122020_15022021-UTCp01h00.csv',
    '13_14_july_events': None,
}

desirables = {
    'MC1': ['1138102', '1151599', '1160964', '1171286', '1249368', '1251053', '1256995', '1257142', '1260304',
            '1267265', '1983396', '2018747', '2179206', '2288344', '2427371', '2901330', '2902689', '3085780', '366014', '367398',
            '402436', '772093', '775700', '776019', '776884', '780007', '923397', '938607'],
    'MC2': ['1130816', '1239222', '1249368', '1249408', '1250189', '1256995', '1257639', '1258609', '1260304',
            '1261324', '1267649', '1269161', '1374298', '1629354', '1630543', '1658112', '1773936', '1987738', '2288344'],
    '13_14_july_events': ['1267265', '2902689', '932942', '1620451', '2179206', '1138102', '402436', '367398', '938903',
                          '776884', '2018747', '775700', '2427371', '1135866', '1249368', '1983396', '3085780', '1151599',
                          '1618407', '923397', '1160964', '938205']
}

undesirables = {
    'MC1': [0, 4, 12, 13, 14, 15, 16, 21, 30, 36, 37],
    'MC2': [2, 5, 7, 9, 12, 17, 19, 20, 21, 22, 23, 26, 30, 31, 33, 34, 35, 36],
    '13_14_july_events': [],
}

total_loggers = {
    'MC1': ['1135866', '1138102', '1151599', '1160964', '1161577', '1171286', '1249368', '1251053', '1256995',
            '1257142', '1260304', '1267265', '1377059', '1618407', '1620451', '1628166', '1629341', '1983396', '2018747',
            '2179206', '2288344', '2426841', '2427371', '2901330', '2902689', '3085780', '366014', '367398', '402436', '772093',
            '773117', '775700', '776019', '776884', '780007', '923397', '932942', '938205', '938607'],
    'MC2': ['1130816', '1239222', '1240717', '1249368', '1249408', '1249633', '1250189', '1251053', '1256995',
            '1257142', '1257639', '1258609', '1259881', '1260304', '1261324', '1267649', '1269161', '1368299', '1374298',
            '1377059', '1422947', '1627949', '1628550', '1628842', '1629354', '1630543', '1631265', '1658112', '1773936',
            '1987738', '2018688', '2018705', '2288344', '2933990', '2940654', '402436', '541993'],
    '13_14_july_events': ['1267265', '2902689', '932942', '1620451', '2179206', '1138102', '402436', '367398', '938903',
                          '776884', '2018747', '775700', '2427371', '1135866', '1249368', '1983396', '3085780', '1151599',
                          '1618407', '923397', '1160964', '938205']
}

sim_durations = {
    # from 01 July, 00:00:00 to 22 July 00:00:00,
    # and from 22 July 00:00:00 to 06 Aug 00:00:00
    'MC1': 21 + 15,
    # from 08 December 00:00:00 to 16 December 00:00:00
    # and from 16 December 00:00:00 to 18 December 00:00:00
    'MC2': 8 + 2,
    # from 1 July, 00:00:00 to 16 July 00:00:00
    '13_14_july_events': 15,
}

measurement_durations = {
    # from 08 July 00:00:00 to 06 Aug 00:00:00
    'MC1': 29,
    # from 08 December 00:00:00 to 18 December 00:00:00
    'MC2': 10,
    # from 1 July, 00:00:00 to 16 July 00:00:00
    '13_14_july_events': 15,
}

start_times = {
    'MC1': (2020, 7, 1, 0, 0, 0, 0),
    'MC2': (2020, 12, 8, 0, 0, 0, 0),
    '13_14_july_events': (2021, 7, 1, 0, 0, 0, 0),
}

shift = {
    # shift + 1, to go from UTC + 1 to UTC + 2
    "MC1": 1,
    # shift 0, to go from UTC + 1 to UTC + 1
    "MC2": 0,
    # shift -1, to go from UTC + 1 to UTC
    "13_14_july_events": -1
}


def get_simulation(name="MC1", duration_in_days=sim_durations["MC1"], start=start_times["MC1"]):
    """
    Get necessary inp file and get reference simulation data
    for a certain period of time.
    :param start:
    :param analyze:
    :param name:
    :param duration_in_days:
    :return:
    """

    total_duration = 0
    real_start = start

    # if no span of days is given,
    # duration is either 0 or set to the max possible duration of the live sim
    if duration_in_days is not None:
        total_duration = nr_of_seconds_in_one_day * duration_in_days

    if os.path.exists(os.path.join(DATA_ROOT, RESOURCES, name, "simulation_data.pkl")):
        index, header, data, elevations = load_from_pickle(
            os.path.join(DATA_ROOT, RESOURCES, name, "simulation_data.pkl"))
    else:
        # run simulation...
        print("Running simulation for inp file", inp_files[name])
        inp_file = os.path.join(DATA_ROOT, INP, inp_files[name])
        wn = wntr.network.WaterNetworkModel(inp_file_name=inp_file)
        wn.options.time.duration = int(total_duration)
        wn.options.time.hydraulic_timestep = timestep_in_mins * 60
        wn.options.time.report_timestep = wn.options.time.hydraulic_timestep
        wn.options.time.pattern_timestep = wn.options.time.hydraulic_timestep
        wn.options.time.rule_timestep = wn.options.time.hydraulic_timestep
        wn.options.hydraulic.damplimit = 0.5
        wn.options.hydraulic.accuracy = 0.02
        sim = wntr.sim.EpanetSimulator(wn)
        WNTR_result = sim.run_sim()
        print("... Done.")

        # get results
        head_res = WNTR_result.node['head']
        head_res.memory_usage(deep=True).sum() / (10 ** 6)

        # get components
        index = head_res.index.to_numpy()
        header = np.array(head_res.columns.values)
        data = head_res.to_numpy()

        # get elevations for each sensor
        elevations = dict()
        for i, s in enumerate(total_loggers[name]):
            elevation = wn.get_node(s).elevation
            elevations[s] = elevation

        coordinates = dict()
        for n in wn.nodes:
            coordinates[n] = wn.get_node(n).coordinates

        dump_to_pickle(coordinates,
                       os.path.join(DATA_ROOT, RESOURCES, "coordinates.pkl"))

        dump_to_pickle((index, header, data, elevations),
                       os.path.join(DATA_ROOT, RESOURCES, name, "simulation_data.pkl"))

    # prep data...

    print("Start prepping data...")

    def convert_seconds(st, seconds):
        start_date = datetime(st[0], st[1], st[2],
                              hour=st[3], minute=st[4], second=st[5],
                              microsecond=st[6])
        timestamp = (start_date.timestamp()) + seconds
        return datetime.fromtimestamp(timestamp).strftime(ts_format)

    timestamps = np.array([convert_seconds(real_start, ts) for ts in index])
    selected_idx = [i for (i, s) in enumerate(header) if s in total_loggers[name]]
    selected_names = [s for (i, s) in enumerate(header) if s in total_loggers[name]]
    original_positions = [total_loggers[name].index(s) for s in selected_names]

    permutation = list(range(len(original_positions)))
    for i in range(len(permutation)):
        permutation[i] = original_positions.index(i)

    print("Selection info:", original_positions, permutation)

    # perform selection
    selected_data = data[:, selected_idx]
    selected_data[:] = selected_data[:, permutation]

    # get rid of undesirable sensors
    to_keep = [i for (i, s) in enumerate(total_loggers[name]) if i not in undesirables[name]]
    final_data = selected_data[:, to_keep]

    # look for starting point
    start_at = convert_seconds(start, 0)
    if start_at in timestamps:
        start_idx = np.where(timestamps == start_at)[0].item()
        end_idx = int(total_duration * (1 / 60) * (1 / timestep_in_mins)) + 1
        timestamps = timestamps[start_idx:start_idx + end_idx]
        final_data = final_data[start_idx:start_idx + end_idx]

    print("... Done.")

    return timestamps, final_data, elevations, permutation


def get_raw_data(name="MC1",
                 duration_in_days=measurement_durations["MC1"], start=start_times["MC1"],
                 interpolate=False):
    """
    Get raw pressure data for a certain period of time.
    :param interpolate:
    :param start:
    :param analyze:
    :param name:
    :param duration_in_days:
    :return:
    """

    if name in ['13_14_july_events']:
        # if clean data dump gets no input, then it will use the data dump
        header, timestamps, pressures_df = clean_data_dump(None, header=total_loggers[name])

        timestamps = pd.DatetimeIndex(timestamps)

        # shift to UTC
        timestamps = timestamps.shift(shift[name], freq='H')
        timestamps = pd.Series(timestamps.format()).to_numpy()

        # set additional info
        desirables[name] = header.tolist()
        total_loggers[name] = header.tolist()
        co_header = np.array([s for (i, s) in enumerate(header) if i in undesirables[name]])
    else:
        # define pressures file
        pressures_file = os.path.join(DATA_ROOT, RESOURCES, "pressures", mobile_pressure_data[name])

        # read pressures and timestamps
        pressures_df = pd.read_csv(pressures_file)
        timestamps = pd.DatetimeIndex(pressures_df['datetime'])
        # shift to UTC
        timestamps = timestamps.shift(shift[name], freq='H')
        timestamps = pd.Series(timestamps.format()).to_numpy()

        # get headers
        header = np.array(list(pressures_df.columns)[1:len(total_loggers[name]) + 1])
        co_header = np.array([s for (i, s) in enumerate(header) if i in undesirables[name]])

        # reindex and drop
        pressures_df = pressures_df.reindex(columns=header)
        pressures_df = pressures_df.drop(columns=co_header)
        pressures_df = pressures_df.to_numpy()

    get_distances(name, header)

    # do simulation
    simulation_ts, simulation_data, elevations, permutation = get_simulation(name, sim_durations[name], start)

    # check sampling frequency and adjust if necessary
    print("Start fixing and converting raw data...")

    gap = datetime.strptime(str(timestamps[1]), ts_format) - \
          datetime.strptime(str(timestamps[0]), ts_format)
    if gap.total_seconds() == 60:
        # find starting point that within the simulation
        first_idx = 0
        while timestamps[first_idx] not in simulation_ts:
            first_idx += 1
        timestamps = timestamps[first_idx::timestep_in_mins]
        pressures_df = pressures_df[first_idx::timestep_in_mins]

    # get maximal duration of experiment
    max_duration_in_n_mins = datetime.strptime(str(timestamps[-1]), ts_format) - \
                             datetime.strptime(str(timestamps[0]), ts_format)
    max_duration_in_n_mins = int(max_duration_in_n_mins.total_seconds() * (1 / timestep_in_mins) * (1 / 60)) + 1
    if duration_in_days is not None:
        max_duration_in_n_mins = int(24 * 60 * (1 / timestep_in_mins) * duration_in_days) + 1

    # start at beginning of simulation, at the earliest
    start_ts = 0
    if simulation_ts[0] in timestamps:
        start_ts = max(start_ts, np.where(timestamps == simulation_ts[0])[0].item())
    # stop at end of simulation, at the latest
    final_ts = start_ts + max_duration_in_n_mins
    if simulation_ts[-1] in timestamps:
        final_ts = min(final_ts, np.where(timestamps == simulation_ts[-1])[0].item())

    # bound sampling period
    timestamps = timestamps[start_ts:final_ts + 1]
    pressures_df = pressures_df[start_ts:final_ts + 1]

    # fix broken values and convert pressures to headloss
    pressures_df[pressures_df <= 1] = np.NAN
    pressures_df[pressures_df <= -5] = np.NAN

    header = np.array([s for (i, s) in enumerate(header) if s not in co_header])
    assert all([s in desirables[name] for s in header])
    for i, s in enumerate(header):
        pressures_df[:, i] *= bar_unit_to_mwc_unit_factor
        pressures_df[:, i] += elevations[s]

    if not interpolate:
        for i, row in enumerate(pressures_df[:]):
            for j, col in enumerate(pressures_df[i]):
                replacement_index = np.where(simulation_ts == timestamps[i])[0].item()
                if np.isnan(pressures_df[i, j]):
                    pressures_df[i, j] = simulation_data[replacement_index, j]

    else:
        # interpolate instead of using simulations
        pressures_df = pd.DataFrame(pressures_df)
        start, end = pressures_df.index[0], pressures_df.index[-1]
        pressures_df = pressures_df.interpolate(limit_direction='both').loc[start:end]
        pressures_df = pressures_df.to_numpy()

    print("... Done.")

    return header, timestamps, pressures_df, simulation_ts, simulation_data
