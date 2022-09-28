import csv
import itertools
import shutil

from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import postprocessing as postprocessing
from data_prep import *
from models.ae_model import CustomAutoEncoder
from static import *
from utils import *

### INFO ###
# Belgium operates in two timezones:
# - Brussels daylight saving time, so UTC + 2
# - Brussels standard time, so UTC + 1 (from 31 october onward)

leaks = {
    # in UTC + 2
    "MC1": [("2020-07-22 14:00:00", "2020-07-23 10:00:00", "pseudo-leak"),
            ("2020-08-05 09:35:00", "2020-08-05 09:55:00", "775125"),
            ("2020-08-05 10:12:00", "2020-08-05 10:32:00", "773051"),
            ("2020-08-05 11:06:00", "2020-08-05 11:28:00", "780176"),
            ("2020-08-05 13:00:00", "2020-08-05 13:20:00", "936678"),
            ("2020-08-05 13:39:00", "2020-08-05 13:59:00", "2288126"),
            ("2020-08-05 14:15:00", "2020-08-05 14:35:00", "2288309"),
            ("2020-08-05 14:50:00", "2020-08-05 15:10:00", "1164867"),
            ("2020-08-05 15:28:00", "2020-08-05 15:48:00", "1135282")],
    # in UTC + 1
    "MC2": [("2020-12-16 09:05:00", "2020-12-16 09:45:00", "1239007"),
            ("2020-12-16 10:00:00", "2020-12-16 10:40:00", "1240389"),
            ("2020-12-16 10:50:00", "2020-12-16 11:30:00", "1657869"),
            ("2020-12-16 11:40:00", "2020-12-16 12:20:00", "402336"),
            ("2020-12-17 07:40:00", "2020-12-17 08:20:00", "1260152"),
            ("2020-12-17 08:30:00", "2020-12-17 09:10:00", "2288323"),
            ("2020-12-17 10:15:00", "2020-12-17 10:55:00", "401772"),
            ("2020-12-17 11:05:00", "2020-12-17 11:45:00", "1811296"),
            ("2020-12-17 11:45:00", "2020-12-17 12:35:00", "2018703")],
    # in UTC:
    "13_14_july_events": [("2021-07-13 06:15:00", "2021-07-13 13:20:00", "393488"),
                          ("2021-07-14 06:17:00", "2021-07-14 13:43:00", "923252")]}


def select_slice_of_data(timestamps, data,
                         beginning=None, end=None,
                         min_size=None, return_new_range=False):
    """
    Grab a slice of data.
    :param timestamps:
    :param data:
    :param beginning:
    :param end:
    :param min_size:
    :param return_new_range:
    :return:
    """
    print("Trying to grab slice from", beginning, "to", end, "...")

    def add_seconds(start, seconds):
        start_date = datetime.strptime(start, ts_format)
        timestamp = (start_date.timestamp()) + seconds
        return datetime.fromtimestamp(timestamp).strftime(ts_format)

    def get_diff(d1, d2):
        gap = datetime.strptime(d1, ts_format) - \
              datetime.strptime(d2, ts_format)
        return gap.total_seconds() // (60 * timestep_in_mins)

    # initially always try to shrink (stay within original interval)
    left_increment, right_increment = 60, -60

    biterations = 0  # make sure we don't get trapped in an endless loop
    if beginning is not None:
        # increment by one minute until we have a match
        while beginning not in timestamps and biterations < 10000:
            beginning = add_seconds(beginning, left_increment)
            biterations += 1
    else:
        beginning = timestamps[0]
    eiterations = 0  # make sure we don't get trapped in an endless loop
    if end is not None:
        # decrement by one minute until we have a match
        while end not in timestamps and eiterations < 10000:
            end = add_seconds(end, right_increment)
            eiterations += 1
    else:
        end = timestamps[-1]

    if min_size is not None and get_diff(end, beginning) <= min_size:
        b = beginning
        e = end

        # widen the interval by going further back
        while get_diff(e, b) <= min_size:
            b = add_seconds(b, -(60 * timestep_in_mins))
        beginning, end = b, e

    bidx = list(timestamps).index(beginning)
    eidx = list(timestamps).index(end)

    print("Selecting period from", timestamps[bidx], "to", timestamps[eidx])
    if return_new_range:
        return data[bidx:eidx], bidx, eidx
    return data[bidx:eidx]


def visualize_slice(name, timestamps, data, beginning=None, end=None, sim=None, rec=None):
    """
    Visualize a slice of data.
    :param name:
    :param timestamps:
    :param data:
    :param beginning:
    :param end:
    :param sim:
    :param rec:
    :return:
    """
    # grab relevant data slice
    data_slice = select_slice_of_data(timestamps, data, beginning, end)
    columns = ['reference']
    if sim is not None:
        sim_slice = select_slice_of_data(timestamps, sim, beginning, end)
        data_slice = np.vstack((data_slice, sim_slice))
        data = np.vstack((data, sim))
        columns.append('simulation')
    if rec is not None:
        rec_slice = select_slice_of_data(timestamps, rec, beginning, end)
        data_slice = np.vstack((data_slice, rec_slice))
        data = np.vstack((data, rec))
        columns.append('reconstruction')
    if len(columns) > 1:
        data_slice = data_slice.T
        data = data.T

    if beginning is None:
        beginning = timestamps[0]
    if end is None:
        end = timestamps[-1]

    # plot data slice in micro format
    def plot_micro(dslice, beginning, end):
        print(beginning, end)
        # ts = list(timestamps)
        df = pd.DataFrame(dslice, columns=columns)
        df.plot(legend=True)
        pyplot.xlabel('Time step [min]')
        pyplot.ylabel('Head [m]')
        pyplot.axvline(x=5)
        pyplot.axvline(x=len(dslice) - 1)
        pyplot.ylim(top=max(dslice[:, 0]) + 5, bottom=min(dslice[:, 0]) - 5)
        pyplot.xticks([5, len(dslice) - 1], [beginning, end])
        try:
            os.makedirs(os.path.join(PLOTS, "/".join(name.split("/")[:-1])))
        except OSError:
            pass
        print("saving fig... " + os.path.join(PLOTS, name + "_micro.png"))
        pyplot.savefig(os.path.join(PLOTS, name + "_micro.png"), dpi=300)
        pyplot.close()

    if len(data_slice.shape) < 2:
        data_slice = np.expand_dims(data_slice, axis=1)
    if len(data.shape) < 2:
        data = np.expand_dims(data, axis=1)

    plot_micro(data_slice, beginning, end)


def get_param_combos(params, selection=None):
    """
    Get all relevant parameter combinations.
    :param params:
    :param selection:
    :return:
    """
    param_lists = list()
    if selection is None:
        selection = params.keys()
    actual_params = [param for param in params if param in selection]
    for param in actual_params:
        param_lists.append(params[param])
    param_combos = list(itertools.product(*param_lists))
    param_combo_dicts = list()
    for i, combo in enumerate(param_combos):
        assert len(combo) == len(actual_params)
        combo_dict = {param: combo[j] for j, param in enumerate(actual_params)}
        param_combo_dicts.append(combo_dict)
    return param_combo_dicts


def train_and_validate(name="MC1", train_with_sim=True):
    """
    Train and validate various models such that
    the best model is returned with the best settings.
    For AEs, return the model with the lowest
    overall reconstruction error on the validation set.
    :param train_with_sim:
    :param name:
    :return:
    """
    header, data_ts, data, sim_ts, sim = get_raw_data(name,
                                                      duration_in_days=measurement_durations[name],
                                                      start=start_times[name])

    # select training data, excluding last timestamp
    if len(leaks[name]) > 0:
        original_train_data = select_slice_of_data(sim_ts, sim, end=leaks[name][0][0])[:-2]
        if not train_with_sim:
            original_train_data = select_slice_of_data(data_ts, data, end=leaks[name][0][0])[:-2]
            # original_train_data = select_slice_of_data(data_ts, data, beginning="2021-07-06 00:00:00", end=leaks[name][0][0])[:-2]
            # original_train_data = select_slice_of_data(data_ts, data, beginning="2021-07-01 00:00:00", end="2021-07-06 00:00:00")[:-2]
    else:
        original_train_data = data

    # all different architecture settings, including both AE-RNN and AE-TGCN
    tests = [
        ('tgcn', {'weighted': 0, 'augmented': False, 'use_in_tgcn': False, 'use_out_tgcn': False, 'gru': True}),
        ('tgcn', {'weighted': 0, 'augmented': True, 'use_in_tgcn': False, 'use_out_tgcn': False, 'gru': True}),
        ('tgcn', {'weighted': 0, 'augmented': False, 'use_in_tgcn': True, 'use_out_tgcn': True, 'gru': True}),
        ('tgcn', {'weighted': 1, 'augmented': False, 'use_in_tgcn': True, 'use_out_tgcn': True, 'gru': True}),
        ('tgcn', {'weighted': 2, 'augmented': False, 'use_in_tgcn': True, 'use_out_tgcn': True, 'gru': True}),
        ('tgcn', {'weighted': 0, 'augmented': False, 'use_in_tgcn': False, 'use_out_tgcn': False, 'gru': False}),
        ('tgcn', {'weighted': 0, 'augmented': True, 'use_in_tgcn': False, 'use_out_tgcn': False, 'gru': False}),
        ('tgcn', {'weighted': 0, 'augmented': False, 'use_in_tgcn': True, 'use_out_tgcn': True, 'gru': False}),
        ('tgcn', {'weighted': 1, 'augmented': False, 'use_in_tgcn': True, 'use_out_tgcn': True, 'gru': False}),
        ('tgcn', {'weighted': 2, 'augmented': False, 'use_in_tgcn': True, 'use_out_tgcn': True, 'gru': False})
        ]

    ranges = {'epochs': [100],
              'dropout_rate': [0.1],
              'l2_regularizer': [0.001, 0.01, 0.1],
              'timesteps': [2, 3, 4, 5],
              'layers': [2, 3],
              'outer_dim': [64, 128],
              'sensor_nodes': [list(header)]}

    selections = {'std': ['epochs', 'dropout_rate', 'l2_regularizer'],
                  'lstm': ['epochs', 'dropout_rate', 'l2_regularizer', 'timesteps'],
                  'tgcn': ['epochs', 'dropout_rate', 'l2_regularizer', 'timesteps', 'layers', 'outer_dim', 'sensor_nodes']}

    with open(os.path.join(DATA_ROOT, RESULTS, name, "performances.csv"), 'w') as csvfile:
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, test_idx = list(tscv.split(original_train_data))[-1]
        actual_train_data, test_data = original_train_data[train_idx], original_train_data[test_idx]
        print("Testing with split",
              len(train_idx) / len(original_train_data), "-",
              len(test_idx) / len(original_train_data))

        test_errors = {}
        for kind, setup in tests:
            param_combos = get_param_combos(ranges, selections[kind])
            performances = {}
            for combo in param_combos:
                params = {**combo, **setup}
                print("Training and validating model", kind, "with params", params, "...")
                print("Performances so far...")
                for setting in performances:
                    print("For setting", setting, "we got", performances[setting])
                ae_model = CustomAutoEncoder(kind=kind, params=params)
                tscv = TimeSeriesSplit(n_splits=3)
                # do grid search using three-split cross validation
                rec_errors = list()
                for i, (train_idx, val_idx) in enumerate(tscv.split(actual_train_data)):
                    print("Validating with split",
                          len(train_idx) / len(actual_train_data), "-",
                          len(val_idx) / len(actual_train_data))
                    train_data, val_data = actual_train_data[train_idx], actual_train_data[val_idx]
                    _, training_rec = ae_model.fit(train_data)
                    # load best settings
                    ae_model = CustomAutoEncoder(kind=kind, params=params, load=True)
                    _, val_rec = ae_model.predict(val_data)
                    rec_errors.append(mean_squared_error(val_data, val_rec))
                    print("Validation MSE:", mean_squared_error(val_data, val_rec))
                avg_error = sum(rec_errors) / len(rec_errors)
                performances[(kind, json.dumps(params))] = avg_error

            best_settings = min(performances.keys(), key=performances.__getitem__)
            print("Best settings found for", best_settings, "with avg valid rec error", min(performances.values()))

            kind, params = best_settings[0], json.loads(best_settings[1])

            # retrain on actual pressures data
            final_ae_model = CustomAutoEncoder(kind=kind, params=params)
            final_ae_model.fit(actual_train_data)
            # load best settings
            final_ae_model = CustomAutoEncoder(kind=kind, params=params, load=True)
            _, test_rec = final_ae_model.predict(test_data)
            test_error = mean_squared_error(test_data, test_rec)
            test_errors[(kind, json.dumps(params))] = test_error

        best_settings = min(test_errors.keys(), key=test_errors.__getitem__)
        print("Best settings found for", best_settings, "with avg test rec error", min(test_errors.values()))

        kind, params = best_settings[0], json.loads(best_settings[1])

        # now retrain on everything
        final_ae_model = CustomAutoEncoder(kind=kind, params=params)
        final_ae_model.fit(original_train_data)
        final_ae_model = CustomAutoEncoder(kind=kind, params=params, load=True)

        wr = csv.writer(csvfile)
        wr.writerow(["kind", "settings", "error"])
        for setting in test_errors:
            wr.writerow([setting[0], json.loads(setting[1]), test_errors[setting]])
        wr.writerow(["best option - " + best_settings[0], best_settings[1],
                     test_errors[(best_settings[0], best_settings[1])]])

    # save everything for future reference
    dump_to_pickle(best_settings, os.path.join(DATA_ROOT, MODELS, name, "best_settings.pkl"))
    shutil.copy(os.path.join(DATA_ROOT, RESOURCES, "checkpoints", "weights-improvement-ae.hdf5"),
                os.path.join(DATA_ROOT, MODELS, name, "weights-improvement-ae.hdf5"))
    shutil.copy(os.path.join(DATA_ROOT, RESOURCES, "checkpoints", "internals.pkl"),
                os.path.join(DATA_ROOT, MODELS, name, "internals.pkl"))
    return final_ae_model, (header, data_ts, data, sim_ts, sim)


def predict(name="MC1", load=False):
    """
    Predict leaks.
    :param name:
    :param load:
    :return:
    """

    if not load:
        print("Start new training procedure...")
        ae_model, data_bundle = train_and_validate(name, train_with_sim=False)
    else:
        print("Load previously trained model...")
        shutil.copy(os.path.join(DATA_ROOT, MODELS, name, "weights-improvement-ae.hdf5"),
                    os.path.join(DATA_ROOT, RESOURCES, "checkpoints", "weights-improvement-ae.hdf5"))
        shutil.copy(os.path.join(DATA_ROOT, MODELS, name, "internals.pkl"),
                    os.path.join(DATA_ROOT, RESOURCES, "checkpoints", "internals.pkl"))
        best_settings = load_from_pickle(os.path.join(DATA_ROOT, MODELS, name, "best_settings.pkl"))
        kind, params = best_settings[0], json.loads(best_settings[1])

        ae_model = CustomAutoEncoder(kind=kind, params=params, load=True)
        print("Get accompanying data...")
        data_bundle = tuple(get_raw_data(name,
                                         duration_in_days=measurement_durations[name],
                                         start=start_times[name]))

    header, data_ts, data, sim_ts, sim = data_bundle

    # get start of leaky period
    _, leaky_period_incept, leaky_period_termin = select_slice_of_data(data_ts, data,
                                                                       leaks[name][0][0], leaks[name][-1][1],
                                                                       return_new_range=True)
    print("Total leaky period:", data_ts[leaky_period_incept], "to", data_ts[leaky_period_termin], "\n\n")
    # predict location for all leaks
    for leak in leaks[name]:
        print("Predict leak", leak[2], "from", leak[0], "to", leak[1])
        # make sure the minimum size of a leak period
        # is never smaller than the time window used by the model
        min_size = None
        if ae_model.kind != "std":
            min_size = ae_model.params["timesteps"]
        p, pb, pe = select_slice_of_data(data_ts, data, leak[0], leak[1], min_size=min_size, return_new_range=True)

        # visualize prediction per sensor
        # _, total_rec = ae_model.predict(data)
        # for i, sensor in enumerate(header):
        #     visualize_slice(os.path.join(name, leak[2], "slice_" + sensor),
        #                     data_ts[leaky_period_incept - 5:leaky_period_termin + 1],
        #                     data[leaky_period_incept - 5:leaky_period_termin + 1, i],
        #                     data_ts[pb - 5], data_ts[pe],
        #                     sim[leaky_period_incept - 5:leaky_period_termin + 1, i],
        #                     total_rec[leaky_period_incept - 5:leaky_period_termin + 1, i])

        try:
            # do leak prediction
            _, reconstruction = ae_model.predict(p)

            # get rec errors and cumulative rec errors
            rec_errors = abs(reconstruction - p)
            rec_errors_signed = reconstruction - p
            cumul_rec_errors = [[sum(rec_errors[:j + 1, i]) for i, val in enumerate(error)] for j, error in
                                enumerate(rec_errors)]
            cumul_rec_errors_signed = [[sum(rec_errors_signed[:j + 1, i]) for i, val in enumerate(error)] for j, error in
                                       enumerate(rec_errors_signed)][-1]

            # get errors based on rec and cumulative rec
            errors = [(header[ref], mean_squared_error(p[:, ref], reconstruction[:, rec])) for (ref, rec)
                      in zip(range(p.shape[-1]), range(reconstruction.shape[-1]))]
            errors_alt = [(e[0], val) for (i, (e, val)) in enumerate(zip(errors, cumul_rec_errors[-1][:]))
                          if cumul_rec_errors_signed[i] > 0]
            errors.sort(key=lambda a: a[1], reverse=True)
            errors_alt.sort(key=lambda a: a[1], reverse=True)

            print("Errors found:", errors, errors_alt)

            print("Distributing errors...")
            distances = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, name, "distances.pkl"))
            error_map = postprocessing.distribute_with_dist(errors_alt, distances)
            outputs = postprocessing.output(name, leak[2], error_map, to_file=True)
        except Exception:
            print("Catastrophic failure!")


# run all measurement campaigns
predict("MC1", load=False)
predict("MC2", load=False)
predict("13_14_july_events", load=False)
