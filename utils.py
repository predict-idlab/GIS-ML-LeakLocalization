import os
import pickle


def dump_to_pickle(data, file_path):
    """
    Write a binary dump to a pickle file of arbitrary size.
    src: https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    :param data:
    :param file_path:
    :return:
    """
    bytes_out = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_path, 'wb') as f_out:
        # 2**31 - 1 is the max nr. of bytes pickle can dump at a time
        for idx in range(0, len(bytes_out), 2 ** 31 - 1):
            f_out.write(bytes_out[idx:idx + 2 ** 31 - 1])
    return


def load_from_pickle(file_path):
    """
    Read from a pickle file of arbitrary size.
    src: https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    :param file_path:
    :return:
    """
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, 2 ** 31 - 1):
            bytes_in += f_in.read(2 ** 31 - 1)
    return pickle.loads(bytes_in)