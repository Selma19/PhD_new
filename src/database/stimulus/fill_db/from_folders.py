"""Utility functions to fill the stimulus database from the data contained
in the former folder arborescence.

Note that these functions return json serializable data.
"""

from typing import List, Tuple
import numpy as np
import h5py, os

from .. import loadData as ld

def block_xp_num(block: str):
    """Given a block as the ones returned by `..loadData.get_list_block`,
    returns the experiment name as well as the block number (order of appearance
    in time), which both correspond to columns in the database table.
    """
    couple = block[2: -5].split(', ')
    xp_name = couple[0][:-1]
    block_num = int(couple[1]) // 10
    return xp_name, block_num

def get_time_steps(agent: str, coh: float, block: str) -> List[float]:
    """Returns the ordered list of time steps in ms."""
    return (ld.load_joystick(block, coh, agent)[0] / 1000).tolist()

def get_joystick(agent: str, coh: float, block: str) -> Tuple[List[float], List[float]]:
    """Returns the joystick response for the time steps
    returned by `get_time_steps` for the same arguments.

    The result is a couple (eccentricity, angle) with the angle in radians.
    """
    angle, ecc = ld.load_joystick(block, coh, agent)[1:]
    return [ecc.tolist(), (angle * np.pi / 180).tolist()]

def get_dot(agent: str, coh: float, block: str) -> Tuple[List[float], List[float]]:
    """Returns the mean dot direction for the time steps
    returned by `get_time_steps` for the same arguments.

    The result is a couple (real part, imaginary part).
    """
    complex_signal = ld.load_signal(block, coh, agent)
    return [complex_signal.real.tolist(), complex_signal.imag.tolist()]

def get_nom(agent: str, coh: float, block: str) -> List[Tuple[float, float]]:
    """Returns the 10 nominal directions -and their time of appearance- that alternate
    over the time interval returned by `get_time_steps` for the same arguments.

    The angles are returned in radians and times in ms.
    """
    xp, block_num = block_xp_num(block)
    line = block_num * 10
    root = os.path.join(
        ld.prefix_path(agent), 'Signal', xp
    )
    list_angle = np.loadtxt(
        os.path.join(root, 'list_direction.txt')
    )[line: line + 10, 0]
    list_angle = list_angle * np.pi / 180

    # get the times of change
    list_time = []
    for num in range(line, line + 10):
        list_time.append(
            np.loadtxt(
                os.path.join(root, f'list_times_{num}.txt')
            )[0] / 1000
        )
    return list(zip(list_time, list_angle))

def get_tgt_times(agent: str, coh: float, block: str) -> List[float]:
    """Returns the ordered list of time appearance in ms of the targets."""
    root = ld.prefix_path_h5(agent)
    xp = block_xp_num(block)[0]
    filename = os.path.join(root, xp + 's.h5')

    with h5py.File(filename, 'r') as f:
        feature = 'INFO_TargetCounter'
        tgt_ts, y = f['time'][feature], f['value'][feature]

        # remove the times for which no target has appeared
        # also convert them in ms
        ind = 0
        while y[ind] == 0:
            ind += 1
        tgt_ts = [el / 1000 for el in tgt_ts[ind:]]

        # note that once x has been cleaned that way, y becomes redundant
        # so let us forget about it
        del y

        # extract the time interval covering the block and coherence value
        # (note that the block actually defines the coherence)
        sub_ts = get_time_steps(agent, coh, block)
        t_min = np.min(sub_ts)
        t_max = np.max(sub_ts)

        # restrict to the targets appearing within that interval
        ind_inf = 0
        while tgt_ts[ind_inf] < t_min:
            ind_inf += 1
        ind_sup = len(tgt_ts) - 1
        while tgt_ts[ind_sup] > t_max:
            ind_sup -= 1
        
        res = tgt_ts[ind_inf: ind_sup + 1]
    return res

def get_agents():
    """Returns the list of solo agent names."""
    path = __file__
    for _ in range(4):
        path = os.path.dirname(path)
    path = os.path.join(
        path, "Data", "Solo"
    )
    list_files = os.listdir(path)
    return [name for name in list_files if '.' not in name and len(name) == 3]
