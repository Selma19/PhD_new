"""The CPR solo data are organized as follows.
The data relative to an agent abc are stored in CPR_psychophysics/abc/

This directory contains a variable number of .h5 files.
Each file has the generic name:
{yyyy}{mm}{dd}_{abc}_CPRsolo_block{i}_psycho{j}_fxs.h5

where the variables can be interpreted as:
- yyyy is the year when was done the experiments
- mm is the month when was done the experiments
- dd is the day when was done the experiments
- abc is the agent name
- i is a number equal to 1 or 2
(refers to whether the experiment was done before or after a break for the player)
- j is a number equal to 3 or 4 (labeling the room in which the experiment was done)

Note that 'block' here is different from what we call a block in the code
(a sequence of 10 nominal directions consecutive in time).

Note also that sometimes 'block' and 'psycho' are swapped in the file name.

All these files share the same internal structure, which is as follows.

# h5 structure

## time steps
The time steps are expressed in microseconds.

- /time/STIM_RDP_direction is the list of time steps at which a new nominal direction is set
- /time/STIM_RDP_coherence is the list of time steps at which a new coherence is set
- /time/STIM_RDP_dotPositions is the list of time steps at which the dots move
- /time/IO_joystickDirection is the list of time steps at which a new joystick direction is measured
- /time/IO_joystickStrength is the list of time steps at which a new joystick eccentricity is measured
- /time/TRIAL_start is the list of time steps at which a trial begins (3 coherences per trial)
- /time/TRIAL_end is the list of time steps at which a trial ends
 - /time/INFO_TargetCounter is the list of time steps at which a target appears

Since the time steps at which the joystick and dots were measured differ, we chose to:
1. keep the dot time steps as reference
2. infer the value of the joystick direction and eccentricity at those times, using linear interpolation

## values
The angles are expressed in degrees (range from 0 to 360).

- /value/STIM_RDP_direction is the list of nominal directions
- /value/STIM_RDP_coherence is the list coherences
- /value/STIM_RDP_dotPositions is the list of dot positions, formatted as a list of lists:
[frame1 --> [dot1_x, dot1_y, dot2_x, ...], frame2 --> [dot1_x, dot1_y, dot2_x, ...], ...]
- /value/IO_joystickDirection is the list of joystick directions in degrees
- /value/IO_joystickStrength is the list of joystick eccentricities

## blocks vs trials
We define a block as a sequence of 10 nominal directions consecutive in time at fixed coherence.
In the h5 files, blocks are not tracked explicitly, but trials are.
There are 3 blocks per trial, each with a different coherence, except for the first or last one,
for which there is less than one block (such data should be discarded).

We also noticed that each start of a trial comes with 2 simultaneous changes in coherence and nominal direction,
separated by less than 0.1 second.
These dummy data should be discarded.
This puts the starting time of the first block in a trial usually at the second change in
coherence / nominal direction in this trial.

## notes
For a very few isolated frames (occurs in agent 'rit'),
we noticed that the dots did not move from one time step to the next.
This time step does not coincide with the start or end of a block.

For many agents, it appear that a few isolated time steps, that do not coincide with the start
or end of a block, the dot positions stored in h5['value']['STIM_RDP_dotPositions']
are a NaN instead of a list of length 1006.

We handled both issues by removing the corresponding time steps.
"""

from typing import List
import numpy as np
import os, h5py, json
import matplotlib.pyplot as plt

def get_agents():
    """Returns the list of solo agent names."""
    path = __file__
    for _ in range(5):
        path = os.path.dirname(path)
    path = os.path.join(
        path, "CPR_psychophysics"
    )
    list_files = os.listdir(path)
    return [name for name in list_files if '.' not in name and len(name) == 3]

def get_time_ind(
    t: int,
    ind: int,
    data: List[int]
):
    while data[ind] < t:
        ind += 1
    return ind

def sorted_find(t: int, joyTimes: List[int]):
    """Returns the largest `k` such that
    `joyTimes[k] <= t < joyTimes[k + 1]`.

    Notes
    -----
    It assumes that `joyTimes` is sorted by increasing order, enabling
    to use dichotomy.
    It also assumes that
    `joyTimes[0] <= t < joyTimes[-1]`.
    """
    k_left = 0
    k_right = len(joyTimes) - 1
    while k_right - k_left > 1:
        k = (k_left + k_right) // 2
        if joyTimes[k] < t:
            k_left = k
        else:
            k_right = k
    if joyTimes[k_right] == t:
        return k_right
    return k_left

def interpolate_angle(t: int, joyTimes: List[int], joyVal: List[float]):
    """Given angle values `joyVal` computed at some times `joyTimes`,
    infers a missing value at a time `t`.

    Parameters
    ----------
    t : int
        the time at which to infer the missing angle
    joyTimes : List[int]
        the times at which the angles are known
    joyVal : List[float]
        the known angles, expressed in degrees and ranging from 0 to 360

    Notes
    -----
    We infer the missing value by using linear interpolation:
    - find the largest `k` such that
    `joyTimes[k] <= t < joyTimes[k + 1]`
    - read the missing value from the segment joining the points
    `(joyTimes[k], joyVal[k])` and `(joyTimes[k + 1], joyVal[k + 1])`
    - take care that we are interpolating angles
    - if `t` is outside of the range of the known times, we use constant interpolation
    """
    if t < joyTimes[0]:
        return joyVal[0]
    elif t >= joyTimes[-1]:
        return joyVal[-1]
    
    k = sorted_find(t, joyTimes)
    t0 = joyTimes[k]
    dt = joyTimes[k + 1] - t0

    if abs(joyVal[k + 1] - joyVal[k]) < 180:
        res = joyVal[k] + (t - t0) * (joyVal[k + 1] - joyVal[k]) / dt
    
    else:
        theta = 360 - joyVal[k + 1]
        res = 360 - joyVal[k] - (t - t0) * (theta - joyVal[k]) / dt
    return res

def interpolate_lin(t: int, joyTimes: List[int], joyVal: List[float]):
    """Given known values `joyVal` computed at some times `joyTimes`,
    infers a missing value at a time `t`.

    Parameters
    ----------
    t : int
        the time at which to infer the missing value
    joyTimes : List[int]
        the times at which the angles are known
    joyVal : List[float]
        the known values

    Notes
    -----
    We infer the missing value using linear interpolation:
    - find the largest `k` such that
    `joyTimes[k] <= t < joyTimes[k + 1]`
    - read the missing value from the segment joining the points
    `(joyTimes[k], joyVal[k])` and `(joyTimes[k + 1], joyVal[k + 1])`
    - if `t` is outside of the range of the known times, we use constant interpolation
    """
    if t < joyTimes[0]:
        return joyVal[0]
    elif t >= joyTimes[-1]:
        return joyVal[-1]
    
    k = sorted_find(t, joyTimes)
    t0 = joyTimes[k]
    dt = joyTimes[k + 1] - t0
    return joyVal[k] + (t - t0) * (joyVal[k + 1] - joyVal[k]) / dt

def extract_all(agent: str):
    path = __file__
    for _ in range(5):
        path = os.path.dirname(path)
    path = os.path.join(
        path, "CPR_psychophysics", agent
    )
    h5_files = [name for name in os.listdir(path) if name[-3:] == '.h5' and '.mwk2' not in name]

    rows = []
    
    for h5_file in h5_files:
        with h5py.File(os.path.join(path, h5_file), "r") as f:
            # split the timeline into trials, meaning:
            # for each trial, get the indices in nom_time and coh_time
            # that belongs to that trial
            trial_starts = f['time']['TRIAL_start']
            trial_ends = f['time']['TRIAL_end']
            trial_bounds = [(start, end) for start, end in zip(trial_starts, trial_ends)]

            # each dict in tab corresponds to one trial
            # this dict is such that dict['nom_dir'] = couple (start index, end index)
            # in nom_times of the times that belong to the trial
            tab = [{'nom_dir': (0, 0), 'coh': (0, 0)} for _ in trial_bounds]

            nom_times = f['time']['STIM_RDP_direction']
            coh_times = f['time']['STIM_RDP_coherence']

            for name, data in zip(['nom_dir', 'coh'], [nom_times, coh_times]):
                iTrial = 0 # running index in trial_bounds
                iData = 0 # running index in data
                # smallest and largest indices in data whose values belong to the current trial:
                # the times in the trial are given by data[startData: endData]
                startData = 0; endData = 0
                while True:
                    # if the data (nom_dir or coh) time is larger than the end of the trial
                    # then we have changed of trial
                    if data[iData] > trial_bounds[iTrial][1]:
                        endData = iData
                        tab[iTrial][name] = (startData, endData)

                        startData = iData
                        iTrial += 1
                    iData += 1

                    if iData == len(data):
                        tab[iTrial][name] = (startData, iData)
                        break

                    elif iTrial == len(trial_bounds):
                        break

            # discard the trials for which less than 3 coherence values are observed
            # (should be 3 blocks per trial)
            indicesToRemove = {
                ind for ind, dic in enumerate(tab) if dic['coh'][1] - dic['coh'][0] < 3
            }
            trial_bounds = [el for ind, el in enumerate(trial_bounds) if ind not in indicesToRemove]
            tab = [el for ind, el in enumerate(tab) if ind not in indicesToRemove]

            # split in blocks:
            # each block is given by the indices of nom_dir belonging to the block,
            # the index of coherence belonging to the block and the final time step (excluded)
            # of the block
            blocks = []
            for iTrial, el in enumerate(tab):
                endCoh = el['coh'][1]
                startNom = el['nom_dir'][0]
                for k in range(3):
                    coh_ind = endCoh - 3 + k
                    if k == 2:
                        end_time = trial_bounds[iTrial][1]
                    else:
                        end_time = coh_times[coh_ind + 1]
                    block = {'nom_dir': [], 'coh': coh_ind, 'end_time': end_time}

                    # there should be 10 nominal directions for this coherence
                    while nom_times[startNom] < coh_times[coh_ind]:
                        startNom += 1

                    # just double check the block is valid
                    if startNom + 9 < len(nom_times):
                        if coh_ind + 1 < len(coh_times):
                            if nom_times[startNom + 9] < coh_times[coh_ind + 1]:
                                block['nom_dir'] = [startNom + k for k in range(10)]
                                blocks.append(block)
                        else:
                            block['nom_dir'] = [startNom + k for k in range(10)]
                            blocks.append(block)

            # now for each block, we can extract:
            # - coh REAL
            # - block INTEGER (appearance order in time)
            # - times TEXT (sampling times of dot and joystick in ms)
            # - nominal_angle TEXT (list of 10 couples (starting time, value)
            # for the nominal direction, angles are in radians)
            # - target_times TEXT (list of appearance times of the targets)
            # - joystick TEXT (couple of lists (eccentricity, orientation))
            # - mean_dot TEXT (couple of lists (real part, imaginary part))

            dot_times = f['time']['STIM_RDP_dotPositions']
            dot_values = f['value']['STIM_RDP_dotPositions']
            nom_values = f['value']['STIM_RDP_direction'].astype('float64')
            coh_values = f['value']['STIM_RDP_coherence'].astype('float64')
            tgt_times = f['time']['INFO_TargetCounter']
            joy_dir_times = f['time']['IO_joystickDirection']
            joy_dir_values = f['value']['IO_joystickDirection'].astype('float64')
            joy_ecc_times = f['time']['IO_joystickStrength']
            joy_ecc_values = f['value']['IO_joystickStrength'].astype('float64')
            
            for num, block in enumerate(blocks):
                # coherence
                coh = coh_values[block['coh']]

                # block number
                block_num = num

                # nominal_angle
                nominal_angle = list(zip(
                    (nom_times[block['nom_dir']].astype('float64') / 1000).tolist(),
                    (nom_values[block['nom_dir']] * np.pi / 180).tolist()
                ))

                # indices in dot times corresponding to the block time interval
                # i.e. dot_times[startDot: endDot] is included in the block time range
                startTime = nom_times[block['nom_dir'][0]]
                endTime = block['end_time']
                startDot = sorted_find(startTime, dot_times)
                endDot = sorted_find(endTime, dot_times)

                # let us extract the mean dot direction
                # it is defined as the average of exp(i theta_dot)
                # exp(i theta_dot) is defined as dx / |dx|
                correctIndices = [ind for ind in range(startDot, endDot) if len(dot_values[ind]) == 1006]
                frames = np.array([dot_values[ind] for ind in correctIndices], dtype=np.float64)
                tab_times = np.array(dot_times[correctIndices], dtype=np.float64) / 1000

                # remove the times t such that the dots did not move btw t and t + 1
                correctIndices = np.nonzero(np.sum(np.abs(frames[1:] - frames[:-1]), axis=1))
                frames = frames[correctIndices].reshape(-1, 503, 2)
                tab_times = tab_times[correctIndices]

                dx = frames[1:, ...] - frames[:-1, ...]
                # identify the points that teleport from one time step to the other:
                # for these points at these times, no moving direction can be defined
                # so they should not contribute to the average
                norm = np.sqrt(dx[:, :, 0] ** 2 + dx[:, :, 1] ** 2)
                dt = tab_times[1:] - tab_times[:-1]
                velocity = norm / np.tile(dt.reshape(-1, 1), (1, 503))
                mask = np.tile(( velocity < 9e-3 ).reshape(-1, 503, 1), (1, 1, 2))

                # compute the average
                mean_dot = np.sum(
                    np.where(mask, dx, np.zeros_like(dx)) /
                    (np.tile(norm.reshape(-1, 503, 1), (1, 1, 2))),
                    axis=1
                ) / np.sum(mask, axis=1)
                mean_dot = [mean_dot[:, 0].tolist(), mean_dot[:, 1].tolist()]

                # take care that the mean dot direction could not be computed for the last time
                # step, and tab_times contains the times at which both the mean dot and
                # joystick data are computed
                tab_times = tab_times[:-1]

                # extract the target times
                startTgt = sorted_find(startTime, tgt_times)
                endTgt = sorted_find(endTime, tgt_times)
                target_times = tgt_times[startTgt: endTgt].astype('float64') / 1000

                # extract the joystick direction and eccentricity
                # for the same time steps as the mean dot direction, i.e.
                # `times`
                startJoyDir = sorted_find(startTime, joy_dir_times)
                endJoyDir = sorted_find(endTime, joy_dir_times)
                joyTimes = joy_dir_times[startJoyDir: endJoyDir].astype('float64') / 1000
                joyVal = joy_dir_values[startJoyDir: endJoyDir]
                joy_angles = [interpolate_angle(t, joyTimes, joyVal) * np.pi / 180 for t in tab_times]

                startJoyEcc = sorted_find(startTime, joy_ecc_times)
                endJoyEcc = sorted_find(endTime, joy_ecc_times)
                joyTimes = joy_ecc_times[startJoyEcc: endJoyEcc].astype('float64') / 1000
                joyVal = joy_ecc_values[startJoyEcc: endJoyEcc]
                joy_ecc = [interpolate_lin(t, joyTimes, joyVal) for t in tab_times]

                # put the joystick data into the desired format
                joystick = [joy_ecc, joy_angles]

                # store the collected data in json format:
                # agent TEXT,
                # coh REAL,
                # xp TEXT,
                # block INTEGER,
                # times TEXT,
                # nominal_angle TEXT,
                # target_times TEXT,
                # joystick TEXT,
                # mean_dot TEXT,
                rows.append(
                    (
                        agent, coh, h5_file[:-3], block_num,
                        json.dumps(tab_times.tolist()), json.dumps(nominal_angle),
                        json.dumps(target_times.tolist()), json.dumps(joystick),
                        json.dumps(mean_dot)
                    )
                )
    return rows

def visu_timeline(path_to_h5: str):
    with h5py.File(path_to_h5, "r") as f:
        nom_times = list(f['time']['STIM_RDP_direction'])
        coh_times = list(f['time']['STIM_RDP_coherence'])
        trial_starts = list(f['time']['TRIAL_start'])
        trial_ends = list(f['time']['TRIAL_end'])

    fig, ax = plt.subplots(1, 1)

    colors = ['blue', 'red', 'green', 'black']
    heights = [1, 2, 3, 3]
    datas = [nom_times, coh_times, trial_starts, trial_ends]
    labels = ['nom_time', 'coh_time', 'trial_start', 'trial_end']
    markers = ['x'] + ['--'] * 3

    for data, height, color, label, marker in zip(datas, heights, colors, labels, markers):
        for t in data:
            ax.plot([t] * 2, [0, height], marker, color=color)
        ax.plot([data[0]] * 2, [0, height], marker, color=color, label=label)
    ax.legend()
    plt.show()
