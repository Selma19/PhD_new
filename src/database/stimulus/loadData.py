"""In this file you will find functions to load the data for the stimulus and the 
joystick response stored in Data/Solo.
"""

from typing import List
import os
import numpy as np

__all__ = [
    "load_signal",
    "load_Joystick",
    "load_list_block_data"
]

# location of the parent of Data/
rootPath = __file__
for _ in range(3):
    rootPath = os.path.dirname(rootPath)

def prefix_path(agent: str):
    """The path to the directory containing the joystick and dot data for `agent`."""
    return os.path.join(
        rootPath,
        'Data', 'Solo', agent, 'Data', 'Formated_Data'
    )

def prefix_path_h5(agent: str):
    """The path to the directory containing the HDF5 files for `agent`."""
    return os.path.join(
        rootPath,
        'Data', 'Solo', 'hdf5', agent
    )

def load_signal(block: str, coh: float, agent: str):
    """Loads the average direction of the dots in complex form,
    and same angle orientation and reference as the nominal and joystick directions.
    """
    return np.loadtxt(
        os.path.join(prefix_path(agent), 'Signal', str( int(coh * 1000) ), block)
    ).view(complex)

def load_joystick(block: str, coh: float, agent: str):
    """Loads the joystick direction in complex form,
    and same angle orientation and reference as the nominal and joystick directions.
    """
    joystick = np.loadtxt(prefix_path(agent) + '/Joystick/' + str(int(coh*1000)) + '/' + block)
    time = joystick[:, 0]
    direction = joystick[:, 1]
    eccentricity = joystick[:, 2]
    return time, direction, eccentricity

def get_list_block(coh: float, agent: str):
    """Returns the list of block names for a given coherence and solo agent.
    
    A given block corresponds to a given coherence value within a gaming session.
    There are usually several blocks from the same session with the same coherence.
    A given block contains data that are contiguous in time so a block should not be split.
    """
    list_block = os.listdir(
        os.path.join(prefix_path(agent), 'Joystick', str(int(coh * 1000)))
    )
    if '.DS_Store' in list_block:
        list_block.remove('.DS_Store')
    return list_block

def load_block_data(block: str, coh: float, agent: str):
    """Returns the joystick and dot directions in complex forms
    for a given block of a gaming session.
    """
    dot = load_signal(block, coh, agent)
    joystick = load_joystick(block, coh, agent)
    joystick = joystick[2] * np.exp(1j * joystick[1] * np.pi / 180)
    return joystick, dot

def load_list_block_data(list_block: List[str], coh: float, agent: str):
    """Returns the joystick and dot directions for a list of block names."""
    joystick_list = []; dot_list = []
    for block in list_block:
        joy, dot = load_block_data(block, coh, agent)
        joystick_list.append(joy)
        dot_list.append(dot)
    return joystick_list, dot_list
