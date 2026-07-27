"""There are 13 distinct coherences, but from the former files, there
were only 7.

Some coherences occur only for some agents!!!
But the 7 coherences we considered before are exactly the coherences common to all agents.

The error was in the filling of the Main table in the Kernel database:
gather all pairs (agent, coh) occurring in the Stimulus db instead of assuming
all combinations occur.
"""

from database import Kernel_db, Stimulus_db
from database.kernel.fill_db import load_dataset
from database.kernel.fill_db._fill_db import splitData
import numpy as np
from tqdm import tqdm
import json, time

def load_dataset_debug(agent, coh):
    db = Stimulus_db()
    db.connect()
    raw_signal = db.cur.execute("""
        SELECT joystick, mean_dot
        FROM Main
        WHERE
            coh = ?
            AND agent = ?
    """, (coh, agent)).fetchall()
    db.close()

    J_list = []
    D_list = []
    for joy_raw, dot_raw in raw_signal:
        tab = np.array(json.loads(joy_raw))
        J_list.append(
            tab[0, :] * np.exp(1j * tab[1, :])
        )

        tab = np.array(json.loads(dot_raw))
        D_list.append(
            tab[0, :] + 1j * tab[1, :]
        )
    return J_list, D_list

def D_m_matrix_single_block_debug(D_m, kernelSize: int):
    mat = np.zeros((len(D_m), kernelSize), dtype=complex)
    # fill the first half
    a, b = np.ogrid[0:kernelSize, 0:-kernelSize:-1]
    mat[:kernelSize, :] = np.tril(D_m[a + b])

    # fill the second half
    a, b = np.ogrid[1:len(D_m)-kernelSize+1, kernelSize-1:-1:-1]
    mat[kernelSize:, :] = D_m[a + b]
    return mat

def Js_Mat_for_blocks_debug(joystick_list, dot_list, kernel_size):
    list_mat = [
        D_m_matrix_single_block_debug(dot, kernel_size)
        for dot in dot_list
    ]
    return np.vstack(list_mat), np.hstack(joystick_list)

if __name__ == "__main__":
    db = Kernel_db()
    db.connect()
    main_rows = db.cur.execute("""SELECT * FROM Main""").fetchall()
    db.close()

    for row in tqdm(main_rows):
        dataset = load_dataset(*row[1:])
        data_folds = splitData(dataset, testRatio=0.3)
