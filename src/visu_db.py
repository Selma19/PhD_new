"""Interactive visualization of the results accumulated in the kernel database, allowing them to be
interpreted.

The idea is that we have 2 figures (for now):
- one where we visualize various kernel functions (implement only multiselect buttons)
- one where we visualize the train and test errors of the plotted kernels
"""

import sqlite3, json, os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from database.kernel.fill_db import read_kernel, read_fit_fct

# for wide figures
st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    """Loads the whole database in the cache, as a single dataframe."""
    query = """SELECT 
        m.agent,
        m.coh,
        m.filtering_method,
        k.kernel_output,
        k.train_error,
        k.test_error,
        k.kernel_type,
        k.kernel_method,
        k.method_param,
        f.kernel_key,
        f.fit_output,
        f.error as fit_error,
        f.fit_type,
        f.fit_method

        FROM Main m
        LEFT JOIN Kernels k ON k.main_key = m.id
        LEFT JOIN Fit_kernels f ON f.kernel_key = k.id
    """
    
    # connect to the kernel database
    db_path = __file__
    for _ in range(1):
        db_path = os.path.dirname(db_path)
    db_path = os.path.join(
        db_path,
        "kernel_extraction", "kernel_data", "solo"
    )
    db_name = "kernel_storage.db"
    conn = sqlite3.connect(os.path.join(db_path, db_name))
    conn.execute("PRAGMA foreign_keys = ON;")

    df = pd.read_sql_query(query, conn)
    conn.close()

    # deserialize the JSON data (restore the Python data types)
    for column in ["kernel_output", "fit_output", "method_param"]:
        df[column] = df[column].apply(json.loads)

    # add the kernel fcts
    k_fcts = []
    for k_out, k_type, k_meth in zip(
        df["kernel_output"], df["kernel_type"], df["kernel_method"]
    ):
        k_fcts.append( read_kernel(k_out, k_type, k_meth) )
    df.insert(2, "kernel", k_fcts)
    
    # add the fit fcts
    f_fcts = []
    for f_out, f_type, f_meth in zip(
        df["fit_output"], df["fit_type"], df["fit_method"]
    ):
        f_fcts.append( read_fit_fct(f_out, f_type, f_meth) )
    df.insert(12, "fit", f_fcts)

    return df

# load the database as a dataframe
df = load_data()

# intializes the session dynamical variables (changed upon user actions)
if 'picking_kernel' not in st.session_state:
    st.session_state.picking_kernel = False

if 'added_kernels' not in st.session_state:
    st.session_state.added_kernels = []

if 'chosen_kernel' not in st.session_state:
    st.session_state.chosen_kernel = None

if 'sub_df' not in st.session_state:
    st.session_state.sub_df = None

if 'fig_ker' not in st.session_state:
    st.session_state.fig_ker = go.Figure()
    st.session_state.fig_ker.update_layout(
        xaxis_title="time (ms)",
        yaxis_title= "kernel modulus",
        template="plotly_white"
    )

if 'fig_err' not in st.session_state:
    st.session_state.cohs = sorted( df["coh"].unique() )

    st.session_state.fig_err = make_subplots(
        rows=3, cols=2, start_cell="bottom-left",
        subplot_titles=tuple(f"coherence: {coh}" for coh in st.session_state.cohs)
    )
    st.session_state.fig_err.update_layout(
        template="plotly_white"
    )
    for num, coh in enumerate(st.session_state.cohs):
        i = num // 2 + 1
        j = num % 2 + 1

        for name, color in zip(
            ["unfiltered", "removeAfterTgt"],
            ["LightSeaGreen", "RoyalBlue"]
        ):
            st.session_state.fig_err.add_scatter(
                x=[], y=[], mode="markers",
                marker=dict(size=7, color=color),
                name=name, row=i, col=j, showlegend=False
            )

    for j in [1, 2]:
        st.session_state.fig_err.update_xaxes(
            dict(title="kernel group"), row=1, col=j
        )
    for i in [1, 2, 3]:
        st.session_state.fig_err.update_yaxes(
            dict(title="test_error"), row=i, col=1
        )
    
    st.session_state.fig_err.update_traces(
        dict(showlegend=True), row=1, col=1
    )

def click_pick_kernel():
    st.session_state.picking_kernel = True

def display_kernels():
    dt = 8.33 # ms
    x = np.arange(300) * dt
    # read the (complex-valued) kernel function
    k_fct = st.session_state.sub_df["kernel"].iloc[0]

    # plot its modulus
    y = np.abs(k_fct)
    new_kernel = st.session_state.added_kernels[-1]
    name = f"filtering_meth: {new_kernel[0]},"
    name += f"<br>coh: {new_kernel[2]}, k_type: {new_kernel[3]},"
    name += f"<br>k_meth: {new_kernel[4]}, reg_param: {new_kernel[5]}"
    st.session_state.fig_ker.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=name
        )
    )

def click_add_kernel():
    st.session_state.picking_kernel = False
    st.session_state.added_kernels.append(
        st.session_state.chosen_kernel
    )
    # visualize all the chosen kernels
    display_kernels()

def display_err(err_group: str):
    """Displays the test error.
    
    The figure has 1 subplot per coherence value.
    At a given x location, we display the test error of every
    kernel and their fit belonging to the group at this location.
    """
    # make the correspondence btw the user chosen group and the
    # attributes of the dataframe
    if err_group == 'type':
        col = 'kernel_type'
    elif err_group == 'extraction method':
        col = 'kernel_method'

    # set the x ticks
    groups = sorted( df[col].unique() )
    tickvals = [i for i in range(len(groups))]

    # set the y data
    for num, coh in enumerate(st.session_state.cohs):
        i = num // 2 + 1
        j = num % 2 + 1

        sub_df = df[ df['coh'] == coh ]

        st.session_state.fig_err.update_xaxes(
            dict(
                tickmode = 'array',
                tickvals = tickvals,
                ticktext = groups
            ), row = i, col = j
        )

        ## add the kernel test error for each trace (no filter and with filter)
        trace_names = ["unfiltered", "removeAfterTgt"]
        row_names = ["unfiltered", "remove_after_tgt"]

        for trace_name, row_name in zip(trace_names, row_names):
            xs = []
            ys = []

            for tickval, group in zip(tickvals, groups):
                dg = sub_df[ sub_df["filtering_method"] == row_name ]
                y = dg[ dg[col] == group ]["test_error"].to_numpy()
                xs.extend( [tickval] * len(y) )
                ys.extend(y)
            
            st.session_state.fig_err.update_traces(
                dict(x=xs, y=ys),
                row = i, col = j,
                selector=dict(name=trace_name)
            )

    ## add the fit test error
    pass

def clear_kernels():
    st.session_state.added_kernels = []
    st.session_state.fig_ker.data = []

# buttons for interacting with the kernel figure
st.button("Pick a kernel", on_click=click_pick_kernel)
st.button("Add the picked kernel", on_click=click_add_kernel)
st.button("Clear the kernels", on_click=clear_kernels)

# session loop: visualize multiple kernels on the same plot
if st.session_state.picking_kernel:
    # define scrolling menus

    # define the features to be selected, in this order
    labels = [
        "Filtering method", "Agent", "Coherence", "Kernel type",
        "Method for extracting the kernel", "Regularization parameter"
    ]
    col_names = [
        "filtering_method", "agent", "coh",
        "kernel_type", "kernel_method", "method_param"
    ]

    chosen_kernel = tuple()
    sub_df = df.copy()

    # select each feature one after the other
    for label, col_name in zip(labels, col_names):
        selected_feature = st.selectbox(label, sorted(sub_df[col_name].unique()))
        try:
            isnan = np.isnan(selected_feature)
        except TypeError:
            isnan = False
        
        if isnan:
            sub_df = sub_df[ sub_df[col_name].isnull() ]
        else:
            sub_df = sub_df[ sub_df[col_name] == selected_feature ]
        
        chosen_kernel += (selected_feature,)
    st.session_state.chosen_kernel = chosen_kernel
    st.session_state.sub_df = sub_df

# display the figures

## the kernels
st.plotly_chart(st.session_state.fig_ker, width='stretch')

# box for interacting with the test error figure
err_group = st.selectbox("Group kernels by", ['type', 'extraction method'])

## the test error
display_err(err_group)
st.plotly_chart(st.session_state.fig_err, width='stretch')
