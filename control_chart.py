import numpy as np
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pymysql
import sqlalchemy
import streamlit as st
import time
from statistics import fmean, stdev

plt.ion()
for i in range(10):
    machine_data = sqlalchemy.create_engine('mysql+pymysql://root:@localhost:3325/machine', pool_recycle=3600)
    machine_1_data = pd.read_sql_table("machine_1", machine_data)
    x = machine_1_data['ID']
    y1 = machine_1_data['humidity']
    y2 = machine_1_data['temperature']
    y3 = machine_1_data['torque']
    y4 = machine_1_data['pressure']

    plt.plot(x, y1, label="Humid", color="brown")
    plt.plot(x, y2, label="Temp", color="blue")
    plt.plot(x, y3, label="Torque", color="orange")
    plt.plot(x, y4, label="Press", color="red")

    # UCL & LCL
    y1_mean = fmean(y1)
    y1_std = stdev(y1)
    y1_ucl = y1_mean + 3 * y1_std
    y1_lcl = y1_mean - 3 * y1_std
    plt.hlines(y=y1_ucl, xmin=0, xmax=x.max(), label="Humid Upper", colors="brown", linewidth=2, linestyles="--")
    plt.hlines(y=y1_lcl, xmin=0, xmax=x.max(), label="Humid Lower", colors="brown", linewidth=2, linestyles="--")

    y2_mean = fmean(y2)
    y2_std = stdev(y2)
    y2_ucl = y2_mean + 3 * y2_std
    y2_lcl = y2_mean - 3 * y2_std
    plt.hlines(y=y2_ucl, xmin=0, xmax=x.max(), label="Temp Upper", colors="blue", linewidth=2, linestyles="--")
    plt.hlines(y=y2_lcl, xmin=0, xmax=x.max(), label="Temp Lower", colors="blue", linewidth=2, linestyles="--")

    y3_mean = fmean(y3)
    y3_std = stdev(y3)
    y3_ucl = y3_mean + 3 * y3_std
    y3_lcl = y3_mean - 3 * y3_std
    plt.hlines(y=y3_ucl, xmin=0, xmax=x.max(), label="Torque Upper", colors="orange", linewidth=2, linestyles="--")
    plt.hlines(y=y3_lcl, xmin=0, xmax=x.max(), label="Torque Lower", colors="orange", linewidth=2, linestyles="--")

    y4_mean = fmean(y4)
    y4_std = stdev(y4)
    y4_ucl = y4_mean + 3 * y4_std
    y4_lcl = y4_mean - 3 * y4_std
    plt.hlines(y=y4_ucl, xmin=0, xmax=x.max(), label="Press Upper", colors="red", linewidth=2, linestyles="--")
    plt.hlines(y=y4_lcl, xmin=0, xmax=x.max(), label="Press Lower", colors="red", linewidth=2, linestyles="--")

    # WARNING
    for y1_idx, y1_value in enumerate(y1):
        if y1_value >= y1_ucl or y1_value <= y1_lcl:
            plt.annotate(f"\u26A0 ID:{y1_idx+1} - Humidity:{y1_value}", xy=(y1_idx + 1, y1_value), xytext=(y1_idx + 3, y1_value + 5),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         )

    for y2_idx, y2_value in enumerate(y2):
        if y2_value >= y2_ucl or y2_value <= y2_lcl:
            plt.annotate(f"\u26A0 ID:{y2_idx+1} - Temperature:{y2_value}", xy=(y2_idx + 1, y2_value), xytext=(y2_idx + 3, y2_value + 5),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         )

    for y3_idx, y3_value in enumerate(y3):
        if y3_value >= y3_ucl or y3_value <= y3_lcl:
            plt.annotate(f"\u26A0 ID:{y3_idx+1} - Torque:{y3_value}", xy=(y3_idx + 1, y3_value), xytext=(y3_idx + 3, y3_value + 5),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         )

    for y4_idx, y4_value in enumerate(y4):
        if y4_value >= y4_ucl or y4_value <= y4_lcl:
            plt.annotate(f"\u26A0 ID:{y4_idx+1} - Pressure:{y4_value}", xy=(y4_idx + 1, y4_value), xytext=(y4_idx + 3, y4_value + 5),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         )

    plt.legend()
    plt.tight_layout()
    plt.get_current_fig_manager().canvas.set_window_title('Control Chart')
    plt.draw()
    plt.pause(1)
    plt.clf()
