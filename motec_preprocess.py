import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from os import listdir
from os.path import isfile, join


def read_all_CSV(path):

    csv_files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    dfs = []
    for f in csv_files:
        df = read_CSV(f)
        if df is not None:
            dfs.append(df)
    return dfs


def read_CSV(path):
    print(path)

    #Fix separator errors
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    fw = open(path, 'w')
    # Parse errors like "2,34" that occurs sometimes in motec csv
    for line in lines:
        line2 = re.sub('("[^",]+),([^"]+")', '\\1.\\2', line)
        if line2 != line:
            print("Fixed line: {}".format(line))
        fw.write(line2)
    fw.close()

    with open(path) as f:
        line = f.readlines()[11].split(',')[1].replace('"', '').strip().split(' ')[0:-1]
        lap_markers = np.array(line, dtype=float)

    if len(lap_markers) >= 20:
        ignore_rows = []
        ignore_rows += [x for x in range(14)] + [15]
        df = pd.read_csv(path, skiprows=ignore_rows, header=0, dtype=float)

        # Remove unnecessary columns
        columns = ["LAP_BEACON", "CLUTCH", "SUS_TRAVEL_LF", "SUS_TRAVEL_RF", "SUS_TRAVEL_LR", "SUS_TRAVEL_RR",
                   "WHEEL_SPEED_LF", "WHEEL_SPEED_RF", "WHEEL_SPEED_LR", "WHEEL_SPEED_RR",
                   "BUMPSTOPUP_RIDE_LF", "BUMPSTOPUP_RIDE_RF", "BUMPSTOPUP_RIDE_LR", "BUMPSTOPUP_RIDE_RR"]
        df.drop(columns, inplace=True, axis=1)

        # Filter lap markers rows
        df_lap_markers = []
        lap_markers = lap_markers[lap_markers != 0]
        for lap_marker in lap_markers:
            df_lap_markers.append(df.loc[(df['Time']-lap_marker).abs().argsort()[0], :])

        # Add Lap_Time columns
        df_lap_markers = pd.DataFrame(df_lap_markers)
        df_lap_markers['Lap_Time'] = df_lap_markers.Time - df_lap_markers.Time.shift(1)
        df_lap_markers['Lap_Time_Avg'] = df_lap_markers['Lap_Time'].ewm(alpha=0.2).mean()

        # plt.plot(df_lap_markers['Lap_Time'], label='normal')
        # plt.plot(df_lap_markers['Lap_Time_Avg'], label='ewm mean')
        # plt.legend()
        # plt.show()

        # calc columns means
        init_idx = 0
        for i in range(len(lap_markers)):
            df_lap_markers.loc[df_lap_markers.index[i], ['G_LAT', 'ROTY', 'STEERANGLE', 'SPEED', 'THROTTLE', 'BRAKE', 'GEAR', 'G_LON', 'RPMS', 'TC', 'ABS', 'BRAKE_TEMP_LF', 'BRAKE_TEMP_RF', 'BRAKE_TEMP_LR', 'BRAKE_TEMP_RR', 'TYRE_PRESS_LF', 'TYRE_PRESS_RF', 'TYRE_PRESS_LR', 'TYRE_PRESS_RR', "TYRE_TAIR_LF","TYRE_TAIR_RF","TYRE_TAIR_LR","TYRE_TAIR_RR"]] \
                = df[['G_LAT', 'ROTY', 'STEERANGLE', 'SPEED', 'THROTTLE', 'BRAKE', 'GEAR', 'G_LON', 'RPMS', 'TC', 'ABS', 'BRAKE_TEMP_LF', 'BRAKE_TEMP_RF', 'BRAKE_TEMP_LR', 'BRAKE_TEMP_RR', 'TYRE_PRESS_LF', 'TYRE_PRESS_RF', 'TYRE_PRESS_LR', 'TYRE_PRESS_RR', "TYRE_TAIR_LF","TYRE_TAIR_RF","TYRE_TAIR_LR","TYRE_TAIR_RR"]].iloc[init_idx:df_lap_markers.index[i]].mean()
            init_idx = df_lap_markers.index[i]

        # Drop first row
        df_lap_markers = df_lap_markers.iloc[1:]
        # drop Lap_Time column
        df_lap_markers.drop('Lap_Time', inplace=True, axis=1)

        return df_lap_markers
    else:
        print("file: {} ignored. Total laps < 20.".format(path))
        return None


def print_ts(df):
    sns.lineplot(x=df.index, y='Lap_Time', data=df)
    sns.lineplot(x=df.index, y='TYRE_TAIR_LF', data=df)
    sns.lineplot(x=df.index, y='TYRE_TAIR_RF', data=df)
    sns.lineplot(x=df.index, y='TYRE_TAIR_LR', data=df)
    sns.lineplot(x=df.index, y='TYRE_TAIR_RR', data=df)
    plt.show()



if __name__ == '__main__':
    # df = read_CSV('./motec_files/nurburgring-porsche_991ii_gt3_r-11-2020.06.27-22.03.49.csv')
    # df = read_CSV('./motec_files/Laguna_Seca-porsche_991ii_gt3_r-0-2020.07.03-00.17.08.csv')
    # df = read_CSV('./motec_files/Laguna_Seca-porsche_991ii_gt3_r-15-2020.07.03-00.03.51.csv')
    # df_input = df[0:10]
    # df_output = df[-1:len(df.index)]
    # print_ts(df)

    path = './motec_files/'
    print(read_all_CSV(path))