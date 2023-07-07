import numpy as np
from scipy.interpolate import CubicSpline

from lib.delineation import ppg_pulse_detection, gap_correction
from lib.filters import filtering_and_normalization, remove_impulse_artifacts
from lib.hrv import time_metrics

if __name__ == '__main__':
    fs = 250

    # Load file
    myFile = np.genfromtxt('dataset/141453_24hz.csv', delimiter=',')
    data_matrix = np.delete(myFile, 0, 0)
    green = data_matrix[:, 1]
    unixtimestamps = data_matrix[:, 3]

    # Interpolate ppg at fs
    tAux = (unixtimestamps - unixtimestamps[0]) / 1000
    t = np.arange(0, tAux[-1], 1 / fs)
    cs = CubicSpline(tAux, -green)
    ppg = cs(t)

    # Baseline removal, filtering and normalization of PPG signal
    ppg_filtered = filtering_and_normalization(ppg, fs)
    ppg_filtered = remove_impulse_artifacts(ppg_filtered)

    # Pulse detection
    ppg_tk = ppg_pulse_detection(ppg_filtered, fs, plotflag=False, fine_search=True)
    print(ppg_tk)

    # HRV
    time_metrics(ppg_tk)
    # ppg_tn = gap_correction(ppg_tk, True)
