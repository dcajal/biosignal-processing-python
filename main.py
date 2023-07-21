import numpy as np
from scipy.interpolate import CubicSpline

from lib.delineation import ppg_pulse_detection, gap_correction
from lib.filters import filtering_and_normalization, remove_impulse_artifacts
from lib.hrv import time_metrics, frequency_metrics, mti
from lib.shared_tools import plot_signal

if __name__ == '__main__':
    fs_ppg = 250
    fs_ihr = 4

    # Load file
    myFile = np.genfromtxt('dataset/141820_24hz.csv', delimiter=',')
    data_matrix = np.delete(myFile, 0, 0)
    green = data_matrix[:, 1]
    unixtimestamps = data_matrix[:, 3]

    # Interpolate ppg at fs
    tAux = (unixtimestamps - unixtimestamps[0]) / 1000
    t = np.arange(0, tAux[-1], 1 / fs_ppg)
    cs = CubicSpline(tAux, -green)
    ppg = cs(t)

    # Baseline removal, filtering and normalization of PPG signal
    ppg_filtered = filtering_and_normalization(ppg, fs_ppg)
    ppg_filtered = remove_impulse_artifacts(ppg_filtered)

    # Pulse detection and correction
    ppg_tk = ppg_pulse_detection(ppg_filtered, fs_ppg, plotflag=False, fine_search=True)
    ppg_tn = gap_correction(ppg_tk, False)
    ppg_tn = ppg_tn - ppg_tn[0]  # Make the series begin at 0

    # Estimation of instantaneous heart rate based on the IPFM model
    sp = mti(ppg_tn, spline_order=4)
    ihr_t = np.arange(ppg_tn[0], ppg_tn[-1], 1 / fs_ihr)
    ihr = sp(ihr_t)*1000  # [mHz]

    # HRV
    time_metrics(ppg_tk)
    frequency_metrics(ppg_tn)
