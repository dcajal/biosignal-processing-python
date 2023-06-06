import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def compute_threshold(rr):
    wind = 30
    if rr.size < wind:
        wind = rr.size
    mf = signal.medfilt(np.concatenate((np.flipud(rr[0:wind // 2]), rr, np.flipud(rr[-(wind // 2):])))[:], wind - 1)
    mf[mf > 1.5] = 1.5
    return mf[(wind // 2):-(wind // 2)]


def plot_response(w, h, title):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(w, np.abs(h))
    ax.set_xlim(0, 10)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain')
    ax.set_title(title)
    ax = fig.add_subplot(212)
    ax.plot(w, np.angle(h))
    ax.set_xlim(0, 10)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase')
    plt.show()


def plot_signal(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, label='data')
    plt.show()
