import numpy as np
from scipy import signal


def filtering_and_normalization(sig, sig_fs):
    b, a = signal.butter(3, 0.3, btype='highpass', fs=sig_fs)
    sig_filtered = signal.filtfilt(b, a, sig)
    b, a = signal.butter(3, 10, btype='lowpass', fs=sig_fs)
    sig_filtered = signal.filtfilt(b, a, sig_filtered)
    sig_filtered = normalize(sig_filtered)
    return sig_filtered


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def remove_impulse_artifacts(sig):
    # Square of second derivative
    aux = np.diff(np.diff(sig)) ** 2
    aux = np.insert(aux, 0, aux[0])
    aux = np.append(aux, aux[-1])

    # Median filter threshold
    wind = 1000
    if aux.size < wind:
        wind = aux.size
    mf = signal.medfilt(aux, wind - 1)

    # Find impulses
    margin = 20
    impulses = np.asarray(np.where(aux > mf + 0.005)).ravel()
    for impulse in impulses:
        impulses = np.append(impulses, np.arange(impulse - margin, impulse + margin + 1))
    impulses = np.sort(impulses)
    impulses = np.unique(impulses)
    impulses = impulses[impulses >= 0]

    # Remove impulses
    output = sig
    output[impulses] = np.nan
    return output
