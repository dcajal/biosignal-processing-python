import numpy as np

from lib.shared_tools import compute_threshold


def time_metrics(tk):
    rr = np.diff(tk)
    rr[rr == 0] = []  # Remove repeated beats

    threshold = compute_threshold(rr)
    dRR = np.diff(rr)  # (ms)
    rr[rr < 0.7*threshold] = np.nan
    rr[rr > 1.3*threshold] = np.nan

    # Compute time domain indices
    mhr = np.nanmean(60. / rr)  # (beats / min)
    sdnn = 1000 * np.nanstd(rr)  # (ms)
    rmssd = 1000 * np.sqrt(np.sum(np.square(dRR[~np.isnan(dRR)])) / dRR[~np.isnan(dRR)].size)  # (ms)
    sdsd = 1000 * np.nanstd(dRR)  # (ms)
    pnn50 = 100 * (np.sum(np.abs(dRR) > 0.05)) / np.sum(~np.isnan(dRR))  # (%)

    # Print metrics
    print("MHR: %.2f beats/min" % mhr)
    print("SDNN: %.2f ms" % sdnn)
    print("RMSSD: %.2f ms" % rmssd)
    print("SDSD: %.2f ms" % sdsd)
    print("pNN50: %.2f%%" % pnn50)
