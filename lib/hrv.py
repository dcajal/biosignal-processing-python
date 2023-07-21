import numpy as np
from scipy.interpolate import splrep, BSpline
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


def frequency_metrics(tn):
    pass


def mti(tn, spline_order=14):
    # Estimation of instantaneous heart rate based on the IPFM model
    # Simplified version without 'ids'. Note that ids is not recommended as
    # it is better to perform gap-filling methods: https://www.mdpi.com/1424-8220/22/15/5774
    #
    # tn: normal beat occurrence time series (in seconds)
    # spline_order: order of the spline for interpolation (default, 14)
    # sp = (1+m(t))/T interpolated with splines of order spline_order

    if spline_order < 2:
        raise ValueError('splineOrder should be a natural value higher than 1')

    if len(tn) == 0:
        raise ValueError('tn is empty')

    tn = np.array(tn).flatten()

    # Calculation of initial and final time intervals
    dt1 = np.median(np.diff(tn[0:9]))
    dt2 = np.median(np.diff(tn[-8:]))

    # Construction of the extended time series tt
    k = 10
    tt = np.concatenate(
        (tn[0] - np.arange(k, 0, -1) * dt1,
         tn,
         tn[-1] + np.arange(1, k + 1) * dt2)) - tn[0]

    # Generation of the interpolating spline
    tck = splrep(tt + tn[0], np.arange(0, len(tt)), k=spline_order)
    sp = BSpline(np.array(tck[0]), np.array(tck[1]), spline_order).derivative()

    return sp
