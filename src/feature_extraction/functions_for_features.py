import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from ..utils import path_project


def extract_heart_rate_features(ecg):
    mean_heart_rate = np.mean(ecg.heart_rate)
    median_heart_rate = np.median(ecg.heart_rate)
    std_heart_rate = np.std(ecg.heart_rate)
    var_coefficient = (std_heart_rate / mean_heart_rate) * 100

    # heart rate volatility calculation
    chg_heart_rate = []
    for i in range(0, len(ecg.heart_rate) - 1):
        chg_heart_rate = np.append(chg_heart_rate, ecg.heart_rate[i + 1] / ecg.heart_rate[i] - 1)
    vol_heart_rate = np.std(chg_heart_rate)
    skew_heart_rate = sp.stats.skew(ecg.heart_rate)

    # heart rate variability calculation
    RMS = 0
    for j in range(0, len(ecg.heart_rate)):
        RMS += (60000 / ecg.heart_rate[j]) ** 2
    RMSSD = 0
    if (len(ecg.heart_rate)) >= 1:
        RMSSD = np.sqrt(RMS / (len(ecg.heart_rate)))
    return mean_heart_rate, median_heart_rate, (
                mean_heart_rate - median_heart_rate) / mean_heart_rate, std_heart_rate, var_coefficient, vol_heart_rate, skew_heart_rate, RMSSD

    # returns mean, median, (mean-median)/mean, std, varcoeff, skewness, RMSSD


# Heart rate variability(RMSSD) calculated based on peaks as an alternative
def extract_heart_rate_variability_peaks(ecg):
    peak_dist = []

    for i in range(0, len(ecg.rpeaks) - 1):
        peak_dist = np.append(peak_dist, (ecg.rpeaks[i + 1] - ecg.rpeaks[i]) * 10 / 3)

    RMS = 0
    for j in range(0, len(peak_dist)):
        RMS += peak_dist[j] ** 2
    RMSSD_peaks = np.sqrt(RMS / (len(ecg.rpeaks) - 1))
    return RMSSD_peaks


# Heart rate correction flag and heart rate correction level
def compare_length_peak_heart_rate(ecg):
    if ((len(ecg.rpeaks) - 1) - len(ecg.heart_rate)) > 0:
        x = 0
    else:
        x = 1

    return x, ((len(ecg.rpeaks) - 1) - len(ecg.heart_rate)) / len((ecg.rpeaks) - 1)

# Heart rate feature calculation based on peaks as an alternative
def heart_rate_derived_from_peaks(ecg):
    heart_rate_peaks = []

    for i in range(0, len(ecg.rpeaks) - 1):
        heart_rate_peaks = np.append(heart_rate_peaks, 18000 / (ecg.rpeaks[i + 1] - ecg.rpeaks[i]))

    mean_heart_rate_peaks = np.mean(heart_rate_peaks)
    median_heart_rate_peaks = np.median(heart_rate_peaks)
    std_heart_rate_peaks = np.std(heart_rate_peaks)
    var_coefficient_peaks = (std_heart_rate_peaks / mean_heart_rate_peaks) * 100
    chg_heart_rate_peaks = []
    for i in range(0, len(heart_rate_peaks) - 1):
        chg_heart_rate_peaks = np.append(chg_heart_rate_peaks, heart_rate_peaks[i + 1] / heart_rate_peaks[i] - 1)
    vol_heart_rate_peaks = np.std(heart_rate_peaks)
    skew_heart_rate_peaks = sp.stats.skew(heart_rate_peaks)
    return mean_heart_rate_peaks, (
                mean_heart_rate_peaks - median_heart_rate_peaks) / mean_heart_rate_peaks, std_heart_rate_peaks, var_coefficient_peaks, vol_heart_rate_peaks, skew_heart_rate_peaks, (
           (np.max(heart_rate_peaks) - np.min(heart_rate_peaks))) / mean_heart_rate_peaks


def extract_template_stats_all(ecg):
    return np.concatenate((np.mean(ecg.templates, axis=0), np.median(ecg.templates, axis=0),(np.mean(ecg.templates, axis=0) - np.median(ecg.templates, axis=0)) / (np.mean(ecg.templates, axis=0) + 0.000000000001), np.std(ecg.templates, axis=0), 100 * (np.std(ecg.templates, axis=0) / (np.mean(ecg.templates, axis=0) + 0.000000000001)), sp.stats.skew(ecg.templates, axis=0)), axis=0)

    # returns all template statistics in one go: mean, median, (mean-median)/mean, std, variation coefficient, skewness

def extract_template_stats_mean(ecg):
    return np.mean(ecg.templates, axis=0)

def extract_template_stats_median(ecg):
    return np.median(ecg.templates, axis=0)

def extract_template_stats_mean_median_diff_ratio(ecg):
    return (np.mean(ecg.templates, axis=0) - np.median(ecg.templates, axis=0)) / (np.mean(ecg.templates, axis=0) + 0.000000000001)

def extract_template_stats_std(ecg):
    return np.std(ecg.templates, axis=0)

def extract_template_stats_var_coeff(ecg):
    return 100 * (np.std(ecg.templates, axis=0) / (np.mean(ecg.templates, axis=0) + 0.000000000001))

def extract_template_stats_skew(ecg):
    return sp.stats.skew(ecg.templates, axis=0)

def create_csv(df, name_path, name_file):

    # dump into a csv file
    path = path_project + name_path
    os.makedirs(path, exist_ok=True)
    np.savetxt(path + name_file + ".csv", df, header='class0,class1,class2,class3,all_classes,min_class0,max_class0,min_class1,max_class1,min_class2,max_class2,min_class3,max_class3,per_10_class0,per_90_class0,per_10_class1,per_90_class1,per_10_class2,per_90_class2,per_10_class3,per_90_class3', comments='', delimiter=",", fmt=['%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f', '%1.10f','%1.10f', '%1.10f', '%1.10f', '%1.10f'])

def create_sav(X_train, y_train, X_test, extension):
    path = path_project + "data/processed/"
    os.makedirs(path, exist_ok=True)

    train_path = path + "X_train_processed" + extension + ".sav"
    pickle.dump((X_train), open(train_path, "wb"))

    y_train_path = path + "y_train" + extension + ".sav"
    pickle.dump((y_train), open(y_train_path, "wb"))

    test_path = path + "X_test_processed" + extension +".sav"
    pickle.dump((X_test), open(test_path, "wb"))

