import numpy as np
from scipy.signal import savgol_filter

def msc(input_data, reference=None):
    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i, :] -= input_data[i, :].mean()

    # Get the reference spectrum. If not given, estimate it from the mean
    if reference is None:
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference

    # Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i, :], 1, full=True)
        # Apply correction
        data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]

    return (data_msc, ref)


def snv(input_data):
    # Define a new array and populate it with the corrected data
    data_snv = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        data_snv[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])

    return data_snv


def refPeak(input_data):
    data_ref = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        data_ref[i, :] = input_data[i, :]/input_data[i, 93]
    return data_ref


def preprocess_data(input_data, method="msc", reference=None, window_length=11, polyorder=2):
    # --- Step 1: Choose preprocessing ---
    if method == "msc":
        if reference is None:
            raise ValueError("Reference spectrum required for MSC.")
        processed = msc(input_data, reference=reference)[0]

    elif method == "snv":
        processed = snv(input_data)

    elif method == "scaling":
        processed = refPeak(input_data)

    elif method == "SG":
        processed = input_data
    else:
        raise ValueError("Method must be 'msc', 'snv', or 'scaling'.")

    # --- Step 2: Apply Savitzky-Golay smoothing ---
    filt = savgol_filter(processed, window_length=window_length, polyorder=polyorder)

    return filt
