# Script 2: this one calculates MSDs
# Lots of code from Chris here:
"""
Created on Mon Aug  5 18:07:37 2024

@author: azaldegc

correction made on 11/21/2024

(Totally gutted 05/2025)
"""

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np
import utils
import sklearn.mixture
# altsklearn is sklearn with some lines of the _gaussian_mixture.py file changed
#import altsklearn.mixture
import matplotlib.pyplot as plt

def has_gap(values, settings):
    sorted_values = values
            
    for i in range(len(sorted_values) - 1):
        if sorted_values[i+1] - sorted_values[i] > (settings['max_gap']+1):
            return True
    return False

def calculate_msd2(x, y, max_tau, settings):
    msd = []
    weights = []
    track_length_steps = len(x) - 1
    for tau in range(1, max_tau + 1):
        #print(range(1, max_tau+1))
        displacements = []
        n_steps = track_length_steps - tau
        #print("tau", tau)
        for t in range(n_steps):
         #   print("t", t)
            dx = (x[t + tau] - x[t])*settings['pixel_size_um']
            dy = (y[t + tau] - y[t])*settings['pixel_size_um']
            displacements.append(dx**2 + dy**2) 
        msd.append(np.mean(displacements))
        weights.append(len(displacements))
    return msd, np.asarray(weights)

def brownian_blur(tau, D, sig): # normal Brownian motion with blurring
    return 4*D*(tau) + 4*(sig)**2

def get_msd(tracks, gen_info, settings):
    """Confusing names, but this is the main function"""
    # Edit from Sam: set up the dataframe to have exactly 1 column labeled 'MSD'
    # Would be easier in pandas but will use numpy here like original code
    if 'MSD' in list(tracks.columns):
        tracks = tracks.drop(columns=['MSD'])
    tracks['MSD'] = np.nan

    min_frames = settings['min_frames']
    t_int = settings['t_int']
    t_delay = settings['t_delay']
    max_tau = max_tau_for_fit = min_frames - 2
    #for index, row in tracks.iterrows():
    for i in range(len(tracks)):
        n = len(tracks.iloc[i]['frame'])
        gap = has_gap(tracks.iloc[i]['frame'], settings)
        if n >= min_frames and gap == False:
            #track_x_coords = row['x'].to_numpy()[:]
            #track_y_coords = row['y'].to_numpy()[:]
            #time = row['frame'].to_numpy()[:]
            track_x_coords = tracks.iloc[i]['x']
            track_y_coords = tracks.iloc[i]['y']
            time = tracks.iloc[i]['frame']
            msd_curve, weights = calculate_msd2(track_x_coords, track_y_coords, max_tau, settings)
            weights[weights == 0] = 1 # Avoid division by zero
            time_lags = np.arange(1,max_tau+1) * (t_int+t_delay)
            sigma = np.sqrt(1 / weights)
            # fit to MSD function
            popt, pcov = curve_fit(brownian_blur, time_lags[:max_tau_for_fit], 
                                   msd_curve[:max_tau_for_fit], 
                                   maxfev = 10000, bounds=(0, [np.inf, 2]),
                                   sigma=sigma[:max_tau_for_fit], absolute_sigma=True)
            # Extract fit parameters
            D, sigma = popt
            # calculate fitted values
            msd_fitted = brownian_blur(time_lags, *popt)
            # Residuals
            residuals = msd_curve[:max_tau_for_fit] - msd_fitted[:max_tau_for_fit]
            # Calculate R^2 and RMSE
            r2 = r2_score(msd_curve[:max_tau_for_fit], msd_fitted[:max_tau_for_fit])
            if r2 > 0.75:
                # at would reference index
                tracks.iat[i, len(tracks.columns)-1] = D
                # Store min and max log-msd values (for extrema on graphs)
    gen_info['min_msd'] = np.log10(min(tracks['MSD'][~np.isnan(tracks['MSD'])]))
    gen_info['max_msd'] = np.log10(max(tracks['MSD'][~np.isnan(tracks['MSD'])]))
    utils.pickle_save(ct=tracks, gi=gen_info)
    return tracks, gen_info

def convert_px_to_um(combined_tracks, settings):
    combined_tracks['x_um'] = combined_tracks['x'].apply(lambda to_convert: [val * settings['pixel_size_um'] for val in to_convert])
    combined_tracks['y_um'] = combined_tracks['y'].apply(lambda to_convert: [val * settings['pixel_size_um'] for val in to_convert])
    return combined_tracks

def calc_squared_displacement(combined_tracks, settings):
    combined_tracks['squared_displacement'] = np.zeros
    if 'x_um' not in combined_tracks.columns:
        combined_tracks = convert_px_to_um(combined_tracks, settings)
    for t in range(len(combined_tracks)):
        x_vals = combined_tracks.at[t, 'x_um']
        y_vals = combined_tracks.at[t, 'y_um']
        frames = combined_tracks.at[t, 'frame']
        sqd = []
        for f in range(len(frames)-1):
            if frames[f+1] != frames[f] + 1:
                sqd.append(np.nan)
                continue
            dx = x_vals[f+1] - x_vals[f]
            dy = y_vals[f+1] - y_vals[f]
            sqd.append(dx**2 + dy**2)
        combined_tracks.at[t, 'squared_displacement'] = sqd
    utils.pickle_save(ct=combined_tracks)
    return combined_tracks


def calc_log_condition_MSDs(combined_tracks, condition):
    """This utility function removes NAs, takes the log10, and returns tracks matching a particular function"""
    condition_data = combined_tracks[combined_tracks['condition'] == condition]
    # condition_MSDs = condition_data['MSD']
    # return np.log10(condition_MSDs.values.astype(float)).reshape(-1, 1)
    condition_MSDs = condition_data['MSD'][~np.isnan(condition_data['MSD'])].to_frame()
    return np.log10(condition_MSDs).values.reshape(-1, 1)


import sklearn
import matplotlib.pyplot as plt


def single_condition_gmm(combined_info, condition, log_condition_MSDs, n_components, GMM_SETTINGS):
    rng = np.random.default_rng()
    gmm_means = np.zeros((GMM_SETTINGS['bootstrap_n'], n_components))
    gmm_weights = np.zeros((GMM_SETTINGS['bootstrap_n'], n_components))
    # Both defines and fits GMMs
    gmm = sklearn.mixture.GaussianMixture(n_components=n_components,
                        covariance_type='full',
                        n_init=GMM_SETTINGS['n_init']).fit(log_condition_MSDs)
    # Now bootstrap to get CIs
    for b in range(GMM_SETTINGS['bootstrap_n']):
        bootstrap_MSDs = rng.choice(log_condition_MSDs, size=len(log_condition_MSDs), replace=True)
        # Both defines and fits GMMs
        bootstrap_working = sklearn.mixture.GaussianMixture(n_components=n_components,
                            covariance_type='full',
                            n_init=GMM_SETTINGS['n_init']).fit(bootstrap_MSDs)
        means = bootstrap_working.means_.flatten()
        gmm_sort_indices = np.argsort(means)
        means = np.power(10, np.take_along_axis(means, gmm_sort_indices, axis=0))
        gmm_means[b] = means
        weights = np.take_along_axis(bootstrap_working.weights_.flatten(), gmm_sort_indices, axis=0)
        gmm_weights[b] = weights
        # Now store info in combined_info, which gets returned
        if len(gmm_means) == 0:
            gmm_means = np.zeros((GMM_SETTINGS['bootstrap_n'], n_components))
            gmm_weights = np.zeros((GMM_SETTINGS['bootstrap_n'], n_components))
        # Sort means and variances to always put the slow pop first
        means = gmm.means_.flatten()
        sort_indices = np.argsort(means)
        means = np.take_along_axis(means, sort_indices, axis=0)
        variances = np.take_along_axis(gmm.covariances_.flatten(), sort_indices, axis=0)
        weights = np.take_along_axis(gmm.weights_.flatten(), sort_indices, axis=0)
        combined_info.at[condition, "gmm_n_comps"] = n_components
        combined_info.at[condition, "gmm_means"] = means      # Means of the two Gaussians
        combined_info.at[condition, "gmm_variances"] = variances # Variances of the two Gaussians (since it's 1D)
        combined_info.at[condition, "gmm_err_NOTLOG"] = [np.percentile(gmm_means[:,z], [2.5, 97.5]) for z in range(n_components)]
        combined_info.at[condition, "gmm_weights"] =  weights  # Mixing proportions of the two Gaussians
        combined_info.at[condition, "gmm_weight_err"] = [np.percentile(gmm_weights[:,z], [2.5, 97.5]) for z in range(n_components)]
        combined_info.at[condition, "gmm_criterium"] = gmm.aic(log_condition_MSDs)
    return combined_info


def calc_gmms(combined_tracks, combined_info, GMM_SETTINGS):
    """This function calculates Gaussian Mixture Model fits for each condition and stores info about them in the combined_info variable"""    
    # Initialize columns in the combined_info dataframe
    # Can't be nan because that throws errors when overwriting
    combined_info['gmm_n_comps'] = np.zeros
    combined_info['gmm_means'] = np.zeros
    combined_info['gmm_err_NOTLOG'] = np.zeros
    combined_info['gmm_variances'] = np.zeros
    combined_info['gmm_weights'] = np.zeros
    combined_info['gmm_weight_err'] = np.zeros
    combined_info['gmm_criterium'] = np.zeros
    combined_info['gmm_optimization_data'] = np.zeros if GMM_SETTINGS['n_components'] == "optimize" else None

    # Define variables used to optimize the # of Gaussians
    crit_array = np.zeros((len(combined_info), 10))

    # For each condition
    for r in range(len(combined_info)):
        condition = combined_info.index[r]
        log_condition_MSDs = calc_log_condition_MSDs(combined_tracks, condition)
        if GMM_SETTINGS['verbose']:
            print(condition)

        # Handle the different options for n components
        # If it's an int (simplest), just use pass info to the function
        if type(GMM_SETTINGS['n_components']) == int:
            n_components = GMM_SETTINGS['n_components']
            combined_info = single_condition_gmm(combined_info, condition, log_condition_MSDs, n_components, GMM_SETTINGS)

        # alternatively, reads in a dictionary specifying a different # of gaussians for each condition
        elif type(GMM_SETTINGS['n_components']) == dict:
            n_components = GMM_SETTINGS['n_components'][condition]
            combined_info = single_condition_gmm(combined_info, condition, log_condition_MSDs, n_components, GMM_SETTINGS)

        # ("optimize") alternatively, optimizes each condition to find the # of gaussians to optimize aic/bic, or
        elif GMM_SETTINGS['n_components'] == "optimize":
            # best criterium # (BIC or AIC) set to infinity... anything will beat that
            best_crit = float('inf')
            # Try fitting 1-10 Gaussians
            for n_components_working in range(1, 11):
                gmm_t = sklearn.mixture.GaussianMixture(n_components=n_components_working,
                                    covariance_type='full',
                                    n_init=GMM_SETTINGS['n_init']).fit(log_condition_MSDs)
                if GMM_SETTINGS['criterium'] == 'aic':
                    crit = gmm_t.aic(log_condition_MSDs)
                elif GMM_SETTINGS['criterium'] == 'bic':
                    crit = gmm_t.bic(log_condition_MSDs)
                else:
                    raise ValueError("Invalid criterium. Choose 'aic' or 'bic'.")
                # If best-scoring criterium, keep that one
                if crit < best_crit:
                    best_crit = crit
                    best_n_gaussians = n_components_working
                    n_components = n_components_working
                # Update criterium array
                crit_array[r, n_components_working-1] = crit
            # Once we've settled on the best case, treat it as if it had been a given since the beginning and bootstrap it to get CIs
            combined_info = single_condition_gmm(combined_info, condition, log_condition_MSDs, n_components, GMM_SETTINGS)
            combined_info.at[condition, "gmm_optimization_data"] = crit_array[r, :]
        
        # Throw an error if there was an invalid option for n_components
        else:
            raise ValueError("Invalid n_components. Choose an int, a dict, or 'optimize'.")
    
    # return and save
    utils.pickle_save(ci=combined_info)
    return combined_info