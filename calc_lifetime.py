import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.optimize import curve_fit
import utils

def single_exp(t, k):
    return np.exp(-k * t)

def count_tracks_for_bootstrap(boot_frames, num_timepoints, max_frame_diff):
    all_rel_frames = np.concatenate([np.array(frames) - frames[0] for frames in boot_frames])
    valid_frames = all_rel_frames[all_rel_frames < max_frame_diff].astype(int)
    counts = np.bincount(valid_frames, minlength=num_timepoints).astype(float)
    return counts

def count_tracks(condition_tracks_series, num_timepoints, max_frame_diff):
    all_rel_frames = np.concatenate([np.array(frames) - frames[0] for frames in condition_tracks_series])
    valid_frames = all_rel_frames[all_rel_frames < max_frame_diff].astype(int)
    counts = np.bincount(valid_frames, minlength=num_timepoints).astype(float)
    return counts

def calc_lifetime(combined_info, combined_tracks, DECAY_SETTINGS):
    frame_length = DECAY_SETTINGS['t_int'] + DECAY_SETTINGS['t_delay']
    times_possible = np.arange(0, 30, frame_length)
    rng = np.random.default_rng()
    # Starting params: a, b, c, d, with the equation (as above) being a*e^(-t/b)+ c*e^(-t/d)
    initial_guess = DECAY_SETTINGS['initial_guess']
    bounds =  DECAY_SETTINGS['bounds']
    # bounds = ((1e-10, 1e-10), (np.inf, np.inf))
    bootstrapped_k_vals = {}
    reference = DECAY_SETTINGS['reference']
    max_frame_diff = 30.0 / frame_length
    num_timepoints = len(times_possible)
    combined_info['timepoints'] = None
    combined_info['params'] = None

    # for each condition...
    for r in range(len(combined_info)):
        condition = combined_info.index[r]
        timepoints = pd.DataFrame({'time_from_track_start': times_possible, 'tracks_surviving': np.zeros(len(times_possible))})
        bootstrapped_survival_curves = np.zeros(shape=(len(times_possible), DECAY_SETTINGS['bootstrap_n']))
        condition_tracks = combined_tracks[combined_tracks['condition'] == condition]
        condition_tracks.reset_index(inplace=True)
        
        # first find the "real" survival #s
        timepoints['tracks_surviving'] = count_tracks(condition_tracks['frame'], num_timepoints, max_frame_diff)

        # The bootstrapping loop
        for s in range(DECAY_SETTINGS['bootstrap_n']):
            bootstrapped_frames = rng.choice(condition_tracks['frame'], size=len(condition_tracks), replace=True)
            bootstrapped_survival_curves[:,s] = count_tracks_for_bootstrap(bootstrapped_frames, num_timepoints, max_frame_diff)
        

        # Update timepoints
        timepoints['bootstrap_std_abs'] = np.std(bootstrapped_survival_curves, axis=1)
        timepoints['normalized_tracks_surviving'] = timepoints['tracks_surviving'] / timepoints.at[2, 'tracks_surviving']
        timepoints['bootstrap_std_normalized'] = timepoints['bootstrap_std_abs'] / timepoints.at[2, 'tracks_surviving']
        
        # Drop places with less than 5 tracks surviving
        timepoints = timepoints[2:].reset_index(drop=True) # drop the first 2 time points
        timepoints['time_from_track_start'] = timepoints['time_from_track_start'] - 2*frame_length # adjust time to reflect dropped points
        std_cut_mask = timepoints['tracks_surviving'] < 5
        timepoints = timepoints[~std_cut_mask].reset_index(drop=True)
        timepoints.to_csv(f"op/{condition}_decay.csv")
        combined_info.at[condition, 'timepoints'] = timepoints
        
        
        ### Curve fitting
        ## Pt 1: base curve fitting
        # Extract x and y arrays
        x_data = timepoints['time_from_track_start']
        y_data = timepoints['normalized_tracks_surviving']
        stdevs = timepoints['bootstrap_std_normalized']

        try:
            # Fit the curve. We use 1e-10 to avoid zero division errors in curve_fit when stdevs are zero.
            popt, pcov = curve_fit(single_exp, x_data, y_data, p0=initial_guess, sigma=stdevs+1e-10, bounds=bounds, absolute_sigma=True, maxfev=5000)
            combined_info.at[condition, 'params'] = popt
            # print(f"Error on fit for condition '{condition}': {np.sqrt(np.diag(pcov))} (vals: {popt})")
        except RuntimeError:
            print(f"Warning: Curve fit failed to converge for condition '{condition}'.")
            continue

        ## Pt 2: bootstrapped curve fitting
        bootstrap_k_vals = []
        boot_x_data = times_possible[2:]-2*frame_length
        boot_x_data = boot_x_data[~std_cut_mask[:]]
        for s in range(DECAY_SETTINGS['bootstrap_n']):
            boot_y_data = bootstrapped_survival_curves[2:,s] / bootstrapped_survival_curves[2,s]
            boot_y_data = boot_y_data[~std_cut_mask[:]]
            try:
                popt_boot, _ = curve_fit(single_exp, boot_x_data, boot_y_data, p0=initial_guess, sigma=stdevs+1e-10, bounds=bounds, absolute_sigma=True, maxfev=5000)
                bootstrap_k_vals.append(popt_boot[0])
            except RuntimeError:
                print(f"Warning: Bootstrapped curve fit failed to converge for condition '{condition}', sample {s}. This may generate missing values.")
                bootstrap_k_vals.append(np.nan)




        # Extract ksb values
        combined_info.at[condition, 'raw_ksb'] = popt[0]
        # print(1/np.max([popt[0], popt[1]]))
        combined_info.at[condition, 'ksb_ci_lower'] = np.nanpercentile(bootstrap_k_vals, 2.5)
        combined_info.at[condition, 'ksb_ci_upper'] = np.nanpercentile(bootstrap_k_vals, 97.5)
        bootstrapped_k_vals[condition] = bootstrap_k_vals
        
    for r in range(len(combined_info)):
        condition = combined_info.index[r]
        if condition != DECAY_SETTINGS['background']: # h2b is the "background" dwell time that we want to subtract out, so we don't calculate a corrected dwell time for it
            combined_info.at[condition, 'corrected_dwell_time'] = 1/(combined_info.at[condition, 'raw_ksb'] - combined_info.at[DECAY_SETTINGS['background'], 'raw_ksb'])
            corrected_bootstrap_dwell_times = 1 / (np.array(bootstrapped_k_vals[condition]) - np.array(bootstrapped_k_vals[DECAY_SETTINGS['background']]))
            combined_info.at[condition, 'corrected_dwell_time_ci_lower'] = np.nanpercentile(corrected_bootstrap_dwell_times, 2.5)
            combined_info.at[condition, 'corrected_dwell_time_ci_upper'] = np.nanpercentile(corrected_bootstrap_dwell_times, 97.5)
        if condition != reference:
            ref_dist = bootstrapped_k_vals[reference]
            cond_dist = bootstrapped_k_vals[condition]
            diff_dist = np.array(ref_dist) - np.array(cond_dist)
            diff_dist = diff_dist[~np.isnan(diff_dist)]
            p_value = 2*np.min([(np.sum(diff_dist <= 0)) / (len(diff_dist)), (np.sum(diff_dist >= 0)) / (len(diff_dist))])
            if DECAY_SETTINGS['verbose']:
                print(condition)
                print(f"direct p val: {p_value}")
            se_ref = np.std(ref_dist)
            se_cond = np.std(cond_dist)
            combined_info.at[condition, 'SE'] = se_cond
            k_ref = combined_info.at[reference, 'raw_ksb']
            k_cond = combined_info.at[condition, 'raw_ksb']
            se_diff = np.sqrt(se_ref**2 + se_cond**2)
            t_stat = (k_ref - k_cond) / se_diff
            n_ref = len(combined_tracks[combined_tracks['condition'] == reference])
            n_cond = len(combined_tracks[combined_tracks['condition'] == condition])
            df = n_ref + n_cond - 2
            hybrid_p_value = t.sf(np.abs(t_stat), df) * 2
            if DECAY_SETTINGS['verbose']:
                print(f"ttest p val: {hybrid_p_value}\n")
    utils.pickle_save(ci=combined_info)
    return combined_info