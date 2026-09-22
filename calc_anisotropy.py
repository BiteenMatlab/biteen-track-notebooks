"""
Written by Sam, Spring 2025
"""
import numpy as np
import utils
import pandas as pd
import pickle



def get_condition_angles(combined_tracks, condition, settings):
    """ Utility function to return angles over min D val for a certain condition"""
    plotable_data = combined_tracks[combined_tracks['MSD'] > settings['min_D']]
    plotable_data = plotable_data[plotable_data['MSD'] < settings['max_D']]
    condition_data = plotable_data[plotable_data['condition'] == condition]
    return condition_data['angle'].sum()


def calc_fold_anisotropy(combined_info, combined_tracks, settings):
    """Calculates fold anisotropy, or the bias towards going backwards as opposed to forwards"""
    combined_info['fold_anisotropy'] = np.zeros
    combined_info['fold_anisotropy_err'] = np.zeros
    rng = np.random.default_rng()
    # For each condition
    for index, row in combined_info.iterrows():
        # Pull angle to look at from configuration information
        ang = settings['angle_from_center']
        # Get usable angles for this condition
        angles = get_condition_angles(combined_tracks, index, settings)
        # Find the number of angles centered around 0 and 180 degrees
        n_around_0 = len(list(filter(lambda a: 0-ang < a < 0+ang, angles)))
        n_around_180 = len(list(filter(lambda a: a > 180-ang or a < -180+ang, angles)))
        # If both exist (to avoid errors from dividing by 0)
        if n_around_0 and n_around_180:
            combined_info.at[index, 'fold_anisotropy'] = n_around_180 / n_around_0
        # Realized this is the one thing we need an error for
        fold_anisotropies = []
        for b in range(settings['bootstrap_n']):
                bootstrap_angles = rng.choice(angles, size=len(angles), replace=True)
                # I love python data types
                n_near_180 = sum(bootstrap_angles > 150)
                n_near_0 = sum(bootstrap_angles < 30)
                # print(f"180: {n_near_180}, 0: {n_near_0}")
                if n_near_0 > 0:
                    fold_anisotropies.append(n_near_180 / n_near_0)
        combined_info.at[index, 'fold_anisotropy_err'] = np.percentile(fold_anisotropies, [2.5,97.5])
    utils.pickle_save(ci=combined_info)
    return combined_info


def calc_angle(tracks, include_na, conditions):
    """find anisotropy for each set of 3 steps"""
    # Set up the dataframe to have exactly 1 angle column and 1 mean_displacement column
    # Would be easier in Pandas, but the original code uses numpy
    if 'displacement' in list(tracks.columns):
        tracks = tracks.drop(columns=['displacement'])
    tracks['displacement'] = np.empty((len(tracks), 0)).tolist()
    if 'angle' in list(tracks.columns):
        tracks = tracks.drop(columns=['angle'])
    tracks['angle'] = np.empty((len(tracks), 0)).tolist()
    if 'mean_displacement' in list(tracks.columns):
        tracks = tracks.drop(columns=['mean_displacement'])
    tracks['mean_displacement'] = np.empty((len(tracks), 0)).tolist()

    # For each track in the dataframe
    for i in range(len(tracks)):
        row = tracks.iloc[i]
        frames = row['frame']
        x_vals = row['x']
        y_vals = row['y']
        displacements = []
        angles = []
        mean_displacements = []

        # For each step in the track
        for t in range(len(frames)):
            # find places where there are 3 continuous steps.
            if frames[t] == frames[t-1]+1:
                first_deltax = x_vals[t-1] - x_vals[t-2]
                second_deltax = x_vals[t] - x_vals[t-1]
                first_deltay = y_vals[t-1] - y_vals[t-2]
                second_deltay = y_vals[t] - y_vals[t-1]
                first_displacement = conditions['pixel_size_um'] * (first_deltax**2 + first_deltay**2)**.5
                displacements.append(first_displacement)
            if (frames[t] == frames[t-1]+1 and
                frames[t] == frames[t-2]+2):
                # calculate angle and mean displacement and append to the dataframe
                # Pythagorian
                mean_displacement = conditions['pixel_size_um'] * 0.5 * ((first_deltax**2 + first_deltay**2)**.5 + (second_deltax**2 + second_deltay**2)**.5)
                first_displacement = conditions['pixel_size_um'] * (first_deltax**2 + first_deltay**2)**.5
                # New angle calculation method
                A = [first_deltax, first_deltay]
                B = [second_deltax, second_deltay]
                # Some math I got Ishika's help with because I thought I wanted to be a biologist when I grew up
                # %theta = arccos( frac {X cdot Y} {X times Y} )
                angle = np.degrees(np.arccos(np.dot(A, B)/((A[0]**2 + A[1]**2)**.5 * (B[0]**2 + B[1]**2)**.5)))
                angles.append(angle)
                mean_displacements.append(mean_displacement)
            elif include_na:
                angle = np.nan
                mean_displacement = np.nan
                angles.append(angle)
                mean_displacements.append(mean_displacement)
        
        # Now append the values we calculated
        tracks.iat[i, len(tracks.columns)-3] = displacements
        tracks.iat[i, len(tracks.columns)-2] = angles
        tracks.iat[i, len(tracks.columns)-1] = mean_displacements
    utils.pickle_save(ct=tracks)
    return tracks



def get_anisotropy_by_displacement(combined_tracks, combined_info, ANISOTROPY_SETTINGS):
    step_size = (ANISOTROPY_SETTINGS['usable_range'][1]-ANISOTROPY_SETTINGS['usable_range'][0])/ANISOTROPY_SETTINGS['disp_n_bins']
    bin_edges = list(np.arange(ANISOTROPY_SETTINGS['usable_range'][0], ANISOTROPY_SETTINGS['usable_range'][1]+step_size, step_size))
    bin_middles = list(np.arange(ANISOTROPY_SETTINGS['usable_range'][0]+.5*step_size, ANISOTROPY_SETTINGS['usable_range'][1]+1.5*step_size, step_size))
    plotable_data = combined_tracks[combined_tracks['MSD'] > ANISOTROPY_SETTINGS['min_D']]
    plotable_data = plotable_data[plotable_data['MSD'] < ANISOTROPY_SETTINGS['max_D']]
    rng = np.random.default_rng()
    combined_info['anisotropy_by_displacement'] = None
    for r in range(len(combined_info)):
        condition = combined_info.index[r]
        # Define a new frame for the condition with displacement (centered on each bin), anisotropy, and error
        data_for_graph = pd.DataFrame(data={'displacement': bin_middles[:-1], 'anisotropy': np.zeros(len(bin_edges)-1), 'anisotropy_err': np.zeros(len(bin_edges)-1)})
        condition_data = plotable_data[plotable_data['condition'] == condition]
        # print(len(condition_data))
        # Filter and get lists of all angles and mean displacement (sum() is a nice workaround to combine all lists into one big list)
        combined_angles = condition_data['angle'].sum()
        combined_md = condition_data['mean_displacement'].sum()
        # Now pull data into a second frame with 2 columns: displacement and angle
        angle_summary = pd.DataFrame(data={'mean_displacement': combined_md, 'angles': combined_angles})
        # For each displacement bin
        for i in range(len(bin_edges)-1):
            # Filter the data
            range_data = angle_summary[angle_summary['mean_displacement'] > bin_edges[i]]
            range_data = range_data[range_data['mean_displacement'] < bin_edges[i+1]]
            # print(len(range_data))

            # Remove bins with very small amounts of data
            # if len(range_data) < 50:
            #     data_for_graph.at[i, 'anisotropy'] = 0
            #     data_for_graph.at[i, 'anisotropy_err'] = 0
            #     continue
            fold_anisotropies = []
            # Now do bootstrapping to generate error bars. Idea here: find fold anisotropy lots of times, and take the outer 2.5% to get a 05% CI
            for b in range(ANISOTROPY_SETTINGS['bootstrap_n']):
                bootstrap_angles = rng.choice(range_data['angles'], size=len(range_data), replace=True)
                # I love python data types
                n_near_180 = sum(bootstrap_angles > 150)
                n_near_0 = sum(bootstrap_angles < 30)
                # print(f"180: {n_near_180}, 0: {n_near_0}")
                if n_near_0 > 0:
                    fold_anisotropies.append(n_near_180 / n_near_0)
            
            # Find mean and CI95 by taking mean and tails of the bootstrap distribution
            mean = np.mean(fold_anisotropies)
            ci_95 = np.percentile(fold_anisotropies, [2.5,97.5])
            data_for_graph.at[i, 'anisotropy'] = mean
            data_for_graph.at[i, 'anisotropy_err'] = mean - ci_95[0]
        combined_info.at[condition, 'anisotropy_by_displacement'] = data_for_graph
    utils.pickle_save(ci=combined_info)
    return combined_info

def summarize_midrange_anisotropy(combined_tracks, combined_info, ANISOTROPY_SETTINGS):
    # Filter data to only look at ones in the proper range of D-values
    plotable_data = combined_tracks[combined_tracks['MSD'].between(ANISOTROPY_SETTINGS['min_D'], ANISOTROPY_SETTINGS['max_D'])]
    # Establish a reference condition (simply the first one) against which to calculate p-values
    ref_condition_data = plotable_data[plotable_data['condition'] == combined_info.index[0]]
    # Want a dataframe with 2 columns: displacement, angle. Taking advantage of the fact that summing these will concatenate them (so NOT adding integers)
    ref_angle_summary = pd.DataFrame(data={'mean_displacement': ref_condition_data['mean_displacement'].sum(), 'angles': ref_condition_data['angle'].sum()})
    # Now a second layer of filters: only look at a certain range of mean displacements
    ref_angle_summary = ref_angle_summary[ref_angle_summary['mean_displacement'].between(ANISOTROPY_SETTINGS['min_mean_displacement'], ANISOTROPY_SETTINGS['max_mean_displacement'])]
    # Create three new blank columns in combined_info in which to store results
    combined_info['intermediate_disp_fold_anisotropy'] = np.zeros
    combined_info['anisotropy_err'] = np.zeros
    combined_info['p-val_against_reference'] = np.zeros
    # Configure a random number generator
    rng = np.random.default_rng()

    # For each condition (including the first... so it compares the reference condition to itself)
    for r in range(len(combined_info)):
        # Do the same filtering, but this time for whatever condition we're actively working on
        condition = combined_info.index[r]

        condition_data = plotable_data[plotable_data['condition'] == condition]
        angle_summary = pd.DataFrame(data={'mean_displacement': condition_data['mean_displacement'].sum(), 'angles': condition_data['angle'].sum()})
        angle_summary = angle_summary[angle_summary['mean_displacement'].between(ANISOTROPY_SETTINGS['min_mean_displacement'], ANISOTROPY_SETTINGS['max_mean_displacement'])]
        
        # Merge angles from the reference and working condition (see comment by permutation loop for explanation)
        joint_angles = pd.concat([ref_angle_summary['angles'], angle_summary['angles']])
        perm_diffs = []
        fold_anisotropies = []
        
        """Bootstrap loop
        Theory: We resample the data many times, calculating fold anisotropy for each one.
        This produces a bootstrap distribution. The mean of the bootstrap distribution
        is roughly the population mean (if our assumptions are met), and we can take the
        of the distribution to get a confidence interval for our graphs.
        """
        for b in range(ANISOTROPY_SETTINGS['bootstrap_n']):
            # size MUST be kept at len(angle_summary) or less (for subsampling)!
            bootstrap_angles = rng.choice(angle_summary['angles'], size=len(angle_summary), replace=True)
            # I ❤ python data types
            n_near_180 = sum(bootstrap_angles > 150)
            n_near_0 = sum(bootstrap_angles < 30)
            if n_near_0 > 0:
                fold_anisotropies.append(n_near_180 / n_near_0)
        mean = np.mean(fold_anisotropies)
        ci_95 = np.percentile(fold_anisotropies, [2.5,97.5])
        combined_info.at[condition, 'intermediate_disp_fold_anisotropy'] = mean
        combined_info.at[condition, 'anisotropy_err'] = mean - ci_95[0]
        
        """Permutation loop
        Theory: The null hypothesis for this test is that all fold anisotropies came from the
        same distribution. We test this by merging all the angles into one big dataset and then
        randomly assigning each point in that set to be either reference or comparison. We 
        calculate the difference between means for each of those randomized groups, and the 
        p-value is the proportion of means greater than the difference calculated between real datasets.
        """
        for i in range(ANISOTROPY_SETTINGS['permutation_n']):
            shuffled_angles = np.random.permutation(joint_angles)
            # lens should both be ref_angle_summary
            ref_angles = shuffled_angles[:len(ref_angle_summary)]
            cond_angles = shuffled_angles[len(ref_angle_summary):]
            n_near_180 = sum(ref_angles > 150)
            n_near_0 = sum(ref_angles < 30)
            ref_anisotropy = n_near_180 / n_near_0
            n_near_180 = sum(cond_angles > 150)
            n_near_0 = sum(cond_angles < 30)
            cond_anisotropy = n_near_180 / n_near_0
            perm_diffs.append(ref_anisotropy - cond_anisotropy)
        observed_diff = combined_info.at[combined_info.index[0], 'intermediate_disp_fold_anisotropy'] - combined_info.at[condition, 'intermediate_disp_fold_anisotropy']
        count_extreme = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff))
        p_value = count_extreme / len(perm_diffs)
        combined_info.at[condition, 'p-val_against_reference'] = p_value
        print(f"{combined_info.index[0]} vs {condition} p-val: {p_value}")
    utils.pickle_save(ci=combined_info)
    return combined_info