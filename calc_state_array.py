from saspt import StateArray
import numpy as np
import utils

def calc_state_array(combined_info, combined_tracks, rbme_likelihood, SASPT_SETTINGS):
    # Need to "pivot longer" before using combined_tracks
    combined_tracks_long = combined_tracks.explode(['x', 'y', 'frame'])
    combined_info['SA'] = np.zeros
    print("The following warnings are from State Arrays itself... don't worry too much about them")
    for index, row in combined_info.iterrows():
        condition_data = combined_tracks_long[combined_tracks_long['condition'] == index]
        SA = StateArray.from_detections(condition_data[['x', 'y', 'trajectory', 'frame']], **SASPT_SETTINGS)
        SA.likelihood = rbme_likelihood
        SA.posterior_assignment_probabilities;
        combined_info.at[index, 'SA'] = SA
    utils.pickle_save(ci=combined_info)
    return combined_info