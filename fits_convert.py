import h5py
import glob
import numpy as np
import pandas as pd

def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

def convert_fits_folder(directory, directory_condition, excluded_files):
    # foldername that contains all of the fits.mat files
    directory = directory + "/*fits.mat"
    # pull in all of the fits.mat files
    mat_files = filepull(directory)

    # for each fits.mat file
    tracks_name = ['frame','y','x','trajectory','ROI_NUM','MOL_ID', 'condition', 'file']
    combined_tracks_df = pd.DataFrame(columns = tracks_name)
    for file in mat_files:
        if file not in excluded_files:
            # read the .mat file
            with h5py.File(file, 'r') as f:
                    if excluded_files:
                        print(file)
                    # assign tracks array to var
                    tracks = f['tracks'][:]
                    tracks = np.concatenate([tracks, np.full((1,tracks.shape[1]), directory_condition), np.full((1,tracks.shape[1]), file)])
            
            # transpose, convert, and save tracks data
            if tracks.shape == (2,):
                tracks = np.zeros((6,1))
            tracks_df = pd.DataFrame(tracks.T, columns=tracks_name)
            tracks_df['frame'] = tracks_df['frame'].astype(float)
            tracks_df['x'] = tracks_df['x'].astype(float)
            tracks_df['y'] = tracks_df['y'].astype(float)
            tracks_df = tracks_df.groupby('trajectory').agg({
                'frame': list,
                'y': list,
                'x': list,
                'ROI_NUM': 'first',
                'MOL_ID': list,
                'trajectory': 'first',
                'condition': 'first',
                'file': 'first'
                })
            combined_tracks_df = pd.concat([combined_tracks_df, tracks_df])
    return combined_tracks_df

def batch_convert_all_folders(data_directories, excluded_files):
    # First generate csvs from matlab files and make experiment condition list
    combined_tracks = pd.DataFrame()
    for dirs in data_directories.keys():
        tracks = convert_fits_folder(dirs, data_directories[dirs], excluded_files)
        if len(tracks) == 0:
            raise RuntimeError(f"No tracks found for condition {data_directories[dirs]}. Check that the file is formatted correctly and not empty.")
        combined_tracks = pd.concat([combined_tracks, tracks])
    conditions = list(dict.fromkeys(data_directories.values()))
    combined_tracks['condition'] = pd.Categorical(combined_tracks.condition, categories=conditions, ordered=True)
    combined_tracks = combined_tracks.sort_values('condition').reset_index()
    combined_tracks = combined_tracks.reset_index(drop=True)
    combined_tracks['trajectory'] = combined_tracks.index
    return combined_tracks

def gen_human_readable_csv(combined_info):
    human_readable_combined_info = combined_info[["gmm_n_comps", "gmm_means", "gmm_variances", "gmm_weights", "fold_anisotropy"]]
    for r in range(len(human_readable_combined_info)):
        condition = human_readable_combined_info.index[r]
        gmm_means_old = human_readable_combined_info.at[condition, "gmm_means"]
        gmm_mean_errors = combined_info.at[condition, "gmm_err_NOTLOG"]
        gmm_means_new = []
        gmm_weights_old = human_readable_combined_info.at[condition, "gmm_weights"]
        gmm_weight_errors = combined_info.at[condition, "gmm_weight_err"]
        gmm_weights_new = []
        fold_anisotropy_old = human_readable_combined_info.at[condition, "fold_anisotropy"]
        fold_anisotropy_errors = combined_info.at[condition, "fold_anisotropy_err"]
        fold_anisotropy_new = f"{fold_anisotropy_old:.3g}±{fold_anisotropy_errors[1]-fold_anisotropy_old:.2g}"
        for m in range(len(gmm_means_old)):
            gmm_means_new.append(f"{10 ** gmm_means_old[m]:.3g}±{gmm_mean_errors[m][1]-(10 ** gmm_means_old[m]):.2g}")
            gmm_weights_new.append(f"{gmm_weights_old[m]:.3g}±{gmm_weight_errors[m][1]-gmm_weights_old[m]:.2g}")
        # Properly "GMM mean D (um^2/s)" now
        human_readable_combined_info.at[condition, "gmm_means"] = gmm_means_new
        human_readable_combined_info.at[condition, "gmm_weights"] = gmm_weights_new
        human_readable_combined_info.at[condition, "fold_anisotropy"] = fold_anisotropy_new
    return human_readable_combined_info