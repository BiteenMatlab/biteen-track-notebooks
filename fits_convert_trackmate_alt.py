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
    directory = directory + "/*tracks.csv"
    # pull in all of the fits.mat files
    csv_files = filepull(directory)

    # for each fits.mat file
    tracks_name = ['frame','y','x','track_num','roi_num', 'condition', 'file']
    combined_tracks_df = pd.DataFrame(columns = tracks_name)
    for file in csv_files:
        if file not in excluded_files:
            tracks_df = pd.read_csv(file)
            if excluded_files:
                print(file)
            tracks_df['condition'] = directory_condition
            tracks_df['file'] = file
            tracks_df['frame'] = tracks_df['frame'].astype(float)
            tracks_df['x'] = tracks_df['x'].astype(float)
            tracks_df['y'] = tracks_df['y'].astype(float)

            tracks_df.sort_values(by="frame", inplace=True)

            # Issue with trackmate tracks from my script: track_num rolls over going from ROI 1 to ROI 2
            # A few lines of AI code here:
            maxes = tracks_df.groupby('roi_num')['track_num'].max()+1
            shifts = maxes.cumsum().shift(1).fillna(0)
            tracks_df['track_num'] += tracks_df['roi_num'].map(shifts)
            
            tracks_df = tracks_df.groupby('track_num').agg({
                'frame': list,
                'y': list,
                'x': list,
                'roi_num': 'first',
                'track_num': 'first',
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