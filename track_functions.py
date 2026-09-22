# Functions that do the heavy lifting for the tracking script
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import imagej
import os
import copy
import re
from java_imports import *


def get_activations(imp):
    # Detects activation frames on an image using FIJI (originally written for PIL and re-written with Gemini)
    # Requires the shutters to be set up so that 561 is off when 406 is on
    # Works by looking at brightness in a 10px border around the image. Yuhan suggested that we eventually try only looking at corners because nuclei *definitely* can't be there, so maybe that's a future step
    stack = imp.getStack()
    
    img_n_frames = stack.getSize()
    masked_img_contents = []
    cell_mm_brightness = []
    
    # img_dims order: (height, width) to match original nd2.imread shape[1:3]
    img_dims = (imp.getHeight(), imp.getWidth())

    # Create a mask to include only a 10px border around the image
    border_mask = np.zeros((img_dims[0], img_dims[1]), dtype="int64")
    # left 10 px
    border_mask[0:img_dims[0],0:10]=1
    # right 10 px
    border_mask[0:img_dims[0],img_dims[1]-10:img_dims[1]]=1
    # top 10 px
    border_mask[0:10,0:img_dims[1]]=1
    # bottom 10 px
    border_mask[img_dims[0]-10:img_dims[0],0:img_dims[1]]=1

    # Go through each frame and apply the mask to it. Then look at the top 1% of pixels in that border area
    # Note: FIJI stacks are 1-indexed
    for i in range(1, img_n_frames + 1):
        ip = stack.getProcessor(i)
        
        # Extract pixels to a 1D numpy array and reshape to 2D image dimensions
        img_flat = np.array(ip.getPixels())
        img = img_flat.reshape(img_dims)
        
        # Original logic: apply mask and find the 99th percentile
        masked_img_contents.append(border_mask*np.array(img, dtype="int16"))
        cell_mm_brightness.append(np.percentile(masked_img_contents[-1], 99))
        
    # We're basically doing a min-max thing here. Find frames where the highest pixels have the lowest value, filter for ones with several in a row, and treat them as activation frames.
    # Set a threshold to be .92 of the mean.
    threshold = 0.92 * np.mean(cell_mm_brightness)
    activation_frames = []
    current_under_threshold = []

    # For each frame... (this code both ensures that the frame is part of a series and thus not an outlier and also only returns one frame per multi-frame activation)
    for i in range(len(cell_mm_brightness)):
        # If brightness of the top 1% of masked pixels is under the threshold, add it to a working list
        if cell_mm_brightness[i] < threshold:
            current_under_threshold.append(i)
        # When past those, check if the working list has at least 2 consecutive hits, save it if it does, and clear the list
        else:
            if len(current_under_threshold) >= 1:
                activation_frames.append(i-1)
            current_under_threshold = []
            
    
    return activation_frames


def optimize_qual_threshold(ij, imp, SPFPA, init_detector_quality, DETECTOR_SETTINGS, DEBUG):
    activation_frames = get_activations(imp)
    num_pre_frames = activation_frames[0]
    target_spot_count = SPFPA * num_pre_frames * imp.getWidth() * imp.getHeight()
    
    min_qual = 0.0
    max_qual = 100.0 # Assuming threshold won't exceed 100; increase if needed
    working_qual = init_detector_quality
    best_qual = working_qual
    min_diff = float('inf')
    
    for i in range(15):
        model = Model()
        if DEBUG:
            model.setLogger(Logger.IJ_LOGGER)
        else:
            model.setLogger(Logger.VOID_LOGGER)
        settings = Settings(imp)
        settings.detectorFactory = LogDetectorFactory()
        settings.tstart = 0
        settings.tend = num_pre_frames
        
        DETECTOR_SETTINGS['THRESHOLD'] = JDouble(working_qual)
        java_detector_settings = ij.py.to_java(DETECTOR_SETTINGS)
        settings.detectorSettings = java_detector_settings
        
        trackmate = TrackMate(model, settings)
        trackmate.execDetection()
        trackmate.computeSpotFeatures(True)
        trackmate.execSpotFiltering(True)
        
        actual_spot_count = model.getSpots().getNSpots(True)
        diff = actual_spot_count - target_spot_count
        
        # Save the closest value we find
        if abs(diff) < min_diff:
            min_diff = abs(diff)
            best_qual = working_qual
            
        # Exit early if we are within a 2% margin
        if abs(diff) <= 0.02 * target_spot_count:
            break
            
        if actual_spot_count > target_spot_count:
            # Too many spots -> threshold is too low
            min_qual = working_qual
        else:
            # Too few spots -> threshold is too high
            max_qual = working_qual
            
        working_qual = (min_qual + max_qual) / 2.0
    return best_qual

def set_rois_from_file(ij, base_filepath, ROI_EXTENSION, ROI_GROW):
    roi_filename = base_filepath + ROI_EXTENSION
    if roi_filename.endswith('.npy'):
        seg_data = np.load(roi_filename, allow_pickle=True).item()
        label_mask = ij.py.to_imageplus(seg_data['masks'])
    else:
        label_mask = ij.IJ.openImage(roi_filename)

    if label_mask is None:
        # TODO improve error handling
        print(f"Missing label mask for {base_filepath}")
        
    rois = []
    max_roi_num = int(label_mask.getStatistics().max)
    
    for i in range(1, max_roi_num + 1):
        label_mask.getProcessor().setThreshold(i, i, ImageProcessor.NO_LUT_UPDATE)
        roi = ThresholdToSelection.run(label_mask)
        
        if roi is not None:
            roi = RoiEnlarger.enlarge(roi, ROI_GROW)
            roi.setName(f"ROI_{i}")
            rois.append(roi)
    label_mask.getProcessor().resetThreshold()
        
    # Free label mask memory
    label_mask.changes = False
    label_mask.close()
    return rois


# Code I changed a bit but got from Lyn, who I'm pretty sure made AI write it all
def combine_trackmate_fov(xml_paths, DEBUG):
    """
    Combine all TrackMate ROI XML files (using filenames in a list)
    """

    # ------------------------------------------------------------
    # First XML becomes the template/base
    # ------------------------------------------------------------

    tree = ET.parse(xml_paths[0])
    root = tree.getroot()

    model = root.find("Model")

    if model is None:
        raise RuntimeError("Could not find <Model>")

    all_spots = model.find("AllSpots")
    all_tracks = model.find("AllTracks")
    filtered_tracks = model.find("FilteredTracks")

    if all_spots is None:
        raise RuntimeError("Could not find <AllSpots>")

    if all_tracks is None:
        raise RuntimeError("Could not find <AllTracks>")

    # ------------------------------------------------------------
    # Determine next available IDs from the first file
    # ------------------------------------------------------------

    next_spot_id = 0

    for frame in all_spots.findall("SpotsInFrame"):
        for spot in frame.findall("Spot"):

            if "ID" in spot.attrib:
                next_spot_id = max(
                    next_spot_id,
                    int(spot.attrib["ID"]) + 1
                )

    next_track_id = 0

    for track in all_tracks.findall("Track"):

        if "TRACK_ID" in track.attrib:
            next_track_id = max(
                next_track_id,
                int(track.attrib["TRACK_ID"]) + 1
            )

    if DEBUG:
        print(f"Initial next spot ID:  {next_spot_id}")
        print(f"Initial next track ID: {next_track_id}")

    # ------------------------------------------------------------
    # Process each additional ROI
    # ------------------------------------------------------------

    for xml_file in xml_paths[1:]:

        if DEBUG:
            print(f"  Adding {xml_file}")

        roi_tree = ET.parse(xml_file)
        roi_root = roi_tree.getroot()

        roi_model = roi_root.find("Model")

        if roi_model is None:
            if DEBUG:
                print("    WARNING: no Model -- skipped")
            continue

        roi_spots = roi_model.find("AllSpots")
        roi_tracks = roi_model.find("AllTracks")
        roi_filtered_tracks = roi_model.find("FilteredTracks")

        if roi_spots is None or roi_tracks is None:
            if DEBUG:
                print("    WARNING: missing spots/tracks -- skipped")
            continue

        # ========================================================
        # SPOTS
        # ========================================================

        # Map:
        #
        # old spot ID -> new spot ID
        #
        spot_id_map = {}

        n_added_spots = 0

        for roi_frame in roi_spots.findall("SpotsInFrame"):

            frame_number = roi_frame.attrib["frame"]

            # Find the corresponding frame in the combined file
            combined_frame = None

            for frame in all_spots.findall("SpotsInFrame"):

                if frame.attrib["frame"] == frame_number:
                    combined_frame = frame
                    break

            # If this frame doesn't exist yet, create it
            if combined_frame is None:

                combined_frame = ET.SubElement(
                    all_spots,
                    "SpotsInFrame",
                    {"frame": frame_number}
                )

            # Copy every spot
            for old_spot in roi_frame.findall("Spot"):

                old_id = int(old_spot.attrib["ID"])

                new_id = next_spot_id
                next_spot_id += 1

                spot_id_map[old_id] = new_id

                new_spot = copy.deepcopy(old_spot)

                new_spot.set("ID", str(new_id))

                combined_frame.append(new_spot)

                n_added_spots += 1

        if DEBUG:
            print(f"    Added {n_added_spots} spots")

        # ========================================================
        # TRACKS
        # ========================================================

        # Map:
        #
        # old TRACK_ID -> new TRACK_ID
        #
        track_id_map = {}

        n_added_tracks = 0

        for old_track in roi_tracks.findall("Track"):

            old_track_id = int(
                old_track.attrib["TRACK_ID"]
            )

            new_track_id = next_track_id
            next_track_id += 1

            track_id_map[old_track_id] = new_track_id

            new_track = copy.deepcopy(old_track)

            # Update track ID
            new_track.set(
                "TRACK_ID",
                str(new_track_id)
            )

            # Track index should also be unique
            if "TRACK_INDEX" in new_track.attrib:
                new_track.set(
                    "TRACK_INDEX",
                    str(new_track_id)
                )

            # ----------------------------------------------------
            # Update every edge's spot IDs
            # ----------------------------------------------------

            for edge in new_track.findall("Edge"):

                source_id = int(
                    edge.attrib["SPOT_SOURCE_ID"]
                )

                target_id = int(
                    edge.attrib["SPOT_TARGET_ID"]
                )

                if source_id not in spot_id_map:
                    raise RuntimeError(
                        f"Could not remap source spot "
                        f"{source_id} in {xml_file}"
                    )

                if target_id not in spot_id_map:
                    raise RuntimeError(
                        f"Could not remap target spot "
                        f"{target_id} in {xml_file}"
                    )

                edge.set(
                    "SPOT_SOURCE_ID",
                    str(spot_id_map[source_id])
                )

                edge.set(
                    "SPOT_TARGET_ID",
                    str(spot_id_map[target_id])
                )

            all_tracks.append(new_track)

            n_added_tracks += 1

        if DEBUG:
            print(f"    Added {n_added_tracks} tracks")

        # ========================================================
        # FILTERED TRACKS
        # ========================================================

        if filtered_tracks is not None and roi_filtered_tracks is not None:

            for old_track_id_element in roi_filtered_tracks.findall("TrackID"):

                old_track_id = int(
                    old_track_id_element.attrib["TRACK_ID"]
                )

                if old_track_id in track_id_map:

                    new_track_id_element = ET.Element(
                        "TrackID",
                        {
                            "TRACK_ID":
                            str(track_id_map[old_track_id])
                        }
                    )

                    filtered_tracks.append(
                        new_track_id_element
                    )

    # ============================================================
    # Update AllSpots nspots attribute
    # ============================================================

    total_spots = sum(
        len(frame.findall("Spot"))
        for frame in all_spots.findall("SpotsInFrame")
    )

    all_spots.set("nspots", str(total_spots))

    # ============================================================
    # Save and elete previous xmls
    # ============================================================

    first_xml_path = xml_paths[0]
    output_file = re.sub(r'_?ROI-\d+', '', first_xml_path)

    # Pretty-print XML if using Python >= 3.9
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(
        output_file,
        encoding="UTF-8",
        xml_declaration=True
    )

    for fpath in xml_paths:
        os.remove(fpath)

    if DEBUG:
        print("\nDone!")
        print(f"Total spots:  {total_spots}")
        print(f"Total tracks: {len(all_tracks.findall('Track'))}")
        print(f"Saved to:\n{output_file}")

    return output_file