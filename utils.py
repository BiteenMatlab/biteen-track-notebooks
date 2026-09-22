import pickle

def pickle_save(ct=False, ci=False, gi=False):
    if ct is not False:
        with open('combined_tracks.pickle', 'wb') as handle:
            pickle.dump(ct, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if ci is not False:
        with open('combined_info.pickle', 'wb') as handle:
            pickle.dump(ci, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if gi is not False:
        with open('general_info.pickle', 'wb') as handle:
            pickle.dump(gi, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load():
    with open('combined_tracks.pickle', 'rb') as handle:
        ct = pickle.load(handle)
    with open('combined_info.pickle', 'rb') as handle:
        ci = pickle.load(handle)
    with open('general_info.pickle', 'rb') as handle:
        gi = pickle.load(handle)
    return ct, ci, gi


def init_graph_colors(colors_in, combined_info):
    colors = [] if type(colors_in) != list else colors_in
    if type(colors_in) == dict:
        for condition in combined_info.index:
            if condition in colors_in:
                colors.append(colors_in[condition])
            else:
                colors.append("gray") # default color if not specified
    elif type(colors_in) == str:
        colors = [colors_in] * len(combined_info)
    return colors


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