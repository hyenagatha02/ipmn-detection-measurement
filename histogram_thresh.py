import sys
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
# # from curve_fit_types import try_gaussian_fit, try_fallback_peak_detection, kde_fit, gamma_fit, try_truncated_gaussian, try_skewed_gaussian
# from curve_fit_types import try_gaussian_fit, try_fallback_peak_detection, kde_fit, try_truncated_gaussian, try_gmm_fitting
# from curve_fit_types import gamma_fit_loc_is_0, gamma_fit_trimmed_loc_is_None
# from curve_fit_types import try_skewed_gaussian_init_mean, try_skewed_gaussian_init_argmax, skewed_gaussian

# Full Width at Half Maximum (FWHM)
import scipy.stats as stats
import scipy.optimize as opt
from scipy.interpolate import interp1d


label_colors = {
    'bduct' : 'tab:orange',     # 'orange',
    'pduct' : 'tab:green',      # 'green',
    'pancreas' : 'tab:blue',    # 'blue',
    'combinedDuct' : 'tab:cyan', # 'tab:red',     # 'red',
}


def plot_histograms(file_id, file_paths, labels, stats):
    """
    Generate and save histograms (individual or overlapped) with optional Gaussian fitting.

    Parameters:
    file_paths (list of str): List of CSV file paths with 'Intensity' and 'Frequency' columns.
    labels (list of str): List of labels for the data (e.g., ['Pancreas', 'Bile Duct', 'Pancreatic Duct']).
    stats (dict): Precomputed mean, median, and std for each label.
    file_id (str): Base ID for output files.
    gaussian_fit (bool): Whether to fit and overlay Gaussian curves on the data.
    """

    plt.figure(figsize=(10, 6))

    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, linewidth=0.7, alpha=0.7)
    # plt.legend(loc='upper right', prop={'size': 8})

    print()
    for file_path, label in zip(file_paths, labels):
        print(f"(Processing: [{label}] {file_path})")
        if not os.path.isfile(file_path):
            print(f"\033[1m ! File not found: {file_path} — Skipping {label} \033[0m")
            continue
        if label not in stats:
            print(f"\033[1m ! Label '{label}' not found in stats. Skipping! \033[0m")
            continue
        
        # # Load the data
        # data = pd.read_csv(file_path)
        ## works for case with empty mask
        try:
            data = pd.read_csv(file_path)
            if data.empty:
                print(f"\033[1m ! CSV is empty: {file_path} — Skipping {label} \033[0m")
                continue
        except Exception as e:
            print(f"\033[1m ! Failed to read CSV: {file_path}, error: {e} — Skipping \033[0m")
            continue
        
        xvals = data['Intensity'].values
        yvals = data['Frequency'].values

        mean_value = stats[label]['mean']
        median_value = stats[label]['median']
        std_value = stats[label]['std']

        plot_line, = plt.plot(xvals, yvals, 'o-', label=label, linewidth=0.92, markersize=2, color=label_colors[label])
        plt.axvline(mean_value, color=plot_line.get_color(), linestyle='--', linewidth=1.0, label=f"{label} Mean: {mean_value:.2f}")
        plt.axvline(median_value, color=plot_line.get_color(), linestyle=':', linewidth=0.9, label=f"{label} Median: {median_value:.2f}")


    # print(f'\n\033[1m* Save plot {labels} *\033[0m')
    if len(file_paths) > 1:
        plt.title('All masks Overlapped Intensity Histograms')
        # plot_name = f"{file_paths[0].replace(labels[0], f'all_masks_{len(file_paths)}').replace('csv', 'png')}"
        plot_name = f"{file_paths[0].replace(labels[0], f'all_mask').replace('.csv', '.png')}"
        # if len(file_paths) > 3 and len(labels) > 3:
        if len(file_paths) > 1 and len(labels) > 1:   ## works for case with empty mask
            thresholds = {}
            lower_bound = stats['pancreas']['mean']
            upper_bound = stats['combinedDuct']['mean']

            '''
            T = lower + (p/100) * (upper - lower)
            '''
            for p in [30, 40, 50, 60, 70]:
                p_threshold = lower_bound + (p / 100) * (upper_bound - lower_bound)
                # thresholds[f"point_{p}"] = p_threshold
                if p in [50]:
                    thresholds[f"point_{p}"] = p_threshold
                #     # plt.axvline(percentile_threshold, color='red', linestyle='-.', linewidth=0.5, label=f"Percentile={p:.2f}: {p_threshold:.2f}")
                #     # plt.text(p_threshold, plt.ylim()[1] * 0.85, f"{p:.1f}th", color='red', fontsize=8, ha='center', va='bottom', rotation=90)
                # else:
                #     plt.axvline(p_threshold, color='gray', linestyle='-.', linewidth=0.5, label=f"Percentile={p:.2f}: {p_threshold:.2f}")
                #     plt.text(p_threshold, plt.ylim()[1] * 0.85, f"{p:.1f}th", color='gray', fontsize=8, ha='center', va='bottom', rotation=90)

            # print('\n- all thresholds')
            # for tname, threshold in thresholds.items():
            #     print(  tname, threshold, type(threshold))

            average_means = (lower_bound + upper_bound) / 2
            assert math.isclose(thresholds["point_50"], average_means, rel_tol=1e-9), print(f"Assertion failed! (point_50: {thresholds['point_50']}, average_means: {average_means})")

            plt.axvline(average_means, color='red', linestyle='-.', linewidth=0.85, label=f"average of mean: {average_means:.2f}")
            plt.text(average_means, plt.ylim()[1] * 0.25, f"average of pacnreas mean & combinedDuct mean", color='red', fontsize=8, ha='center', va='bottom', rotation=90)

            # save into csv
            df_thresholds = pd.DataFrame(list(thresholds.items()), columns=['Threshold Name', 'Value'])
            # threshold_csv_name = f"{file_paths[2].replace('Histogram', 'threshold')}"
            # threshold_csv_name = f"{file_paths[2].replace('.csv', '_threshold.csv')}"
            threshold_csv_name = os.path.join(os.path.dirname(file_paths[2]), f"{file_id}_t2_threshold.csv")
            df_thresholds.to_csv(threshold_csv_name, index=False)
            print(f"\033[36m-- Saved threshold to: {threshold_csv_name}\033[0m")

    else:
        plt.title(f'Intensity Histogram {labels[0]}')
        # plot_name = os.path.splitext(file_paths[0])[0]
        plot_name = file_paths[0].replace('csv', 'png')
    
    plt.legend(loc='upper right', prop={'size': 8})
    plt.savefig(plot_name)
    print(f"\033[36m-- Saved plot to: {plot_name}\033[0m")

    plt.close()
    

def main(file_id, file_paths, means, medians, stds):
    # print("\033[1m ## Running gaussian_curve_fit_final.py ## \033[0m")

    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    # file_id = os.path.splitext(file_paths[0])[0]
    # file_id = os.path.basename(file_paths[0]).split('_')[0]
    # file_id = os.path.basename(os.path.dirname(file_paths[0]))

    # file_labels = stats_labels[:len(file_paths)]
    # file_labels = [os.path.basename(file_path).split('_')[2] for file_path in file_paths]
    file_labels = []
    for file_path in file_paths:
        if 'bduct' in os.path.basename(file_path): 
            file_labels.append('bduct')
        if 'pduct' in os.path.basename(file_path): 
            file_labels.append('pduct')
        if 'pancreas' in os.path.basename(file_path): 
            file_labels.append('pancreas')
        if 'combinedDuct' in os.path.basename(file_path): 
            file_labels.append('combinedDuct') 

    # if len(file_paths) != len(means) or len(file_paths) != len(medians) or len(file_paths) != len(stds):
    #     # raise ValueError("Number of file paths and statistics (mean, median, std) must match!")
    #     # stats_labels = ["bduct", "pduct", "pancreas"]
    #     stats_labels = ["bduct", "pduct", "pancreas", "combinedDuct"]
    # else:
    #     stats_labels = file_labels
    ## works for case with empty mask
    stats_labels = []
    filtered_means, filtered_medians, filtered_stds = [], [], []

    for label, mean, median, std in zip(file_labels, means, medians, stds):
        if label in file_labels:
            stats_labels.append(label)
            filtered_means.append(mean)
            filtered_medians.append(median)
            filtered_stds.append(std)

    # stats = {}
    # for label, mean, median, std in zip(stats_labels, means, medians, stds):
    #     stats[label] = {
    #         'mean': float(mean),
    #         'median': float(median),
    #         'std': float(std)
    #     }
    ## works for case with empty mask
    stats = {}
    for label, mean, median, std in zip(stats_labels, filtered_means, filtered_medians, filtered_stds):
        stats[label] = {
            'mean': float(mean),
            'median': float(median),
            'std': float(std)
        }

    print(f'- file_id : {file_id}')
    print(f'- file_paths : {file_paths}')
    print(f'- stats_labels : {stats_labels}')
    print(f'- file_labels : {file_labels}')
    print(f'- stats : ')
    for label, stat in stats.items():
        print(f"  {label}: Mean={stat['mean']:.2f}, Median={stat['median']:.2f}, Std={stat['std']:.2f}")

    plot_histograms(file_id, file_paths, file_labels, stats)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate single or overlapped histogram plots from CSV files.")
    parser.add_argument("--file_id", type=str, required=True, help="ID of processing files.")
    parser.add_argument("--file_paths", type=str, nargs='+', help="List of histogram CSV file paths.")
    parser.add_argument("--means", type=str, nargs='+', help="Mean values for each mask.")
    parser.add_argument("--medians", type=str, nargs='+', help="Median values for each mask.")
    parser.add_argument("--stds", type=str, nargs='+', help="Standard deviation values for each mask.")

    args = parser.parse_args()

    main(args.file_id, args.file_paths, means=args.means, medians=args.medians, stds=args.stds)
