# coding: utf-8

import argparse
import sys
import numpy as np
import random
from os import listdir
from os.path import isfile, join
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# CONSTANTS
DATA_DIR = 'data/'
MAX_LENGTH = 10000 # Max length of one file (note: some are longer than 10K!)
N_SITES = 95
N_INSTANCES = 4  # number of instances per site

# Threshold Constants 
# the range of thresholds is 0 to 1, and divide these values by MAX_THRESHOLD
MIN_THRESHOLD = 900
MAX_THRESHOLD = 1000
STEP = 1

orig_times = dict()
gen_times = dict()

# 3 arrays to store correlation values, 3 categories of correlations to track
# 1: same instance
same_instance_corrs = list()
# 2: same site, different instance
same_site_corrs = list()
# 3: different site and instance
different_corrs = list()


def read_times():
    ######################################################################################
    #                                                                                    #
    # 1. Open up the files in the data directory. File names are in the form             #
    #    <site>-<instance>, where <site> is the website the user went to, and <instance> #
    #    is the count.                                                                   #
    #                                                                                    #
    # 2. Read in the original times from the file and save them into the orig_times      #
    #    dictionary (one entry per file)                                                 #
    #                                                                                    #
    # 3. For each time, add some random delay based on the parameter a to create the     #
    #    generated times. Save these into the gen_times dictionary (one entry per file). #  
    #                                                                                    #
    # 4. For both orig and gen entries, make sure that all are filled out to MAX_LENGTH  #
    #    number of time values with "0.0".                                               #  
    #                                                                                    #
    ######################################################################################
    data_files = [f for f in listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
    for data_file in data_files:
        
        # LIMIT FOR TESTING
        if int(data_file.split("-")[1]) > N_INSTANCES:
            continue

        orig_times[data_file] = list()
        gen_times[data_file] = list()

        with open(join(DATA_DIR, data_file)) as f:
            random_value = random.uniform(1, RANDOM_PARAMETER)
            for _ in range(MAX_LENGTH):
                try:
                    line = next(f)
                    packet_time = float(line.split()[0]) 
                    #generated_time = packet_time + random.uniform(1, RANDOM_PARAMETER)
                    generated_time = packet_time + random_value

                except StopIteration:
                    packet_time = 0.0
                    generated_time = 0.0
                
                orig_times[data_file].append(packet_time)
                gen_times[data_file].append(generated_time)


def compute_correlation():
    ######################################################################################
    #                                                                                    #
    # 1. Open up pairs of entries, one each from orig_times and gen_times.               #
    #                                                                                    #
    # 2. Compute the statistical correlation between the pair of times.                  #
    #    Note: both time arrays must be the same length!                                 #
    #                                                                                    #
    # 3. Save the correlation values into the arrays defined above.                      #
    #                                                                                    #
    ######################################################################################
    for key_orig in orig_times:
        for key_gen in gen_times:
            correlation = pearsonr(orig_times[key_orig], gen_times[key_gen])
            
            if key_orig == key_gen:
                # they are from the same instance
                same_instance_corrs.append(correlation[0])

            elif key_orig.split("-")[0] == key_gen.split("-")[0]:
                # they are from the same site but different instances
                same_site_corrs.append(correlation[0])

            else:
                # completely different sites
                different_corrs.append(correlation[0])



def display_results():
    # for each threshold correlation value, output the rate of TP, FP (same site), FP (different sites), FP (total)
    # non-verbose output format is: threshold TP FP_same FP_diff FP_total
    # generally use TP and FP_total to generate a ROC curve
    true_positive_rate = list()
    false_positive_rate = list()

    for threshold in range(MIN_THRESHOLD, MAX_THRESHOLD, STEP):

        threshold = float(threshold) / float(MAX_THRESHOLD)  # threshold must be between 0 and 1
        same_instance_count = same_site_count = different_count = 0.0
        same_instance_rate = same_site_rate = different_rate = total_d_rate = 0.0

        for corr in same_instance_corrs:
            if corr > threshold:
                same_instance_count += 1.0

        for corr in same_site_corrs:
            if corr > threshold:
                same_site_count += 1.0

        for corr in different_corrs:
            if corr > threshold:
                different_count += 1.0

        if len(same_instance_corrs) > 0:
            same_instance_rate = same_instance_count / len(same_instance_corrs)
        if len(same_site_corrs) > 0:
            same_site_rate = same_site_count / len(same_site_corrs)
        if len(different_corrs) > 0:
            different_rate = different_count / len(different_corrs)
        if len(same_site_corrs) + len(different_corrs) > 0:
            #total_d_rate = (same_site_count + different_count)/(len(same_site_corrs)+len(different_corrs))
            total_d_rate = same_site_rate + different_rate
    
        print("threshold={}, same_instance_rate={:.3f}, same_site_rate={:.3f} diff_site_rate={:.3f}, total_d={:.3f}".format(
            threshold, same_instance_rate, same_site_rate, different_rate, total_d_rate))
    
        true_positive_rate.append(same_instance_rate)
        false_positive_rate.append(total_d_rate)
    
    # calculate AUC and display ROC curbe
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure()
    plt.plot(
        false_positive_rate, 
        true_positive_rate, 
        color='darkorange', 
        lw=1, 
        label="ROC curve (area = {:.2f})".format(roc_auc)
    )
    #plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Timing analysis for Tor traffic', add_help=True)
    parser.add_argument('random', nargs='?', help='The port for the server to bind to', type=float)
    args = parser.parse_args()

    RANDOM_PARAMETER = args.random
    random.seed(RANDOM_PARAMETER)

    print("Reading in times...")
    read_times()
    print("Done")

    print("Computing correlation...")
    compute_correlation()
    print("Done")

    print("Displaying results...")
    display_results()
    print("Done")