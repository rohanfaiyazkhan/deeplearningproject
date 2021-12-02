import numpy as np

ground_truth = ['pol_dat_us', 'pol_dat_ca', 'pol_dat_uk', 'pol_fb_us']

def label_func(row):
    '''
    Checks each column of ground_truth and extracts whichever column is not null as the label.
    0 represents liberal and 1 represents conservative
    '''
    for i in ground_truth:
        if ~np.isnan(row[i]):
            return row[i]
    return np.nan

def get_labels(data):
    '''
    Returns array of labels from entire dataset
    '''
    return data.apply(lambda row: label_func(row), axis=1).to_numpy()

