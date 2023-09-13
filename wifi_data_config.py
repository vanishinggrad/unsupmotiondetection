#!/usr/bin/python3
from dataset_config import pets_conf as date_test_conf


# folder directory where all the pets data are stored
log_file_prefix = 'data/house/raw/'

# folder used to store processed data
file_prefix = 'data/house/preprocessed/'

# folder used to load store model (path relative to repo directory)
model_folder = 'models/'


ntx_max = nrx_max = 3
nsubcarrier_max = 56
n_timestamps = 128
time_offset_ratio = 1.0/10.0
D = 1
step_size = 33 #30, 23
ntx = 3
nrx = 3
nsubcarrier = 14
normalization = False
do_fft = True
fft_shape = (128, nsubcarrier)
data_shape_to_nn = (50, nsubcarrier)+(nrx*ntx, 2)
print('n_timestamps {}, D {}, nsubcarrier {}, step size {} final shape {}'.format(n_timestamps, D, nsubcarrier, step_size, data_shape_to_nn))
print('fft shape {}'.format(fft_shape))


label = [(0,0),(1,1),(2,2)]
total_classes = 3
win_len = 128
win_detect_cnt = 1


frame_dur = 10 # milli
skip_time = 5000
skip_frames = int(skip_time/frame_dur)

mixed_dates = [date for date in date_test_conf.keys()]

# Replace the string below with all possible strings below to go over each case of labels/classes
# e.g. replace 'hazel' with 'apollo' and re-run the code again!
mixed_label_name = ['empty', 'human', 'apollo', 'hazel', 'lincoln_hazel'][4]

# for training a label-generating model, if one is not already being used
training_date = []
training_validate_date = []
# make sure validation data and training data come from disjoint days
for d in training_validate_date:
    if d in training_date:
        raise ValueError('validation date {} should not appear in training date'.format(d))
training_date = training_date + training_validate_date
# print("training date {}".format(training_date))


# test date
test_date = []
training_classes = {}
training_classes[0] = {'label name': ['empty',],
                        'test': []}
training_classes[1] = {'label name': ['human',],
                        'test': []}
training_classes[2] = {'label name': ['hazel', 'lincoln_hazel', 'apollo'],
                        'test': []}

for c,i in label:
    for v in training_date:
        m = []
        for l in training_classes[c]['label name']:
            if v in training_validate_date:
                if l in date_test_conf[v]:
                    m.append(date_test_conf[v][l])
                else:
                    m.append(None)
            else:
                m.append(0)
        training_classes[c]['test'].append(m)

test_classes = {}
test_classes[0] = {'label name': ['empty',],
                        'test': []}
test_classes[1] = {'label name': ['human',],
                        'test': []}
test_classes[2] = {'label name': ['hazel', 'lincoln_hazel', 'apollo'],
                        'test': []}
for c,i in label:
    for v in test_date:
        m = []
        for l in test_classes[c]['label name']:
            if l in date_test_conf[v]:
                m.append(date_test_conf[v][l])
            else:
                m.append(None)
        test_classes[c]['test'].append(m)


epochs = 10
test_result_filename = "test_result.dat"
use_existing_model = False
model_name = model_folder + 'wifi_motion_labeling_model.h5'