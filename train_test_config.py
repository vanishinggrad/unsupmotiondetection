
# refer to the data collection notes for more information.

# train set
dates_train = ['May28', 'Jun3', 'Jun4', 'Jun9']

# exclude in training set
exclude_dates_train = [['May28', 'human', 1], ['May28', 'human', 2], ['May28', 'empty', 2],
                       ['Jun5', 'lincoln_hazel', 3], ['Jun5', 'lincoln_hazel', 4],
                       ['Jun8', 'apollo', 2],
                       ['Jun9', 'empty', 1], ['Jun9', 'human', 2],
                       ['Jun12', 'apollo', 1],
                       ['Jun15', 'empty', 2], ['Jun15', 'human', 1], ['Jun15', 'human', 2], ['Jun15', 'human', 3],
                       ['Jun15', 'human', 4],
                       ]

# test set
dates_test = ['May28', 'Jun10', 'Jun15']

# exclude in test set
exclude_dates_test = [['May28', 'empty', 1], ['May28', 'human', 2],
                      ['Jun10', 'hazel', 1],
                      ['Jun15', 'empty', 1], ['Jun15', 'human', 1], ['Jun15', 'human', 2], ['Jun15', 'human', 4],
                      ['Jun18', 'human', 2], ['Jun18', 'human', 3], ['Jun18', 'human', 4],
                      ]
