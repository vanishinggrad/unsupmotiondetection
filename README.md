Code supplied with our paper
<br>
`Tahir, Naveed, et al. "An Unsupervised Approach to Motion Detection Using WiFi Signals." 2023 22nd IEEE International Conference on Machine Learning and Applications (ICMLA). IEEE, 2023.`


## Requirements

We used TensorFlow 2.3.1 with a Python 3.7 interpreter on Ubuntu 22.04. Other required Python packages include NumPy and Matplotlib.

## Usage

These instructions will guide you through the steps needed to reproduce our results.

### Configuration

Most configuration parameters discussed in the paper are specified in the files ``wifi_data_config.py`` and ``dcn/dcn_config.py``. Please refer to these files if you need to change any configuration parameters.


The directories ``data/house/preprocessed`` and ``data/lab/preprocessed`` contain pre-processed datasets included with the draft. Further data collection notes for the house dataset are included at ``Data_Collection_Notes_House_Data.xlsx`` Moreover, reports (plots or numerical summaries) will be available mainly in the ``results`` directory. 


### Training and Evaluation
To train and evaluate the proposed model on a dataset, please execute the corresponding shell script. For example, the shell script for the house dataset is included as ``house.sh`` and that for the lab dataset is included as ``lab.sh``. Model performance statistics will also be generated using the same script.

Run ``get_baselines.sh`` to get baseline results as reported in the paper.
