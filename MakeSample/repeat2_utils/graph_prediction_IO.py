import os
import sys
# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )

def get_new_InputDirs(p_new, n_new):
    from setting_param import Evaluation_repeat1_link_prediction_new_AGATE_DEAL_InputDir as new_InputDir
    if p_new == 'repeat1_AGATE_DEAL':
        probability_new_InputDir = new_InputDir

    from setting_param import Evaluation_prediction_num_of_edge_new_LSTM_InputDir as num_new_LSTM_InputDir
    from setting_param import Evaluation_prediction_num_of_edge_new_Baseline_InputDir as num_new_Baseline_InputDir
    if n_new == 'Baseline':
        num_new_InputDir = num_new_Baseline_InputDir
    elif n_new == 'LSTM':
        num_new_InputDir = num_new_LSTM_InputDir

    return probability_new_InputDir, num_new_InputDir