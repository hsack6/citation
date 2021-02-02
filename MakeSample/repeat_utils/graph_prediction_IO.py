import os
import sys
# 一つ上の階層のmoduleをインポートできるようにする
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../' )

def get_new_InputDirs(p_new, n_new):
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_Baseline_mix_InputDir as new_COSSIMMLP_Baseline_mix_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_Baseline_inference_InputDir as new_COSSIMMLP_Baseline_inference_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_FNN_mix_InputDir as new_COSSIMMLP_FNN_mix_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_FNN_inference_InputDir as new_COSSIMMLP_FNN_inference_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_DeepMatchMax_mix_InputDir as new_COSSIMMLP_DeepMatchMax_mix_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_DeepMatchMax_inference_InputDir as new_COSSIMMLP_DeepMatchMax_inference_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_PROSER_mix_InputDir as new_COSSIMMLP_PROSER_mix_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_COSSIMMLP_PROSER_inference_InputDir as new_COSSIMMLP_PROSER_inference_InputDir
    from setting_param import \
        Evaluation_link_prediction_new_DEAL_PROSER_inference_InputDir as new_DEAL_PROSER_inference_InputDir
    if p_new == 'COSSIMMLP_Baseline_mix':
        probability_new_InputDir = new_COSSIMMLP_Baseline_mix_InputDir
    elif p_new == 'COSSIMMLP_Baseline_inference':
        probability_new_InputDir = new_COSSIMMLP_Baseline_inference_InputDir
    elif p_new == 'COSSIMMLP_FNN_mix':
        probability_new_InputDir = new_COSSIMMLP_FNN_mix_InputDir
    elif p_new == 'COSSIMMLP_FNN_inference':
        probability_new_InputDir = new_COSSIMMLP_FNN_inference_InputDir
    elif p_new == 'COSSIMMLP_DeepMatchMax_mix':
        probability_new_InputDir = new_COSSIMMLP_DeepMatchMax_mix_InputDir
    elif p_new == 'COSSIMMLP_DeepMatchMax_inference':
        probability_new_InputDir = new_COSSIMMLP_DeepMatchMax_inference_InputDir
    elif p_new == 'COSSIMMLP_PROSER_mix':
        probability_new_InputDir = new_COSSIMMLP_PROSER_mix_InputDir
    elif p_new == 'COSSIMMLP_PROSER_inference':
        probability_new_InputDir = new_COSSIMMLP_PROSER_inference_InputDir
    elif p_new == 'DEAL_PROSER_inference':
        probability_new_InputDir = new_DEAL_PROSER_inference_InputDir

    from setting_param import Evaluation_prediction_num_of_edge_new_LSTM_InputDir as num_new_LSTM_InputDir
    from setting_param import Evaluation_prediction_num_of_edge_new_Baseline_InputDir as num_new_Baseline_InputDir
    if n_new == 'Baseline':
        num_new_InputDir = num_new_Baseline_InputDir
    elif n_new == 'LSTM':
        num_new_InputDir = num_new_LSTM_InputDir

    return probability_new_InputDir, num_new_InputDir