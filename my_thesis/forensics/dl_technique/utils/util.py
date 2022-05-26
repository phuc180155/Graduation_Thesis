from logging import raiseExceptions
import numpy as np
from tqdm import tqdm
from glob import glob

def save_list_to_file(saved_list, file_name: str, overwrite=False):
    if isinstance(saved_list, np.ndarray):
        saved_list = saved_list.tolist()
    assert isinstance(saved_list, list), "Type of iteration should be a list!"
    with open('result/' + file_name, 'w' if overwrite else 'a') as f:
        f.write('=====================================\n')
        f.write(','.join([str(int(ele)) for ele in saved_list]))
        f.write('\n*************************************')


def is_refined_model(model_name: str):
    BASELINES = ['xception', 'dual_efficient', 'capsulenet', 'srm_2_stream', 'meso4', 'vit', 'swin_vit', 'm2tr']
    PROPOSALS = ['dual_attn_cnn', 'efficient_vit', \
                 'origin_dual_efficient_vit', 'origin_dual_efficient_vit_remove_ifft', \
                 'dual_cma_cnn_vit', 'dual_cnn_vit',  'pairwise_dual_cnn_vit', \
                 'dual_cnn_feedforward_vit', 'pairwise_dual_cnn_feedforward_vit', \
                 'dual_cma_cnn_attn', 'dual_cnn_cma_transformer', \
                 'dual_patch_cnn_cma_vit', 'pairwise_dual_patch_cnn_cma_vit', \
                 'dual_cnn_vit_test']
    if model_name in BASELINES:
        return False
    if model_name in PROPOSALS:
        return True
    raise Exception("Model not exists.")
            
        