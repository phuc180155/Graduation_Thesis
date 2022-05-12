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
            
        