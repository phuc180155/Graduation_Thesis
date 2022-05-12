import torch
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def write_scalars(self, scalar_dict: dict, global_step: int, tag: str):
        self.writer.add_scalars(main_tag=tag, tag_scalar_dict=scalar_dict, global_step=global_step)
            
    def write_scalar(self, scalar_dict: dict, global_step: int):
        for k in scalar_dict.keys():
            self.writer.add_scalar(tag=k, scalar_value=scalar_dict[k], global_step=global_step)

    def write_image(self, image_dict: dict, global_step: int):
        for k in image_dict.keys():
            self.writer.add_image(tag=k, img_tensor=image_dict[k], global_step=global_step)