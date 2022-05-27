"""
Created on Wed Jun 19 17:06:48 2019

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
import numpy as np

from misc_functions import get_example_params, convert_to_grayscale, save_gradient_images
import os
import sys
sys.path.insert(0, "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print(sys.path)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # see issue #152

class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            # print(grad_in[0].shape)
            # print("aaaaaaaaaa")
            self.gradients = grad_in[0]

        # Register hook to the first layer
        # print(list(self.model.base[0][:-1]._modules.items())[0][1].conv1.conv)
        first_layer = list(self.model.base_net[0][:-1]._modules.items())[0][1].conv1.conv  # xception
        # print(list(self.model.layer1._modules.items()))
        # first_layer = self.model.conv1  # resnext50

        # first_layer = list(self.model.vgg_ext.vgg_1._modules.items())[0][1]  # capsule
        # print(list(self.model.vgg_ext.vgg_1._modules.items())[0])
        # print(first_layer.register_backward_hook(hook_function))
        first_layer.register_backward_hook(hook_function)

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and steps
        step_list = np.arange(steps+1)/steps
        # Generate scaled xbar images
        xbar_list = [input_image*step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        print("Model output: ", model_output)
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        print(one_hot_output.shape)
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

    def generate_integrated_gradients(self, input_image, target_class, steps):
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        # Initialize an iamge composed of zeros
        integrated_grads = np.zeros(input_image.size())
        for xbar_image in xbar_list:
            # Generate gradients from xbar images
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            # Add rescaled grads from xbar images
            integrated_grads = integrated_grads + single_integrated_grad/steps
        # [0] to get rid of the first channel (1,3,224,224)
        return integrated_grads[0]

from torch.autograd import Variable
import glob
import os
from dl_technique.model.cnn.xception_net.model import xception

if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake

    model = xception(pretrained=False)
    # print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_folder = '/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/CAM_visualization/data/dfdc/*.jpg'
    model_path = "/mnt/disk1/doan/phucnp/Graduation_Thesis/my_thesis/forensics/dl_technique/checkpoint/datasetv5/dfdcv5/xception/(0.1393_0.9459_0.8476)_lr_0.0003_batch_32_es_val_loss_loss_bce_pre_1_seed_0_drmlp_0.0_aug_1/step/best_test_acc_42000_0.847615.pt"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load("/hdd/tam/dfd_benmark/code/dfd_benchmark/xception_reenact_checkpoint/model_pytorch_3.pt",map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load("../../../model/resnext50/model_pytorch_4.pt",map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load(os.path.join("/hdd/tam/dfd_benmark/code/dfd_benchmark/capsule_df_checkpoint/",'capsule_' + str(3) + '.pt'),map_location=torch.device('cpu')))
    print("Load xong ... ")
    model.eval()
    # print(model)
    # print(model.base[0][:-1][0].conv1.conv)
    IG = IntegratedGradients(model)

    for target_example in range(len(glob.glob(data_folder))):
        (original_image, prep_img, target_class, file_name_to_export,original_image) = get_example_params(target_example, data_folder)

        # print(original_image)
        # Vanilla backprop
        # Generate gradients
        integrated_grads = IG.generate_integrated_gradients(prep_img, 0, 5)
        # Convert to grayscale
        grayscale_integrated_grads = convert_to_grayscale(integrated_grads)
        # Save grayscale gradients
        # file_name_to_export = ""
        save_gradient_images(grayscale_integrated_grads, file_name_to_export + '_Integrated_G_gray_'+ 'dfdc',original_image, 'dfdc')
        print('Integrated gradients completed.')
