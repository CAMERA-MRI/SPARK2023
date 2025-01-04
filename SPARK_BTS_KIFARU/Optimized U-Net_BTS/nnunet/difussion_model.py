import logging
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
import os
import model.networks as networks
import sys

model_path = sys.os.path('/home/guest189/SPARK_Stater/Optimized U-Net_BTS/nnunet/baseline_model.py')
# import baseline_model


logger = logging.getLogger('base')

torch.set_printoptions(precision=10)


class DDM2(baseline_model):
    def __init__(self, opt):
        super(DDM2, self).__init__(opt)
        self.opt = opt

        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))

        self.schedule_phase = None

        self.loss_type = opt['model']['loss_type']

        # set loss and load resume state
        self.set_loss()

        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')

        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if k.find('denoise_fn') >= 0:
                        continue
                    if k.find('noise_model_variance') >= 0:
                        continue
                    optim_params.append(v)
                #optim_params = list(self.netG.parameters())
            print('Optimizing: '+str(len(optim_params))+' params')
            
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])

            self.log_dict = OrderedDict()
        
        #self.print_network()
        self.load_network()
        self.counter = 0

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()

        outputs = self.netG(self.data)
        
        if torch.is_tensor(outputs):
            l_pix = outputs
            l_pix.backward()
            self.optG.step()

        elif type(outputs) is dict:
            l_pix = outputs['total_loss']

            total_loss = l_pix
            total_loss.backward()
            self.optG.step()
    
        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        if self.use_ttt:
            optim_params = []
            for k, v in self.netG.named_parameters():
                if k.find('denoise_fn') >= 0:
                    continue
                optim_params.append(v)
            
            ttt_opt = torch.optim.Adam(
                optim_params, lr=self.opt['TTT']["optimizer"]["lr"])
        else:
            self.netG.eval()
            ttt_opt = None

        if isinstance(self.netG, nn.DataParallel):
            self.denoised = self.netG.module.denoise(
                self.data, continous, ttt_opt=ttt_opt)
        else:
            self.denoised = self.netG.denoise(
                self.data, continous, ttt_opt=ttt_opt)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
       


    def sample(self, data, continous=False):
        # Get a list of image files with "_SSA_" in their names
        image_files = [file for file in os.listdir(data) if "_SSA_" in file]

        # Sample randomly from the available image files
        if batch_size > len(image_files):
            raise ValueError("Requested batch size is larger than the number of available images.")
    selected_files = random.sample(image_files, batch_size)

        # Perform denoising on the selected image files
        denoised_images = []
        for file in selected_files:
        
        
        # Perform denoising operation on the image file (e.g., using denoising algorithm or network)
        denoised_image = denoise_image(os.path.join(data, file))
        denoised_images.append(denoised_image)

    return denoised_images
