import importlib
import logging
from os import path
import copy
from torch import nn
import torch
from coala.models.model import BaseModel
logger = logging.getLogger(__name__)

def load_model(model_name: str):
    dir_path = path.dirname(path.realpath(__file__))
    model_file = path.join(dir_path, "{}.py".format(model_name))
    if not path.exists(model_file):
        logger.error("Please specify a valid model.")
    model_path = "models.{}".format(model_name)
    model_lib = importlib.import_module(model_path)
    model = getattr(model_lib, "Model")
    # TODO: maybe return the model class initiator
    return model

class ViewLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, -1)
        out = input.view(shape)
        return out

class BaseSFLModel(BaseModel):
    def __init__(self):
        super(BaseSFLModel, self).__init__()
        
        # define model as three nn.sequential objects in separate model scripts
        self.model = None

        self.view_identifier = "ViewLayer"
    
    def split(self, cut_layer = 1):
        '''
        Split the self.model, nn.Sequential object.
        '''
        if self.model is None:
            logger.error("Must provide self.model: nn.Sequential object! Exit without split")
            return

        self.cut_layer = cut_layer
        
        model_list = list(self.model.children())
        total_layer = 0
        for name, module in enumerate(model_list):
            if self.check_if_module_is_layer(module):
                total_layer += 1
        
        self.total_layer = total_layer
        
        if self.cut_layer == 0:
            logger.error("cut_layer must be greater than 0!")
            return
        if self.cut_layer == self.total_layer:
            logger.error(f"cut_layer must be smaller than {self.total_layer}!")
            return
        client_list = []
        server_list = []
        
        client_layer_count = 0
        for name, module in enumerate(model_list):
            
            if self.check_if_module_is_layer(module):
                client_layer_count += 1
            
            if client_layer_count <= self.cut_layer:
                client_list.append(module)
            else:
                server_list.append(module)

        self.client_cloud_copy = [nn.Sequential(*client_list)]  # use list to wrap client-side model (cloud copy) to avoid registering it under state_dict

        self.server = [nn.Sequential(*server_list)] # use list to wrap server-side model to avoid registering it under state_dict
        
        logger.info("original model is:")
        logger.info(self.model) # this is a ordered dict
        logger.info(f"total number of layer is {total_layer}, cut_layer is {self.cut_layer}")
        logger.info("client model is:")
        logger.info(self.client_cloud_copy[0])
        logger.info("server model is:")
        logger.info(self.server[0])
    
    def initialize_weights(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None: 
                    m.bias.data.zero_()
            if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None: 
                    m.bias.data.zero_()
        
        if self.client_cloud_copy[0] is not None:
            self.client_cloud_copy[0].apply(init_weights)
        if self.server[0] is not None:
            self.server[0].apply(init_weights)
    
    def forward(self, x): # if -1, meaning use client-side model located at the server to perform forward
        x = self.model(x)
        return x
    
    def get_num_of_server_layer(self):
        num_of_server_layer = 0
        if self.server[0] is not None:
            list_of_layers = list(self.server[0].children())
            for i, module in enumerate(list_of_layers):
                if self.check_if_module_is_layer(module):
                    num_of_server_layer += 1
        return num_of_server_layer
    
    def get_num_of_client_layer(self, cid = 0):
        num_of_client_layer = 0
        if self.client_cloud_copy[0] is not None:
            list_of_layers = list(self.client_cloud_copy[0].children())
            for i, module in enumerate(list_of_layers):
                if self.check_if_module_is_layer(module):
                    num_of_client_layer += 1
        return num_of_client_layer
    
    def get_smashed_data_size(self, batch_size = 1, channel_size = 3, height = 32, width = 32):
        '''
        Get the size of feature [aka, activation, smashed data]
        '''
        with torch.no_grad():
            noise_input = torch.randn([batch_size, channel_size, height, width])
            try:
                device = next(self.client_cloud_copy[0].parameters()).device
                noise_input = noise_input.to(device)
            except:
                pass
            smashed_data = self.client_cloud_copy[0](noise_input)
        return smashed_data.size()


    def check_if_module_is_layer(self, module):
        '''
        Override this method if model includes modules other than Conv2d and Linear (i.e. BasicBlock)
        '''
        valid_layer_list = ["Conv2d", "Linear"]
        for valid_layer in valid_layer_list:
            if valid_layer in str(module):
                return True
        return False
        
    def train(self):
        if self.server[0] is not None:
            self.server[0].train()
        if self.client_cloud_copy[0] is not None:
            self.client_cloud_copy[0].train()
    
    def eval(self):
        if self.server[0] is not None:
            self.server[0].eval()
        if self.client_cloud_copy[0] is not None:
            self.client_cloud_copy[0].eval()
    
    def to(self, device):
        if self.server[0] is not None:
            self.server[0].to(device)
        if self.client_cloud_copy[0] is not None:
            self.client_cloud_copy[0].to(device)