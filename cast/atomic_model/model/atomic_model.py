import os
import argparse
import time
import pdb

import torch
import torch.nn as nn

class AtomicModel(nn.Module):

    def __init__(self, vision_encoder, 
                       action_head, 
                       ):
        super(AtomicModel, self).__init__()


        self.vision_encoder = vision_encoder            
        self.action_head = action_head

    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder":      
            output = self.vision_encoder(kwargs["obs_img"], kwargs["lang_embed"])        
        elif func_name == "action_head":
            output = self.action_head(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        else:
            raise NotImplementedError
        return output