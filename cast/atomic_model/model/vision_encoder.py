import torch
import torch.nn as nn
from typing import Optional, Callable
from efficientnet_pytorch import EfficientNet
from cast.atomic_model.model.self_attention import PositionalEncoding


class VisionLanguageEncoder(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        lang_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        Atomic Model class
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.lang_encoding_size = lang_encoding_size
        self.final_encoding_size = obs_encoding_size//4
        self.context_size = context_size

        # Initialize FiLM Model
        self.film_model = make_model(self.lang_encoding_size, 3*(self.context_size +1), 8, 128, self.final_encoding_size)

        # Initialize the observation encoder
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError
        
        # Initialize compression layers if necessary
        if self.num_obs_features != self.final_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.final_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()


        self.compress_final_enc = nn.Linear(self.lang_encoding_size, self.final_encoding_size)

        # Initialize positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(self.final_encoding_size, max_seq_len=self.context_size+2) #no context
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.final_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.final_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

    def forward(self, obs_img: torch.tensor, feat_text: torch.tensor) -> torch.Tensor:
        # Get obs encoding
        obs_img = obs_img.reshape(-1, 3, obs_img.shape[-2], obs_img.shape[-1])
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.final_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        lang_token = self.compress_final_enc(feat_text) if feat_text.shape[1] != self.final_encoding_size else feat_text
        tokens = torch.cat((obs_encoding, lang_token), dim=1)
        if self.positional_encoding:
            tokens = self.positional_encoding(tokens)
        obs_encoding_tokens = self.sa_encoder(tokens)
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)
        return obs_encoding_tokens


# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

#From FiLM
def conv(ic, oc, k, s, p):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(oc),
    )


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, n_channels):
        super(FeatureExtractor, self).__init__()
        
        self.model = nn.Sequential(
            conv(in_channels, n_channels, 5, 2, 2),
            conv(n_channels, n_channels, 3, 2, 1),
            conv(n_channels, n_channels, 3, 2, 1),

        )
        
    def forward(self, x):
        return self.model(x)

class FeatureExtractor_last(nn.Module):
    def __init__(self, n_channels):
        super(FeatureExtractor_last, self).__init__()
        
        self.model = nn.Sequential(          
            conv(n_channels, n_channels*2, 3, 2, 1),
            conv(n_channels*2, n_channels*4, 3, 2, 1),
            conv(n_channels*4, n_channels*8, 3, 2, 1),
            conv(n_channels*8, n_channels*8, 3, 2, 1),                                
        )
        
    def forward(self, x):
        return self.model(x)
  
class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()
        
    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        x = gamma * x + beta
        
        return x
        
        
class ResBlock(nn.Module):
    def __init__(self, in_place, out_place):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_place, out_place, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_place, out_place, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_place)
        self.film = FiLMBlock()
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, beta, gamma)
        x = self.relu2(x)
        
        x = x + identity
        
        return x
        
        
class FiLM(nn.Module):
    def __init__(self, input_dim, input_channels, n_res_blocks, n_channels, output_dim):
        super(FiLM, self).__init__()
        
        self.film_generator = nn.Linear(input_dim, 2 * n_res_blocks * n_channels)
        self.feature_extractor = FeatureExtractor(input_channels, n_channels)
        self.res_blocks = nn.ModuleList()
        self.feature_extractor_last = FeatureExtractor_last(n_channels)
        self.output_layer = nn.Linear(n_channels*n_res_blocks, output_dim)

        for _ in range(n_res_blocks):
            self.res_blocks.append(ResBlock(n_channels + 2, n_channels))
    
        self.n_res_blocks = n_res_blocks
        self.n_channels = n_channels
        
    def forward(self, x, question):
        batch_size = question.size(0)
        device = question.device
        x = self.feature_extractor(x)

        film_vector = self.film_generator(question).view(
            batch_size, self.n_res_blocks, 2, self.n_channels)
        
        d = x.size(2)
        coordinate = torch.arange(-1, 1 + 0.00001, 2 / (d-1)).to(device)
        coordinate_x = coordinate.expand(batch_size, 1, d, d)
        coordinate_y = coordinate.view(d, 1).expand(batch_size, 1, d, d)
        
        for i, res_block in enumerate(self.res_blocks):
            beta = film_vector[:, i, 0, :]
            gamma = film_vector[:, i, 1, :]
            
            x = torch.cat([x, coordinate_x, coordinate_y], 1)
            x = res_block(x, beta, gamma)
        
        feature = self.feature_extractor_last(x)
        if len(feature.shape) != 2:
            feature = feature.squeeze(2).squeeze(2)
        feature = self.output_layer(feature)
        return feature

def make_model(input_dim, input_channels, n_res, n_channels, output_dim):
    return FiLM(input_dim, input_channels, n_res, n_channels, output_dim)
               


    
