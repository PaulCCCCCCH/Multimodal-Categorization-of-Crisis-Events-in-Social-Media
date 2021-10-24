import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import os

class MMModel(nn.Module):
    def __init__(self, imageEncoder, textEncoder):
        self.imageEncoder = imageEncoder
        self.textEncoder = textEncoder

    def forward(self, x):
        raise NotImplemented



class DenseNetBertMMModel(MMModel):
    def __init__(self, dim_visual_repr=756, dim_text_repr=756, dim_proj=100, num_class=2, save_dir='.'):
        self.save_dir = save_dir

        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr

        # DenseNet: https://pytorch.org/hub/pytorch_vision_densenet/
        # The authors did not mention which one they used.
        imageEncoder = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True)

        # Bert model: https://huggingface.co/transformers/model_doc/auto.html 
        config = BertConfig()
        textEncoder = BertModel(config).from_pretrained('bert-base-uncased')

        super(MMModel, self).__init__(textEncoder, imageEncoder)

        # Linear layers used to project embeddings to fixed dimension (eqn. 3)
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        # Linear layers to produce attention masks (eqn. 4)
        self.layer_attn_visual = nn.Linear(dim_proj, dim_proj)
        self.layer_attn_text = nn.Linear(dim_proj, dim_proj)

        # An extra fully-connected layer for classification
        # The authors wrote "we add self-attention in the fully-connected networks"
        # Here it is assumed that they mean 'we added a fully-connected layer as self-attention'.
        self.fc_as_self_attn = nn.Linear(2*dim_proj, 2*dim_proj)

        # Classification layer
        self.cls_layer = nn.Linear(2*dim_proj, num_class)


    def forward(self, x):
        image, text = x

        # Getting feature map (eqn. 1)
        f_i = nn.Flatten(self.imageEncoder(image)) # N, dim_visual_repr
        
        # Getting sentence representation (eqn. 2)
        hidden_states = self.textEncoder(text, image)  # N, T, dim_text_repr
        # The authors used embedding associated with [CLS] to represent the whole sentence
        e_i = hidden_states[:, 0, :]  # N, dim_text_repr

        # Getting linear projections (eqn. 3)
        f_i_tilde = nn.ReLU(self.proj_visual(f_i))  # N, dim_proj
        e_i_tilde = nn.ReLU(self.proj_text(e_i))  # N, dim_proj

        # Getting attention masks
        # The authors seemed to have made a mistake in eqn. 4: they said alpha_v_i is 
        # completely dependent on e_i, and alpha_e_i is completely dependent on alpha_v_i,
        # while the equations mean the opposite. The implementation will stick to the text
        # instead of the equations.
        alpha_v_i = nn.Sigmoid(self.layer_attn_visual(e_i))  # N, dim_proj
        alpha_e_i = nn.Sigmoid(self.layer_attn_visual(f_i))  # N, dim_proj

        # The authors concatenated masked embeddings to get a joint representation
        masked_v_i = torch.dot(alpha_v_i, f_i_tilde)
        masked_e_i = torch.dot(alpha_e_i, e_i_tilde)
        joint_repr = torch.cat((masked_v_i, masked_e_i), dim=1) # N, 2*dim_proj

        # Get class label prediction logits with final fully-connected layers
        return self.cls_layer(nn.ReLU(self.fc_as_self_attn(joint_repr)))

    def save(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(self.save_dir, filename + '.pt'))

    def load(self, filepath):
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict)