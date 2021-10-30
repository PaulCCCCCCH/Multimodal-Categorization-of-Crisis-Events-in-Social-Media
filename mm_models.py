import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import os
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, save_dir='.'):
        super(BaseModel, self).__init__()
        self.save_dir = save_dir

    def save(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(self.save_dir, filename + '.pt'))

    def load(self, filepath):
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict)


class MMModel(BaseModel):
    def __init__(self, imageEncoder, textEncoder):
        super(MMModel, self).__init__()
        self.imageEncoder = imageEncoder
        self.textEncoder = textEncoder

    def forward(self, x):
        raise NotImplemented


class TextOnlyModel(BaseModel):
    def __init__(self, dim_text_repr=768, num_class=2, save_dir='.'):
        super(TextOnlyModel, self).__init__()
        config = BertConfig()

        self.textEncoder = BertModel(
            config).from_pretrained('bert-base-uncased')

        self.linear = nn.Linear(dim_text_repr, num_class)

    def forward(self, x):
        _, text = x

        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        e_i = F.dropout(hidden_states[1])  # N, dim_text_repr

        return self.linear(e_i)


class ImageOnlyModel(BaseModel):
    def __init__(self, dim_visual_repr=1000, num_class=2, save_dir='.'):
        super(ImageOnlyModel, self).__init__()

        self.imageEncoder = torch.hub.load(
            'pytorch/vision:v0.8.0', 'densenet201', pretrained=True)
        self.flatten_vis = nn.Flatten()
        self.linear = nn.Linear(dim_visual_repr, num_class)

    def forward(self, x):
        image, _ = x

        f_i = F.dropout(self.flatten_vis(self.imageEncoder(image)))

        return self.linear(f_i)


class DenseNetBertMMModel(MMModel):
    def __init__(self, dim_visual_repr=1000, dim_text_repr=768, dim_proj=100, num_class=2, save_dir='.'):
        self.save_dir = save_dir

        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr

        # DenseNet: https://pytorch.org/hub/pytorch_vision_densenet/
        # The authors did not mention which one they used.
        # imageEncoder = torch.hub.load(
        #     'pytorch/vision:v0.8.0', 'densenet121', pretrained=True)
        # imageEncoder = torch.hub.load('pytorch/vision:v0.8.0', 'densenet169', pretrained=True)
        imageEncoder = torch.hub.load(
            'pytorch/vision:v0.8.0', 'densenet201', pretrained=True)
        # imageEncoder= torch.hub.load('pytorch/vision:v0.8.0', 'densenet161', pretrained=True)

        # Bert model: https://huggingface.co/transformers/model_doc/auto.html
        config = BertConfig()
        textEncoder = BertModel(config).from_pretrained('bert-base-uncased')

        super(DenseNetBertMMModel, self).__init__(imageEncoder, textEncoder)

        # Flatten image features to 1D array
        self.flatten_vis = torch.nn.Flatten()

        # Linear layers used to project embeddings to fixed dimension (eqn. 3)
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)

        # Linear layers to produce attention masks (eqn. 4)
        self.layer_attn_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.layer_attn_text = nn.Linear(dim_text_repr, dim_proj)

        # An extra fully-connected layer for classification
        # The authors wrote "we add self-attention in the fully-connected networks"
        # Here it is assumed that they mean 'we added a fully-connected layer as self-attention'.
        self.fc_as_self_attn = nn.Linear(2*dim_proj, 2*dim_proj)
        self.self_attn_bn = nn.BatchNorm1d(2*dim_proj)

        # Classification layer
        self.cls_layer = nn.Linear(2*dim_proj, num_class)

    def forward(self, x):
        image, text = x

        # Getting feature map (eqn. 1)
        # N, dim_visual_repr
        f_i = F.dropout(self.flatten_vis(self.imageEncoder(image)))

        # Getting sentence representation (eqn. 2)
        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        # The authors used embedding associated with [CLS] to represent the whole sentence
        e_i = F.dropout(hidden_states[1])  # N, dim_text_repr

        # Getting linear projections (eqn. 3)
        f_i_tilde = F.relu(self.proj_visual_bn(
            self.proj_visual(f_i)))  # N, dim_proj
        e_i_tilde = F.relu(self.proj_text_bn(
            self.proj_text(e_i)))  # N, dim_proj

        # Getting attention masks
        # The authors seemed to have made a mistake in eqn. 4: they said alpha_v_i is
        # completely dependent on e_i, and alpha_e_i is completely dependent on alpha_v_i,
        # while the equations mean the opposite. The implementation will stick to the text
        # instead of the equations.
        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i))  # N, dim_proj
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i))  # N, dim_proj

        # The authors concatenated masked embeddings to get a joint representation
        masked_v_i = torch.multiply(alpha_v_i, f_i_tilde)
        masked_e_i = torch.multiply(alpha_e_i, e_i_tilde)
        joint_repr = torch.cat((masked_v_i, masked_e_i),
                               dim=1)  # N, 2*dim_proj

        # Get class label prediction logits with final fully-connected layers
        return self.cls_layer(F.dropout(F.relu(self.self_attn_bn(self.fc_as_self_attn(joint_repr)))))
