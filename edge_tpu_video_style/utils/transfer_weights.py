import onnx
import tensorflow as tf
import torch
import PIL
import numpy as np

from network import ReCoNet as reconet_torch
from reconet import build_reconet as build_reconet_tf
from itertools import zip_longest


reconet_state_dict = torch.load('saved_models/finetune_20200917163819.pt', map_location=torch.device('cpu'))
torch_reconet = reconet_torch()
torch_reconet.load_state_dict(state_dict=reconet_state_dict, strict=True)

dummy_input = torch.autograd.Variable(torch.rand(1, 3, 216, 512, dtype=torch.float32, device='cpu'))
torch_reconet(dummy_input)

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    print(tensor.dtype)
    print(tensor.shape)

    return PIL.Image.fromarray(tensor)

tf_reconet = build_reconet_tf()

torch_reco_params = dict()
weight_queue = None
for name, layer in reconet_state_dict.items():
    if 'conv' in name or 'inst' in name:
        if 'deconv' in name:
            continue
        if 'weight' in name:
            assert weight_queue is None
            weight_queue = layer
        if 'bias' in name:
            assert weight_queue is not None
            torch_reco_params[name.replace('bias','')] = {'weight': weight_queue, 'bias': layer}
            weight_queue = None


conv_layers = [layer for layer in tf_reconet.layers if len(layer.get_weights()) > 0]

for tf_layer, pt_layer in zip_longest(conv_layers, torch_reco_params.items()):
    if tf_layer is not None:
        params = tf_layer.get_weights()
        if len(pt_layer[1]['weight'].shape) > 1:
            weight = pt_layer[1]['weight'].permute(2, 3, 1, 0).numpy()
            bias = pt_layer[1]['bias'].numpy()
        else:
            weight = pt_layer[1]['weight'].numpy()
            bias = pt_layer[1]['bias'].numpy()
        tf_layer.set_weights((weight, bias))
    else:
        print(pt_layer[0])


tf_reconet.compile()
tf_reconet.save('saved_models/reconet_tf_in.pb')
