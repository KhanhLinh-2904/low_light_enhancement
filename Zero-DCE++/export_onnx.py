import torch
import torch.nn as nn
import torch.nn.functional as F
import model

scale_factor = 12
torch_model = model.enhance_net_nopool(scale_factor)
print(type(torch_model))
torch_input = torch.randn(1, 3, 1200, 900)

onnx_program = torch.onnx.export(torch_model, torch_input, 'output1.onnx')
# onnx_program.save("output.onnx")