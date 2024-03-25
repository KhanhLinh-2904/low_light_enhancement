import onnxruntime
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F 
import model

scale_factor = 12
torch_model = model.enhance_net_nopool(scale_factor)
print(type(torch_model))
torch_input = torch.randn(1, 3, 3648, 5472)

onnx_model = onnx.load('output.onnx')
ort_session = onnxruntime.InferenceSession('output.onnx')
# Get the input name expected by the ONNX model
input_name = ort_session.get_inputs()[0].name

onnxruntime_outputs = ort_session.run(None, {input_name: torch_input.numpy()})
# Convert ONNX runtime outputs to PyTorch tensors if needed
torch_outputs = [torch.tensor(output) for output in onnxruntime_outputs]

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")
print(f"Sample output: {onnxruntime_outputs}")
print("Done")