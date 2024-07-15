##### Quantizied the model ####
import onnx
from onnxruntime.quantization import shape_inference, QuantFormat, QuantType
import vai_q_onnx
import os


# As docs it is recommended to do quantizie preprocess
shape_inference.quant_pre_process(
    'forestfire_model.onnx', #input model
    'forestfire_model_pre.onnx', #output model
)

#since model contains FLOAT 32, better quantized to supported format
vai_q_onnx.quantize_static(
    'forestfire_model_pre.onnx',
    'forestfire_model_quant.onnx',
    calibration_data_reader=None,
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    input_nodes=[],
    output_nodes=[],
    op_types_to_quantize=[],
    random_data_reader_input_shape=[],
    per_channel=False,
    reduce_range=False,
    activation_type=vai_q_onnx.QuantType.QInt8,
    weight_type=vai_q_onnx.QuantType.QInt8,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    optimize_model=True,
    use_external_data_format=False,
    execution_providers=['CPUExecutionProvider'],
    convert_fp16_to_fp32=False,
    convert_nchw_to_nhwc=False,
    include_cle=False,
    enable_dpu=True,
    extra_options={'ActivationSymmetric': True}
)

os.system('copy forestfire_model_quant.onnx ..\models')

#check models
'''
model = onnx.load('forestfire_model_quant.onnx')
output =[node.name for node in model.graph.output]

input_all = [node.name for node in model.graph.input]
input_initializer = [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
print('Outputs: ', output)
'''
