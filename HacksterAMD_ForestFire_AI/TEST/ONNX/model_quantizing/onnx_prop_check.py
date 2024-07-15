import onnxruntime

options = onnxruntime.SessionOptions()
session = onnxruntime.InferenceSession(
              'forestfire_model_quant.onnx',
               providers=["VitisAIExecutionProvider"],
               provider_options=[{"config_file":"C:/AMD/voe-4.0-win_amd64/vaip_config.json"}])

model_inputs = session.get_inputs()
input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name


# Load inputs and do preprocessing by input_shape
input_data = [...]
result = session.run([], {input_name: input_data})