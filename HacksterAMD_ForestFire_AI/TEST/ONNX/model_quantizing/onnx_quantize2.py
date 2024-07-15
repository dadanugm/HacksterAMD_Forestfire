##### Quantizied the model ####

from vitis_customop.preprocess import generic_preprocess as pre
from vitis_customop.postprocess_resnet import generic_post_process as post
import time


input_node_name = "blob.1"
preprocessor = pre.PreProcessor('forestfire_model.onnx', 'forestfire_model_pre.onnx', input_node_name)
#preprocessor.resize([640, 640])
#preprocessor.normalize(1, 1, 1)
#preprocessor.set_resnet_params(1, 1, 1)
preprocessor.build()

output_node_name = "1327"
postprocessor = post.PostProcessor('forestfire_model_pre.onnx', 'forestfire_model_quant.onnx', output_node_name)
postprocessor.ResNetPostProcess()
postprocessor.build()