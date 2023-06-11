"""
.pth to .onnx
"""
import torch
import onnx
from onnxsim import simplify

pth_path = "/opt/ml/level2_cv_semanticsegmentation-cv-09/checkpoint/additional_nested_unet_best.pt"
onnx_path = '/opt/ml/level2_cv_semanticsegmentation-cv-09/converted_model/additional_nested_unet_best.onnx'


model = torch.load(pth_path).cuda()
model.eval()

# 모델을 ONNX로 변환
input_sample = torch.randn(1, 3, 512, 512).cuda()  # 입력 샘플 생성
torch.onnx.export(model, input_sample, onnx_path, opset_version=12)

# load your predefined ONNX model
model = onnx.load(onnx_path)

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

# use model_simp as a standard ONNX model object
onnx.save(model_simp, "best_simplified.onnx")

"""
.onnx to .pb
"""
import onnx 
import torch
from onnx_tf.backend import prepare

onnx_model_path = "/opt/ml/level2_cv_semanticsegmentation-cv-09/converted_model/additional_nested_unet_best.onnx"
tf_model_path = "/opt/ml/level2_cv_semanticsegmentation-cv-09/converted_model/additional_nested_unet_best.pb"

onnx_model = onnx.load(onnx_model_path)

tf_rep = prepare(onnx_model)
    
tf_rep.export_graph(tf_model_path)

"""
.pb to .tflite
"""
import tensorflow as tf

# pb 모델 경로
pb_model_path = "/opt/ml/level2_cv_semanticsegmentation-cv-09/converted_model/additional_nested_unet_best.pb"
# TFLite 모델 저장 경로
tflite_model_path = "/opt/ml/level2_cv_semanticsegmentation-cv-09/converted_model/additional_nested_unet_best.tflite"

# TensorFlow Lite 포맷으로 모델 변환
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_model_path, #TensorFlow freezegraph .pb model file
                                                      input_arrays=['input.1'], # name of input arrays as defined in torch.onnx.export function before.
                                                      output_arrays=['367']  # name of output arrays defined in torch.onnx.export function before.
                                                      )

converter.optimizations = [tf.lite.Optimize.DEFAULT]	# 최적화
tflite_model = converter.convert()	# tflite로 변환

# TensorFlow Lite 모델 저장
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)