# Yolov5模型转换
转换过程可以在训练和测试过程中进行，该说明是在 test.py 执行过程中进行转换。


## torch -> torchscript (torch.jit.trace)
- config.py setting
```python3
from deepvac import AttrDict

config.cast.TraceCast = AttrDict()
config.cast.TraceCast.model_dir = <output-trace-pt-file>
# 静态量化(not available)
# config.cast.TraceCast.static_quantize_dir = <output-static-quantize-qt-file>
# 动态量化(not available)
# config.cast.TraceCast.dynamic_quantize_dir = <output-dynamic-quantize-qt-file>
# Yolov5Test为测试类名
config.core.Yolov5Test.cast2cpu = True
```

- cast process
```python3
python3 test.py <trained-model-file(required)> <test-sample-path(required)> <test-label-path(option)>
```

- trace model forward
```python3
# 加在config.py或者test.py中
config.core.Yolov5Test.jit_model_path = <output-trace-pt-file>
# jit_model_path会将model_path覆盖,trace模型测试:
python3 test.py <trained-model-file(required)> <test-sample-path(required)> <test-label-path(option)>
```

- attention please
```python3
# 目前trace pt模型只接受固定输入尺寸
# 输入大小取决于测试过程self.config.sample size
# 可以通过测试log获取输入尺寸,log示例:
2021-06-28 11:16:20,887 93019:main INFO     TEST: [input shape: torch.Size([1, 3, 256, 416])] [1/1]
```


## torch -> torchscript (torch.jit.script)
- config.py setting
```python3
from deepvac import AttrDict

config.cast.ScriptCast = AttrDict()
config.cast.ScriptCast.model_dir = <output-script-pt-file>
# 静态量化(not available)
# config.cast.ScriptCast.static_quantize_dir = <output-static-quantize-qt-file>
# 动态量化(not available)
# config.cast.ScriptCast.dynamic_quantize_dir = <output-dynamic-quantize-qt-file>
# Yolov5Test为测试类名
config.core.Yolov5Test.cast2cpu = True
```

- cast process
```python3
python3 test.py <trained-model-file(required)> <test-sample-path(required)> <test-label-path(option)>
```

- script model forward
```python3
# 加在config.py或者test.py中
config.core.Yolov5Test.jit_model_path = <output-script-pt-file>
# jit_model_path会将model_path覆盖,script模型测试:
python3 test.py <trained-model-file(required)> <test-sample-path(required)> <test-label-path(option)>
```

- attention please
```python3
# 目前script pt模型接受任意尺寸输入
```


## torch -> coreml
- config.py setting
```python3
import coremltools
from deepvac import AttrDict

# 当前yolov5只支持通过trace方式转换coreml模型
config.cast.TraceCast = AttrDict()
config.cast.CoremlCast = AttrDict()
config.cast.TraceCast.model_dir = <output-trace-pt-file>
config.cast.CoremlCast.model_dir = <output-coreml-mlmodel-file>
# if setting None, input_type=TensorType, you alse can set input_type="image"
config.cast.CoremlCast.input_type = None
config.cast.CoremlCast.scale = 1.0 / 255.0
config.cast.CoremlCast.color_layout = 'RGB'
config.cast.CoremlCast.blue_bias = 0
config.cast.CoremlCast.green_bias = 0
config.cast.CoremlCast.red_bias = 0
config.cast.CoremlCast.minimum_deployment_target = coremltools.target.iOS13
config.cast.CoremlCast.classfier_config = ["cls{}".format(i) for i in range(config.core.Yolov5Test.class_num)]
# Yolov5Test为测试类名
config.core.Yolov5Test.cast2cpu = True
```

- cast process
```python3
python3 test.py <trained-model-file(required)> <test-sample-path(required)> <test-label-path(option)>
```

- [coreml model forward](https://zhuanlan.zhihu.com/p/110269410)
```python3
# 推理环境: macos13 or later
import sys
try:
    import coremltools
except:
    pip3 install coremltools
    import coremltools

spec = coremltools.utils.load_spec('syszux_scene.mlmodel')
# 以上代码可执行在linux上,以下代码只能运行在macos13 or later上
# load mlmodel
mlmodel = coremltools.models.MLModel("output/coreml.mlmodel")
# img process
img = cv2.imread("test.jpg", 1)
# input shape = (1, 3, 256, 416)
img = cv2.resize(img, (416, 256))
img = img.transpose(2, 0, 1).div(255)
img = np.expand_dims(img, axis=0)
# forward
out = mlmodel.predict({'input': img})
当前out结果为归一化结果,暂时无法对应到torch结果上
```

- attention please
```python3
# 目前mlmodel模型只接收固定输入,接收输入尺寸与trace模型保持一致
# 模型转换环境
torch == 1.8.1
numpy == 1.19.5
coremltools == 4.1
```


# torch -> onnx
- config.py setting
```python3
config.cast.OnnxCast = AttrDict()
config.cast.OnnxCast.onnx_model_dir = <output-onnx-onnx-file>
config.cast.OnnxCast.onnx_version = 11
config.cast.OnnxCast.onnx_input_names = ["input"]
config.cast.OnnxCast.onnx_output_names = ["output"]
config.cast.OnnxCast.onnx_dynamic_ax = {"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 1: "anchors"}}
# Yolov5Test为测试类名
config.core.Yolov5Test.cast2cpu = True
```
 
- cast process
```python3
python3 test.py <trained-model-file(required)> <test-sample-path(required)> <test-label-path(option)>
```

- onnx forward
```python3
import numpy as np
try:
    import onnxruntime
    from onnxruntime.datasets import get_example
except:
    pip3 install onnxruntime
    import onnxruntime
    from onnxruntime.datasets import get_example

x = np.random.randn(1, 3, 416, 416).astype(np.float32)
example_model = get_example("onnx-file-abspath")
sess = onnxruntime.InferenceSession(example_model)
onnx_out = sess.run(None, {"input": x})
```

- attention please
```python3
# onnx模型可接收任意尺寸输入
# 当前onnx模型只限于CPU上执行
```


# torch -> ncnn
- config.py setting
```python3
config.cast.NcnnCast = AttrDict()
config.cast.NcnnCast.onnx_model_dir = <output-onnx-onnx-file>
config.cast.NcnnCast.onnx_version = 11
config.cast.NcnnCast.onnx_input_names = ["input"]
config.cast.NcnnCast.onnx_output_names = ["output"]
# not support onnx_dynamic_ax now
# config.cast.NcnnCast.onnx_dynamic_ax = {"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 1: "anchors"}}
config.cast.NcnnCast.model_dir = <output-ncnn-bin-file>
# you should build onnx2ncnn first
config.cast.NcnnCast.onnx2ncnn = <onnx2ncnn-binary-file>
```
 
- cast process
```python3
python3 test.py <trained-model-file(required)> <test-sample-path(required)> <test-label-path(option)>
```

- ncnn forward
```python3
import cv2
import numpy as np
try:
    import ncnn
except:
    pip3 install ncnn
    import ncnn


# Net
net = ncnn.Net()
net.opt.use_vulkan_compute = False
net.opt.num_threads = 1
net.load_param(<output-ncnn-param-file>)
net.load_model(<output-ncnn-bin-file>)
print(net.input_names())
print(net.output_names())

# input
cv_img = cv2.imread("test.jpg", 1)
h0, w0 = cv_img.shape[:2]
h , w  = 256, 416
mat_in = ncnn.Mat.from_pixels_resize(cv_img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w0, h0, w, h)
means = ()
norms = (1/255.0, 1/255.0, 1/255.0)
mat_in.substract_mean_normalize(means, norms)

# forward
ex = net.create_extractor()
ex.input("input", mat_in)
ret, mat_out = ex.extract("output")
# ex.input("data", mat_in)
```

- attention please
```
当前ncnn模型不支持动态输入
当前ncnn模型可以转换成功,但是推理结果错误
[问题](https://github.com/DeepVAC/yolov5/issues/23)
```


# torch -> tnn
- config.py setting
```python3
config.cast.TnnCast.onnx_model_dir = <output-onnx-onnx-file>
config.cast.TnnCast.onnx_version = 11
config.cast.TnnCast.onnx_input_names = ["input"]
config.cast.TnnCast.onnx_output_names = ["output"]
config.cast.TnnCast.model_dir = <output-tnn-tnnmodel-file>
# config.cast.TnnCast.optimize = True
```

- cast process
```python3
python3 test.py <trained-model-file(required)> <test-sample-path(required)> <test-label-path(option)>
```

- tnn forward

- attention please


# torch -> mnn
- config.py setting
```python3
config.cast.MnnCast.onnx_model_dir = <output-onnx-onnx-file>
config.cast.MnnCast.onnx_version = 11
config.cast.MnnCast.onnx_input_names = ["input"]
config.cast.MnnCast.onnx_output_names = ["output"]
config.cast.MnnCast.onnx2mnn = <MNNConvert-binary-file>
config.cast.MnnCast.model_dir = <output-mnn-mnn-file>
# not support save_static_model yet
config.cast.MnnCast.save_static_model = False
```

- cast process
```python3
python3 test.py <trained-model-file(required)> <test-sample-path(required)> <test-label-path(option)>
```

- tnn forward
```python3
import numpy as np
try:
    import MNN
except:
    pip3 install mnn
    import MNN
import cv2


# Net
net = MNN.nn.load_module_from_file(sys.argv[1], ["input"], ["output"])

# Input
image = cv2.imread("images/porn_0.jpg", 1)
image = cv2.resize(image, (416, 256))
image = image[..., ::-1].transpose(2, 0, 1)
image = image.astype(np.float32) / 255.
input_var = MNN.expr.placeholder([1, 3, 256, 416], MNN.expr.NCHW)
input_var.write(image)
input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)

# Forward
output_var = net.forward(input_var)
output_var = MNN.expr.convert(output_var, MNN.expr.NHWC)
```

- attention please
```
当前mnn模型可以转换成功
当前mnn模型推理结构与onnx模型不一致，具体原因研究中
```


# torch -> tensorrt
- config.py setting
 ```python3
config.cast.TensorrtCast.onnx_model_dir = <output-onnx-onnx-file>
config.cast.TensorrtCast.onnx_version = 11
config.cast.TensorrtCast.onnx_input_names = ["input"]
config.cast.TensorrtCast.onnx_output_names = ["output"]
config.cast.onnx_dynamic_ax = {"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 1: "anchors"}}
config.cast.TensorrtCast.model_dir = <output-tensorrt-trt-file>
config.cast.TensorrtCast.input_min_dims = (1, 3, 1, 1)
config.cast.TensorrtCast.input_opt_dims = (1, 3, 416, 416)
config.cast.TensorrtCast.input_max_dims = (1, 3, 2000, 2000)
```

- cast process
```python3
python3 test.py <trained-model-file(required)> <test-sample-path(required)> <test-label-path(option)>
```

- tnn forward
```python3
import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

# build engine
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
with open("output/yolov5.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# cv_img
cv_img = cv2.imread("/home/liyang/Pictures/porn_0.jpg", 1)
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
cv_img = cv2.resize(cv_img, (416, 256))
img = np.transpose(cv_img, (2, 0, 1)).astype(np.float32)
img = np.expand_dims(img, axis=0)
img /= 255.0

create_execution_context() as context:
h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)

# Allocate device memory for inputs and outputs.
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
# Create a stream in which to copy inputs/outputs and run inference.
stream = cuda.Stream()
np.copyto(h_input, img.ravel())

# Transfer input data to the GPU.
cuda.memcpy_htod_async(d_input, h_input, stream)
# Run inference.
context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
# Transfer predictions back from the GPU.
cuda.memcpy_dtoh_async(h_output, d_output, stream)
# Synchronize the stream
stream.synchronize()
# print(h_output)
```

- attention please
```
当前tensorrt模型可以转换成功
当前tensorrt模型推理结构与onnx模型一致
当前tensorrt模型只接受固定尺寸输入
```
