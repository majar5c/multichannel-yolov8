import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
from ultralytics.utils import (ARM64, DEFAULT_CFG, LINUX, LOGGER, MACOS, ROOT, WINDOWS, __version__, callbacks,
                               colorstr)
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.files import file_size
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

from ultralytics.engine.exporter import Exporter, export_formats

class MultiChannelExporter(Exporter):
    @smart_inference_mode()
    def __call__(self, model=None):
        """Returns list of exported files/dirs after running callbacks."""
        self.run_callbacks('on_export_start')
        t = time.time()
        format = self.args.format.lower()  # to lowercase
        if format in ('tensorrt', 'trt'):  # 'engine' aliases
            format = 'engine'
        if format in ('mlmodel', 'mlpackage', 'mlprogram', 'apple', 'ios'):  # 'coreml' aliases
            format = 'coreml'
        fmts = tuple(export_formats()['Argument'][1:])  # available export formats
        flags = [x == format for x in fmts]
        if sum(flags) != 1:
            raise ValueError(f"Invalid export format='{format}'. Valid formats are {fmts}")
        jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, ncnn = flags  # export booleans

        # Device
        if format == 'engine' and self.args.device is None:
            LOGGER.warning('WARNING ⚠️ TensorRT requires GPU export, automatically assigning device=0')
            self.args.device = '0'
        self.device = select_device('cpu' if self.args.device is None else self.args.device)

        # Checks
        model.names = check_class_names(model.names)
        if self.args.half and onnx and self.device.type == 'cpu':
            LOGGER.warning('WARNING ⚠️ half=True only compatible with GPU export, i.e. use device=0')
            self.args.half = False
            assert not self.args.dynamic, 'half=True not compatible with dynamic=True, i.e. use only one.'
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.args.optimize:
            assert not ncnn, "optimize=True not compatible with format='ncnn', i.e. use optimize=False"
            assert self.device.type == 'cpu', "optimize=True not compatible with cuda devices, i.e. use device='cpu'"
        if edgetpu and not LINUX:
            raise SystemError('Edge TPU export only supported on Linux. See https://coral.ai/docs/edgetpu/compiler/')

        # Input
        im = torch.zeros(self.args.batch, next(model.named_parameters())[1].shape[1], *self.imgsz).to(self.device)
        file = Path(
            getattr(model, 'pt_path', None) or getattr(model, 'yaml_file', None) or model.yaml.get('yaml_file', ''))
        if file.suffix in ('.yaml', '.yml'):
            file = Path(file.name)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for m in model.modules():
            if isinstance(m, (Detect, RTDETRDecoder)):  # Segment and Pose use Detect base class
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
            elif isinstance(m, C2f) and not any((saved_model, pb, tflite, edgetpu, tfjs)):
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split

        y = None
        for _ in range(2):
            y = model(im)  # dry runs
        if self.args.half and (engine or onnx) and self.device.type != 'cpu':
            im, model = im.half(), model.half()  # to FP16

        # Filter warnings
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
        warnings.filterwarnings('ignore', category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
        warnings.filterwarnings('ignore', category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

        # Assign
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = tuple(y.shape) if isinstance(y, torch.Tensor) else \
            tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        self.pretty_name = Path(self.model.yaml.get('yaml_file', self.file)).stem.replace('yolo', 'YOLO')
        data = model.args['data'] if hasattr(model, 'args') and isinstance(model.args, dict) else ''
        description = f'Ultralytics {self.pretty_name} model {f"trained on {data}" if data else ""}'
        self.metadata = {
            'description': description,
            'author': 'Ultralytics',
            'license': 'AGPL-3.0 https://ultralytics.com/license',
            'date': datetime.now().isoformat(),
            'version': __version__,
            'stride': int(max(model.stride)),
            'task': model.task,
            'batch': self.args.batch,
            'imgsz': self.imgsz,
            'names': model.names}  # model metadata
        if model.task == 'pose':
            self.metadata['kpt_shape'] = model.model[-1].kpt_shape

        LOGGER.info(f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
                    f'output shape(s) {self.output_shape} ({file_size(file):.1f} MB)')

        # Exports
        f = [''] * len(fmts)  # exported filenames
        if jit or ncnn:  # TorchScript
            f[0], _ = self.export_torchscript()
        if engine:  # TensorRT required before ONNX
            f[1], _ = self.export_engine()
        if onnx or xml:  # OpenVINO requires ONNX
            f[2], _ = self.export_onnx()
        if xml:  # OpenVINO
            f[3], _ = self.export_openvino()
        if coreml:  # CoreML
            f[4], _ = self.export_coreml()
        if any((saved_model, pb, tflite, edgetpu, tfjs)):  # TensorFlow formats
            self.args.int8 |= edgetpu
            f[5], keras_model = self.export_saved_model()
            if pb or tfjs:  # pb prerequisite to tfjs
                f[6], _ = self.export_pb(keras_model=keras_model)
            if tflite:
                f[7], _ = self.export_tflite(keras_model=keras_model, nms=False, agnostic_nms=self.args.agnostic_nms)
            if edgetpu:
                f[8], _ = self.export_edgetpu(tflite_model=Path(f[5]) / f'{self.file.stem}_full_integer_quant.tflite')
            if tfjs:
                f[9], _ = self.export_tfjs()
        if paddle:  # PaddlePaddle
            f[10], _ = self.export_paddle()
        if ncnn:  # ncnn
            f[11], _ = self.export_ncnn()

        # Finish
        f = [str(x) for x in f if x]  # filter out '' and None
        if any(f):
            f = str(Path(f[-1]))
            square = self.imgsz[0] == self.imgsz[1]
            s = '' if square else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not " \
                                  f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(' ', '')
            predict_data = f'data={data}' if model.task == 'segment' and format == 'pb' else ''
            q = 'int8' if self.args.int8 else 'half' if self.args.half else ''  # quantization
            LOGGER.info(f'\nExport complete ({time.time() - t:.1f}s)'
                        f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                        f'\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q} {predict_data}'
                        f'\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}'
                        f'\nVisualize:       https://netron.app')

        self.run_callbacks('on_export_end')
        return f  # return list of exported files/dirs