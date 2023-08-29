from ultralytics.data.augment import Mosaic, CopyPaste, RandomPerspective, LetterBox, Compose, MixUp, RandomFlip, RandomHSV
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version

import numpy as np
import random

class Albumentations:
    """Albumentations transformations. Optional, uninstall package to disable.
    Applies Blur, Median Blur, convert to grayscale, Contrast Limited Adaptive Histogram Equalization,
    random change of brightness and contrast, RandomGamma and lowering of image quality by compression."""

    def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        prefix = colorstr('albumentations: ')
        try:
            import albumentations as A

            check_version(A.__version__, '1.0.3', hard=True)  # version requirement
            
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                
                ## Not supported for more than 3 channels
                # A.ToGray(p=0.01), # Only works on 3 channels
                # A.CLAHE(p=0.01), # Only works on 1 or 3 channels
                # A.ImageCompression(quality_lower=75, p=0.0)  # cannot reshape array of size 49152 into shape (128,128,4)
                ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    def __call__(self, labels):
        """Generates object detections and returns a dictionary with detection results."""
        im = labels['img']
        cls = labels['cls']
        if len(cls):
            labels['instances'].convert_bbox('xywh')
            labels['instances'].normalize(*im.shape[:2][::-1])
            bboxes = labels['instances'].bboxes
            # TODO: add supports of segments and keypoints
            if self.transform and random.random() < self.p:
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                if len(new['class_labels']) > 0:  # skip update if no bbox in new im
                    labels['img'] = new['image']
                    labels['cls'] = np.array(new['class_labels'])
                    bboxes = np.array(new['bboxes'], dtype=np.float32)
            labels['instances'].update(bboxes=bboxes)
        return labels


def multichannel_transforms(dataset, imgsz, hyp, stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = Compose([
        Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
        CopyPaste(p=hyp.copy_paste),
        RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
        )])
    flip_idx = dataset.data.get('flip_idx', [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get('kpt_shape', None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')

    return Compose([
        pre_transform,
        MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
        Albumentations(p=1.0),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction='vertical', p=hyp.flipud),
        RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)])  # transforms