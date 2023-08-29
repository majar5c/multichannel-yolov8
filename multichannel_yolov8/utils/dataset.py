from multichannel_yolov8.utils.transform import multichannel_transforms

from ultralytics.utils import colorstr
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox

class MutlichannelDataset(YOLODataset):
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = multichannel_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32, augment=True):
    """Build YOLO Dataset"""
    return MutlichannelDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment= mode=='train' if augment else False,  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0)