from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.models.yolo.segment.val import SegmentationValidator

from ultralytics.utils.plotting import output_to_target
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, callbacks, colorstr, emojis
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.nn.autobackend import AutoBackend

import torch
from tqdm import tqdm
import json
from copy import copy

from multichannel_yolov8.utils.plot import plot_images
from multichannel_yolov8.utils.dataset import build_yolo_dataset

class MultiChannelSegmentationTrainer(SegmentationTrainer):
  def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)
  def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        plot_images(batch['img'],
                    batch['batch_idx'],
                    batch['cls'].squeeze(-1),
                    batch['bboxes'],
                    batch['masks'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

  def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss'
        return MultiChannelSegmentationValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

class MultiChannelSegmentationValidator(SegmentationValidator):
  @smart_inference_mode()
  def __call__(self, trainer=None, model=None):
      """
      Supports validation of a pre-trained model if passed or a model being trained
      if trainer is passed (trainer gets priority).
      """
      self.training = trainer is not None
      augment = self.args.augment and (not self.training)
      if self.training:
          self.device = trainer.device
          self.data = trainer.data
          model = trainer.ema.ema or trainer.model
          self.args.half = self.device.type != 'cpu'  # force FP16 val during training
          model = model.half() if self.args.half else model.float()
          self.model = model
          self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
          self.args.plots = trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
          model.eval()
      else:
          callbacks.add_integration_callbacks(self)
          self.run_callbacks('on_val_start')
          assert model is not None, 'Either trainer or model is needed for validation'
          model = AutoBackend(model,
                              device=select_device(self.args.device, self.args.batch),
                              dnn=self.args.dnn,
                              data=self.args.data,
                              fp16=self.args.half)
          self.model = model
          self.device = model.device  # update device
          self.args.half = model.fp16  # update half
          stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
          imgsz = check_imgsz(self.args.imgsz, stride=stride)
          if engine:
              self.args.batch = model.batch_size
          elif not pt and not jit:
              self.args.batch = 1  # export.py models default to batch-size 1
              LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

          if isinstance(self.args.data, str) and self.args.data.split('.')[-1] in ('yaml', 'yml'):
              self.data = check_det_dataset(self.args.data)
          elif self.args.task == 'classify':
              self.data = check_cls_dataset(self.args.data, split=self.args.split)
          else:
              raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

          if self.device.type == 'cpu':
              self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
          if not pt:
              self.args.rect = False
          self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

          model.eval()
          model.warmup(imgsz=(1 if pt else self.args.batch, next(model.named_parameters())[1].shape[1], imgsz, imgsz))  # warmup

      dt = Profile(), Profile(), Profile(), Profile()
      n_batches = len(self.dataloader)
      desc = self.get_desc()
      # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
      # which may affect classification task since this arg is in yolov5/classify/val.py.
      # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
      bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
      self.init_metrics(de_parallel(model))
      self.jdict = []  # empty before each val
      for batch_i, batch in enumerate(bar):
          self.run_callbacks('on_val_batch_start')
          self.batch_i = batch_i
          # Preprocess
          with dt[0]:
              batch = self.preprocess(batch)

          # Inference
          with dt[1]:
              preds = model(batch['img'], augment=augment)

          # Loss
          with dt[2]:
              if self.training:
                  self.loss += model.loss(batch, preds)[1]

          # Postprocess
          with dt[3]:
              preds = self.postprocess(preds)

          self.update_metrics(preds, batch)
          if self.args.plots and batch_i < 3:
              self.plot_val_samples(batch, batch_i)
              self.plot_predictions(batch, preds, batch_i)

          self.run_callbacks('on_val_batch_end')
      stats = self.get_stats()
      self.check_stats(stats)
      self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
      self.finalize_metrics()
      self.print_results()
      self.run_callbacks('on_val_end')
      if self.training:
          model.float()
          results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
          return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
      else:
          LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                      tuple(self.speed.values()))
          if self.args.save_json and self.jdict:
              with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                  LOGGER.info(f'Saving {f.name}...')
                  json.dump(self.jdict, f)  # flatten and save
              stats = self.eval_json(stats)  # update stats
          if self.args.plots or self.args.save_json:
              LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
          return stats

  def plot_val_samples(self, batch, ni):
        """Plots validation samples with bounding box labels."""
        plot_images(batch['img'],
                    batch['batch_idx'],
                    batch['cls'].squeeze(-1),
                    batch['bboxes'],
                    batch['masks'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_labels.jpg',
                    names=self.names,
                    on_plot=self.on_plot)

  def plot_predictions(self, batch, preds, ni):
      """Plots batch predictions with masks and bounding boxes."""
      plot_images(
          batch['img'],
          *output_to_target(preds[0], max_det=15),  # not set to self.args.max_det due to slow plotting speed
          torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
          paths=batch['im_file'],
          fname=self.save_dir / f'val_batch{ni}_pred.jpg',
          names=self.names,
          on_plot=self.on_plot)  # pred
      self.plot_masks.clear()