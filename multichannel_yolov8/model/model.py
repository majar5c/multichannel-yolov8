import inspect
import ultralytics
from ultralytics.nn import SegmentationModel

from multichannel_yolov8.model.model_util import MultiChannelSegmentationTrainer, MultiChannelSegmentationValidator

class MultiChannelSegmentationModel(ultralytics.engine.model.Model):
  def smart_load(self, key):
    """Load model/trainer/validator/predictor."""
    map = {
            'model': SegmentationModel,
            'trainer': MultiChannelSegmentationTrainer,
            'validator': MultiChannelSegmentationValidator,
            'predictor': ultralytics.models.yolo.segment.SegmentationPredictor}
    try:
        return map[key]
    except Exception:
        name = self.__class__.__name__
        mode = inspect.stack()[1][3]  # get the function name.
        raise NotImplementedError(
            f'WARNING ⚠️ `{name}` model does not support `{mode}` mode for `{self.task}` task yet.')