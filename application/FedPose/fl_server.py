import logging
from coala.distributed.distributed import CPU
import torch
import numpy as np
from coala.tracking import metric
from coala.server.base import BaseServer
from utils.loss import JointsMSELoss
from utils.transforms import flip_back
from utils.inference import get_final_preds
from utils.evaluate import get_accuracy, pose_evaluate
from coala.utils.metric import AverageMeter

logger = logging.getLogger(__name__)


class PoseServer(BaseServer):
    def __init__(self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999):
        super(PoseServer, self).__init__(conf, test_data, val_data, is_remote, local_port)

    def test_in_server(self, device=CPU):
        self.model.eval()
        self.model.to(device)
        loss_fn = self.load_loss_fn(self.conf.client)
        test_loader = self.test_data.loader(self.conf.server.batch_size, shuffle=False, seed=self.conf.seed)

        num_samples = self.test_data.size()
        all_preds = np.zeros((num_samples, self.conf.model.num_joints, 3), dtype=np.float32)
        all_boxes = np.zeros((num_samples, 6))
        image_path = []
        idx = 0
        losses = AverageMeter()
        acc = AverageMeter()
        with torch.no_grad():
            for input, target, target_weight, meta in test_loader:
                # compute output
                input, target, target_weight = input.to(device), target.to(device), target_weight.to(device)
                output = self.model(input)
                if self.conf.test.flip_test:
                    # this part is ugly, because pytorch has not supported negative index
                    # input_flipped = model(input[:, :, :, ::-1])
                    input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                    input_flipped = torch.from_numpy(input_flipped).cuda()
                    output_flipped = self.model(input_flipped)
                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                               self.test_data.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if self.conf.test.shift_heatmap:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                loss = loss_fn(output, target, target_weight)

                # measure accuracy and record loss
                num_images = input.size(0)
                losses.update(loss.item(), num_images)
                # measure accuracy and record loss
                _, avg_acc, cnt, pred = get_accuracy(output.cpu().numpy(),
                                                     target.cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()

                preds, maxvals = get_final_preds(self.conf, output.clone().cpu().numpy(), c, s)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])

                idx += num_images

            name_values = pose_evaluate(self.conf, all_preds)

            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(name_value)
            else:
                _print_name_value(name_values)

        name_values['accuracy'] = 100 * acc.avg
        name_values['loss'] = losses.avg

        test_results = {metric.TEST_METRIC: name_values}

        return test_results

    def load_loss_fn(self, conf):
        criterion = JointsMSELoss()
        return criterion


def _print_name_value(name_value):
    names = name_value.keys()
    values = name_value.values()
    logger.info(
        '|' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info(
        '|' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )
