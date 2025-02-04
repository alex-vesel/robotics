import torch


def process_delta_angle(delta_angle):
    return delta_angle / 30


class TrainConfigModule():
    def __init__(self, name, loss_fn, process_gnd_truth_fn, head, type, gt_key, weight=1, mask=None):
        self.name = name
        self.loss_fn = loss_fn
        self.process_gnd_truth_fn = process_gnd_truth_fn
        self.head = head
        self.type = type
        self.gt_key = gt_key
        self.weight = weight
        self.mask = mask


    def get_loss(self, outputs, y, split=None, weight=None):
        if self.type == "likelihood":
            losses = self.loss_fn(outputs[:, :2], self.process_gnd_truth_fn(delta_angle), var=outputs[:, 2])
        else:
            losses = self.loss_fn(outputs.squeeze(), self.process_gnd_truth_fn(y))

        losses *= self.weight

        if weight is not None:
            losses = losses.mean(dim=1) * weight

        return losses.mean()


    def __str__(self):
        return f'{self.name} - {self.loss_fn} - {self.head}'