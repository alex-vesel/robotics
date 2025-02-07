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


    def get_loss(self, outputs, batch, split=None, weight=None):
        if self.type == "likelihood":
            losses = self.loss_fn(outputs[:, :2], self.process_gnd_truth_fn(delta_angle), var=outputs[:, 2])
        else:
            losses = self.loss_fn(outputs.squeeze(), self.process_gnd_truth_fn(batch[self.gt_key]))

        losses *= self.weight
        if len(losses.shape) > 1:
            losses = losses.mean(dim=1)

        mask = torch.ones_like(losses)
        for mask_fn in self.mask:
            mask *= mask_fn(batch)

        losses = losses[mask.bool()]

        if weight is not None:
            weight = weight[mask.bool()]
            losses = losses * weight

        return losses.mean()


    def __str__(self):
        return f'{self.name} - {self.loss_fn} - {self.head}'