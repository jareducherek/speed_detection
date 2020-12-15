# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, pred_dim, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            pred_dim: dimension of prediction per timestep
            num_queries: number of predictions per output. This is the maximum number of predictions for a single output.
                         We recommend no more than 100.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.pred_embed = nn.Linear(hidden_dim, pred_dim)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.queries_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "preds": the prediction vectors for all queries.
                          Shape= [batch_size x num_queries x (pred_dim + 1)]
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        output_preds = self.pred_embed(hs)
        out = {'preds': output_preds}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_preds)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, output_preds):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'preds': a}
                for a in zip(output_preds[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    """
    def __init__(self, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            TODO: delete num_classes, eos_coef
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_l2(self, outputs, targets, log=True):
        """Mean Squared Error Loss (MSE)
        targets dicts must contain the key "preds" containing a tensor of dim [nb_target_preds]
        """
        assert 'preds' in outputs
        preds = outputs['preds']

        loss_mse = F.mse_loss(preds, targets)
        losses = {'loss_mse': loss_mse}

        if log:
            # TODO add logging loss for MSE
            pass
        return losses

    @torch.no_grad()
    def loss_l1(self, outputs, targets):
        """ Compute the mean absolute error (L1)
        This is not currently used to propagate gradients, for logging purposes only
        """
        assert 'preds' in outputs
        preds = outputs['preds']

        loss_mae = F.l1_loss(preds, targets)
        losses = {'loss_mae': loss_mae}

        return losses

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'l1': self.loss_l1,
            'l2': self.loss_l2,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_preds = sum(len(t["targets"]) for t in targets)
        num_preds = torch.as_tensor([num_preds], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_preds)
        num_preds = torch.clamp(num_preds / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'targets':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        pred_dim=args.pred_dim,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    weight_dict = {'loss_mse': 1, 'loss_mae': 0}
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['l2', 'l1']
    criterion = SetCriterion(weight_dict=weight_dict, losses=losses)
    criterion.to(device)

    return model, criterion
