# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3576
mATE: 0.6071
mASE: 0.2684
mAOE: 0.4157
mAVE: 0.3928
mAAE: 0.2021
NDS: 0.4902
Eval time: 129.7s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.559   0.465   0.157   0.110   0.350   0.205
truck   0.285   0.633   0.205   0.101   0.304   0.209
bus     0.373   0.667   0.204   0.076   0.896   0.345
trailer 0.167   0.956   0.228   0.482   0.289   0.100
construction_vehicle    0.077   0.869   0.454   1.024   0.108   0.335
pedestrian      0.402   0.652   0.299   0.821   0.493   0.253
motorcycle      0.321   0.544   0.255   0.484   0.529   0.159
bicycle 0.276   0.466   0.272   0.522   0.173   0.011
traffic_cone    0.551   0.432   0.321   nan     nan     nan
barrier 0.565   0.386   0.287   0.121   nan     nan
"""
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

from exps.bev_stereo_lss_r50_256x704_128x128_24e_2key import \
    BEVStereoLightningModel as BEVStereoLightningModel
from layers.backbones.lss_fpn import LSSFPN as BaseLSSFPN
from layers.heads.bev_stereo_head import BEVStereoHead
from models.bev_stereo import BEVStereo as BaseBEVStereo


class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class LSSFPN(BaseLSSFPN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth_aggregation_net = self._configure_depth_aggregation_net()

    def _configure_depth_aggregation_net(self):
        """build pixel cloud feature extractor"""
        return DepthAggregation(self.output_channels, self.output_channels,
                                self.output_channels)

    def _forward_voxel_net(self, img_feat_with_depth):
        # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
        img_feat_with_depth = img_feat_with_depth.permute(
            0, 3, 1, 4, 2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
        n, h, c, w, d = img_feat_with_depth.shape
        img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
        img_feat_with_depth = (
            self.depth_aggregation_net(img_feat_with_depth).view(
                n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float())
        return img_feat_with_depth


class BEVStereo(BaseBEVStereo):
    def __init__(self, backbone_conf, head_conf, is_train_depth=True):
        super(BaseBEVStereo, self).__init__()
        self.backbone = LSSFPN(**backbone_conf)
        self.head = BEVStereoHead(**head_conf)
        self.is_train_depth = is_train_depth


class BEVStereoLightningModel(BEVStereoLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = BEVStereo(self.backbone_conf,
                               self.head_conf,
                               is_train_depth=True)
        self.data_use_cbgs = True
        self.basic_lr_per_img = 2e-4 / 32

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-2)
        scheduler = MultiStepLR(optimizer, [16, 19])
        return [[optimizer], [scheduler]]


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = BEVStereoLightningModel(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = BEVStereoLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=20,
                        accelerator='ddp',
                        num_sanity_val_steps=0,
                        gradient_clip_val=5,
                        limit_val_batches=0,
                        enable_checkpointing=True,
                        precision=16,
                        default_root_dir='./outputs/bev_stereo_lss_r50_'
                        '256x704_128x128_20e_cbgs_2key_da')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
