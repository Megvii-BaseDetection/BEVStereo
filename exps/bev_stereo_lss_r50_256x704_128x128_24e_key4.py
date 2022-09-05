# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3427
mATE: 0.6560
mASE: 0.2784
mAOE: 0.5982
mAVE: 0.5347
mAAE: 0.2228
NDS: 0.4423
Eval time: 116.3s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.508   0.518   0.163   0.188   0.534   0.230
truck   0.268   0.709   0.214   0.215   0.510   0.226
bus     0.379   0.640   0.207   0.142   1.049   0.315
trailer 0.151   0.953   0.240   0.541   0.618   0.113
construction_vehicle    0.092   0.955   0.514   1.360   0.113   0.394
pedestrian      0.350   0.727   0.300   1.013   0.598   0.328
motorcycle      0.371   0.576   0.259   0.777   0.634   0.175
bicycle 0.325   0.512   0.261   0.942   0.221   0.002
traffic_cone    0.489   0.503   0.345   nan     nan     nan
barrier 0.495   0.468   0.280   0.206   nan     nan
"""
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl

from exps.bev_stereo_lss_r50_256x704_128x128_24e_2key import \
    BEVStereoLightningModel as BaseBEVStereoLightningModel


class BEVStereoLightningModel(BaseBEVStereoLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_sweeps = 2
        self.sweep_idxes = [4]
        self.key_idxes = list()


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
    parser.set_defaults(
        profiler='simple',
        deterministic=False,
        max_epochs=24,
        accelerator='ddp',
        num_sanity_val_steps=0,
        gradient_clip_val=5,
        limit_val_batches=0,
        enable_checkpointing=True,
        precision=16,
        default_root_dir='./outputs/bev_stereo_lss_r50_256x704_'
        '128x128_24e_key4')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
