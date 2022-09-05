# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3435
mATE: 0.6585
mASE: 0.2757
mAOE: 0.5792
mAVE: 0.5034
mAAE: 0.2163
NDS: 0.4485
Eval time: 159.3s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.511   0.517   0.165   0.190   0.517   0.227
truck   0.287   0.700   0.211   0.205   0.464   0.231
bus     0.383   0.670   0.214   0.129   1.015   0.296
trailer 0.148   1.020   0.238   0.688   0.471   0.087
construction_vehicle    0.087   0.878   0.485   1.329   0.097   0.365
pedestrian      0.350   0.720   0.299   0.992   0.583   0.324
motorcycle      0.355   0.594   0.255   0.658   0.661   0.192
bicycle 0.334   0.503   0.269   0.827   0.219   0.007
traffic_cone    0.486   0.498   0.350   nan     nan     nan
barrier 0.494   0.486   0.271   0.196   nan     nan
"""

from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl

from callbacks.ema import EMACallback
from exps.bev_stereo_lss_r50_256x704_128x128_24e_2key_ema import \
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
    train_dataloader = model.train_dataloader()
    ema_callback = EMACallback(len(train_dataloader.dataset) * args.max_epochs)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback])
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
        enable_checkpointing=False,
        precision=16,
        default_root_dir='./outputs/bev_stereo_lss_r50_256x704_'
        '128x128_24e_key4_ema')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
