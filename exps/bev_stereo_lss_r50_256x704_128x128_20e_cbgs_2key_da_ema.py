# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3721
mATE: 0.5980
mASE: 0.2701
mAOE: 0.4381
mAVE: 0.3672
mAAE: 0.1898
NDS: 0.4997
Eval time: 138.0s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.567   0.457   0.156   0.104   0.343   0.204
truck   0.299   0.650   0.205   0.103   0.321   0.197
bus     0.394   0.613   0.203   0.106   0.643   0.252
trailer 0.178   0.991   0.239   0.433   0.345   0.070
construction_vehicle    0.102   0.826   0.458   1.055   0.114   0.372
pedestrian      0.402   0.653   0.297   0.803   0.479   0.249
motorcycle      0.356   0.553   0.251   0.450   0.512   0.168
bicycle 0.311   0.440   0.265   0.779   0.180   0.006
traffic_cone    0.552   0.420   0.336   nan     nan     nan
barrier 0.561   0.377   0.291   0.111   nan     nan
"""
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch

from callbacks.ema import EMACallback
from exps.bev_stereo_lss_r50_256x704_128x128_20e_cbgs_2key_da import \
    BEVStereoLightningModel as BaseBEVStereoLightningModel


class BEVStereoLightningModel(BaseBEVStereoLightningModel):
    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-2)
        return [optimizer]


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
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=20,
                        accelerator='ddp',
                        num_sanity_val_steps=0,
                        gradient_clip_val=5,
                        limit_val_batches=0,
                        enable_checkpointing=False,
                        precision=16,
                        default_root_dir='./outputs/bev_stereo_lss_r50_256x704'
                        '_128x128_20e_cbgs_2key_da_ema')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
