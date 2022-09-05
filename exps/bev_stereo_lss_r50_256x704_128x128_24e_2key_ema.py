# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3494
mATE: 0.6672
mASE: 0.2785
mAOE: 0.5607
mAVE: 0.4687
mAAE: 0.2295
NDS: 0.4542
Eval time: 166.7s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.509   0.522   0.163   0.187   0.507   0.228
truck   0.287   0.694   0.213   0.202   0.449   0.229
bus     0.390   0.681   0.207   0.152   0.902   0.261
trailer 0.167   0.945   0.248   0.491   0.340   0.185
construction_vehicle    0.087   1.057   0.515   1.199   0.104   0.377
pedestrian      0.351   0.729   0.299   0.987   0.575   0.321
motorcycle      0.368   0.581   0.262   0.721   0.663   0.226
bicycle 0.338   0.494   0.258   0.921   0.209   0.008
traffic_cone    0.494   0.502   0.341   nan     nan     nan
barrier 0.502   0.467   0.278   0.185   nan     nan
"""
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed

from callbacks.ema import EMACallback
from exps.bev_stereo_lss_r50_256x704_128x128_24e_2key import \
    BEVStereoLightningModel as BaseBEVStereoLightningModel


class BEVStereoLightningModel(BaseBEVStereoLightningModel):
    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
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
        '128x128_24e_2key_ema')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
