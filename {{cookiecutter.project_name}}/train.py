import os
import logging
import tarfile
from shutil import copytree, ignore_patterns

import torch
import torch.nn as nn
import torch.distributed as dist

from polyaxon.tracking import Run

from phobos.loss import get_loss
from phobos.runner import Runner
from phobos.grain import Grain

from models.unet import UNet
from dataloader import get_dataloaders


def local_testing():
    if 'POLYAXON_NO_OP' in os.environ:
        if os.environ['POLYAXON_NO_OP'] == 'true':
            return True
    else:
        False


experiment = None
if not local_testing():
    experiment = Run()

grain_exp = Grain(polyaxon_exp=experiment)
args = grain_exp.parse_args_from_json('metadata.json')

logging.basicConfig(level=logging.INFO)
"""
Set up environment: define paths, download data, and set device
"""

if not local_testing():
    if not os.path.exists(args.local_artifacts_path):
        os.makedirs(args.local_artifacts_path)
    tf = tarfile.open(os.path.join(args.nfs_data_path, 'train.tar.gz'))
    tf.extractall(args.local_artifacts_path)
    tf = tarfile.open(os.path.join(args.nfs_data_path, 'test.tar.gz'))
    tf.extractall(args.local_artifacts_path)
    args.dataset_dir = os.path.join(args.local_artifacts_path)

    # log code to artifact/code folder
    # code_path = os.path.join(experiment.get_artifacts_path(), 'code')
    # copytree('.', code_path, ignore=ignore_patterns('.*'))

    # set artifact/weight folder
    args.weight_dir = os.path.join(experiment.get_artifacts_path(), 'weights')

if not os.path.exists(args.weight_dir):
    os.makedirs(args.weight_dir)

train_loader, val_loader = get_dataloaders(args)
"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')

if args.model == 'unet':
    model = grain_exp.load_model(UNet,
                                 n_channels=len(args.band_ids),
                                 n_classes=1)

if args.pretrained_checkpoint:
    pretrained = torch.load(args.pretrained_checkpoint)
    model.load_state_dict(pretrained)

if args.gpu > -1:
    model = model.to(args.gpu)
    if args.num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

if args.resume_checkpoint:
    weight = torch.load(args.resume_checkpoint)
    model.load_state_dict(weight)



criterion = get_loss(args.loss, args.loss_args)
# optimizer = optim.SGD(model.parameters(), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)

runner = Runner(device=0,
                model=model,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                distributed=args.distributed,
                metrics=args.metrics,
                optimizer=args.optimizer,
                optimizer_args=args.optimizer_args,
                polyaxon_exp=experiment)

best_dc = -1
best_metrics = None

logging.info('STARTING training')
for epoch in range(args.epochs):
    """
    Begin Training
    """
    logging.info('SET model mode to train!')
    runner.set_epoch_metrics()
    train_metrics = runner.train_model()
    eval_metrics = runner.eval_model()
    print(train_metrics)
    print(eval_metrics)
    """
    Store the weights of good epochs based on validation results
    """
    if eval_metrics['val_dc'] > best_dc:
        cpt_path = os.path.join(args.weight_dir,
                                'checkpoint_epoch_' + str(epoch) + '.pt')
        torch.save(model.state_dict(), cpt_path)
        best_dc = eval_metrics['val_dc']

        best_metrics = {**train_metrics, **eval_metrics}
        if not local_testing():
            experiment.log_outputs(**best_metrics)

if not local_testing():
    experiment.log_outputs(**best_metrics)