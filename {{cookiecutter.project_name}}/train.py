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

from models.model import Dummy
from dataloader import get_dataloaders


################### Polyaxon / Local ###################
"""
Initialization to use datalab or local system for training.
"""
def local_testing():
    if 'POLYAXON_NO_OP' in os.environ:
        if os.environ['POLYAXON_NO_OP'] == 'true':
            return True
    else:
        False


experiment = None
if not local_testing():
    experiment = Run()


################### Polyaxon / Local ###################

################### Arguments ###################

"""Initialize all arguments passed via metadata.json
"""
grain_exp = Grain(polyaxon_exp=experiment)
args = grain_exp.parse_args_from_yaml('metadata.yaml')

################### Arguments ###################

############## Distributed w/d polyaxon ################
if args.distributed and not local_testing():
    Runner.distributed()
########################################################


################### Setup Data and Weight ###################

if not local_testing():
    """
    When using datalab for training, we need to see how data is stored in datastore 
    and copy, untar, or pass url properly depending on how we use the datastore. 
    This will require a bit of effort in understanding the structure of the dataset, 
    like are train, val tarred together or are they different. Are we using webdataset
    shards, etc. We will eventually move to a unified framwork under webdataset-aistore
    for all dataset coming from Europa and all third party open-source datasets.
    """
    if not os.path.exists(args.local_artifacts_path):
        os.makedirs(args.local_artifacts_path)
    #tf = tarfile.open(os.path.join(args.nfs_data_path, 'train.tar.gz'))
    #tf.extractall(args.local_artifacts_path)
    #tf = tarfile.open(os.path.join(args.nfs_data_path, 'test.tar.gz'))
    #tf.extractall(args.local_artifacts_path)
    #args.dataset_dir = os.path.join(args.local_artifacts_path)

    # log code to artifact/code folder
    # code_path = os.path.join(experiment.get_artifacts_path(), 'code')
    # copytree('.', code_path, ignore=ignore_patterns('.*'))

    # set artifact/weight folder
    args.weight_dir = os.path.join(experiment.get_artifacts_path(), 'weights')

if not os.path.exists(args.weight_dir):
    os.makedirs(args.weight_dir)


train_loader, val_loader = get_dataloaders(args)

################### Setup Data and Weight Directories ###################


################### Intialize Model ###################
"""
Load Model then define other aspects of the model
"""

if args.model == 'dummy':
    """
    Make sure that all args that are passed to model class are passed
    via grain load_model functions. This allows us to later use the 
    arguments as it is during inference.
    """
    model = grain_exp.load_model(Dummy,
                                 n_channels=len(args.band_ids),
                                 n_classes=args.num_classes)

if args.pretrained_checkpoint:
    """
    If you have any pretrained weights that you want to load for the model, this 
    is the place to do it.
    """
    pretrained = torch.load(args.pretrained_checkpoint)
    model.load_state_dict(pretrained)

if args.gpu > -1:
    """Ad-hoc method to use gpu(s). We will be moving to DistributedParallel for all 
    datalab trainings.
    """
    model = model.to(args.gpu)
    if args.num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

elif args.distributed:
    model = model.to(args.gpu)
    model = nn.parallel.DistributedDataParallel(model,find_unused_parameters=False)

if args.resume_checkpoint:
    """If we want to resume training from some checkpoints.
    """
    weight = torch.load(args.resume_checkpoint)
    model.load_state_dict(weight)

################### Intialize Model ###################

################### Intialize Runner ###################

criterion = get_loss(args.loss, args.loss_args)

runner = Runner(
    model=model,
    device=0,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=args.optimizer,
    optimizer_args=args.optimizer_args,
    scheduler=args.scheduler,
    scheduler_args=args.scheduler_args,
    distributed=args.distributed,
    distributed_val=args.distributed_val,
    max_iters = args.max_iters,
    mode = args.mode,
    val_frequency=args.val_frequency,
    tensorboard_logging = True,
    polyaxon_exp=experiment,
    metrics = args.metrics,
    num_classes = args.num_classes
)

################### Intialize Runner ###################

################### Train ###################
"""Dice coeffiecient is used to select best model weights.
Use metric as you think is best for your problem.
"""
best_dc = -1 
best_metrics = None

logging.info('STARTING training')

for step,train_metrics,eval_metrics in runner.trainer():
    """
    Begin Training
    """
    print(train_metrics)
    print(eval_metrics)
    if (not args.distributed or local_testing) or (args.distributed and dist.get_rank() == 0):
        """
        Store the weights of good epochs based on validation results
        """
        if eval_metrics['val_dc'] > best_dc:
            cpt_path = os.path.join(args.weight_dir,
                                    'checkpoint_epoch_' + str(step) + '.pt')
            
            model_dict = None
            if args.distributed and not local_testing:
                model_dict = model.module.state_dict()
            else:
                model_dict = model.state_dict()
            
            torch.save(model_dict, cpt_path)
            best_dc = eval_metrics['val_dc']

            best_metrics = {**train_metrics, **eval_metrics}
            if not local_testing():
                experiment.log_outputs(**best_metrics)

if not local_testing():
    experiment.log_outputs(**best_metrics)

################### Train ###################