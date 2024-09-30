import os
import argparse
import torch
import random
import numpy as np
import wandb
from models.model import AudioModel, FusionModel, CrossAttentionModel, FusionModelAudioCLIP, VisionModel
from models.engine import test

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_user_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--checkpoint_file', default='model/checkpoints/checkpoint_fusion_42_1lay.pth', type=str, help='Full path to the checkpoint file to load. Include the file extension.')

    # Optional arguments
    parser.add_argument('--device', default=None, type=str, help='Overrides the device argument in the loaded checkpoint')
    parser.add_argument('--balance_classes', default=None, type=bool, help='Overrides the balance_classes argument in the loaded checkpoint')
    parser.add_argument('--batch_size', default=None, type=int, help='Overrides the batch_size argument in the loaded checkpoint')
    parser.add_argument('--num_workers', default=None, type=int, help='Overrides the num_workers argument in the loaded checkpoint')
    parser.add_argument('--dataset', default=None, type=str, help='Overrides the dataset argument in the loaded checkpoint')

    return parser.parse_args()

def main():
    user_args = set_user_args()

    # Load checkpoint
    load_file = user_args.checkpoint_file
    checkpoint = torch.load(load_file)

    # Override arguments
    if user_args.device is not None:
        checkpoint['args'].device = user_args.device
    if user_args.balance_classes is not None:
        checkpoint['args'].balance_classes = user_args.balance_classes
    if user_args.batch_size is not None:
        checkpoint['args'].batch_size = user_args.batch_size
    if user_args.num_workers is not None:
        checkpoint['args'].num_workers = user_args.num_workers
    if user_args.dataset is not None:
        checkpoint['args'].dataset = user_args.dataset

    args = checkpoint['args']

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    print(f'Pytorch is using the following device: {device}')

    if args.seed is not None and args.seed >= 0:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    wandb_tags = ['final metrics']
    wandb_notes = f'Testing {user_args.checkpoint_file} on {args.dataset}'

    if args.wandb_tag is not None:
        wandb_tags.append(args.wandb_tag)
    
    # Testing currently only works in the wandb offline mode, using the online mode may possibly break the wandb workspace
    wandb.init(project='ba', notes=wandb_notes, tags=wandb_tags, config=vars(args), mode='offline')
    wandb.watch_called = False

    if args.model == 'fusion':
        model = FusionModel(args)
    elif args.model == 'cross_attention':
        model = CrossAttentionModel(args)
    elif args.model == 'fusion_audioclip':
        model = FusionModelAudioCLIP(args)
    elif args.model == 'vision':
        model = VisionModel(args)
    elif args.model == 'audio':
        model = AudioModel(args)

    model.load_state_dict(checkpoint['model'])

    model.to(device)
    wandb.watch(model, log='all')

    wandb.define_metric('epoch', hidden=True)

    for set in ['test']:
        wandb.define_metric(f'{set}/loss', step_metric='epoch')
        wandb.define_metric(f'{set}/acc', step_metric='epoch')
        wandb.define_metric(f'{set}/valence_acc', step_metric='epoch')
        wandb.define_metric(f'{set}/arousal_acc', step_metric='epoch')
        wandb.define_metric(f'{set}/auc_ovr', step_metric='epoch')
        wandb.define_metric(f'{set}/auc_ovo', step_metric='epoch')
        wandb.define_metric(f'{set}/f1', step_metric='epoch')
        wandb.define_metric(f'{set}/pre', step_metric='epoch')
        wandb.define_metric(f'{set}/rec', step_metric='epoch')
        wandb.define_metric(f'{set}/cm', step_metric='epoch')

    test(args, model, device)

if __name__ == '__main__':
    main()
