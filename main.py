import os
import argparse
import torch
import random
import numpy as np
import wandb
from models.model import AudioModel, FusionModel, CrossAttentionModel, FusionModelAudioCLIP, VisionModel
from models.engine import train, test

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_args():
    parser = argparse.ArgumentParser()
    
    ## general
    parser.add_argument('--model', default='fusion', type=str, help='Choose between audio, vision, fusion, fusion_audioclip, and cross_attention')
    parser.add_argument('--device', default='0', type=str, help='device')

    ## training
    parser.add_argument('--audio_lr', default=None, type=float, help='learning rate for audio parameters')
    parser.add_argument('--vision_lr', default=None, type=float, help='learning rate for vision parameters')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate for non pretrained parameters')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for regularization')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warmup proportion for learning rate scheduler')
    parser.add_argument('--dropout_rate', default=0.3, type=float, help='dropout probability')
    parser.add_argument('--loss_function', default='cross_entropy', type=str, help='Warning: the custom loss function currently does not work correctly. Please use cross_entropy.')
    parser.add_argument('--custom_loss_ce_factor', default=1.0, type=float, help='Factor for the cross entropy part of the custom loss function. Only applicable if you are using the custom loss function.')
    parser.add_argument('--custom_loss_l2_factor', default=1.0, type=float, help='Factor for the MSE part of the custom loss function. Only applicable if you are using the custom loss function.')
    parser.add_argument('--balance_classes', default=False, type=bool, help='Choose True to optimize the model and calculate metrics per-class, choose False to do so globally, without balancing classes.')

    ## model
    parser.add_argument('--pretrained_vision_model', default="openai/clip-vit-base-patch32", type=str, help="source for pretrained vision model")
    parser.add_argument('--pretrained_audio_model', default="laion/clap-htsat-unfused", type=str, help="source for pretrained audio model")
    parser.add_argument('--freeze_pretrained_model', default=False, action='store_true', help='Whether to freeze the pretrained models')
    parser.add_argument('--num_train_epochs', default=25, type=int, help='number of epochs for training')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size for training and testing')
    parser.add_argument('--num_heads_ca', default=8, type=int, help='number of heads for cross attention')
    parser.add_argument('--label_number', default=9, type=int, help='Number of classes')

    ## experiment
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    ## data
    parser.add_argument('--num_workers', default=24, type=int, help='number of workers')
    parser.add_argument('--model_output_directory', default='model/checkpoints', type=str, help='Folder to save model checkpoint files to')
    parser.add_argument('--path_to_pt', default='data/liris_accede/preprocessed/', type=str, help='path to preprocessed .pt files')
    parser.add_argument('--wandb_tag', default=None, type=str, help='Additional wandb tag to add')

    ## ablation
    parser.add_argument('--dataset', default='mediaeval', type=str, help='Choose between mediaeval, holdout, holdout_accede, random and stratified. See preprocessing notebook for details.')
    parser.add_argument('--pooling', default='projection_output', type=str, help='At which stage to extract the pooled embeddings from the CLAP model. Choose between pooler_output and projection_output')
    parser.add_argument('--num_unfreeze_layers', default=2, type=int, help='number of layers to unfreeze (not implemented)')

    return parser.parse_args()

def main():
    args = set_args()

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

    wandb_tags = ['final metrics', 'final testing']

    if args.wandb_tag is not None:
        wandb_tags.append(args.wandb_tag)
    
    wandb.init(project='ba', notes='final', tags=wandb_tags, config=vars(args))
    wandb.watch_called = False

    ## define learning rate
    if not args.audio_lr:
        args.audio_lr = args.lr

    if not args.vision_lr:
        args.vision_lr = args.lr

    # Freeze encoder
    if args.model == 'fusion':
        model = FusionModel(args)
        if args.freeze_pretrained_model == True:
            for _, p in model.model_vision.vision_model.named_parameters():
                p.requires_grad = False
            for _, p in model.model_audio.audio_model.named_parameters():
                p.requires_grad = False
            
    elif args.model == 'cross_attention':
        model = CrossAttentionModel(args)
        if args.freeze_pretrained_model == True:
            for _, p in model.audioclip.named_parameters():
                p.requires_grad = False

    elif args.model == 'fusion_audioclip':
        model = FusionModelAudioCLIP(args)
        if args.freeze_pretrained_model == True:
            for _, p in model.audioclip.named_parameters():
                p.requires_grad = False

    elif args.model == 'audio':
        model = AudioModel(args)
        if args.freeze_pretrained_model == True:
            for _, p in model.model_audio.audio_model.named_parameters():
                p.requires_grad = False
                
    elif args.model == 'vision':
        model = VisionModel(args)
        if args.freeze_pretrained_model == True:
            for _, p in model.model_vision.vision_model.named_parameters():
                p.requires_grad = False

    else:
        raise ValueError('Illegal argument for --model: ' + str(args.model))


    model.to(device)
    wandb.watch(model, log='all')

    wandb.define_metric('epoch', hidden=True)

    for set in ['train', 'val', 'test']:
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

    train(args, model, device)
    test(args, model, device)

if __name__ == '__main__':
    main()
