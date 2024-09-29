import os
from utils.loss_function import CustomLoss
import wandb
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import CLIPImageProcessor, ClapProcessor
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.utils.logging import set_verbosity_error

from data.dataset import LirisAccedeDataset
from utils.metrics import evaluate
from utils.utils import calculate_metrics, calculate_weights
from utils.processor import AudioCLIPProcessor
from utils.early_stopping import EarlyStopping

logging.basicConfig(filename='logs/last_run.log', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# Prevents unnecessary warnings when loading pretrained models
set_verbosity_error()

def train(args, model, device):
    if not os.path.exists(args.model_output_directory):
        os.mkdir(args.model_output_directory)

    # Initialize processor(s)
    if args.model in ['cross_attention', 'fusion_audioclip']:
        processor = AudioCLIPProcessor()
    elif args.model == 'fusion':
        audio_processor = ClapProcessor.from_pretrained(args.pretrained_audio_model)
        vision_processor = CLIPImageProcessor.from_pretrained(args.pretrained_vision_model)
    elif args.model == 'audio':
        audio_processor = ClapProcessor.from_pretrained(args.pretrained_audio_model)
    elif args.model == 'vision':
        vision_processor = CLIPImageProcessor.from_pretrained(args.pretrained_vision_model)

    # Initialize dataset and dataloader
    train_data = LirisAccedeDataset(
        args, device,
        f'data/liris_accede/preprocessed/videos/',
        f'data/liris_accede/preprocessed/audios/',
        f'data/liris_accede/preprocessed/labels/',
        f'data/liris_accede/preprocessed/data_splits/train_ids_{args.dataset}.pt'
        )
    val_data = LirisAccedeDataset(
        args, device,
        f'data/liris_accede/preprocessed/videos/',
        f'data/liris_accede/preprocessed/audios/',
        f'data/liris_accede/preprocessed/labels/',
        f'data/liris_accede/preprocessed/data_splits/val_ids_{args.dataset}.pt'
    )
    train_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, collate_fn=LirisAccedeDataset.collate_func,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
    
    total_steps = int(len(train_loader) * args.num_train_epochs)
    model.to(device)

    # Get class weights
    if args.balance_classes:
        train_weights = calculate_weights(args, train_data).to(device)
    else:
        train_weights = None

    ## custom AdamW optimizer
    if args.model in ['cross_attention', 'fusion_audioclip']:
        base_params = [param for name, param in model.named_parameters() if 'visual' not in name and 'audio' not in name]
        optimizer = AdamW([{'params': base_params},
                        {'params': model.audioclip.visual.parameters(), 'lr': args.vision_lr*0.1},
                        {'params': model.audioclip.audio.parameters(), 'lr': args.audio_lr*0.1}], 
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.model == 'fusion': 
        base_params = [param for name, param in model.named_parameters() if 'model_vision' not in name and 'model_audio' not in name]
        optimizer = AdamW([{'params': base_params},
                        {'params': model.model_vision.parameters(), 'lr': args.vision_lr*0.1},
                        {'params': model.model_audio.parameters(), 'lr': args.audio_lr*0.1}], 
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.model == 'audio':
        base_params = [param for name, param in model.named_parameters() if 'model_audio' not in name]
        optimizer = AdamW([{'params': base_params},
                        {'params': model.model_audio.parameters(), 'lr': args.audio_lr*0.1}], 
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.model == 'vision':
        base_params = [param for name, param in model.named_parameters() if 'model_vision' not in name]
        optimizer = AdamW([{'params': base_params},
                           {'params': model.model_vision.parameters(), 'lr': args.vision_lr*0.1}],
                           lr=args.lr,
                           weight_decay=args.weight_decay)
        
    if args.loss_function == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=train_weights)
    elif args.loss_function == 'custom':
        criterion = CustomLoss(weight=train_weights, cross_entropy_factor=args.custom_loss_ce_factor, l2_factor=args.custom_loss_l2_factor)
    else:
        raise ValueError(f'Invalid argument for loss_function: {args.loss_function}. Please choose either cross_entropy or custom')

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps
        )

    max_accuracy = 0.

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.01, maximize=True)

    for i_epoch in tqdm(range(0, int(args.num_train_epochs)), desc='epoch', total=args.num_train_epochs, position=0, leave=True, disable=False):
        wandb.log({'epoch': i_epoch})

        model.train()

        prob = []
        y_pred = []
        targets = []
        running_loss = 0

        # Pretrained model unfreezing
        if i_epoch == 10:
            if args.model in ['cross_attention', 'fusion_audioclip']:
                for param in model.audioclip.audio.layer4.parameters():
                    param.requires_grad = True
                for param in model.audioclip.visual.layer4.parameters():
                    param.requires_grad = True

            elif args.model == 'fusion':
                for i in range(10, 12):
                    for param in model.model_vision.vision_model.encoder.layers[i].parameters():
                        param.requires_grad = True
                for param in model.model_audio.audio_model.audio_encoder.layers[3].parameters():
                    param.requires_grad = True

            elif args.model == 'audio':
                for param in model.model_audio.audio_model.audio_encoder.layers[3].parameters():
                    param.requires_grad = True

            elif args.model == 'vision':
                for i in range(10, 12):
                    for param in model.model_vision.vision_model.encoder.layers[i].parameters():
                        param.requires_grad = True


        for step, batch in tqdm(enumerate(train_loader), desc='batch', total=len(train_loader), position=1, leave=False):
            if args.model in ['cross_attention', 'fusion_audioclip']:
                videos, audios, labels = batch

                frames = torch.flatten(videos, start_dim=0, end_dim=1)
                video_inputs = processor.preprocess_images(frames).to(device, non_blocking=True)

                audio_inputs = processor.preprocess_audio(audios).to(device, non_blocking=True)
                
                batch_length = len(labels)

            elif args.model == 'fusion':
                videos, audios, labels = batch

                frames = torch.flatten(videos, start_dim=0, end_dim=1).to(device, non_blocking=True)

                audio_inputs = audio_processor(audios=audios, sampling_rate=48_000, return_tensors='pt').to(device)
                video_inputs = vision_processor(images=frames, padding=True, truncation=True, return_tensors='pt').to(device)

                batch_length = len(labels)
                
            elif args.model == 'audio':
                _, audios, labels = batch

                audio_inputs = audio_processor(audios=audios, sampling_rate=48_000, return_tensors='pt').to(device)

            elif args.model == 'vision':
                videos, _, labels = batch

                frames = torch.flatten(videos, start_dim=0, end_dim=1).to(device, non_blocking=True)

                video_inputs = vision_processor(images=frames, padding=True, truncation=True, return_tensors='pt').to(device)

                batch_length = len(labels)

            target = torch.tensor(labels).to(device, non_blocking=True)
            targets.extend(labels)

            with torch.autocast(device_type=device.type):
                if args.model in ['cross_attention', 'fusion_audioclip']:
                    pred = model(audio_inputs, video_inputs, batch_length)
                if args.model == 'fusion':
                    pred = model(audio_inputs, video_inputs, batch_length)
                elif args.model == 'audio':
                    pred = model(audio_inputs)
                elif args.model == 'vision':
                    pred = model(video_inputs, batch_length)
                prob.extend(torch.nn.functional.softmax(pred, dim=-1).detach().cpu())

                loss = criterion(pred, target)

            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        ## stats
        epoch_loss = running_loss / len(train_loader)

        targets = np.array(targets)
        prob = np.array(prob)

        acc, valence_acc, arousal_acc, auc_ovr, auc_ovo, f1, precision, recall, cm = calculate_metrics(args, targets, prob, num_labels=args.label_number)

        ## train results
        cm.title = 'train confusion matrix'
        wandb.log({
                f'train/loss': epoch_loss, 
                f'train/acc': acc, 
                f'train/valence_acc': valence_acc, 
                f'train/arousal_acc': arousal_acc, 
                f'train/auc_ovr': auc_ovr, 
                f'train/auc_ovo': auc_ovo,
                f'train/f1': f1, 
                f'train/pre': precision, 
                f'train/rec': recall, 
                f'train/cm': cm})
        logging.info('i_epoch is {}, train_loss is {}, train_acc is {}, train_f1 is {}, train_auc_ovr is {}, train_auc_ovo is {}, train_pre is {}, train_rec is {}'.format(i_epoch, epoch_loss, acc, f1, auc_ovr, auc_ovo, precision, recall))

        ## validation results
        if args.model in ['cross_attention', 'fusion_audioclip']:
            validation_acc = validate(args, model, device, val_data, processor=processor)
        elif args.model == 'fusion':
            validation_acc = validate(args, model, device, val_data, audio_processor=audio_processor, vision_processor=vision_processor)
        elif args.model == 'audio':
            validation_acc = validate(args, model, device, val_data, audio_processor=audio_processor)
        elif args.model == 'vision':
            validation_acc = validate(args, model, device, val_data, vision_processor=vision_processor)

        ## save best model
        if validation_acc > max_accuracy:
            max_accuracy = validation_acc

            if not os.path.exists(args.model_output_directory):
                os.mkdir(args.model_output_directory)

            checkpoint_file = f'checkpoint_{args.model}_{args.seed}_1lay.pth'

            dict_to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args
            }

            output_dir = os.path.join(args.model_output_directory, checkpoint_file)
            torch.save(dict_to_save, output_dir)

        if early_stopping.early_stop(validation_acc):
            break

    logger.info('Train done')

def validate(args, model, device, val_data, audio_processor=None, vision_processor=None, processor=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=LirisAccedeDataset.collate_func, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.balance_classes:
        val_weights = calculate_weights(args, val_data).to(device)
    else:
        val_weights = None

    # Initialize loss function
    if args.loss_function == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=val_weights)
    elif args.loss_function == 'custom':
        criterion = CustomLoss(weight=val_weights, cross_entropy_factor=args.custom_loss_ce_factor, l2_factor=args.custom_loss_l2_factor)

    if args.model in ['cross_attention', 'fusion_audioclip']:
        val_loss, val_acc, val_valence_acc, val_arousal_acc, val_auc_ovr, val_auc_ovo, val_f1, val_pre, val_rec, val_cm = evaluate(args, model, device, criterion, val_loader, processor=processor)
    elif args.model == 'fusion':
        val_loss, val_acc, val_valence_acc, val_arousal_acc, val_auc_ovr, val_auc_ovo, val_f1, val_pre, val_rec, val_cm = evaluate(args, model, device, criterion, val_loader, audio_processor=audio_processor, vision_processor=vision_processor)
    elif args.model == 'audio':
        val_loss, val_acc, val_valence_acc, val_arousal_acc, val_auc_ovr, val_auc_ovo, val_f1, val_pre, val_rec, val_cm = evaluate(args, model, device, criterion, val_loader, audio_processor=audio_processor)
    elif args.model == 'vision':
        val_loss, val_acc, val_valence_acc, val_arousal_acc, val_auc_ovr, val_auc_ovo, val_f1, val_pre, val_rec, val_cm = evaluate(args, model, device, criterion, val_loader, vision_processor=vision_processor)
        
    ## val results
    val_cm.title = 'val confusion matrix'
    wandb.log({
            f'val/loss': val_loss,
            f'val/acc': val_acc, 
            f'val/valence_acc': val_valence_acc,
            f'val/arousal_acc': val_arousal_acc,
            f'val/auc_ovr': val_auc_ovr, 
            f'val/auc_ovo': val_auc_ovo,
            f'val/f1': val_f1, 
            f'val/pre': val_pre, 
            f'val/rec': val_rec,
            f'val/cm': val_cm})
    logging.info('val_loss is {}, val_acc is {}, val_f1 is {}, val_auc_ovr is {}, val_auc_ovo is {}, val_pre is {}, val_rec is {}'.format(val_loss, val_acc, val_f1, val_auc_ovr, val_auc_ovo, val_pre, val_rec))

    return val_acc
    # return val_acc, val_loss

def test(args, model, device, custom_load_file=None):

    if args.model in ['cross_attention', 'fusion_audioclip']:
        processor = AudioCLIPProcessor()
    elif args.model == 'fusion':
        audio_processor = ClapProcessor.from_pretrained(args.pretrained_audio_model)
        vision_processor = CLIPImageProcessor.from_pretrained(args.pretrained_vision_model)
    elif args.model == 'audio':
        audio_processor = ClapProcessor.from_pretrained(args.pretrained_audio_model)
    elif args.model == 'vision':
        vision_processor = CLIPImageProcessor.from_pretrained(args.pretrained_vision_model)

    if custom_load_file is None:
        load_file = os.path.join(args.model_output_directory, f'checkpoint_{args.model}_{args.seed}_1lay.pth')
    else:
        load_file = custom_load_file
    checkpoint = torch.load(load_file, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model'])

    if args.model in ['cross_attention', 'fusion_audioclip']:
        base_params = [param for name, param in model.named_parameters() if 'visual' not in name and 'audio' not in name]
        optimizer = AdamW([{'params': base_params},
                        {'params': model.audioclip.visual.parameters(), 'lr': args.vision_lr},
                        {'params': model.audioclip.audio.parameters(), 'lr': args.audio_lr}], 
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.model == 'fusion': 
        base_params = [param for name, param in model.named_parameters() if 'model_vision' not in name and 'model_audio' not in name]
        optimizer = AdamW([{'params': base_params},
                        {'params': model.model_vision.parameters(), 'lr': args.vision_lr},
                        {'params': model.model_audio.parameters(), 'lr': args.audio_lr}], 
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.model == 'audio':
        base_params = [param for name, param in model.named_parameters() if 'model_audio' not in name]
        optimizer = AdamW([{'params': base_params},
                        {'params': model.model_audio.parameters(), 'lr': args.audio_lr}], 
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.model == 'vision':
        base_params = [param for name, param in model.named_parameters() if 'model_vision' not in name]
        optimizer = AdamW([{'params': base_params},
                           {'params': model.model_vision.parameters(), 'lr': args.vision_lr*0.1}],
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()

    test_data = LirisAccedeDataset(
        args, device,
        f'data/liris_accede/preprocessed/videos/',
        f'data/liris_accede/preprocessed/audios/',
        f'data/liris_accede/preprocessed/labels/',
        f'data/liris_accede/preprocessed/data_splits/test_ids_{args.dataset}.pt'
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=LirisAccedeDataset.collate_func, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.balance_classes:
        test_weights = calculate_weights(args, test_data).to(device)
    else:
        test_weights = None

    # Initialize loss function
    if args.loss_function == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=test_weights)
    elif args.loss_function == 'custom':
        criterion = CustomLoss(weight=test_weights, cross_entropy_factor=args.custom_loss_ce_factor, l2_factor=args.custom_loss_l2_factor)

    if args.model in ['cross_attention', 'fusion_audioclip']:
        loss, acc, valence_acc, arousal_acc, auc_ovr, auc_ovo, f1, pre, rec, cm = evaluate(args, model, device, criterion, test_loader, processor=processor)
    elif args.model == 'fusion':
        loss, acc, valence_acc, arousal_acc, auc_ovr, auc_ovo, f1, pre, rec, cm = evaluate(args, model, device, criterion, test_loader, audio_processor=audio_processor, vision_processor=vision_processor)
    elif args.model == 'audio':
        loss, acc, valence_acc, arousal_acc, auc_ovr, auc_ovo, f1, pre, rec, cm = evaluate(args, model, device, criterion, test_loader, audio_processor=audio_processor)
    elif args.model == 'vision':
        loss, acc, valence_acc, arousal_acc, auc_ovr, auc_ovo, f1, pre, rec, cm = evaluate(args, model, device, criterion, test_loader, vision_processor=vision_processor)

    ## test results
    cm.title = 'test confusion matrix'
    wandb.log({
            f'test/loss': loss,
            f'test/acc': acc, 
            f'test/valence_acc': valence_acc,
            f'test/arousal_acc': arousal_acc,
            f'test/auc_ovr': auc_ovr,
            f'test/auc_ovo': auc_ovo,
            f'test/f1': f1, 
            f'test/pre': pre, 
            f'test/rec': rec,
            f'test/cm': cm})
    logging.info('test_loss is {}, test_acc is {}, test_f1 is {}, test_auc_ovr is {}, test_auc_ovo is {}, test_pre is {}, test_rec is {}'.format(loss, acc, f1, auc_ovr, auc_ovo, pre, rec))

    logger.info('Test done')