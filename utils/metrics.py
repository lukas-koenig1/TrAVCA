import torch
import numpy as np
from tqdm import tqdm
from utils.utils import calculate_metrics

def evaluate(args, model, device, criterion, dataloader, audio_processor=None, vision_processor=None, processor=None):
    model.eval()

    prob = []
    y_pred = []
    targets = []
    running_loss = 0
    steps = 0

    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), desc='batch', total=len(dataloader), position=1, leave=False):
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
                if args.model in ['fusion']:
                    pred = model(audio_inputs, video_inputs, batch_length)
                elif args.model == 'audio':
                    pred = model(audio_inputs)
                elif args.model == 'vision':
                    pred = model(video_inputs, batch_length)
                prob.extend(torch.nn.functional.softmax(pred, dim=-1).detach().cpu())

                loss = criterion(pred, target)
                running_loss += loss.item()
                steps += 1

    loss = running_loss / len(dataloader)

    targets = np.array(targets)
    prob = np.array(prob)

    acc, valence_acc, arousal_acc, auc_ovr, auc_ovo, f1, precision, recall, cm = calculate_metrics(args, targets, prob, num_labels=args.label_number)

    return loss, acc, valence_acc, arousal_acc, auc_ovr, auc_ovo, f1, precision, recall, cm