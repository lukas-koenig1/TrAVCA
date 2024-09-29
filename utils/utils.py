import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import wandb

def flatten(list_to_flatten):
    ## flatten text from List[List[Strings]] -> List[Strings]
    lengths = [ len(i) for i in list_to_flatten]
    flattened_list = [item for sublist in list_to_flatten for item in sublist]

    return flattened_list, lengths

def unflatten(flattened_list, lengths):
    pooled_embeddings = []
    start_idx = 0
    for size in lengths:
        end_idx = start_idx + size
        pooled_embedding = torch.mean(flattened_list[start_idx:end_idx, :], dim=0)
        pooled_embeddings.append(pooled_embedding)
        start_idx = end_idx
    
    pooled_tensor = torch.stack(pooled_embeddings, dim=0)
    
    return pooled_tensor

def text_mean_pooling(model_output, attention_mask):
    ## pool embeddings of 'last_hidden_state'
    text_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeddings.size()).float()
    
    return torch.sum(text_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def text_max_pooling(model_output, attention_mask):
    text_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeddings.size()).float()

    return torch.max(text_embeddings * input_mask_expanded, 1)[0]

def attention_pooling(max_pooled_embeddings, original_lengths, dimension, num_heads):
    attention = nn.MultiheadAttention(embed_dim=dimension, num_heads=num_heads, batch_first=True)
    pooled_results = []
    start_idx = 0

    for length in original_lengths:
        segment = max_pooled_embeddings[start_idx:start_idx + length]

        query = torch.mean(segment, dim=0, keepdim=True)
        
        att_token, _ = attention(query, segment, segment)
        # attention_scores = torch.matmul(query, segment.T)
        # attention_weights = F.softmax(attention_scores, dim=-1)
        # attention_pooled = torch.matmul(attention_weights, segment)
        # pooled_results.append(attention_pooled.squeeze(0))
        start_idx += length
    
    pooled_tensor = torch.stack(att_token, dim=0)

    return pooled_tensor


def get_combined_labels(valence_labels, arousal_labels):
    assert len(valence_labels) == len(arousal_labels)

    combined_labels = []

    for valenceClass, arousalClass in zip(valence_labels, arousal_labels):
        if(valenceClass == -1):
            if(arousalClass == -1): vaClass = 0 # negative calm
            elif(arousalClass == 0): vaClass = 1 # negative neutral
            elif(arousalClass == 1): vaClass = 2 # negative active
            else: raise ValueError('Illegal arousal label value')
        elif(valenceClass == 0):
            if(arousalClass == -1): vaClass = 3 # neutral calm
            elif(arousalClass == 0): vaClass = 4 # both neutral
            elif(arousalClass == 1): vaClass = 5 # neutral active
            else: raise ValueError('Illegal arousal label value')
        elif(valenceClass == 1):
            if(arousalClass == -1): vaClass = 6 # positive calm
            elif(arousalClass == 0): vaClass = 7 # positive neutral
            elif(arousalClass == 1): vaClass = 8 # positive active
            else: raise ValueError('Illegal arousal label value')
        else: raise ValueError('Illegal valence label value')
        combined_labels.append(vaClass)
    
    return np.array(combined_labels)

def get_separate_labels(labels):
    valence_labels = []
    arousal_labels = []

    for label in labels:
        if(label == 0): valence_labels.append(-1), arousal_labels.append(-1)
        elif(label == 1): valence_labels.append(-1), arousal_labels.append(0)
        elif(label == 2): valence_labels.append(-1), arousal_labels.append(1)
        elif(label == 3): valence_labels.append(0), arousal_labels.append(-1)
        elif(label == 4): valence_labels.append(0), arousal_labels.append(0)
        elif(label == 5): valence_labels.append(0), arousal_labels.append(1)
        elif(label == 6): valence_labels.append(1), arousal_labels.append(-1)
        elif(label == 7): valence_labels.append(1), arousal_labels.append(0)
        elif(label == 8): valence_labels.append(1), arousal_labels.append(1)
        else: raise ValueError('Illegal label value')
    
    return np.array(valence_labels), np.array(arousal_labels)

def calculate_weights(args, dataset):
    labels = []
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        labels.append(label)

    labels = torch.tensor(labels).long()

    n_samples = len(labels)
    n_classes = args.label_number
    samples_per_label = torch.bincount(labels)
    weights = n_samples / (n_classes * samples_per_label)

    return weights

def calculate_metrics(args, targets, prob, num_labels):
    y_pred = np.argmax(prob, axis=-1)
    labels_list = np.arange(num_labels)

    if args.balance_classes == True:
        acc = metrics.balanced_accuracy_score(targets, y_pred)

        valence_targets, arousal_targets = get_separate_labels(targets)
        valence_y_pred, arousal_y_pred = get_separate_labels(y_pred)
        valence_acc = metrics.balanced_accuracy_score(valence_targets, valence_y_pred)
        arousal_acc = metrics.balanced_accuracy_score(arousal_targets, arousal_y_pred)

        auc_ovr = metrics.roc_auc_score(targets, prob, labels=labels_list, multi_class='ovr', average='macro')
        auc_ovo = metrics.roc_auc_score(targets, prob, labels=labels_list, multi_class='ovo', average='macro')

        # F1, precision and recall are set to 0 for classes with no predicted samples (zero_division parameter)
        f1 = metrics.f1_score(targets, y_pred, labels=labels_list, average='macro', zero_division=0.0)
        precision = metrics.precision_score(targets, y_pred, labels=labels_list, average='macro', zero_division=0.0)
        recall = metrics.recall_score(targets, y_pred, labels=labels_list, average='macro', zero_division=0.0)

    else:
        acc = metrics.accuracy_score(targets, y_pred)

        valence_targets, arousal_targets = get_separate_labels(targets)
        valence_y_pred, arousal_y_pred = get_separate_labels(y_pred)
        valence_acc = metrics.accuracy_score(valence_targets, valence_y_pred)
        arousal_acc = metrics.accuracy_score(arousal_targets, arousal_y_pred)

        auc_ovr = metrics.roc_auc_score(targets, prob, labels=labels_list, multi_class='ovr', average='micro')
        auc_ovo = 0 # ROC AUC is not supported for the micro average

        # F1, precision and recall are set to 0 for classes with no predicted samples (zero_division parameter)
        f1 = metrics.f1_score(targets, y_pred, labels=labels_list, average='micro', zero_division=0.0)
        precision = metrics.precision_score(targets, y_pred, labels=labels_list, average='micro', zero_division=0.0)
        recall = metrics.recall_score(targets, y_pred, labels=labels_list, average='micro', zero_division=0.0)

    cm = wandb.plot.confusion_matrix(preds=y_pred, y_true=targets)

    return acc, valence_acc, arousal_acc, auc_ovr, auc_ovo, f1, precision, recall, cm