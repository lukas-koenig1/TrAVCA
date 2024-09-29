import torch
import torch.nn as nn
from transformers import CLIPVisionModel, ClapModel

from pretrained_model.audio_clip.model import AudioCLIP

class AudioModel(nn.Module):
    def __init__(self, args):
        super(AudioModel, self).__init__()
        self.args = args

        self.model_audio = ClapModel.from_pretrained(args.pretrained_audio_model, output_attentions=False)

        if self.args.pooling == 'pooler_output':
            self.audio_embed_dim = self.model_audio.audio_model.config.hidden_size
        else:
            self.audio_embed_dim = self.audio_embed_dim = self.model_audio.audio_model.config.projection_dim

        self.classification_head = nn.Sequential(
            nn.Linear(self.audio_embed_dim, 256),
            nn.Dropout(p=args.dropout_rate),
            nn.ReLU(),
            nn.Linear(256, self.args.label_number)
        )

    def forward(self, audio_input):
        if self.args.pooling == 'pooler_output':
            audio_model_output = self.model_audio.audio_model(**audio_input, output_attentions=False)
        else:
            audio_model_output = self.model_audio.get_audio_features(**audio_input, output_attentions=False)
        

        if self.args.pooling == 'pooler_output':
            audio_output_pooled = audio_model_output.pooler_output
        else:
            audio_output_pooled = audio_model_output

        ## classification head
        output = self.classification_head(audio_output_pooled)

        return output


class VisionModel(nn.Module):
    def __init__(self, args):
        super(VisionModel, self).__init__()
        self.args = args

        self.model_vision = CLIPVisionModel.from_pretrained(args.pretrained_vision_model, output_attentions=False)

        self.vision_embed_dim = self.model_vision.config.hidden_size

        self.classification_head = nn.Sequential(
            nn.Linear(self.vision_embed_dim, 256),
            nn.Dropout(p=args.dropout_rate),
            nn.ReLU(),
            nn.Linear(256, self.args.label_number)
        )
    
    def forward(self, video_input, batch_length):
        vision_model_output = self.model_vision(**video_input, output_attentions=False)

        ## get frames cls tokens
        nb_tokens = vision_model_output[0].shape[-2]
        embed_dim = vision_model_output[0].shape[-1]
        vision_model_output_video = vision_model_output[0].reshape(batch_length, -1, nb_tokens, embed_dim)
        frames_cls_token = vision_model_output_video[..., 0, :] ## cls token of all frames

        ## average cls token of all frames (per video)
        num_frames = vision_model_output_video.shape[1]
        video_cls_token = frames_cls_token.sum(dim=1) / num_frames

        ## classification head
        output = self.classification_head(video_cls_token)

        return output


class FusionModel(nn.Module):
    def __init__(self, args):
        super(FusionModel, self).__init__()
        self.args = args

        self.model_audio = ClapModel.from_pretrained(args.pretrained_audio_model, output_attentions=False)
        self.model_vision = CLIPVisionModel.from_pretrained(args.pretrained_vision_model, output_attentions=False)
        
        if self.args.pooling == 'pooler_output':
            self.audio_embed_dim = self.model_audio.audio_model.config.hidden_size
        else:
            self.audio_embed_dim = self.audio_embed_dim = self.model_audio.audio_model.config.projection_dim
        self.vision_embed_dim = self.model_vision.config.hidden_size
        self.fusion_embed_dim = self.audio_embed_dim + self.vision_embed_dim
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.fusion_embed_dim, 256),
            nn.Dropout(p=args.dropout_rate),
            nn.ReLU(),
            nn.Linear(256, self.args.label_number)
        )

    def forward(self, audio_input, video_input, batch_length):  
        
        if self.args.pooling == 'pooler_output':
            audio_model_output = self.model_audio.audio_model(**audio_input, output_attentions=False)
        else:
            audio_model_output = self.model_audio.get_audio_features(**audio_input, output_attentions=False)
        vision_model_output = self.model_vision(**video_input, output_attentions=False)

        if self.args.pooling == 'pooler_output':
            audio_output_pooled = audio_model_output.pooler_output
        else:
            audio_output_pooled = audio_model_output

        ## get frames cls tokens
        nb_tokens = vision_model_output[0].shape[-2]
        embed_dim = vision_model_output[0].shape[-1]
        vision_model_output_video = vision_model_output[0].reshape(batch_length, -1, nb_tokens, embed_dim)
        frames_cls_token = vision_model_output_video[..., 0, :] ## cls token of all frames

        ## average cls token of all frames (per video)
        num_frames = vision_model_output_video.shape[1]
        video_cls_token = frames_cls_token.sum(dim=1) / num_frames

        ## concatenate audio tokens and frame tokens
        fusion_token = torch.cat((audio_output_pooled, video_cls_token), dim=1)
        
        ## classification head
        output = self.classification_head(fusion_token)

        return output

class CrossAttentionModel(nn.Module):
    def __init__(self, args):
        super(CrossAttentionModel, self).__init__()
        self.args = args

        self.audioclip = AudioCLIP(pretrained='pretrained_model/audio_clip/assets/AudioCLIP-Full-Training.pt')

        self.vision_embed_dim = self.audioclip.embed_dim
        self.audio_embed_dim = self.audioclip.embed_dim
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.vision_embed_dim,
                                                        num_heads=self.args.num_heads_ca,
                                                        batch_first=True)

        self.classification_head = nn.Sequential(
            nn.Linear(self.vision_embed_dim, 256),
            nn.Dropout(p=args.dropout_rate),
            nn.ReLU(),
            nn.Linear(256, self.args.label_number)
        )

    def forward(self, audio_input, video_input, batch_length):
        ((audio_features, _, _), _), _  = self.audioclip(audio=audio_input)
        ((_, image_features, _), _), _ = self.audioclip(image=video_input)

        audio_features = audio_features.unsqueeze(dim=1)

        # frame embeddings
        frame_embeddings = image_features.reshape(batch_length, -1, self.vision_embed_dim)

        ## compute cross attention between audio features (query) and frame features (key, value)
        ca_token, _ = self.cross_attention(audio_features, frame_embeddings, frame_embeddings)
        ca_token = ca_token.squeeze()

        ## classification head
        output = self.classification_head(ca_token)        

        return output


class FusionModelAudioCLIP(nn.Module):
    def __init__(self, args):
        super(FusionModelAudioCLIP, self).__init__()
        self.args = args

        self.audioclip = AudioCLIP(pretrained='pretrained_model/audio_clip/assets/AudioCLIP-Full-Training.pt')

        self.audio_embed_dim = self.audioclip.embed_dim
        self.vision_embed_dim = self.audioclip.embed_dim
        self.fusion_embed_dim = self.audio_embed_dim + self.vision_embed_dim

        self.attention_pooling = nn.MultiheadAttention(embed_dim=self.audio_embed_dim, 
                                                       num_heads=self.args.num_heads_ca, 
                                                       batch_first=True)
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.fusion_embed_dim, 256),
            nn.Dropout(p=args.dropout_rate),
            nn.ReLU(),
            nn.Linear(256, self.args.label_number)
        )

    def forward(self, audio_input, video_input, batch_length):  
        ((audio_features, _, _), _), _  = self.audioclip(audio=audio_input)
        ((_, image_features, _), _), _ = self.audioclip(image=video_input)

        # frame embeddings
        frame_embeddings = image_features.reshape(batch_length, -1, self.vision_embed_dim)

        ## average embeddings of all frames (per video)
        num_frames = frame_embeddings.shape[1]
        video_embeddings = frame_embeddings.sum(dim=1) / num_frames

        ## concatenate audio and video embeddings
        fusion_embeddings = torch.cat((audio_features, video_embeddings), dim=1)
        
        ## classification head
        output = self.classification_head(fusion_embeddings)

        return output
