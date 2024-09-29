import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class LirisAccedeDataset(Dataset):
    def __init__(self, args, device, videos_path, audios_path, labels_path, ids_path):
        self.args = args
        self.device = device

        self.ids = torch.load(ids_path, weights_only=False)
        
        self.videos_path = videos_path
        self.audios_path = audios_path
        self.labels_path = labels_path

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        video = torch.load(self.videos_path + self.ids[index] + '.pt', weights_only=False)
        audio = torch.load(self.audios_path + self.ids[index] + '.pt', weights_only=False)
        label = torch.load(self.labels_path + self.ids[index] + '.pt', weights_only=False)
        return video, audio, label
    
    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
        if batch_size == 0:
            return {}

        videos = []
        audios = []
        labels = []

        for instance in batch_data:
            videos.append(instance[0])
            audios.append(instance[1])
            labels.append(instance[2])

        videos = torch.stack(videos, dim=0)

        return videos, audios, labels
    