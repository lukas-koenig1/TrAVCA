import torch
import torchvision as tv
import torchaudio as ta
import numpy as np

from pretrained_model.audio_clip.utils.transforms import ToTensor1D


class AudioCLIPProcessor():
    def __init__(self):
        # derived from ESResNeXt
        SAMPLE_RATE = 44100
        # derived from CLIP
        IMAGE_SIZE = 224
        IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
        IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

        self.audio_resample = ta.transforms.Resample(orig_freq=48_000, new_freq=SAMPLE_RATE)

        self.audio_transforms = ToTensor1D()

        self.image_transforms = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Resize(IMAGE_SIZE, interpolation=tv.transforms.InterpolationMode.BICUBIC, antialias=True),
            tv.transforms.CenterCrop(IMAGE_SIZE),
            tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
        ])

        self.sample_rate = SAMPLE_RATE

    def preprocess(self, images, audios):
        images = self.preprocess_images(images)
        audios = self.preprocess_audio(audios)
        return images, audios

    def preprocess_audio(self, audios):
        audios = torch.from_numpy(np.array(audios))

        # Resample audio
        audios = self.audio_resample(audios.float())

        audios = torch.stack([self.audio_transforms(np.array(audio).reshape(1, -1)) for audio in audios])
        return audios

    def preprocess_images(self, images):
        images = torch.stack([self.image_transforms(np.array(image)) for image in images])
        return images