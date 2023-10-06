import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    spec_lengths = []
    text_encoded_length = []
    for ds in dataset_items:
        spec_lengths.append(ds['spectrogram'].shape[1])
        text_encoded_length.append(ds['encoded_text'].shape[0])

    spec_dim = dataset_items[0]['spectrogram'].shape[0]
    batch_spectrogram = torch.zeros(len(spec_lengths), spec_dim, max(spec_lengths))
    batch_encoded_text = torch.zeors(len(text_encoded_length), max(text_encoded_length))

    texts = []
    for i, ds in enumerate(dataset_items):
        batch_spectrogram[i, :spec_lengths[i]] = ds['spectrogram']
        batch_encoded_text[i, :text_encoded_length[i]] = ds['encoded_text']
        texts.append(ds['text'])

    text_encoded_length = torch.tensor(text_encoded_length).long()

    return {
        'spectrogram': batch_spectrogram,
        'text_encoded': batch_encoded_text,
        'text_encoded_legnth': text_encoded_length,
        'text': texts
    }
