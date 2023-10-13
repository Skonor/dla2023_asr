from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamsearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []

        if not hasattr(self.text_encoder, "ctc_beam_search"):
            raise NotImplementedError

        probs = torch.exp(log_probs.detach()).cpu()
        lengths = log_probs_length.detach().numpy()
        for prob_mat, length, target_text in zip(probs, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            hypots = self.text_encoder.ctc_beam_search(probs=prob_mat, probs_length=length, beam_size=self.beam_size)
            pred_text = hypots[0].text
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)