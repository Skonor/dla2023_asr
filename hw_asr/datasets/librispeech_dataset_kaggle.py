import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDatasetKaggle(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == 'train_all'

        self.index_path = ROOT_PATH / "data" / "datasets" / "librispeech"
        self.index_path.mkdir(exist_ok=True, parents=True)

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        else:
            data_dir = Path(data_dir)

        self._data_dir = data_dir
        if part == 'train_all':
            index = sum([self._get_or_load_index(part)
                         for part in URL_LINKS if 'train' in part], [])
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self.index_path / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        trans_dir = self._data_dir / 'meta' / part
        split_dir = self._data_dir / part
        
        if not split_dir.exists():
            self._load_part(part)

        trans_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(trans_dir)):
            if any([f.endswith(".txt") for f in filenames]):
                trans_dirs.add(dirpath)
        for trans_dir in tqdm(
                list(trans_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            trans_dir = Path(trans_dir)
            trans_path = list(trans_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    wav_path = split_dir / f"{f_id}.wav"
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(wav_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
