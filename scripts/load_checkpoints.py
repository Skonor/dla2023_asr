import gdown
import shutil
import os
from pathlib import Path


URL_LINKS = {
    'DeepSpeech2_clean': 'https://drive.google.com/uc?id=16N0g78RAdXeJkRFmPwzk2BstvMA1jB0A'
}

def main():
    dir = Path(__file__).absolute().resolve().parent.parent
    for name in URL_LINKS:
        checkpoint_dir = dir / 'saved' / 'models' / 'checkpoints' / URL_LINKS[name]
        checkpoint_dir.mkdir(exists_ok=True, parents=True)
        zip_pth = checkpoint_dir / (URL_LINKS[name] + '.zip')
        model_pth = checkpoint_dir / 'model_weights.pth'
        if not model_pth.exists():
            gdown.download(URL_LINKS[name], zip_pth, quiet=False)
            shutil.unpack_archive(zip_pth, checkpoint_dir, "zip")
            os.remove(zip_pth)

if __name__ == "__main__":
    main()