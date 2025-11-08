import os, torch, requests
from pathlib import Path

os.makedirs("models", exist_ok=True)

CANDIDATE_URLS = [
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt",
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.3/model_swinvit.pt",
]

def download(url, dest):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

tmp_path = Path("models/_swin_raw.pt")
ok = False
for url in CANDIDATE_URLS:
    try:
        print(f"Trying {url}")
        download(url, tmp_path)
        ok = True
        break
    except Exception as e:
        print(f"Failed: {e}")

if not ok:
    raise SystemExit("Could not download Swin UNETR weights from known locations.")

ckpt = torch.load(tmp_path, map_location="cpu")
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    out = ckpt
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    out = {"model_state_dict": ckpt["state_dict"]}
else:
    out = {"model_state_dict": ckpt}

torch.save(out, "models/swin_unetr_pretrained.pth")
tmp_path.unlink()
print("Saved models/swin_unetr_pretrained.pth")

