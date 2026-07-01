"""Auto-loading integration test: convert an official checkpoint to a HF directory,
then load it back via ``AutoModelForCausalLM.from_pretrained`` with zero custom code,
and verify logits match the official reference.

Requires the convert step to have run (or RWKV7_TEST_MODEL pointing at an official .pth).
"""

import os
import shutil
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
import transformers_rwkv7  # noqa: F401,E402  — registers Auto classes
from transformers import AutoModelForCausalLM  # noqa: E402

SRC_MODEL = os.environ.get(
    "RWKV7_TEST_MODEL",
    r"F:\rwkv\models\rwkv7-g1d-0.1b-20260129-ctx8192.pth",
)
VOCAB = r"F:\rwkv\RWKV-LM\RWKV-v7\rwkv_vocab_v20230424.txt"


def main():
    if not os.path.isfile(SRC_MODEL):
        print(f"SKIP: source model not found at {SRC_MODEL}")
        return

    hf_dir = os.path.join(tempfile.mkdtemp(), "rwkv7-0.1b-hf")
    py = sys.executable

    # 1. convert via the CLI entry point
    print(f"converting {SRC_MODEL} -> {hf_dir} ...")
    cmd = [py, "-m", "transformers_rwkv7.convert_checkpoint", "--src", SRC_MODEL,
           "--dst", hf_dir, "--dtype", "fp32"]
    if os.path.isfile(VOCAB):
        cmd += ["--vocab", VOCAB]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"convert failed:\n{r.stderr[-1000:]}"
    files = sorted(os.listdir(hf_dir))
    print(f"  output files: {files}")
    assert "config.json" in files and "model.safetensors" in files

    # 2. zero-code load via AutoModel
    model = AutoModelForCausalLM.from_pretrained(hf_dir).eval()
    print(f"  loaded as: {type(model).__name__}")
    assert type(model).__name__ == "Rwkv7ForCausalLM"

    # 3. logits match the official reference top-5 (from test_alignment.py)
    ids = torch.tensor([[0, 5248, 478, 2214, 30, 15025, 19, 358, 4457, 311]])
    with torch.no_grad():
        top5 = torch.topk(model(ids).logits[0, -1].float(), 5).indices.tolist()
    print(f"  top-5: {top5}")
    assert top5 == [33, 30, 40, 47, 42], f"top-5 mismatch: {top5}"

    # 4. generate end-to-end through the AutoModel
    with torch.no_grad():
        gen = model.generate(ids[:, :4], max_new_tokens=8, do_sample=False, use_cache=True)
    print(f"  generated: {gen[0].tolist()}")

    shutil.rmtree(os.path.dirname(hf_dir), ignore_errors=True)
    print("\nAUTO-LOAD TEST PASSED")


if __name__ == "__main__":
    main()
