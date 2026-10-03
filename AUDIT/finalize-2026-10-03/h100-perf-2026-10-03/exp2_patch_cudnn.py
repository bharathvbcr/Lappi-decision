"""Throwaway (role 3), runs on the box: make a perf overlay's padded training attention run on
SDPA's cuDNN backend instead of the default pick (the sm80 mem-efficient kernel the 23:16Z profile
found). The mask is kept, so what is attended is unchanged; only the kernel differs. Refuses
unless the exact production text is found once. Usage: python exp2_patch_cudnn.py <overlay root>"""

import sys
from pathlib import Path

path = Path(sys.argv[1]) / "python" / "qd_train" / "backbone.py"
text = path.read_text(encoding="utf-8")
old = (
    '        if self.train_attention_mask == "padding":\n'
    "            return contextlib.nullcontext()\n"
)
new = (
    '        if self.train_attention_mask == "padding":\n'
    "            from torch.nn.attention import SDPBackend, sdpa_kernel\n"
    "\n"
    "            return sdpa_kernel([SDPBackend.CUDNN_ATTENTION])\n"
)
if text.count(old) != 1:
    raise SystemExit(f"expected the production branch once in {path}, found {text.count(old)}")
path.write_text(text.replace(old, new), encoding="utf-8")
print(f"patched {path}: padded training attention -> SDPBackend.CUDNN_ATTENTION")
