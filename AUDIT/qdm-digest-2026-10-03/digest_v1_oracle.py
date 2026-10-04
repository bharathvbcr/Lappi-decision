"""Independent oracle for qd-metal's v1 prefix-state digest record (model.rs:166-177 at 3d68484):

    sha256( b"qd-metal.prefix-state.v1\\0" ++ u32le(tokens)
            ++ for each buffer in order: tag ++ u64le(len) ++ sha256(bytes) )

Prints the golden hex for the fixed input the Rust unit test uses.
"""

import hashlib
import struct

TOKENS = 7
PARTS = [
    (b"conv", b"abc"),
    (b"gdn", b""),
    (b"k", bytes(range(256)) * 3),
    (b"v", b"\x00"),
]

acc = b"qd-metal.prefix-state.v1\x00" + struct.pack("<I", TOKENS)
for tag, data in PARTS:
    acc += tag + struct.pack("<Q", len(data)) + hashlib.sha256(data).digest()
print(hashlib.sha256(acc).hexdigest())
