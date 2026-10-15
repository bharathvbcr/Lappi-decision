# Fable ruling: safetensors coverage and duplicate keys

Date: 2026-10-14. Task: ft-e0211ba7412fee41b32f8c6cc152091a.
Closes the residual in GAP-SAFETENSORS-READERS-ACCEPT-EMPTY-TENSOR-SET-AND-DUPLICATE-KEYS-2026-10-12.
The 2026-10-12 ruling refused `header_len == 0`. It did not rule on a JSON object `{}`.

## What was measured before the change

The local Qwen3.5-2B-Base shard
`models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/model.safetensors-00001-of-00001.safetensors`
has 632 tensors. The first `data_offsets` start is 0, there are 0 gaps and 0 overlaps, and the
last end equals the data-section length (4,548,144,832 bytes). Header length is 76,648.

On the pre-change Python reader, a gap, trailing bytes, a tensor that starts at offset 4, `{}`
with 16 data bytes, and a repeated top-level key were all accepted. `{}` with no data bytes was
accepted too.

## Ruling

1. The data section is one contiguous span. The first tensor starts at offset 0, each next tensor
   starts where the previous one ends, and the last tensor ends at the end of the section.
   Touching edges are contiguous. Two zero-length tensors at the same cursor are contiguous.
   A hole, including one byte, is refused (`gap`). Bytes after the last tensor, and a non-empty
   data section with no tensors, are refused (`not covered`). A tensor whose first byte is not
   offset 0 is refused (`not 0`). Overlap keeps the existing words (`overlap`).
2. An empty tensor map with an empty data section is a file. The upstream `safetensors` reader
   accepts it (`Metadata::validate` returns 0 and the length check passes). No caller of such a
   file was found, and refusing it would reject a file that reader accepts. This case is not in
   the shared list, because that list's ok cases require a non-empty tensor map. Both Rust
   readers and `tensor_ranges` have a dedicated test.
3. A repeated JSON object key is refused at every depth, including inside `__metadata__`, inside
   an object the tensor reader ignores, and inside a top-level array. `serde_json` and
   `json.loads` keep the last value. The official writer never emits a repeated key, and
   last-wins on a weight file is the bug. This is stricter than the upstream reader, which uses
   `HashMap` and also last-wins. The scan runs after a successful parse, so a header that is not
   JSON keeps that message. A duplicate nested in a non-object is reported as a duplicate, which
   is the message the Python parser produces while it is still parsing.
4. `qd_export::safetensors` owns the Rust check. qd-metal stays an adapter. Python's
   `safetensors_header` refuses duplicate keys. `tensor_ranges` refuses coverage failures.
   `run_control.Checkpoint.read_weights` calls `tensor_ranges`. `verify_checkpoint_inventory`
   and `checkpoint_tensor_index` stay header readers: they refuse a repeated key and do not
   inspect the data section.
5. No new dependency. The duplicate-key scan is copied from `qd-prep`'s config scanner. qd-prep
   does not depend on qd-export, and qd-export does not depend on qd-prep.
6. No benchmark. Header parsing runs once per open. The added work is one linear scan of the
   header. On the 2B shard that header is 76,648 bytes. There is no per-token path here, so a
   before/after bench was not run.

## After the change

`tensor_ranges` on the same 2B shard returns 632 tensors. The first absolute start is 76,656
(8 + 76,648) and the last end equals the file size 4,548,221,488.
