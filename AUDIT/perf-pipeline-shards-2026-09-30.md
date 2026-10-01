# AUDIT: shard-pipeline cProfile evidence (perf-pipeline-shards, 2026-09-30)

These are the verbatim stdlib cProfile summaries behind the profile tables in
`HANDOFF/perf-pipeline-shards-2026-09-30.md`. The dumps are session scratch files; this file
is what survives. Timings are profiled seconds (cProfile overhead included) on the Mac CPU under
a machine shared with other sessions; they rank hot spots, they are not speedups. Summaries
were produced by a throwaway pstats script: total, top 40 by cumulative time, top 40 by
tottime, `file:line(function)`.

## 1. Phase 3, pure Python, before the qd-prep port

Run 2026-10-01T02:38:52Z (UTC) from the main checkout, before bd47939: the dump itself shows
100,018 calls to the pure-Python `minhash.py:175(signature)`. Ledger row 389888e8, written at
03:32Z from the same checkout, records code_commit e3477eba. The load average was not recorded.

```
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /usr/bin/time -l /Users/bharath/.venvs/ml/bin/python -B \
  -m cProfile -o <scratch>/prof-p3-main.out /Users/bharath/Code/research/Lappi-decision/tools/real_tokenizer_pipeline.py \
  --out <scratch>/p3-prof-main --no-repo-history \
  --defect-class /Users/bharath/Code/research/Lappi-decision/data/pool/commitpackft-corpus-v2 \
  --val-shards --memo-limit 0 --vocab full --rev be3073300cf4efb664f7065a32a44e4b9c12bd37
```

```
total profiled time: 626.1 s

== top cumulative ==
    cum_s     tot_s      ncalls  function
    627.9       0.2        4468  ~:0(<built-in method builtins.exec>)
    627.9       0.0           1  real_tokenizer_pipeline.py:1(<module>)
    627.7       0.1           1  real_tokenizer_pipeline.py:1933(main)
    627.6       0.5           1  real_tokenizer_pipeline.py:1370(run)
    268.2       4.2      100018  minhash.py:175(signature)
    255.9       6.3    12902322  minhash.py:191(<genexpr>)
    249.6      71.5    13000402  ~:0(<built-in method builtins.min>)
    183.4       0.9           1  dedupe.py:245(dedupe)
    178.0     178.0   598540416  minhash.py:192(<genexpr>)
    171.4       1.1           3  shards.py:1312(write_shards)
    163.4      16.6      463704  real_tokenizer_pipeline.py:582(_encode)
    146.1       2.2      463704  tokenization_utils_base.py:2419(__call__)
    141.4       3.5      463704  tokenization_utils_tokenizers.py:857(_encode_plus)
    141.2       1.1      186284  shards.py:836(encode_slot)
    125.9     125.9      463704  ~:0(<method 'encode_batch' of 'tokenizers.Tokenizer' objects>)
    120.4       0.4           1  split.py:267(split)
    119.2       0.4           1  split.py:419(_cross_split_near_duplicates)
    109.2       0.5      281759  shards.py:787(_tokenize_checked)
    105.9       2.4      281759  real_tokenizer_pipeline.py:601(tokenize)
    101.2       1.4      281759  shards.py:1028(_span_token_positions)
     99.8       1.2           2  real_tokenizer_pipeline.py:756(census)
     60.9       0.9      181945  real_tokenizer_pipeline.py:604(offsets)
     37.0       4.7     1929150  real_tokenizer_pipeline.py:607(decode)
     36.4       1.3      191057  render.py:667(render)
     35.2       6.0      149538  shards.py:916(_assert_spans_decode_to_their_text)
     35.0       1.4      140880  shards.py:557(training_texts)
     32.3       2.2     1929165  tokenization_utils_base.py:2855(decode)
     27.0      17.9     2101627  render.py:274(_escape)
     20.2       0.2      191057  render.py:310(escape_block)
     19.7       5.8     1929165  generic.py:314(to_py_obj)
     19.0       5.2           2  minhash.py:320(candidate_pairs)
     13.6       5.1      100018  minhash.py:94(shingle)
     13.0       0.3           1  real_tokenizer_pipeline.py:1065(_read_back)
     12.5       9.3    65649162  generic.py:324(<genexpr>)
     12.1       0.0           1  mixture.py:1240(build_mixture)
     11.9      11.9     2723410  shards.py:319(_token_index_for_char)
     11.7       0.5     2665412  shards.py:1099(<genexpr>)
     11.6      11.3           1  remap.py:241(count_corpus_tokens)
     11.3       5.8     1600290  ~:0(<method 'join' of 'bytes' objects>)
     10.1       0.3      573171  render.py:727(<genexpr>)

== top tottime ==
    178.0     178.0   598540416  minhash.py:192(<genexpr>)
    125.9     125.9      463704  ~:0(<method 'encode_batch' of 'tokenizers.Tokenizer' objects>)
    249.6      71.5    13000402  ~:0(<built-in method builtins.min>)
     27.0      17.9     2101627  render.py:274(_escape)
    163.4      16.6      463704  real_tokenizer_pipeline.py:582(_encode)
     11.9      11.9     2723410  shards.py:319(_token_index_for_char)
     11.6      11.3           1  remap.py:241(count_corpus_tokens)
     10.0      10.0   139379422  ~:0(<method 'append' of 'list' objects>)
     12.5       9.3    65649162  generic.py:324(<genexpr>)
      8.5       8.5     1929165  ~:0(<method 'decode' of 'tokenizers.Tokenizer' objects>)
    255.9       6.3    12902322  minhash.py:191(<genexpr>)
      7.1       6.1    91929680  ~:0(<built-in method builtins.isinstance>)
     35.2       6.0      149538  shards.py:916(_assert_spans_decode_to_their_text)
      6.1       6.0      463704  tokenization_utils_tokenizers.py:670(_convert_encoding)
      6.7       5.9      431298  artifacts.py:851(line_start_indices)
     11.3       5.8     1600290  ~:0(<method 'join' of 'bytes' objects>)
     19.7       5.8     1929165  generic.py:314(to_py_obj)
      8.1       5.7     4576079  minhash.py:171(base_hash)
     19.0       5.2           2  minhash.py:320(candidate_pairs)
     13.6       5.1      100018  minhash.py:94(shingle)
     37.0       4.7     1929150  real_tokenizer_pipeline.py:607(decode)
    268.2       4.2      100018  minhash.py:175(signature)
    141.4       3.5      463704  tokenization_utils_tokenizers.py:857(_encode_plus)
      3.8       3.3     7804207  ~:0(<method 'join' of 'str' objects>)
      5.5       3.2    14402592  minhash.py:349(<genexpr>)
      2.9       2.9      100890  ~:0(<method 'sub' of 're.Pattern' objects>)
      4.7       2.8      283319  ~:0(<built-in method builtins.max>)
      2.8       2.8     6790092  ~:0(<method 'digest' of '_blake2.blake2b' objects>)
      2.7       2.7      990824  ~:0(<built-in method numpy.asarray>)
      2.4       2.4    13375735  ~:0(<method 'to_bytes' of 'int' objects>)
    105.9       2.4      281759  real_tokenizer_pipeline.py:601(tokenize)
      2.2       2.2      193359  encoder.py:207(iterencode)
    146.1       2.2      463704  tokenization_utils_base.py:2419(__call__)
     32.3       2.2     1929165  tokenization_utils_base.py:2855(decode)
      2.0       2.0      181298  decoder.py:351(raw_decode)
      2.0       1.9    20575039  ~:0(<built-in method builtins.len>)
      1.9       1.9     7989314  ~:0(<method 'encode' of 'str' objects>)
      1.9       1.9    38131825  shards.py:1091(<genexpr>)
      9.9       1.5      382114  render.py:625(_slot_suffix)
      1.5       1.5         219  ~:0(<built-in method _imp.create_dynamic>)
```

## 2. Clinc-only general record, this branch, after the port (qd-prep + shingle table)

Run 2026-10-01T04:39:23Z (UTC) from this worktree with HEAD between 3b5678f and 95984e4; the
tool file is identical across both (`tools/real_tokenizer_pipeline.py` sha256 c345882, the
same file as ledger rows 75cbd709 / 1cec1569 / f5664ba2). It overlapped the ledgered rebuilds
N2 and N3; `uptime` read a load average of 14.0 at its start and 6.3 near its end.
`TOKENIZERS_PARALLELISM` was left unset here but was false in section 1, so the
tokenizer times of the two profiles are not comparable. The output
directory was byte-identical to ledger row 5972b896's set except the timestamp fields.

```
HF_HUB_OFFLINE=1 QD_PREP_BIN=<worktree>/target/release/qd-prep /Users/bharath/.venvs/ml/bin/python -B -u \
  -m cProfile -o <scratch>/prof-cl-memo.out <worktree>/tools/real_tokenizer_pipeline.py \
  --out <scratch>/cl-prof-memo --no-repo-history --defect-class data/pool/commitpackft-corpus-v2 \
  --val-shards --memo-limit 0 --vocab full --rev be3073300cf4efb664f7065a32a44e4b9c12bd37 \
  --general-record /Users/bharath/.cache/qd-decision/general/fetch-record-clinc-only-2026-09-30.json
```

```
total profiled time: 358.2 s

== top cumulative ==
    cum_s     tot_s      ncalls  function
    360.3       0.2        4468  ~:0(<built-in method builtins.exec>)
    360.3       0.0           1  real_tokenizer_pipeline.py:1(<module>)
    360.0       0.1           1  real_tokenizer_pipeline.py:2167(main)
    359.9       0.1           1  real_tokenizer_pipeline.py:1603(run)
    167.6       1.3           3  shards.py:1312(write_shards)
    159.9      16.1      729409  real_tokenizer_pipeline.py:586(_encode)
    143.2       2.3      729409  tokenization_utils_base.py:2419(__call__)
    138.0       3.8      729409  tokenization_utils_tokenizers.py:857(_encode_plus)
    125.5       1.1      361421  shards.py:836(encode_slot)
    122.4       0.6      547464  shards.py:787(_tokenize_checked)
    121.5     121.5      729409  ~:0(<method 'encode_batch' of 'tokenizers.Tokenizer' objects>)
    118.5       2.5      547464  real_tokenizer_pipeline.py:605(tokenize)
    102.0       1.6           2  real_tokenizer_pipeline.py:760(census)
     71.2       1.0      547464  shards.py:1028(_span_token_positions)
     57.4       2.3      636731  render.py:667(render)
     53.8       1.9      491154  shards.py:557(training_texts)
     44.6       0.7      181945  real_tokenizer_pipeline.py:608(offsets)
     35.5      23.3     9120953  render.py:274(_escape)
     30.3       4.0     2098288  real_tokenizer_pipeline.py:611(decode)
     30.1       0.4     1464519  render.py:727(<genexpr>)
     29.7       2.3      827788  render.py:625(_slot_suffix)
     26.3       1.6     2098303  tokenization_utils_base.py:2855(decode)
     25.0       4.3      149538  shards.py:916(_assert_spans_decode_to_their_text)
     21.7       0.1           1  mixture.py:1240(build_mixture)
     20.8       1.4     8484222  render.py:320(escape_inline)
     17.3      16.7           1  remap.py:241(count_corpus_tokens)
     16.3       0.2      636731  render.py:310(escape_block)
     16.1       4.6     2098303  generic.py:314(to_py_obj)
     15.8       2.6      827788  render.py:609(_render_option_lines)
     13.7       0.3           1  real_tokenizer_pipeline.py:1069(_read_back)
     12.3       3.5           2  minhash.py:320(candidate_pairs)
     12.1       0.0           1  mixture.py:1014(drop_contradictory_prompts)
     12.1       0.4           1  mixture.py:847(_group_prompts)
     12.0      12.0   251582403  ~:0(<method 'append' of 'list' objects>)
     10.5       7.9    80739221  generic.py:324(<genexpr>)
      9.3       0.1      145577  mixture.py:1534(_dispatch)
      9.0       0.2           1  split.py:267(split)
      8.6       1.0      515004  render.py:422(permutation)
      8.4       0.1           1  split.py:419(_cross_split_near_duplicates)
      8.2       0.8     2098303  tokenization_utils_tokenizers.py:1018(_decode)

== top tottime ==
    121.5     121.5      729409  ~:0(<method 'encode_batch' of 'tokenizers.Tokenizer' objects>)
     35.5      23.3     9120953  render.py:274(_escape)
     17.3      16.7           1  remap.py:241(count_corpus_tokens)
    159.9      16.1      729409  real_tokenizer_pipeline.py:586(_encode)
     12.0      12.0   251582403  ~:0(<method 'append' of 'list' objects>)
      8.0       8.0     2723410  shards.py:319(_token_index_for_char)
     10.5       7.9    80739221  generic.py:324(<genexpr>)
      7.0       7.0     2098303  ~:0(<method 'decode' of 'tokenizers.Tokenizer' objects>)
      6.2       6.0      729409  tokenization_utils_tokenizers.py:670(_convert_encoding)
      6.2       5.3   116917477  ~:0(<built-in method builtins.isinstance>)
     16.1       4.6     2098303  generic.py:314(to_py_obj)
     25.0       4.3      149538  shards.py:916(_assert_spans_decode_to_their_text)
      6.2       4.1     7370272  render.py:403(_next_u64)
      4.7       4.1      431298  artifacts.py:851(line_start_indices)
     30.3       4.0     2098288  real_tokenizer_pipeline.py:611(decode)
    138.0       3.8      729409  tokenization_utils_tokenizers.py:857(_encode_plus)
      3.7       3.7      559264  encoder.py:207(iterencode)
     12.3       3.5           2  minhash.py:320(candidate_pairs)
      6.9       3.2     3581449  ~:0(<method 'join' of 'bytes' objects>)
      2.9       2.9     1931141  ~:0(<built-in method numpy.asarray>)
      4.0       2.8    14534271  ~:0(<method 'join' of 'str' objects>)
     15.8       2.6      827788  render.py:609(_render_option_lines)
    118.5       2.5      547464  real_tokenizer_pipeline.py:605(tokenize)
      2.4       2.4    46978792  ~:0(<built-in method builtins.len>)
    143.2       2.3      729409  tokenization_utils_base.py:2419(__call__)
     57.4       2.3      636731  render.py:667(render)
      3.7       2.3    31569840  minhash.py:349(<genexpr>)
     29.7       2.3      827788  render.py:625(_slot_suffix)
      3.2       1.9      283333  ~:0(<built-in method builtins.max>)
     53.8       1.9      491154  shards.py:557(training_texts)
      1.9       1.9    35774071  ~:0(<method 'to_bytes' of 'int' objects>)
      4.2       1.9      518141  ~:0(<built-in method builtins.sorted>)
      1.9       1.9    11540336  ~:0(<method 'digest' of '_blake2.blake2b' objects>)
      1.7       1.7      205153  decoder.py:351(raw_decode)
     26.3       1.6     2098303  tokenization_utils_base.py:2855(decode)
    102.0       1.6           2  real_tokenizer_pipeline.py:760(census)
      1.4       1.4    14472900  ~:0(<method 'encode' of 'str' objects>)
     20.8       1.4     8484222  render.py:320(escape_inline)
      7.5       1.3     7370272  render.py:410(below)
    167.6       1.3           3  shards.py:1312(write_shards)
```
