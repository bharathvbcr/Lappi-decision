"""The answer-side wire contract, pinned as literals.

Every table below is **generated** from ``crates/qd-runtime/src/{schema,refusal}.rs``
by ``python -m qd_wire.rust_source``, never hand-transcribed, and
``python/tests/test_wire_contract_matches_rust.py`` re-derives it and asserts
equality on every run. That test is the seam: when the XLANG-RS lane renames a
field or adds a refusal, this file stops matching the Rust and Python fails,
naming the field. A paragraph in a document would not.

The Rust types are carried alongside the field names on purpose. A field renamed
is one kind of drift; a field whose type changed from ``usize`` to ``String``
under the same name is the other, and it is the one a name-only table would miss.

Regenerate with::

    PYTHONPATH=python python -m qd_wire.rust_source

Source of record: ``crates/qd-runtime/src/schema.rs`` and
``crates/qd-runtime/src/refusal.rs``. Do not edit these tables by hand.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "BACKEND_ERROR_KINDS",
    "HASH_KINDS",
    "MAX_SLOT_NAME_BYTES",
    "NOUL_LABEL",
    "REFUSAL_KINDS",
    "ROUTES",
    "SLOT_KINDS",
    "STATUS_TAGS",
    "STRUCT_FIELDS",
    "SUPPORTED_SCHEMA_VERSIONS",
]

#: The abstain label. ``crates/qd-runtime/src/schema.rs::NOUL_LABEL``.
NOUL_LABEL: Final[str] = "noul"

#: Every response struct the runtime serializes, field name -> Rust type.
STRUCT_FIELDS: Final[dict[str, dict[str, str]]] = {
    'AnswerEnvelope': {
        'backend': 'String',
        'degraded': 'bool',
        'schema_version': 'u32',
        'slots': 'BTreeMap<String, SlotAnswer>',
    },
    'ErrorEnvelope': {
        'error': 'BackendError',
        'message': 'String',
        'schema_version': 'u32',
    },
    'HashExpectation': {
        'calibration_hash': 'Option<String>',
        'head_hash': 'Option<String>',
        'label_set_hash': 'Option<String>',
        'tokenizer_hash': 'Option<String>',
        'weight_hash': 'Option<String>',
    },
    'RefusalEnvelope': {
        'message': 'String',
        'refusal': 'Refusal',
        'schema_version': 'u32',
    },
    'SlotAnswer': {
        'conformal_set': 'Option<ConformalSet>',
        'degraded': 'bool',
        'noul': 'bool',
        'score': 'f64',
        'value': 'Option<SlotValue>',
    },
    'SpanValue': {
        'end_line': 'usize',
        'start_line': 'usize',
    },
}

#: The `status` discriminant of `Response`, tag -> the struct whose fields flatten beside it.
STATUS_TAGS: Final[dict[str, str]] = {
    'error': 'ErrorEnvelope',
    'ok': 'AnswerEnvelope',
    'refused': 'RefusalEnvelope',
}

#: `Refusal::kind()` -> that variant's fields. 36 conditions, not the 4 the doc tabulates.
REFUSAL_KINDS: Final[dict[str, dict[str, str]]] = {
    'ambiguous_envelope': {},
    'bins_out_of_range': {
        'actual': 'u32',
        'max': 'u32',
        'min': 'u32',
        'slot': 'String',
    },
    'calibration_entry_missing': {
        'rows': 'usize',
        'slot': 'String',
        'slot_type': 'String',
    },
    'context_empty': {
        'len': 'usize',
        'non_whitespace': 'usize',
    },
    'context_len_missing': {
        'found': 'String',
    },
    'context_length_mismatch': {
        'actual': 'usize',
        'declared': 'usize',
    },
    'context_not_base64': {
        'expected': 'String',
        'found': 'String',
        'offset': 'usize',
    },
    'context_not_bytes': {
        'got': 'String',
    },
    'context_not_utf8': {
        'invalid_len': 'usize',
        'offset': 'usize',
    },
    'context_over_cap': {
        'actual': 'usize',
        'cap': 'usize',
    },
    'duplicate_option': {
        'duplicate_index': 'usize',
        'first_index': 'usize',
        'option': 'String',
        'slot': 'String',
    },
    'duplicate_slot_name': {
        'duplicate_index': 'usize',
        'first_index': 'usize',
        'name': 'String',
    },
    'empty_option': {
        'option_index': 'usize',
        'slot': 'String',
    },
    'empty_slot_name': {
        'index': 'usize',
    },
    'empty_slots': {},
    'empty_task': {},
    'hash_mismatch': {
        'actual': 'String',
        'expected': 'String',
        'which': 'HashKind',
    },
    'malformed_request': {
        'detail': 'String',
    },
    'option_text_over_cap': {
        'actual': 'usize',
        'cap': 'usize',
        'option_index': 'usize',
        'slot': 'String',
    },
    'payload_over_cap': {
        'actual': 'usize',
        'cap': 'usize',
    },
    'question_over_cap': {
        'actual': 'usize',
        'cap': 'usize',
    },
    'registered_head_missing': {
        'available': 'Vec<String>',
        'task': 'String',
    },
    'registered_head_slot_missing': {
        'available': 'Vec<String>',
        'slot': 'String',
        'task': 'String',
    },
    'registered_route_span_unsupported': {
        'rows': 'usize',
        'slot': 'String',
    },
    'rendered_prompt_over_cap': {
        'actual': 'usize',
        'cap': 'usize',
    },
    'reserved_option_name': {
        'option_index': 'usize',
        'reserved': 'String',
        'slot': 'String',
    },
    'slot_field_missing': {
        'field': 'String',
        'slot': 'String',
        'slot_type': 'String',
    },
    'slot_field_not_allowed': {
        'field': 'String',
        'slot': 'String',
        'slot_type': 'String',
    },
    'slot_name_over_cap': {
        'actual': 'usize',
        'cap': 'usize',
        'index': 'usize',
    },
    'task_over_cap': {
        'actual': 'usize',
        'cap': 'usize',
    },
    'too_few_options': {
        'actual': 'usize',
        'min': 'usize',
        'slot': 'String',
    },
    'too_many_options': {
        'actual': 'usize',
        'limit': 'usize',
        'slot': 'String',
    },
    'too_many_slots': {
        'actual': 'usize',
        'cap': 'usize',
    },
    'unknown_op': {
        'actual': 'String',
        'supported': 'Vec<String>',
    },
    'unknown_route': {
        'actual': 'String',
        'supported': 'Vec<String>',
    },
    'unknown_schema_version': {
        'actual': 'u32',
        'supported': 'Vec<u32>',
    },
    'unknown_slot_type': {
        'actual': 'String',
        'slot': 'String',
        'supported': 'Vec<String>',
    },
}

#: `BackendError::kind()` -> that variant's fields.
BACKEND_ERROR_KINDS: Final[dict[str, dict[str, str]]] = {
    'deadline_exceeded': {
        'limit_ms': 'u64',
    },
    'decode_failed': {
        'detail': 'String',
    },
    'logit_kind_mismatch': {
        'expected': 'String',
        'slot': 'String',
    },
    'logit_shape_mismatch': {
        'actual': 'usize',
        'expected': 'usize',
        'slot': 'String',
    },
    'non_finite_logit': {
        'row': 'usize',
        'slot': 'String',
    },
    'overloaded': {
        'limit': 'usize',
    },
    'poisoned': {
        'reason': 'String',
    },
    'prefill_failed': {
        'detail': 'String',
    },
    'readonly_violated': {
        'after': 'String',
        'before': 'String',
        'slot_index': 'usize',
    },
    'rebuild_failed': {
        'detail': 'String',
    },
    'reference_backend_not_enabled': {
        'name': 'String',
    },
    'unavailable': {
        'detail': 'String',
    },
}

#: `HashKind`, the value of `refusal.which` on a `hash_mismatch`.
HASH_KINDS: Final[tuple[str, ...]] = (
    'calibration',
    'head',
    'head_backbone_binding',
    'label_set',
    'tokenizer',
    'weights',
)

#: `schema.rs::SUPPORTED_SCHEMA_VERSIONS`. Anything else is a refusal, never a lenient read.
SUPPORTED_SCHEMA_VERSIONS: Final[tuple[int, ...]] = (1,)

#: `SlotKind`.
SLOT_KINDS: Final[tuple[str, ...]] = ('choice', 'score', 'span',)

#: `Route`.
ROUTES: Final[tuple[str, ...]] = ('generic', 'registered',)

#: `schema.rs::MAX_SLOT_NAME_BYTES` — bytes a slot name may occupy, UTF-8, before escaping.
#:
#: Generated from the Rust line, not transcribed from it, and re-derived and compared on every
#: test run. That is the whole mechanism: a cap one lane enforces and the other does not is not a
#: cap, it is a disagreement, and this repository has already paid for that twice
#: (``GAP-RT-WIRE-CONTEXT-ENCODING``, ``GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS``).
#: ``GAP-RT-SLOT-NAME-UNCAPPED``.
MAX_SLOT_NAME_BYTES: Final[int] = 256

