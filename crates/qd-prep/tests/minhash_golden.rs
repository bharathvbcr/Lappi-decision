//! `qd_prep::minhash::MinHasher` against signatures printed by the Python reference,
//! `qd_data.minhash.MinHasher(num_perm=16, seed=...).signature(frozenset(items))`.
//!
//! A golden here, so `cargo test` alone catches a drift; `python/tests/test_qd_prep_parity.py`
//! compares the binary to the live reference on real corpus rows and adversarial sets.

use qd_prep::minhash::MinHasher;

const ACCENTED: &[u8] = "\u{e9} \u{4e2d} \u{1F600} x y".as_bytes();

/// `(seed spelling, shingle set, the reference's signature)`.
type Golden<'a> = (&'a str, &'a [&'a [u8]], [u64; 16]);

#[test]
fn signatures_match_the_python_reference() {
    let cases: &[Golden<'_>] = &[
        (
            "20260919",
            &[b"def f ( x ) :"],
            [
                360056607288688376,
                1840163801616580912,
                946922563885154601,
                1922719973146194705,
                1321724288455754531,
                496101134031197056,
                1523029555525657352,
                2195034299662040920,
                481459033377992034,
                883929215704428136,
                924843995688674891,
                1730858787543152537,
                1424210412511068899,
                1059704193359218151,
                1976753101261470343,
                1659940263041577286,
            ],
        ),
        (
            "20260919",
            &[b"a b c d e", b"b c d e f", ACCENTED],
            [
                2075406474593983263,
                599603013629110785,
                77677604669439608,
                689946699475997191,
                499588497428159001,
                1321803494607262281,
                229185634531881420,
                1210894611538865023,
                412569408688715259,
                534175390209579037,
                21930531086706173,
                655813516147813039,
                228430739479177047,
                793913512783605378,
                108937139432172668,
                1057934793407724923,
            ],
        ),
        (
            "-7",
            &[b"x"],
            [
                1877598487761485362,
                165744527286367247,
                2200841171022053989,
                1124631672464442133,
                275034899571536094,
                1628571814237438574,
                161002918472058311,
                916471324666508574,
                106282031783493201,
                1771574525173027958,
                1788275095128241903,
                635146779667822655,
                594219094052440526,
                265528603016855387,
                1758511677769132501,
                1881977396094974474,
            ],
        ),
        (
            // Wider than any machine integer: the seed only ever travels as its spelling.
            "123456789012345678901234567890",
            &[b"", b"\xff\xfe"],
            [
                365189940074293238,
                320431244954001890,
                933262387819994581,
                137857743327449819,
                528133178649247886,
                1551958462360650407,
                224400266324995474,
                889976964725617541,
                203991838179411547,
                1547537859308644734,
                84647091658310331,
                1604840989573674529,
                859832609837251828,
                687688142211494709,
                1031508628066688480,
                1232033556306681624,
            ],
        ),
    ];
    for (seed, items, want) in cases {
        let h = MinHasher::new(16, seed).unwrap();
        let got = h.signature(items.iter().copied()).unwrap();
        assert_eq!(got, want.to_vec(), "seed {seed}");
        // The minimum does not depend on the order a frozenset happens to iterate in.
        let reversed = h.signature(items.iter().rev().copied()).unwrap();
        assert_eq!(reversed, got, "seed {seed}, reversed");
    }
}
