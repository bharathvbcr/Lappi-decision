//! `qd_prep::blake2b` against digests printed by CPython's `hashlib.blake2b` (3.14.7).
//!
//! Data byte `i` is `(7*i + 3) % 256` and key byte `i` is `(13*i + 5) % 256`. The lengths
//! straddle every boundary the implementation branches on: empty, one block short, exactly
//! one block, one block plus one byte, two blocks, and a keyed message (whose key is its
//! own first block). Regenerate with:
//!
//! ```text
//! python -c "import hashlib; d=lambda n: bytes((i*7+3)%256 for i in range(n)); \
//!   k=lambda n: bytes((i*13+5)%256 for i in range(n)); \
//!   print(hashlib.blake2b(d(129), key=k(32), digest_size=16).hexdigest())"
//! ```

use qd_prep::blake2b::blake2b_into;

const VECTORS: &[(usize, usize, usize, &str)] = &[
    (
        0,
        0,
        64,
        "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce",
    ),
    (0, 1, 8, "75b1f99a891d83a3"),
    (0, 32, 16, "ea0400402cdd07ca52e467326a8777d6"),
    (
        0,
        64,
        32,
        "a5e682185af9cfa0295905ef9bd0deea3d6dbafcf1a906e64554f506058b5826",
    ),
    (
        1,
        0,
        64,
        "4fe4da61bcc756071b226843361d74944c72245d23e8245ea678c13fdcd7fe2ae529cf999ad99cc24f7a73416a18ba53e76c0afef83b16a568b12fbfc1a2674d",
    ),
    (1, 1, 8, "8900a767d1926c1e"),
    (1, 32, 16, "6ecbd25cd21cc20055c2bafad1d3057a"),
    (
        1,
        64,
        32,
        "b6a9698778d88f9c20084d76f29abb4c6657f34a0cbf2d4037a4e4d232556bd8",
    ),
    (
        3,
        0,
        64,
        "76ce7e90c577d436a65b691e51f0a1ad536130d70a18df3bf5368551327258b6f706f4a47c45f2d0d8147cfbb5623077e798247972e4f9338603a13abc05a4db",
    ),
    (3, 1, 8, "d06ad31db8a1622a"),
    (3, 32, 16, "c9ddadef8c585d64ea974e5479e054a7"),
    (
        3,
        64,
        32,
        "88e4d88ea695c00f798b8effecfd813de1c0aab41db75edc426ebd383ed72832",
    ),
    (
        8,
        0,
        64,
        "c23ae16bf2655c6ecb4d70fd2959931ac6757a4324d963bb2be7bcb6b08cd16b2eeaa23acc7389db62f85d0f53377683f6b67bd2dab59c10fe4cc98a2e3e2e7f",
    ),
    (8, 1, 8, "2dbc6df424238e7e"),
    (8, 32, 16, "385ff5cd814bcbdbc0c37b8d96820637"),
    (
        8,
        64,
        32,
        "568ac1f2d6ebc6bfdbd177b582b50280a937b071520d43e1456e2e6afb0c922b",
    ),
    (
        63,
        0,
        64,
        "70b2a0e6daecac22c7a2df82c06e3fc0b4c66bd5ef8098e4ed54e723b393d79ef3bceba079a01a14c6ef2ae2ed1171df1662cd14ef38e6f77b01c7f48144dd09",
    ),
    (63, 1, 8, "38a2487faf340cf9"),
    (63, 32, 16, "0be4e55670b96fafbad89859c5242c96"),
    (
        63,
        64,
        32,
        "a870b1104dccaf9ee58ff41eee054746abf8f70bac530f63d19e68051f535f1d",
    ),
    (
        127,
        0,
        64,
        "71546bbf9110ad184cc60f2eb120fcfd9b4dbbca7a7f1270045b8a23a6a4f4330f65c1f030dd2f5fabc6c57617242c37cf427bd90407fac5b9deffd3ae888c39",
    ),
    (127, 1, 8, "4fa193a9b1c4e23d"),
    (127, 32, 16, "48271caef2440f6b51f306a74c905200"),
    (
        127,
        64,
        32,
        "15f0d86d48e8eea6d221916f1cdb6f962bcec7f44267bbd515228551f8abd3b3",
    ),
    (
        128,
        0,
        64,
        "2d9e329f42afa3601d646692b81c13e87fcaff5bf15972e9813d7373cb6d181f9599f4d513d4af4fd6ebd37497aceb29aba5ee23ed764d8510b552bd088814fb",
    ),
    (128, 1, 8, "356a81c7cad61370"),
    (128, 32, 16, "826a9fec0d30c909ce46c46fe287fd9c"),
    (
        128,
        64,
        32,
        "897d6c3860d463802870971d3c49322d441a237f65443947a7c10f695a720f50",
    ),
    (
        129,
        0,
        64,
        "47889df9eb4d717afc5019df5c6a83df00a0b8677395e078cd5778ace0f338a618e68b7d9afb065d9e6a01ccd31d109447e7fae771c3ee3e105709194122ba2b",
    ),
    (129, 1, 8, "c0ab91917d725e8e"),
    (129, 32, 16, "7685c6e295666c556ea79c30afa8c9eb"),
    (
        129,
        64,
        32,
        "8655c050b5e743ed00456df8e47739e2963c896e7f3e68e1436b8898ab51831c",
    ),
    (
        255,
        0,
        64,
        "1a5199ac66a00e8a87ad1c7fbad30b33137dd8312bf6d98602dacf8f40ea2cb623a7fbc63e5a6bfa434d337ae7da5ca1a52502a215a3fe0297a151be85d88789",
    ),
    (255, 1, 8, "1ca1402ee25f3b1a"),
    (255, 32, 16, "32f0e6b3edfc0c60f48dbf5ec7712ef7"),
    (
        255,
        64,
        32,
        "04744dc30402c78768bb6472c1d77b49e68785b467c6487cb3882aef5168f0ef",
    ),
    (
        256,
        0,
        64,
        "91019c558584980249ca43eceed27e19f1c3c24161b93eed1eee2a6a774f60bf8a81b43750870bee1698feac9c5336ae4d5c842e7ead159bf3916387e8ded9ae",
    ),
    (256, 1, 8, "8535cd4d23243837"),
    (256, 32, 16, "0bb583fcaeb54e3f66143743edc746e8"),
    (
        256,
        64,
        32,
        "ca46976458ba0f96933d3881389f21c9e598b9a242de3e00df101314ed00a7dd",
    ),
    (
        257,
        0,
        64,
        "9f1975efca45e7b74b020975d4d2c22802906ed8bfefca51ac497bd23147fc8f303890d8e5471ab6caaa02362e831a9e8d3435279912ccd4842c7806b096c348",
    ),
    (257, 1, 8, "82417964235e812d"),
    (257, 32, 16, "6bb36f46386f14757163b10108bddec9"),
    (
        257,
        64,
        32,
        "ce239816fca14fa457580ec3eb0d8d8ee21585402e12036f732b7de3e703323f",
    ),
    (
        1000,
        0,
        64,
        "4bdd2c9cf31d797a81d245c989ffb7515143ca345c66f73087dd5c58bf642bf083ba16894eab79e3b08d5126404d833e7510271b50be36a7b7cbbb46f5c89fac",
    ),
    (1000, 1, 8, "ff3b8748d275a835"),
    (1000, 32, 16, "e32be4681ba371e1a9b5f15084407f06"),
    (
        1000,
        64,
        32,
        "e33929e6bcc278fe54f0603bcc2f5241f6d4edb48078303e21061102adac15be",
    ),
];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn agrees_with_hashlib() {
    for &(data_len, key_len, digest, want) in VECTORS {
        let data: Vec<u8> = (0..data_len).map(|i| ((i * 7 + 3) % 256) as u8).collect();
        let key: Vec<u8> = (0..key_len).map(|i| ((i * 13 + 5) % 256) as u8).collect();
        let mut out = vec![0u8; digest];
        blake2b_into(&mut out, &key, &data);
        assert_eq!(
            hex(&out),
            want,
            "data={data_len} key={key_len} digest={digest}"
        );
    }
}
