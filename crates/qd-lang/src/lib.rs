//! `qd-lang` — the source languages, owned once.
//!
//! | Item | Who reads it |
//! | --- | --- |
//! | [`LangId`] | `qd-mutate` (every operator and facade), `qd-runtime`'s admission check |
//! | [`language_from_path`] | `qd-mutate`'s pool reader and CLI, `qd-runtime`'s admission check; `python/qd_data/pool_builder.py` mirrors it and `python/tests/test_pool_builder.py` reads this file to keep the mirror honest |
//! | [`DEFECT_CLASS_POOL_LANGUAGES`] | `qd-noul-rows` (the scrambled source), `qd-runtime`'s admission check |
//!
//! These lived in `qd-mutate` until the serving runtime needed them; `qd-mutate` links
//! tree-sitter's grammars, which have no business in the product runtime, so the language facts
//! moved here and `qd-mutate` re-exports them.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LangId {
    Rust,
    Go,
    Python,
    TypeScript,
    Swift,
}

impl LangId {
    pub const ALL: [LangId; 5] = [
        LangId::Rust,
        LangId::Go,
        LangId::Python,
        LangId::TypeScript,
        LangId::Swift,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            LangId::Rust => "rust",
            LangId::Go => "go",
            LangId::Python => "python",
            LangId::TypeScript => "typescript",
            LangId::Swift => "swift",
        }
    }

    pub fn parse(s: &str) -> Option<LangId> {
        match s {
            "rust" | "rs" => Some(LangId::Rust),
            "go" => Some(LangId::Go),
            "python" | "py" => Some(LangId::Python),
            "typescript" | "ts" => Some(LangId::TypeScript),
            "swift" => Some(LangId::Swift),
            _ => None,
        }
    }

    /// Extension used when a formatter needs a filename to decide its dialect.
    pub fn extension(self) -> &'static str {
        match self {
            LangId::Rust => "rs",
            LangId::Go => "go",
            LangId::Python => "py",
            LangId::TypeScript => "ts",
            LangId::Swift => "swift",
        }
    }
}

impl std::fmt::Display for LangId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The language a path names, or `None`.
///
/// `.d.ts` is excluded on purpose: a declaration file has no function bodies, so it would parse,
/// yield nothing, and inflate the "files seen, zero examples" column with files that never could
/// have produced one.
pub fn language_from_path(path: &str) -> Option<LangId> {
    if path.ends_with(".d.ts") {
        return None;
    }
    let ext = std::path::Path::new(path).extension()?.to_str()?;
    match ext {
        "rs" => Some(LangId::Rust),
        "go" => Some(LangId::Go),
        "py" | "pyi" => Some(LangId::Python),
        "ts" | "tsx" | "mts" | "cts" => Some(LangId::TypeScript),
        "swift" => Some(LangId::Swift),
        _ => None,
    }
}

/// The languages the `code.defect_class` pool holds: what the model was trained on.
///
/// Narrower than [`LangId::ALL`]. The pool *reader* accepts Swift, but the pool that was built
/// holds none: `data/pool/commitpackft-pool-v2.manifest.json`'s `by_language` is go 4,502,
/// python 48,722, rust 2,694, typescript 5,275, and `data/pool/commitpackft-corpus-v3/manifest.json`
/// records swift `files_seen: 0, examples: 0` (`AUDIT/commitpackft-pool-2026-09-21.md` §2: four
/// languages fetched). A language listed here is one the model has seen a `code.defect_class`
/// row in; Swift is not, so a Swift diff is out of distribution however readable its path is.
pub const DEFECT_CLASS_POOL_LANGUAGES: [LangId; 4] =
    [LangId::Go, LangId::Python, LangId::Rust, LangId::TypeScript];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_names_its_language_and_a_declaration_file_names_none() {
        assert_eq!(language_from_path("a/b.rs"), Some(LangId::Rust));
        assert_eq!(language_from_path("a/b.go"), Some(LangId::Go));
        assert_eq!(language_from_path("a/b.py"), Some(LangId::Python));
        assert_eq!(language_from_path("a/b.tsx"), Some(LangId::TypeScript));
        assert_eq!(language_from_path("a/b.swift"), Some(LangId::Swift));
        assert_eq!(language_from_path("a/b.d.ts"), None, "no bodies to mutate");
        assert_eq!(language_from_path("Makefile"), None);
    }

    #[test]
    fn the_pool_held_four_of_the_five_languages_and_not_swift() {
        assert!(
            DEFECT_CLASS_POOL_LANGUAGES
                .iter()
                .all(|lang| LangId::ALL.contains(lang))
        );
        assert!(!DEFECT_CLASS_POOL_LANGUAGES.contains(&LangId::Swift));
        assert_eq!(DEFECT_CLASS_POOL_LANGUAGES.len(), 4);
    }

    /// The pool manifest is the evidence the constant rests on; if it is rebuilt with another
    /// language mix, this fails and names the difference rather than letting the constant rot.
    #[test]
    fn the_constant_is_the_pool_manifests_language_set() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/pool/commitpackft-pool-v2.manifest.json");
        let text = std::fs::read_to_string(&manifest)
            .unwrap_or_else(|e| panic!("{}: {e}", manifest.display()));
        let doc: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", manifest.display()));
        let held: std::collections::BTreeSet<&str> = doc["by_language"]
            .as_object()
            .unwrap_or_else(|| panic!("{} has no by_language object", manifest.display()))
            .iter()
            .filter(|(_, rows)| rows.as_u64().is_some_and(|n| n > 0))
            .map(|(name, _)| name.as_str())
            .collect();
        let ours: std::collections::BTreeSet<&str> = DEFECT_CLASS_POOL_LANGUAGES
            .iter()
            .map(|lang| lang.as_str())
            .collect();
        assert_eq!(ours, held, "{}", manifest.display());
    }
}
