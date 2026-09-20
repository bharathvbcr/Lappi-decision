//! Grammar probe: dumps the s-expression tree for a sample in each language.
//! Evidence for the node kinds the facade names. Not part of the library.

fn dump(label: &str, lang: tree_sitter::Language, src: &str) {
    let mut p = tree_sitter::Parser::new();
    if let Err(e) = p.set_language(&lang) {
        println!("=== {label} === SET_LANGUAGE FAILED: {e}");
        return;
    }
    match p.parse(src, None) {
        Some(t) => println!("=== {label} ===\n{}\n", t.root_node().to_sexp()),
        None => println!("=== {label} === NO TREE"),
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_default();
    let rust_src = r#"
use std::io::Read;
use std::fmt;
/// doc
pub async fn f<T>(a: usize, b: usize) -> Result<Vec<u8>, Error> where T: Clone {
    // a comment
    let mut total = 0;
    for i in 0..10 {
        if a < b { total += 1; } else { total -= 1; }
    }
    let g = m.lock().unwrap();
    let v = read_it(a, b)?;
    other(1, 2).await;
    Ok(vec![])
}
trait T { fn sig(&self) -> u32; fn deflt(&self) -> u32 { 7 } }
"#;
    let go_src = r#"
package main
import (
	"fmt"
	"os"
)
// comment
func F(a int, b int) ([]byte, error) {
	total := 0
	for i := 0; i < 10; i++ {
		if a < b {
			total++
		} else {
			total--
		}
	}
	mu.Lock()
	v, err := readIt(a, b)
	if err != nil {
		return nil, err
	}
	return v, nil
}
type I interface { Sig() int }
"#;
    let py_src = r#"
import os
import sys
def f(a, b):
    """doc"""
    # comment
    total = 0
    for i in range(10):
        if a < b:
            total += 1
        else:
            total -= 1
    v = read_it(a, b)
    return []
async def g(a):
    await h(a)
class C:
    def m(self): ...
"#;
    let ts_src = r#"
import { a } from "./a";
import b from "./b";
// comment
export async function f(a: number, b: number): Promise<number[]> {
    let total = 0;
    for (let i = 0; i < 10; i++) {
        if (a < b) { total += 1; } else { total -= 1; }
    }
    try { await readIt(a, b); } catch (e) { throw e; }
    return [];
}
interface I { sig(): number }
abstract class C { abstract m(): void; n(): number { return 1; } }
"#;
    let swift_src = r#"
import Foundation
import os
// comment
func f(a: Int, b: Int) throws -> [UInt8] {
    var total = 0
    for i in 0..<10 {
        if a < b { total += 1 } else { total -= 1 }
    }
    lock.lock()
    let v = try readIt(a, b)
    return []
}
protocol P { func sig() -> Int }
"#;
    let run = |name: &str| which.is_empty() || which == name;
    if run("rust") {
        dump("rust", tree_sitter_rust::LANGUAGE.into(), rust_src);
    }
    if run("go") {
        dump("go", tree_sitter_go::LANGUAGE.into(), go_src);
    }
    if run("python") {
        dump("python", tree_sitter_python::LANGUAGE.into(), py_src);
    }
    if run("typescript") {
        dump(
            "typescript",
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            ts_src,
        );
    }
    if run("swift") {
        dump("swift", tree_sitter_swift::LANGUAGE.into(), swift_src);
    }
}
