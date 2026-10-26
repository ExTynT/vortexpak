#[path = "../src/dxt.rs"]
mod dxt;

use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn collect_dds(root: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_dds(&path, files)?;
        } else if file_type.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("dds"))
        {
            files.push(path);
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let root = env::args_os().nth(1).ok_or("usage: dxt_roundtrip <root>")?;
    let mut files = Vec::new();
    collect_dds(Path::new(&root), &mut files)?;
    files.sort();
    let mut split = 0usize;
    let mut skipped = 0usize;
    let mut raw = 0u64;
    let mut rec = 0u64;
    let mut mismatches = 0usize;
    for path in &files {
        let input = fs::read(path)?;
        match dxt::dxt_split(&input) {
            Some(record) => {
                split += 1;
                raw += input.len() as u64;
                rec += record.len() as u64;
                match dxt::dxt_merge(&record) {
                    Ok(restored) if restored == input => {}
                    _ => {
                        println!("MISMATCH {}", path.display());
                        mismatches += 1;
                    }
                }
            }
            None => skipped += 1,
        }
    }
    println!(
        "DXT files={} split={} skipped={} raw={} rec={} mismatches={} secs={:.6}",
        files.len(),
        split,
        skipped,
        raw,
        rec,
        mismatches,
        start.elapsed().as_secs_f64()
    );
    if mismatches != 0 {
        std::process::exit(2);
    }
    Ok(())
}
