#[rustfmt::skip]
#[path = "../src/vxcm.rs"] mod vxcm;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

struct Entry {
    path: String,
    disk: PathBuf,
}

struct ResultRow {
    path: String,
    raw: u64,
    comp: u64,
    enc_s: f64,
    dec_s: f64,
    mismatch: bool,
}

fn collect(root: &Path, dir: &Path, files: &mut Vec<Entry>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_dir() {
            collect(root, &path, files)?;
        } else if kind.is_file() {
            let relative = path.strip_prefix(root).map_err(io::Error::other)?;
            files.push(Entry {
                path: relative.to_string_lossy().replace('\\', "/"),
                disk: path,
            });
        }
    }
    Ok(())
}

fn quoted(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0}'..='\u{1f}' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 4 {
        return Err("usage: vxcm_bench <root> <K> <out.json>".into());
    }
    let root = Path::new(&args[1]);
    let k: usize = args[2].to_str().ok_or("invalid K")?.parse()?;
    if k == 0 {
        return Err("K must be positive".into());
    }
    let mut all = Vec::new();
    collect(root, root, &mut all)?;
    all.sort_by(|a, b| a.path.as_bytes().cmp(b.path.as_bytes()));
    let files: Vec<_> = all
        .into_iter()
        .enumerate()
        .filter_map(|(i, entry)| (i % k == 0).then_some(entry))
        .collect();
    let index = AtomicUsize::new(0);
    let rows = Mutex::new(Vec::with_capacity(files.len()));
    let errors = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..16 {
            let files = &files;
            let index = &index;
            let rows = &rows;
            let errors = &errors;
            scope.spawn(move || loop {
                let i = index.fetch_add(1, Ordering::Relaxed);
                let Some(entry) = files.get(i) else {
                    break;
                };
                let data = match fs::read(&entry.disk) {
                    Ok(data) => data,
                    Err(error) => {
                        errors
                            .lock()
                            .unwrap()
                            .push(format!("{}: {}", entry.path, error));
                        continue;
                    }
                };
                let start = Instant::now();
                let comp = vxcm::vxcm_compress(&data, 9);
                let enc_s = start.elapsed().as_secs_f64();
                let start = Instant::now();
                let decoded = vxcm::vxcm_decompress(&comp, data.len());
                let dec_s = start.elapsed().as_secs_f64();
                let mismatch = decoded.as_ref().map_or(true, |out| out != &data);
                if mismatch {
                    println!("MISMATCH {}", entry.path);
                }
                rows.lock().unwrap().push(ResultRow {
                    path: entry.path.clone(),
                    raw: data.len() as u64,
                    comp: comp.len() as u64,
                    enc_s,
                    dec_s,
                    mismatch,
                });
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    if !errors.is_empty() {
        return Err(errors.join("\n").into());
    }
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by(|a, b| a.path.as_bytes().cmp(b.path.as_bytes()));
    let raw: u64 = rows.iter().map(|r| r.raw).sum();
    let comp: u64 = rows.iter().map(|r| r.comp).sum();
    let enc_s: f64 = rows.iter().map(|r| r.enc_s).sum();
    let dec_s: f64 = rows.iter().map(|r| r.dec_s).sum();
    let mismatches = rows.iter().filter(|r| r.mismatch).count();
    let per_file = rows
        .iter()
        .map(|r| {
            format!(
                "{{\"path\":{},\"raw\":{},\"comp\":{}}}",
                quoted(&r.path),
                r.raw,
                r.comp
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let json = format!("{{\"files\":{},\"raw\":{},\"comp\":{},\"enc_s\":{},\"dec_s\":{},\"mismatches\":{},\"per_file\":[\n{}\n]}}\n", rows.len(), raw, comp, enc_s, dec_s, mismatches, per_file);
    let output = Path::new(&args[3]);
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, json)?;
    let enc_speed = if enc_s > 0.0 {
        raw as f64 / enc_s / 1e6
    } else {
        0.0
    };
    let dec_speed = if dec_s > 0.0 {
        raw as f64 / dec_s / 1e6
    } else {
        0.0
    };
    println!(
        "VXCM files={} raw={} comp={} enc_MBps={:.3} dec_MBps={:.3} mismatches={}",
        rows.len(),
        raw,
        comp,
        enc_speed,
        dec_speed,
        mismatches
    );
    Ok(mismatches == 0)
}

fn main() {
    match run() {
        Ok(true) => (),
        Ok(false) => std::process::exit(2),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
