#[path = "../src/vxj.rs"]
#[allow(dead_code)]
mod vxj;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct ResultRow {
    path: String,
    raw: usize,
    rec: Option<usize>,
    reason: Option<String>,
    enc_s: f64,
    dec_s: f64,
    mismatch: bool,
}
fn collect(root: &Path, dir: &Path, files: &mut Vec<(String, PathBuf)>) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        let path = entry.path();
        if kind.is_dir() {
            collect(root, &path, files)?;
        } else if kind.is_file()
            && path
                .extension()
                .is_some_and(|s| s.eq_ignore_ascii_case("jpg") || s.eq_ignore_ascii_case("jpeg"))
        {
            let relative = path
                .strip_prefix(root)
                .map_err(|e| e.to_string())?
                .to_str()
                .ok_or("non-UTF8 path")?
                .replace('\\', "/");
            files.push((relative, path));
        }
    }
    Ok(())
}
fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}
fn run() -> Result<bool, String> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: vxj_bench <root> <out.json>".into());
    }
    let root = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let mut files = Vec::new();
    collect(&root, &root, &mut files)?;
    files.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let next = AtomicUsize::new(0);
    let mut rows = std::thread::scope(|scope| -> Result<Vec<ResultRow>, String> {
        let mut workers = Vec::new();
        for _ in 0..16 {
            workers.push(scope.spawn(|| -> Result<Vec<ResultRow>, String> {
                let mut local = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= files.len() {
                        break;
                    }
                    let (path, full) = &files[i];
                    let jpeg = fs::read(full).map_err(|e| format!("{path}: {e}"))?;
                    let start = Instant::now();
                    let encoded = vxj::vxj_encode(&jpeg);
                    let enc_s = start.elapsed().as_secs_f64();
                    let mut row = ResultRow {
                        path: path.clone(),
                        raw: jpeg.len(),
                        rec: None,
                        reason: None,
                        enc_s,
                        dec_s: 0.0,
                        mismatch: false,
                    };
                    match encoded {
                        Ok(rec) => {
                            row.rec = Some(rec.len());
                            let start = Instant::now();
                            let decoded = vxj::vxj_decode(&rec);
                            row.dec_s = start.elapsed().as_secs_f64();
                            row.mismatch = match decoded {
                                Ok(v) => v != jpeg,
                                Err(_) => true,
                            };
                        }
                        Err(reason) => row.reason = Some(reason),
                    }
                    local.push(row);
                }
                Ok(local)
            }));
        }
        let mut rows = Vec::new();
        for worker in workers {
            rows.extend(worker.join().map_err(|_| "worker panicked")??);
        }
        Ok(rows)
    })?;
    rows.sort_by(|a, b| a.path.as_bytes().cmp(b.path.as_bytes()));
    let raw: usize = rows.iter().map(|r| r.raw).sum();
    let supported = rows.iter().filter(|r| r.rec.is_some()).count();
    let supported_raw: usize = rows.iter().filter(|r| r.rec.is_some()).map(|r| r.raw).sum();
    let vxj_bytes: usize = rows.iter().filter_map(|r| r.rec).sum();
    let combined = vxj_bytes + raw - supported_raw;
    let enc_s: f64 = rows.iter().map(|r| r.enc_s).sum();
    let dec_s: f64 = rows.iter().map(|r| r.dec_s).sum();
    let mismatches = rows.iter().filter(|r| r.mismatch).count();
    let mut reasons = BTreeMap::<String, usize>::new();
    for r in &rows {
        if r.mismatch {
            println!("MISMATCH {}", r.path);
        }
        if let Some(reason) = &r.reason {
            *reasons
                .entry(reason.chars().take(40).collect())
                .or_default() += 1;
        }
    }
    let mut json = format!(
        "{{\n  \"files\": {},\n  \"raw\": {raw},\n  \"supported\": {supported},\n  \"supported_raw\": {supported_raw},\n  \"vxj_bytes\": {vxj_bytes},\n  \"combined\": {combined},\n  \"enc_s\": {enc_s:.9},\n  \"dec_s\": {dec_s:.9},\n  \"mismatches\": {mismatches},\n  \"reasons\": {{", rows.len());
    for (i, (reason, count)) in reasons.iter().enumerate() {
        if i != 0 {
            json.push(',');
        }
        json.push_str(&format!("\n    {}: {count}", quote(reason)));
    }
    json.push_str("\n  },\n  \"per_file\": [");
    for (i, r) in rows.iter().enumerate() {
        if i != 0 {
            json.push(',');
        }
        let rec = r
            .rec
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".into());
        let reason = r
            .reason
            .as_deref()
            .map(quote)
            .unwrap_or_else(|| "null".into());
        json.push_str(&format!(
            "\n    {{\"path\": {}, \"raw\": {}, \"rec\": {rec}, \"reason\": {reason}}}",
            quote(&r.path),
            r.raw
        ));
    }
    json.push_str("\n  ]\n}\n");
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(output, json).map_err(|e| e.to_string())?;
    let ratio = if raw == 0 {
        0.0
    } else {
        combined as f64 / raw as f64
    };
    let supported_pct = if raw == 0 {
        0.0
    } else {
        supported_raw as f64 / raw as f64 * 100.0
    };
    let enc_mbps = if enc_s == 0.0 {
        0.0
    } else {
        supported_raw as f64 / enc_s / 1e6
    };
    let dec_mbps = if dec_s == 0.0 {
        0.0
    } else {
        supported_raw as f64 / dec_s / 1e6
    };
    println!("VXJ files={} supported={supported} raw={raw} combined={combined} ratio={ratio:.6} supported_pct={supported_pct:.3} enc_MBps={enc_mbps:.3} dec_MBps={dec_mbps:.3} mismatches={mismatches}", rows.len());
    Ok(mismatches != 0)
}
fn main() {
    match run() {
        Ok(false) => {}
        Ok(true) => std::process::exit(2),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
