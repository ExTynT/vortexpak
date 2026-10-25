use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use vortexpak_native::vfs::VortexVfs;

fn print_usage() {
    println!("VortexPak Native High-Throughput Engine (Rust + AVX2 SIMD)");
    println!("Usage:");
    println!("  vortexpak-native info <archive.vpak>");
    println!("  vortexpak-native vfs-read <archive.vpak> <asset_path> [output_file]");
    println!("  vortexpak-native vfs-bench <archive.vpak> [num_reads]");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "info" => {
            if args.len() < 3 {
                eprintln!("Error: archive file required");
                return Ok(());
            }
            let vpak_path = &args[2];
            let vfs =
                VortexVfs::open_file(vpak_path).map_err(|e| format!("VFS open error: {e}"))?;

            println!("Archive Mode:       {}", vfs.mode);
            println!("Total Assets:       {}", vfs.entries.len());
            println!("Total Raw Bytes:    {} B", vfs.total_raw_bytes);
            println!("Parallel Blocks:    {}", vfs.blocks.len());
            println!("\nBlocks Summary:");
            for b in &vfs.blocks {
                println!(
                    "  Block #{:<2} | {:<22} | Strat #{:<2} | Raw: {:>10} B | Comp: {:>10} B | Ratio: {:.2}x | File Offset: 0x{:08X}",
                    b.block_id,
                    b.lane_name,
                    b.strategy_id,
                    b.raw_size,
                    b.comp_size,
                    if b.comp_size > 0 { b.raw_size as f64 / b.comp_size as f64 } else { 1.0 },
                    b.file_offset
                );
            }
        }
        "vfs-read" => {
            if args.len() < 4 {
                eprintln!(
                    "Usage: vortexpak-native vfs-read <archive.vpak> <asset_path> [output_file]"
                );
                return Ok(());
            }
            let vpak_path = &args[2];
            let asset_path = &args[3];
            let mut vfs =
                VortexVfs::open_file(vpak_path).map_err(|e| format!("VFS open error: {e}"))?;

            let t0 = Instant::now();
            let data = vfs
                .read_asset(asset_path, true)
                .map_err(|e| format!("VFS read error: {e}"))?;
            let elapsed = t0.elapsed();

            println!(
                "[VFS On-Demand] Successfully read asset '{}' ({} B) in {:.3} ms ({:.2} MB/s)",
                asset_path,
                data.len(),
                elapsed.as_secs_f64() * 1000.0,
                (data.len() as f64 / 1_000_000.0) / elapsed.as_secs_f64().max(1e-6)
            );

            if args.len() >= 5 {
                let out_path = &args[4];
                let mut out_file = File::create(out_path)?;
                out_file.write_all(&data)?;
                println!("Saved to {}", out_path);
            }
        }
        "vfs-bench" => {
            if args.len() < 3 {
                eprintln!("Usage: vortexpak-native vfs-bench <archive.vpak> [num_reads]");
                return Ok(());
            }
            let vpak_path = &args[2];
            let num_reads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);

            let mut vfs =
                VortexVfs::open_file(vpak_path).map_err(|e| format!("VFS open error: {e}"))?;

            let asset_list: Vec<String> = vfs
                .list_assets()
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            if asset_list.is_empty() {
                println!("Archive has no assets.");
                return Ok(());
            }

            println!(
                "Running VFS Seekable On-Demand Benchmark on {} ({} assets available, {} random access reads)...",
                vpak_path,
                asset_list.len(),
                num_reads
            );

            let mut total_bytes = 0usize;
            let t0 = Instant::now();

            for i in 0..num_reads {
                let asset = &asset_list[i % asset_list.len()];
                let bytes = vfs
                    .read_asset(asset, false)
                    .map_err(|e| format!("Failed to read {asset}: {e}"))?;
                total_bytes += bytes.len();
            }

            let elapsed = t0.elapsed();
            let avg_lat_us = (elapsed.as_secs_f64() * 1_000_000.0) / (num_reads as f64);
            let throughput_gbps = (total_bytes as f64 / 1_000_000_000.0) / elapsed.as_secs_f64();

            println!("--- VFS On-Demand Benchmark Results ---");
            println!("  Total Reads:       {}", num_reads);
            println!(
                "  Total Streamed:    {:.2} MB",
                total_bytes as f64 / 1_000_000.0
            );
            println!(
                "  Elapsed Time:      {:.3} ms",
                elapsed.as_secs_f64() * 1000.0
            );
            println!(
                "  Average Latency:   {:.2} microseconds / asset",
                avg_lat_us
            );
            println!("  Effective Speed:   {:.2} GB/s", throughput_gbps);
        }
        unknown => {
            eprintln!("Unknown command: {}", unknown);
            print_usage();
        }
    }

    Ok(())
}
