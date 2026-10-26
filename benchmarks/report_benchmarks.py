"""Build neutral HTML from validated reports without changing evidence."""

import argparse
import sys
from html import escape
import os
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import protocol as bp

MEMORY_KEY = "os_process_lifetime_peak_working_set_bytes"


def load_reports(inputs, legacy=False):
    """Load explicit folders or discover immediate report children of roots.

    Only incomplete discovered children are skipped. Explicit incomplete inputs,
    unsupported schemas and integrity failures are errors, not exclusions.
    """
    experiments, seen = [], set()
    for source in map(Path, inputs):
        if not source.is_dir():
            raise ValueError(f"report input is not a directory: {source}")
        explicit = (source / "report.json").is_file()
        candidates = (
            [source]
            if explicit
            else sorted(
                child
                for child in source.iterdir()
                if child.is_dir() and (child / "report.json").is_file()
            )
        )
        if not candidates:
            raise ValueError(f"no report.json in requested folder/root: {source}")
        for folder in candidates:
            identity = folder.resolve()
            if identity in seen:
                continue
            try:
                report = bp.load_report(folder, legacy=legacy)
            except bp.IncompleteReportError:
                if explicit:
                    raise
                continue
            experiments.append((folder, report))
            seen.add(identity)
    if not experiments:
        raise ValueError("no complete validated reports in requested inputs")
    return experiments


def is_legacy(report):
    return report.get("schema", report.get("schema_version")) == 1


def track(report):
    if is_legacy(report):
        return (
            "quality"
            if report.get("track") == "lossy_textures_not_byte_exact"
            else "archive"
        )
    return report["track"]


def number(value, digits=3):
    return "N/A" if value is None else f"{value:,.{digits}f}"


def metric_lines(metric, scale=1, digits=3, include_ci=True):
    """Accept exact metrics and unresolved CPU without fabricating CIs."""
    value = metric["median"]
    if value is None:
        return ("N/A",)
    result = (number(value / scale, digits),)
    ci = metric.get("bootstrap_95ci") if include_ci else None
    if ci is not None:
        bounds = [number(v / scale, digits) if v is not None else "N/A" for v in ci]
        result += (f"[{bounds[0]}, {bounds[1]}]",)
    return result


def interval(metric, scale=1):
    parts = metric_lines(metric, scale)
    return escape(parts[0]) + "".join(
        f" <small>{escape(part)}</small>" for part in parts[1:]
    )


def tail_flags(report):
    """Return coverage booleans, never interpreting them as qualification."""
    flags = []

    def visit(values, prefix=""):
        for key, value in sorted(values.items()):
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                visit(value, name + ".")
            elif isinstance(value, bool) and (
                "tail" in name.lower() or "p99" in name.lower()
            ):
                flags.append((name, value))

    visit(report.get("adequacy", {}))
    return flags


def exploratory(report):
    flags = tail_flags(report)
    return (
        report.get("settings", {}).get("profile") == "diagnostic"
        or not flags
        or not all(value for _, value in flags)
    )


def report_table(report, legacy=False):
    """Canonical alphabetical cells shared by HTML and PNG; no ranking."""
    old = is_legacy(report)
    if old and not legacy:
        raise ValueError("schema 1 presentation requires explicit --legacy")
    rows = []
    configurations = sorted(report["summary"], key=lambda name: (name.casefold(), name))
    if track(report) == "quality":
        headers = [
            "Configuration",
            "bpp",
            "RGB PSNR dB",
            "Worst image RGB PSNR dB",
            "FLIP mean",
            "FLIP worst",
            "Encode ms\n[95% CI]",
        ]
        keys = (
            "bpp",
            "rgb_psnr_srgb_db",
            "worst_image_rgb_psnr_db",
            "flip_image_mean",
            "flip_worst_image",
        )
        for config in configurations:
            summary = report["summary"][config]
            cells = [(config,)] + [
                metric_lines(
                    summary[key],
                    digits=5 if "flip" in key else 3,
                    include_ci=False,
                )
                for key in keys
            ]
            rows.append(cells + [metric_lines(summary["encode_ms"])])
        return headers, rows

    memory_label = (
        "Legacy sampled RSS" if old else "OS process-lifetime peak working set"
    )
    latency_label = "EXPLORATORY " if exploratory(report) else ""
    headers = [
        "Configuration",
        "Archive MiB\n(exact; no CI)",
        "Ratio ×",
        "Pack wall ms\n[95% CI]",
        "Restore wall ms\n[95% CI]",
        "Pack CPU ms",
        "Restore CPU ms",
        f"Pack {memory_label} MiB",
        f"Restore {memory_label} MiB",
        f"{latency_label}Reset\nmedian per-round p99 ms\n[95% CI]",
        f"{latency_label}Repeat\nmedian per-round p99 ms\n[95% CI]",
    ]
    for config in configurations:
        summary = report["summary"][config]
        size = summary["archive_bytes"]["median"]
        ratio = report["manifest"]["bytes"] / size if size else None
        cells = [
            (config,),
            metric_lines(summary["archive_bytes"], 2**20, include_ci=False),
            (number(ratio),),
            metric_lines(summary["pack"]["wall_ms"]),
            metric_lines(summary["restore"]["wall_ms"]),
            metric_lines(summary["pack"]["cpu_ms"], include_ci=False),
            metric_lines(summary["restore"]["cpu_ms"], include_ci=False),
        ]
        for phase in ("pack", "restore"):
            key = "rss_peak_sampled_bytes" if old and legacy else MEMORY_KEY
            cells.append(
                metric_lines(
                    summary[phase][key],
                    2**20,
                    include_ci=False,
                )
            )
        latency = summary.get("latency")
        for state in ("codec_reset", "repeat_after_one_prime"):
            cells.append(
                metric_lines(latency[state]["p99"])
                if latency
                else ("N/A (no random-access score)",)
            )
        rows.append(cells)
    return headers, rows


def value_text(value):
    if value is None:
        return "N/A"
    if isinstance(value, dict):
        return "; ".join(
            f"{key}={value_text(item)}" for key, item in sorted(value.items())
        )
    if isinstance(value, (list, tuple)):
        return ", ".join(map(value_text, value))
    return str(value)


def report_metadata(folder, report):
    settings, manifest = report.get("settings", {}), report["manifest"]
    environment = report["environment"]
    host = environment.get("host", {})
    metadata = [
        f"Source: {folder}; track: {track(report)}; schema: {report.get('schema')}; configurations: {len(report['summary'])}",
        f"Profile: {value_text(settings.get('profile'))}; case: {value_text(settings.get('case'))}",
        f"Corpus: {len(manifest['files'])} files; {manifest['bytes']} original bytes ({number(manifest['bytes'] / 2**20)} MiB)",
        f"Corpus SHA-256: {manifest['sha256']}",
        "; ".join(
            f"{key}: {value_text(settings.get(key))}"
            for key in ("repeats", "warmups", "threads", "requests")
        ),
    ]
    metadata += [
        f"{key}: {value_text(report.get(key))}"
        for key in ("run_id", "protocol_id", "evidence_head")
    ]
    metadata += [
        f"CPU: {host.get('cpu_model', environment.get('cpu', 'N/A'))}; physical/logical: {host.get('physical_cpus')}/{host.get('logical_cpus')}",
        f"OS: {host.get('os', {}).get('platform', environment.get('platform', 'N/A'))}; Python: {environment.get('python', '').splitlines()[0]}",
        f"Power scheme fingerprint: {host.get('power_scheme', {}).get('sha256', 'N/A')}",
        f"UTC: {environment.get('utc', 'N/A')}; complete environment/source/build details in raw evidence.",
    ]
    return metadata


def report_notes(report, legacy=False):
    notes = []
    if is_legacy(report):
        if not legacy:
            raise ValueError("schema 1 presentation requires --legacy")
        notes.append("LEGACY schema 1: explicitly requested; not v2 protocol evidence.")
    if report.get("settings", {}).get("profile") == "diagnostic":
        notes.append(
            "DIAGNOSTIC profile: exploratory only; never reference evidence "
            "or a qualified score."
        )
    if track(report) == "archive":
        notes += [
            "Archive size is an exact metric with median/min/max in raw data "
            "and no bootstrap CI. CPU N/A means unresolved, not zero.",
            (
                "LEGACY sampled RSS can miss transient peaks; "
                "it is not an OS lifetime peak."
                if is_legacy(report)
                else "OS process-lifetime peak working set is cumulative, NOT a "
                "phase-local RSS peak or a phase memory delta."
            ),
            "Latency p50/p95/p99/p999/max summaries are medians of per-round "
            "statistics, not quantiles or maxima of a pooled request "
            "population. Reset clears codec cache, NOT OS/disk cache; "
            "repeat follows one prime, not a guaranteed cache hit.",
        ]
        flags = tail_flags(report)
        if exploratory(report):
            notes.append(
                "EXPLORATORY latency: insufficient or unestablished tail "
                "coverage, or diagnostic profile; not a qualified score, "
                "population p99 estimate, or SLO evidence."
            )
        notes.append(
            "Tail adequacy flags are COVERAGE ONLY, never SLO qualification: "
            + (
                "; ".join(f"{name}={str(value).lower()}" for name, value in flags)
                or "N/A"
            )
        )
    else:
        notes.append(
            "Lossy quality is separate from byte-exact archives. Rate, RGB "
            "PSNR, FLIP and encode cost are different trade-offs. FLIP has "
            "no universal invisibility threshold; these tables establish "
            "no cross-vendor or actual-GPU quality claim."
        )
    notes.append(
        "Bootstrap 95% intervals, where present, describe rounds conditional "
        "on this host/corpus. No population significance, production SLO, "
        "cross-hardware claim or universal winner. Consult raw observations "
        "and protocol for measured boundaries and limitations."
    )
    return notes


def section(folder, report, legacy=False, output=None, anchor=None):
    folder = Path(folder)
    anchor = folder.name if anchor is None else anchor
    content = [
        f'<section id="{escape(anchor, quote=True)}">' f"<h2>{escape(folder.name)}</h2>"
    ]
    content += [
        f'<p class="metadata">{escape(line)}</p>'
        for line in report_metadata(folder, report)
    ]
    for note in report_notes(report, legacy):
        warning = note.startswith(("EXPLORATORY", "DIAGNOSTIC", "LEGACY"))
        style = "warning" if warning else ""
        content.append(f'<p class="{style}">{escape(note)}</p>')
    links = []
    for filename, label in (
        ("report.json", "Raw observations (original order)"),
        ("report.json.gz", "Existing gzip raw observations"),
        ("report.md", "Markdown"),
    ):
        target = folder / filename
        if filename != "report.json" and not target.is_file():
            continue
        relative = (
            Path(os.path.relpath(target, Path(output).parent)).as_posix()
            if output is not None
            else f"{folder.name}/{filename}"
        )
        href = escape(quote(relative), quote=True)
        links.append(f'<a href="{href}">{label}</a>')
    content.append("<p>" + " · ".join(links) + "</p>")
    headers, rows = report_table(report, legacy)
    content.append('<div class="table-scroll"><table><thead><tr>')
    content += [
        "<th>" + escape(header).replace("\n", "<br>") + "</th>" for header in headers
    ]
    content.append("</tr></thead><tbody>")
    for row in rows:
        content.append("<tr>")
        for index, cell in enumerate(row):
            tag = 'th scope="row"' if index == 0 else "td"
            closing = "th" if index == 0 else "td"
            content.append(
                f"<{tag}>"
                + escape(cell[0])
                + "".join(f"<small>{escape(part)}</small>" for part in cell[1:])
                + f"</{closing}>"
            )
        content.append("</tr>")
    content.append("</tbody></table></div></section>")
    return "\n".join(content)


def html_document(experiments, output=None, legacy=False):
    total = sum(len(report["summary"]) for _, report in experiments)
    intro = (
        f"{len(experiments):,} validated complete reports; "
        f"{total:,} configuration rows."
    )
    links, sections = [], []
    for index, (folder, report) in enumerate(experiments):
        anchor = f"report-{index}"
        links.append(f'<a href="#{anchor}">{escape(folder.name)}</a>')
        sections.append(section(folder, report, legacy, output, anchor))
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VortexPak — validated evidence</title><style>
:root{color-scheme:light dark;font:15px/1.55 system-ui,sans-serif}
body{max-width:1800px;margin:40px auto;padding:0 20px}h1{font-size:2rem}
nav{display:flex;gap:24px;flex-wrap:wrap;margin:24px 0}
a{color:light-dark(#1254a5,#83baff)}
section{border-top:1px solid #8886;padding:30px 0}
.table-scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;border-bottom:1px solid #8884;padding:10px;
white-space:nowrap;font-variant-numeric:tabular-nums}
th:first-child{text-align:left}thead{background:#8882}
tbody tr:nth-child(odd){background:#8881}small{display:block;font-size:11px}
.metadata{font-family:monospace;overflow-wrap:anywhere;font-size:12px}
.warning{font-weight:650;border-left:4px solid #888;padding-left:12px}
</style></head><body><h1>VortexPak — validated evidence</h1>
<p>Alphabetical, neutral configuration rows. No aggregate score or universal
winner. Raw files remain unchanged in their original order; only complete
validated inputs are shown.</p>
""" + (
        f"<p>{intro}</p><nav>"
        + " ".join(links)
        + "</nav>"
        + "\n".join(sections)
        + "</body></html>"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "folders",
        type=Path,
        nargs="+",
        help="report folders or roots to discover",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new HTML path; never overwritten",
    )
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="explicitly allow schema 1 evidence",
    )
    args = ap.parse_args()
    if args.output.exists():
        ap.error(f"output already exists: {args.output}")
    experiments = load_reports(args.folders, legacy=args.legacy)
    document = html_document(experiments, args.output, args.legacy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(document)
    print(args.output)


if __name__ == "__main__":
    main()
