"""Render validated, canonical benchmark tables to a high-resolution PNG."""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import report_benchmarks as reports


def make_image(experiments, legacy=False):
    """Lay out already validated reports in memory, measuring before drawing."""
    experiments = list(experiments)
    if not experiments:
        raise ValueError("no validated reports to render")
    fonts = Path("C:/Windows/Fonts")
    regular, bold = fonts / "segoeui.ttf", fonts / "segoeuib.ttf"
    if not regular.is_file() or not bold.is_file():
        regular, bold = "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"

    def font(size, heavy=False):
        return ImageFont.truetype(str(bold if heavy else regular), size)

    title, heading = font(50, True), font(34, True)
    body, label, small = font(25), font(23, True), font(21)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    margin, padding = 90, 18
    ink, muted, rule = "#222222", "#555555", "#cccccc"
    operations = []

    def text_width(value, face):
        box = measure.textbbox((0, 0), value, font=face, anchor="lt")
        return box[2] - min(0, box[0])

    def line_height(face):
        return face.size + 12

    def wrap(value, face, width):
        """Pixel wrapping also splits long hashes and unbroken config names."""
        result = []
        for original in str(value).splitlines() or [""]:
            remaining = original
            while remaining and text_width(remaining, face) > width:
                lo, hi = 1, len(remaining)
                while lo < hi:
                    middle = (lo + hi + 1) // 2
                    if text_width(remaining[:middle], face) <= width:
                        lo = middle
                    else:
                        hi = middle - 1
                end = remaining.rfind(" ", 0, lo + 1)
                if end <= 0:
                    end = lo
                result.append(remaining[:end])
                remaining = remaining[end:].lstrip()
            result.append(remaining)
        return result

    def column_widths(headers, rows):
        widths = []
        for index, header in enumerate(headers):
            needed = max(text_width(part, label) for part in header.splitlines())
            for row in rows:
                for part_index, part in enumerate(row[index]):
                    face = body if part_index == 0 else small
                    needed = max(needed, text_width(part, face))
            cap = 520 if index == 0 else 360
            widths.append(int(max(180, min(cap, needed + 2 * padding))) + 1)
        return widths

    tables = []
    for folder, report in experiments:
        headers, rows = reports.report_table(report, legacy=legacy)
        widths = column_widths(headers, rows)
        tables.append((Path(folder), report, headers, rows, widths))
    content_width = max([2600] + [sum(table[4]) for table in tables])
    canvas_width = content_width + 2 * margin
    y, rows_drawn = margin, 0

    def text(x, yy, value, face, color=ink):
        operations.append(("text", (x, yy), value, face, color))

    def paragraph(value, face=small, color=muted, gap=10):
        nonlocal y
        for line in wrap(value, face, content_width):
            text(margin, y, line, face, color)
            y += line_height(face)
        y += gap

    def cell_lines(parts, width, header=False):
        result = []
        for index, part in enumerate(parts):
            face = label if header else body if index == 0 else small
            result.extend(
                (line, face) for line in wrap(part, face, width - 2 * padding)
            )
        return result

    def table_row(cells, widths, fill, header=False):
        nonlocal y
        laid_out = [
            cell_lines(cell, width, header) for cell, width in zip(cells, widths)
        ]
        height = (
            max(sum(line_height(face) for _, face in cell) for cell in laid_out)
            + 2 * padding
        )
        operations.append(
            (
                "rectangle",
                (margin, y, margin + sum(widths), y + height),
                fill,
            )
        )
        x = margin
        for cell, width in zip(laid_out, widths):
            yy = y + padding
            for value, face in cell:
                text(x + padding, yy, value, face)
                yy += line_height(face)
            x += width
        y += height + 1

    total = sum(len(report["summary"]) for _, report in experiments)
    paragraph("VortexPak — validated benchmark evidence", title, ink, gap=20)
    paragraph(
        f"{len(experiments):,} validated complete reports; "
        f"{total:,} configuration rows. Counts, corpora and settings come "
        "only from these inputs.",
        body,
    )
    paragraph(
        "Alphabetical neutral rows; each configuration appears once per "
        "experiment. No aggregate score or universal winner. Raw evidence "
        "is unchanged in its original order. Incomplete discovered children "
        "are excluded; explicit incomplete inputs are rejected."
    )
    y += 24
    for folder, report, headers, rows, widths in tables:
        paragraph(folder.name, heading, ink, gap=15)
        for value in reports.report_metadata(folder, report):
            paragraph(value, gap=5)
        paragraph(
            f"Raw observations (original order): {folder / 'report.json'}",
            gap=12,
        )
        for note in reports.report_notes(report, legacy=legacy):
            warning = note.startswith(("EXPLORATORY", "DIAGNOSTIC", "LEGACY"))
            paragraph(note, label if warning else small, ink if warning else muted)
        y += 12
        table_row([(header,) for header in headers], widths, "#e6e6e6", True)
        for index, row in enumerate(rows):
            table_row(row, widths, "#f5f5f5" if index % 2 == 0 else "white")
            rows_drawn += 1
        y += 24
        operations.append(
            (
                "line",
                (margin, y, margin + content_width, y),
                rule,
            )
        )
        y += 34
    if rows_drawn != total:
        raise AssertionError("canonical configuration row count differs")
    canvas = Image.new("RGB", (canvas_width, y + margin), "white")
    draw = ImageDraw.Draw(canvas)
    for operation in operations:
        kind, position, *details = operation
        if kind == "text":
            value, face, color = details
            box = draw.textbbox(position, value, font=face, anchor="lt")
            if not (
                0 <= box[0] <= box[2] <= canvas.width
                and 0 <= box[1] <= box[3] <= canvas.height
            ):
                raise AssertionError(f"text would clip: {value!r}")
            draw.text(position, value, font=face, fill=color, anchor="lt")
        elif kind == "rectangle":
            draw.rectangle(position, fill=details[0])
        else:
            draw.line(position, fill=details[0], width=2)
    return canvas


def render(root, output, legacy=False):
    """Validate a root (or explicit report folder) and save to a NEW path."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    experiments = reports.load_reports([root], legacy=legacy)
    image = make_image(experiments, legacy=legacy)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        image.save(stream, "PNG", dpi=(200, 200), optimize=True)
    return image.size


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("benchmarks/reports"),
        help="discover immediate report children, or use an explicit report folder",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new PNG path; never overwritten",
    )
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="explicitly allow schema 1 evidence",
    )
    args = ap.parse_args()
    print(args.output, render(args.root, args.output, legacy=args.legacy))


if __name__ == "__main__":
    main()
