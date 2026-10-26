"""Fetch the complete official Silesia corpus; never place corpora in the repo."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request
import zipfile

URL = "https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"
OFFICIAL = {
    "dickens": (10192446, "88334708559f6db57d79096bc0aca07e"),
    "mozilla": (51220480, "c7789a2097f1ff944b0c737430a339b3"),
    "mr": (9970564, "38e623e3093b7bf2003ca4b1bbc19927"),
    "nci": (33553445, "31f85bc8706f3c921104e7c169e2e2e1"),
    "ooffice": (6152192, "573c4ae915e36631d8f2dcffb9b9b66d"),
    "osdb": (10085684, "e734b0c48e6a982adfb5802da3032ecd"),
    "reymont": (6627202, "d8f54d78105079775f32d76dc55fc671"),
    "samba": (21606400, "154eaea7ea70e89f6339ff0abf4112ca"),
    "sao": (7251944, "79e95a22e18cd82b7e42bf91b380d30b"),
    "webster": (41458703, "474931ad907ac27bf962c75ded46c069"),
    "xml": (5345280, "9b09c0c80104adb8aae910b7d7db003e"),
    "x-ray": (8474240, "9baec32ad14ec3eff487d254382cb91c"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        ap.error("output exists; will not overwrite")
    repo = Path(__file__).resolve().parents[1]
    if repo == out or repo in out.parents:
        ap.error("corpus must stay outside repository")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        temp = Path(temp)
        archive = temp / "silesia.zip"
        with urllib.request.urlopen(URL, timeout=120) as response, archive.open("xb") as target:
            total = 0
            while data := response.read(1024 * 1024):
                total += len(data)
                if total > 100 * 1024 * 1024:
                    raise ValueError("download exceeds expected archive size bound")
                target.write(data)
        unpacked = temp / "corpus"
        unpacked.mkdir()
        with zipfile.ZipFile(archive) as z:
            if sorted(z.namelist()) != sorted(OFFICIAL):
                raise ValueError("archive members differ from official 12-file corpus")
            for name, (size, expected_md5) in OFFICIAL.items():
                if z.getinfo(name).file_size != size:
                    raise ValueError("unexpected member size: " + name)
                data = z.read(name)
                if hashlib.md5(data).hexdigest() != expected_md5:
                    raise ValueError("official legacy identity checksum differs: " + name)
                (unpacked / name).write_bytes(data)
        unpacked.rename(out)
    print(out)


if __name__ == "__main__":
    main()
