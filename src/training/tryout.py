from pathlib import Path
import shutil


def main():
    prefix = "A-SSS-X"
    name = "A-SSS-X-999-hallo.2"
    name1 = name[len(prefix) + 1 :]
    print("--", prefix, name, name1)


def rename():
    print("-- rename")
    indir = Path.home() / "tmp" / "sumosim" / "results1"
    outdir = Path.home() / "tmp" / "sumosim" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    for f in indir.iterdir():
        if f.name.startswith("P"):
            stem = f.name[1:]
            new_name = f"Q0-{stem}"
            target = outdir / new_name
            print(f"{str(f.name):50s} ---> {target}")
            if not target.exists():
                shutil.copy(f, target)
                print(f"copied to {target}")
