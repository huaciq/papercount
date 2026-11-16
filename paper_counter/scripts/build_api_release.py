"""Build a wheel and a shareable ZIP without datasets or historical outputs."""
from pathlib import Path
import hashlib
import os
import zipfile

from setuptools.build_meta import build_wheel


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    # Keep the distributed configurations identical to the current explicit files.
    for name in ("default.yaml", "paper_layers_v4.yaml", "regions.yaml", "tracks_preprocessed.yaml", "corrugated.yaml"):
        config_path = root / "configs" / name
        if config_path.exists():
            (root / "paper_counter" / "presets" / name).write_bytes(config_path.read_bytes())
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    wheel = dist / build_wheel(str(dist))
    archive = dist / "paper_counter_api_0.1.0.zip"
    source_files = sorted((root / "paper_counter").rglob("*.py"))
    source_files += sorted((root / "paper_counter" / "presets").glob("*.yaml"))
    source_files += sorted((root / "paper_counter" / "recipes").rglob("*.yaml"))
    source_files += [root / "pyproject.toml", root / "API_README.md", root / "requirements.txt",
                     root / "examples" / "example.py", Path(__file__).resolve()]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(wheel, wheel.name)
        bundle.write(root / "API_README.md", "README.md")
        bundle.write(root / "examples" / "example.py", "example.py")
        bundle.writestr("SHA256SUMS.txt", hashlib.sha256(wheel.read_bytes()).hexdigest() + "  " + wheel.name + "\n")
        for path in source_files:
            bundle.write(path, "source/" + path.relative_to(root).as_posix())
    print(archive)


if __name__ == "__main__":
    main()
