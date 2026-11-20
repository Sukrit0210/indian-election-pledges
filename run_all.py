"""Run the numbered scripts in order."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
STEPS = [
    "01_build_quasi_sentences.py",
    "02_encode_statements.py",
    "03_detect_pledges.py",
    "04_classify_topics.py",
    "05_forecast_topics.py",
    "06_retrieve_pledges.py",
    "07_train_cross_encoder.py",
    "08_compare_pledges.py",
    "09_evaluate.py",
    "10_make_plots.py",
]

def project_python():
    if sys.version_info[:2] != (3, 14):
        raise SystemExit("Select Python 3.14 in VS Code, then run this file again.")

    env = ROOT / ".venv314"
    python = env / "Scripts" / "python.exe"
    requirements = ROOT / "requirements.txt"
    marker = env / ".setup-complete"
    key = hashlib.sha256(requirements.read_bytes()).hexdigest()

    if not python.is_file():
        print("Creating the pledge project's Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(env)

    if not marker.is_file() or marker.read_text(encoding="ascii").strip() != key:
        pip = [str(python), "-m", "pip"]
        subprocess.run(
            [
                *pip,
                "install",
                "torch==2.10.0+cu126",
                "--index-url",
                "https://download.pytorch.org/whl/cu126",
            ],
            check=True,
        )
        subprocess.run([*pip, "install", "-r", str(requirements)], check=True)
        subprocess.run(
            [
                str(python),
                "-c",
                "import pysbd, torch; print(torch.cuda.get_device_name(0)); "
                "print('GPU test:', torch.ones(3, device='cuda').sum().item())",
            ],
            check=True,
        )
        marker.write_text(key, encoding="ascii")

    return str(python)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passages",
        type=Path,
        default=ROOT / "data" / "processed" / "master_passages.csv",
    )
    parser.add_argument("--start", type=int, default=1, choices=range(1, 11))
    parser.add_argument("--stop", type=int, default=10, choices=range(1, 11))
    args = parser.parse_args()

    if args.start > args.stop:
        parser.error("--start must be less than or equal to --stop")

    python = project_python()

    for number in range(args.start, args.stop + 1):
        script = STEPS[number - 1]
        command = [python, str(ROOT / "src" / script)]

        if number == 1:
            command.extend(["--passages", str(args.passages.resolve())])

        print(f"Running {script}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
