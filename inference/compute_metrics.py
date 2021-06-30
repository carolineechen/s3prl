import os
import tempfile
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--predict_dir", required=True)
parser.add_argument("--truth_dir", required=True)
parser.add_argument("--output_file", required=True)
args = parser.parse_args()

args.predict_dir = Path(args.predict_dir)
args.truth_dir = Path(args.truth_dir)
args.output_file = open(args.output_file, "w")
assert args.predict_dir.is_dir()
assert args.truth_dir.is_dir()

def read_file(path, callback=lambda x: x):
    content = {}
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            filename, value = line.strip().split(maxsplit=1)
            content[filename] = callback(value)
    return content

# PR
if (args.predict_dir / "pr").is_dir():
    predict_file = args.predict_dir / "pr" / "predict.ark"
    truth_file = args.truth_dir / "pr" / "truth.ark"
    assert predict_file.is_file()
    assert truth_file.is_file()

    predict = read_file(predict_file)
    truth = read_file(truth_file)

    filenames = sorted(predict.keys())
    predict_values = [predict[filename] for filename in filenames]
    truth_values = [truth[filename] for filename in filenames]

    from downstream.ctc.metric import wer
    print(f"PR: per {wer(predict_values, truth_values)}", file=args.output_file)

# ASR
if (args.predict_dir / "asr").is_dir():
    predict_file = args.predict_dir / "asr" / "predict.ark"
    truth_file = args.truth_dir / "asr" / "truth.ark"
    assert predict_file.is_file()
    assert truth_file.is_file()

    predict = read_file(predict_file)
    truth = read_file(truth_file)

    filenames = sorted(predict.keys())
    predict_values = [predict[filename] for filename in filenames]
    truth_values = [truth[filename] for filename in filenames]

    from downstream.ctc.metric import wer
    print(f"ASR: wer {wer(predict_values, truth_values)}", file=args.output_file)

# SF
if (args.predict_dir / "sf").is_dir():
    predict_file = args.predict_dir / "sf" / "predict.ark"
    truth_file = args.truth_dir / "sf" / "truth.ark"
    assert predict_file.is_file()
    assert truth_file.is_file()

    predict = read_file(predict_file)
    truth = read_file(truth_file)

    filenames = sorted(predict.keys())
    predict_values = [predict[filename] for filename in filenames]
    truth_values = [truth[filename] for filename in filenames]

    from downstream.ctc.metric import slot_type_f1, slot_value_cer
    f1 = slot_type_f1(predict_values, truth_values)
    cer = slot_value_cer(predict_values, truth_values)
    print(f"SF: slot_type_f1 {f1}, slot_value_cer {cer}", file=args.output_file)

# SV
if (args.predict_dir / "sv").is_dir():
    predict_file = args.predict_dir / "sv" / "predict.txt"
    truth_file = args.truth_dir / "sv" / "truth.txt"
    assert predict_file.is_file()
    assert truth_file.is_file()

    predict = read_file(predict_file, lambda x: float(x))
    truth = read_file(truth_file, lambda x: float(x))

    pairnames = sorted(predict.keys())
    predict_scores = np.array([predict[name] for name in pairnames])
    truth_scores = np.array([truth[name] for name in pairnames])

    from downstream.sv_voxceleb1.utils import EER
    eer, *other = EER(truth_scores, predict_scores)
    print(f"SV: eer {eer}", file=args.output_file)

# SD
if (args.predict_dir / "sd").is_dir():
    with tempfile.TemporaryDirectory() as scoring_dir:
        os.system(f"bash ./inference/truth/sd/score.sh {scoring_dir} {args.predict_dir} | tail -n 1 | awk '{{print $4}}' > {scoring_dir}/result.log")
        with open(f"{scoring_dir}/result.log", "r") as result:
            der = result.readline().strip()
    print(f"SD: der {der}", file=args.output_file)

args.output_file.close()
