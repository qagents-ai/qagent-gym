"""Optional faster-whisper process; uses only an explicitly supplied local model."""

import argparse
import json
import sys


def main():
    from faster_whisper import WhisperModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    model = WhisperModel(
        args.model, device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True
    )
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request["language"] not in {"en", "el"}:
                raise ValueError("unsupported language")
            segments, _ = model.transcribe(
                request["path"],
                language=request["language"],
                beam_size=1,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            print(json.dumps({"text": " ".join(s.text for s in segments).strip()}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
