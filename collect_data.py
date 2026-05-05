import argparse
import time
import os
import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 22050


def record_clip(duration: int) -> np.ndarray:
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def main():
    parser = argparse.ArgumentParser(description="Record labeled audio samples.")
    parser.add_argument(
        "--label",
        required=True,
        choices=["normal", "abnormal"],
        help="Label for this batch of recordings.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration of each clip in seconds (default: 2).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Number of clips to record (default: 30).",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.5,
        help="Pause between clips in seconds (default: 0.5).",
    )
    args = parser.parse_args()

    out_dir = os.path.join("data", args.label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Recording {args.count} clips of {args.duration}s each → data/{args.label}/")
    print("Press Ctrl+C to stop early.\n")

    for i in range(1, args.count + 1):
        print(f"  [{i}/{args.count}] Recording...", end="", flush=True)
        audio = record_clip(args.duration)
        timestamp = int(time.time() * 1000)
        filename = os.path.join(out_dir, f"sample_{timestamp}.wav")
        sf.write(filename, audio, SAMPLE_RATE)
        print(f" saved → {filename}")
        if i < args.count:
            time.sleep(args.gap)

    print(f"\nDone. {args.count} clips saved to data/{args.label}/")


if __name__ == "__main__":
    main()
