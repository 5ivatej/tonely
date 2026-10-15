"""Command line: ``tonely serve`` (web app) or ``tonely analyze FILE``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import Config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tonely", description="Multimodal audio sentiment & emotion analysis.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--profile", default="full", choices=["full", "lite", "acoustic"])
    s.add_argument("--no-browser", action="store_true")

    a = sub.add_parser("analyze", help="analyze a file and print/save JSON")
    a.add_argument("file")
    a.add_argument("-o", "--out", help="write JSON here instead of stdout")
    a.add_argument("--profile", default="full", choices=["full", "lite", "acoustic"])
    a.add_argument("--whisper", help="override Whisper size (tiny, base, small, medium, large-v3)")
    a.add_argument("--device", help="cpu, mps or cuda (default: auto)")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3", "transformers", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd == "serve":
        from .server import run

        run(args.host, args.port, Config.profile(args.profile), open_browser=not args.no_browser)
        return 0

    from .pipeline import analyze

    cfg = Config.profile(args.profile)
    if args.whisper:
        cfg = cfg.with_(whisper_model=args.whisper)
    if args.device:
        cfg = cfg.with_(device=args.device)

    def progress(stage, frac, label):
        print(f"\r[{frac * 100:5.1f}%] {label:<40}", end="", file=sys.stderr, flush=True)

    result = analyze(args.file, cfg, progress)
    print(file=sys.stderr)
    text = json.dumps(result, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        s = result["fused"]["summary"]
        print(f"{result['file']['name']}: {s['dominant_emotion']} · {s['sentiment']} · arc: {s['arc']['shape']}", file=sys.stderr)
        for r in result["analyzers"]:
            print(f"  {r['label']:<36} {r['status']:<14} {r['seconds']}s", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
