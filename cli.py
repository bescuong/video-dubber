"""Command-line usage:  python cli.py video.mp4 --target vi [--source zh] [--whisper medium]"""
import argparse

from dubber.config import Settings, LANGUAGES, WHISPER_MODELS
from dubber import pipeline


def main():
    ap = argparse.ArgumentParser(description="Lồng tiếng + thay phụ đề video")
    ap.add_argument("video")
    ap.add_argument("--target", default="vi", choices=list(LANGUAGES))
    ap.add_argument("--source", default="auto", choices=["auto"] + list(LANGUAGES))
    ap.add_argument("--whisper", default="medium", choices=WHISPER_MODELS)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-blur", action="store_true")
    ap.add_argument("--no-subs", action="store_true")
    ap.add_argument("--region-top", type=float, default=0.55)
    ap.add_argument("--reuse", action="store_true", help="dùng lại bản dịch đã lưu (project.json)")
    ap.add_argument("-o", "--output")
    a = ap.parse_args()

    s = Settings(source_lang=a.source, target_lang=a.target, whisper_model=a.whisper,
                 use_gpu=not a.cpu, blur_old_subs=not a.no_blur, burn_new_subs=not a.no_subs,
                 sub_region_top=a.region_top)
    proj = pipeline.load_project(a.video) if a.reuse else None
    if proj:
        segs = proj["segments"]
    else:
        segs, _ = pipeline.analyze(a.video, s)
    pipeline.render(a.video, s, segs, output=a.output)


if __name__ == "__main__":
    main()
