"""FFmpeg helpers."""
import json
import subprocess
import numpy as np

CREATE_NO_WINDOW = 0x08000000 if hasattr(subprocess, "STARTUPINFO") else 0


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       creationflags=CREATE_NO_WINDOW)
    if p.returncode != 0:
        raise RuntimeError("Lệnh thất bại: " + " ".join(map(str, cmd)) + "\n"
                           + p.stderr.decode("utf-8", "ignore")[-3000:])
    return p


def probe(path):
    p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
             "-show_format", str(path)])
    info = json.loads(p.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise RuntimeError("File không có luồng video")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "25/1").split("/")
    fps = float(num) / float(den) if float(den) else 25.0
    if not (1 <= fps <= 240):
        num, den = v.get("r_frame_rate", "25/1").split("/")
        fps = float(num) / float(den)
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) in (90, 270):
        w, h = h, w
    return {
        "width": w, "height": h, "fps": fps,
        "fps_str": v.get("avg_frame_rate") if v.get("avg_frame_rate", "0/0") != "0/0" else v.get("r_frame_rate"),
        "duration": float(info["format"].get("duration", 0) or 0),
        "has_audio": a is not None,
    }


def load_audio(path, sr=44100, channels=2):
    """Decode any media file to float32 numpy array of shape (channels, n)."""
    p = run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-f", "f32le",
             "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "-"])
    data = np.frombuffer(p.stdout, dtype=np.float32)
    return data.reshape(-1, channels).T.copy()


def save_wav(path, audio, sr=44100):
    import soundfile as sf
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        a = a.T
    sf.write(str(path), a, sr, subtype="FLOAT")


def read_frames(path, fps, width, height, scale_width=None, sample_fps=None):
    """Yield (index, frame BGR uint8) decoded by ffmpeg at constant frame rate."""
    vf = []
    out_fps = sample_fps or fps
    vf.append(f"fps={out_fps}")
    w, h = width, height
    if scale_width and scale_width < width:
        w = int(scale_width) // 2 * 2
        h = int(round(height * w / width)) // 2 * 2
        vf.append(f"scale={w}:{h}")
    else:
        vf.append(f"scale={w}:{h}")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf", ",".join(vf),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=CREATE_NO_WINDOW, bufsize=w * h * 3 * 4)
    size = w * h * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()
