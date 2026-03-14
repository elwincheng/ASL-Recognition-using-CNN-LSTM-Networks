"""
collect_youtube_data.py

Downloads ASL tutorial videos from YouTube for each of the 10 target words.
Uses yt-dlp for search (metadata only) and pytubefix for actual downloads.
Applies MediaPipe hand-detection as a quality filter.

Usage:
    python collect_youtube_data.py

Outputs (appended to):
    data/youtube_frames.npy   : (M, 30, 224, 224, 3) float32 frames
    data/youtube_labels.npy   : (M,) int64 labels
    data/youtube_done.json    : checkpoint of downloaded video IDs
"""

import os, json, subprocess, tempfile, time, random
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
from pytubefix import YouTube

# ── Config ───────────────────────────────────────────────────
WORDS = ["drink", "computer", "before", "go", "who",
         "cousin", "help", "thin", "cool", "tall"]

TARGET_PER_WORD   = 100   # collect until we have this many good clips per word
MAX_SEARCH_EACH   = 60    # yt-dlp ytsearch<N> per query
MIN_DURATION_S    = 2
MAX_DURATION_S    = 90
MIN_HAND_FRAMES   = 8     # out of 30: discard clips with very few hands detected
NUM_FRAMES        = 30
IMG_SIZE          = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

YTDLP    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".venv/bin/yt-dlp")
OUT_DIR     = "data"
DONE_FILE   = "data/youtube_done.json"
FRAMES_FILE = "data/youtube_frames.npy"
LABELS_FILE = "data/youtube_labels.npy"

# Multiple search queries per word to maximise diversity
QUERIES = {
    "drink":    ["ASL drink sign language", "how to sign drink in ASL",
                 "sign language drink tutorial", "ASL drink cup"],
    "computer": ["ASL computer sign language", "how to sign computer ASL",
                 "sign language computer tutorial", "ASL computer technology"],
    "before":   ["ASL before sign language", "how to sign before ASL",
                 "sign language before tutorial", "ASL time signs before"],
    "go":       ["ASL go sign language", "how to sign go in ASL",
                 "sign language go tutorial", "ASL go leave"],
    "who":      ["ASL who sign language", "how to sign who ASL question",
                 "sign language who wh-question", "ASL question words who"],
    "cousin":   ["ASL cousin sign language", "how to sign cousin ASL",
                 "sign language cousin tutorial", "ASL family signs cousin"],
    "help":     ["ASL help sign language", "how to sign help ASL",
                 "sign language help tutorial", "ASL help assist"],
    "thin":     ["ASL thin sign language", "how to sign thin ASL",
                 "sign language thin skinny", "ASL thin slim tutorial"],
    "cool":     ["ASL cool sign language", "how to sign cool ASL",
                 "sign language cool neat", "ASL cool awesome"],
    "tall":     ["ASL tall sign language", "how to sign tall ASL",
                 "sign language tall height", "ASL tall short height"],
}

label_map = {w: i for i, w in enumerate(WORDS)}

os.makedirs(OUT_DIR, exist_ok=True)


# ── MediaPipe setup ──────────────────────────────────────────
MODEL_PATH = "./hand_landmarker.task"
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
        )
        _detector = mp_vision.HandLandmarker.create_from_options(opts)
    return _detector


def count_hand_frames(frames_norm):
    """Count how many of the 30 frames have at least one hand detected."""
    det = get_detector()
    count = 0
    for f in frames_norm:
        img_u8 = np.clip(f * STD + MEAN, 0, 1)
        img_u8 = (img_u8 * 255).astype(np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_u8)
        res    = det.detect(mp_img)
        if res.hand_landmarks:
            count += 1
    return count


# ── Frame extraction ─────────────────────────────────────────
def extract_frames(video_path):
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None
    if total >= NUM_FRAMES:
        indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    else:
        indices = [i % total for i in range(NUM_FRAMES)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frame = (frame - MEAN) / STD
            frames.append(frame)
        else:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
    cap.release()
    return np.stack(frames).astype(np.float32)


# ── Search with yt-dlp (metadata only, no download) ──────────
def search_yt(query, n=MAX_SEARCH_EACH):
    """Return list of {id, title, duration, url} dicts from YouTube search."""
    cmd = [
        YTDLP, f"ytsearch{n}:{query}",
        "--dump-json", "--no-download", "--flat-playlist",
        "--quiet",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        results = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line: continue
            try:
                info = json.loads(line)
                dur  = info.get("duration") or 0
                if MIN_DURATION_S <= dur <= MAX_DURATION_S:
                    vid_id = info.get("id", "")
                    results.append({
                        "id":       vid_id,
                        "title":    info.get("title", ""),
                        "duration": dur,
                        "url":      f"https://www.youtube.com/watch?v={vid_id}",
                    })
            except Exception:
                pass
        return results
    except Exception as e:
        print(f"    [search error] {e}")
        return []


# ── Download with pytubefix ───────────────────────────────────
def download_video(url, out_path):
    """Download a single YouTube video using pytubefix. Returns True on success."""
    try:
        yt     = YouTube(url)
        stream = (yt.streams
                    .filter(progressive=True, file_extension="mp4")
                    .order_by("resolution")
                    .first())
        if stream is None:
            return False
        out_path = Path(out_path)
        stream.download(output_path=str(out_path.parent),
                        filename=out_path.name)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


# ── Checkpoint ───────────────────────────────────────────────
def load_done():
    if os.path.exists(DONE_FILE):
        return json.load(open(DONE_FILE))
    return {}   # {word: [vid_id, ...]}

def save_done(done):
    with open(DONE_FILE, "w") as f:
        json.dump(done, f, indent=2)


def load_accumulated():
    frames_list, labels_list = [], []
    if os.path.exists(FRAMES_FILE):
        frames_list = [np.load(FRAMES_FILE)]
    if os.path.exists(LABELS_FILE):
        labels_list = [np.load(LABELS_FILE)]
    return frames_list, labels_list

def save_accumulated(frames_list, labels_list):
    if frames_list:
        np.save(FRAMES_FILE, np.concatenate(frames_list, axis=0))
    if labels_list:
        np.save(LABELS_FILE, np.concatenate(labels_list, axis=0))


# ── Main ─────────────────────────────────────────────────────
def main():
    done = load_done()
    frames_list, labels_list = load_accumulated()

    # Count already-accumulated per word
    already = {w: 0 for w in WORDS}
    if labels_list:
        lbl_arr = np.concatenate(labels_list)
        for i, w in enumerate(WORDS):
            already[w] = int((lbl_arr == i).sum())
    for w in WORDS:
        done.setdefault(w, [])

    print("Current YouTube clip counts per word:")
    for w in WORDS:
        print(f"  {w:12s}: {already[w]} accumulated, {len(done[w])} IDs processed")

    with tempfile.TemporaryDirectory(prefix="asl_dl_") as tmp_dir:
        for word in WORDS:
            label_idx = label_map[word]
            need = TARGET_PER_WORD - already[word]
            if need <= 0:
                print(f"\n[{word}] Already have {already[word]} >= {TARGET_PER_WORD}. Skipping.")
                continue

            print(f"\n{'='*60}")
            print(f"[{word}] Need {need} more clips (have {already[word]})")
            print(f"{'='*60}")
            collected = 0

            for query in QUERIES[word]:
                if collected >= need:
                    break
                print(f"  Searching: '{query}'")
                candidates = search_yt(query)
                # Deduplicate by video id
                seen_ids = set()
                unique_cands = []
                for c in candidates:
                    if c["id"] not in seen_ids:
                        seen_ids.add(c["id"])
                        unique_cands.append(c)
                random.shuffle(unique_cands)
                print(f"  Found {len(unique_cands)} unique candidates with valid duration")

                for cand in unique_cands:
                    if collected >= need:
                        break
                    vid_id = cand["id"]
                    if vid_id in done[word] or not vid_id:
                        continue
                    done[word].append(vid_id)

                    vid_path = Path(tmp_dir) / f"{vid_id}.mp4"
                    ok = download_video(cand["url"], vid_path)
                    if not ok:
                        print(f"    [{vid_id}] download failed, skipping")
                        continue

                    frames = extract_frames(vid_path)
                    try: vid_path.unlink()
                    except: pass

                    if frames is None:
                        print(f"    [{vid_id}] frame extraction failed, skipping")
                        continue

                    # MediaPipe quality filter
                    hand_count = count_hand_frames(frames)

                    if hand_count < MIN_HAND_FRAMES:
                        print(f"    [{vid_id}] only {hand_count}/30 frames with hands, skip")
                        continue

                    # Accept this clip
                    frames_list.append(frames[np.newaxis])  # (1, 30, 224, 224, 3)
                    labels_list.append(np.array([label_idx], dtype=np.int64))
                    already[word] += 1
                    collected += 1
                    print(f"    [{vid_id}] OK hands={hand_count}/30  "
                          f"word={word}  total={already[word]}")

                    # Save checkpoint every 10 accepted clips
                    if collected % 10 == 0:
                        save_accumulated(frames_list, labels_list)
                        save_done(done)

                time.sleep(1)  # brief pause between queries

            print(f"  [{word}] Collected {collected} new clips (total {already[word]})")
            save_accumulated(frames_list, labels_list)
            save_done(done)

    # Final save
    save_accumulated(frames_list, labels_list)
    save_done(done)

    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    if frames_list:
        total_labels = np.concatenate(labels_list)
        for i, w in enumerate(WORDS):
            n = int((total_labels == i).sum())
            print(f"  {w:12s}: {n} clips")
        print(f"  TOTAL: {len(total_labels)} clips")
    print("Done!")


if __name__ == "__main__":
    main()
