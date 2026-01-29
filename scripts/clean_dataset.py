import os
import json

TRAIN_PATH = "data/train.txt"
DATA_DIR = "data"
OUT_PATH = "data/clean_train.txt"
REPORT_PATH = "data/clean_report.json"

def check_text(guid):
    candidates = [
        os.path.join(DATA_DIR, "dataset", f"{guid}.txt"),
        os.path.join(DATA_DIR, f"{guid}.txt"),
        os.path.join(DATA_DIR, "dataset", f"{guid}.text"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                if os.path.getsize(p) == 0:
                    return False, "empty"
                # try read small chunk with common encodings
                with open(p, "rb") as f:
                    raw = f.read(2048)
                for enc in ("utf-8", "utf-8-sig", "latin-1", "gbk"):
                    try:
                        txt = raw.decode(enc)
                        if txt.strip():
                            return True, p
                    except Exception:
                        continue
            except Exception:
                return False, "error"
    return False, "missing"

def check_image(guid):
    candidates = [
        os.path.join(DATA_DIR, "dataset", f"{guid}.jpg"),
        os.path.join(DATA_DIR, f"{guid}.jpg"),
        os.path.join(DATA_DIR, "dataset", f"{guid}.png"),
        os.path.join(DATA_DIR, f"{guid}.png"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                if os.path.getsize(p) == 0:
                    return False, "empty"
                return True, p
            except Exception:
                return False, "error"
    return False, "missing"

def main():
    if not os.path.exists(TRAIN_PATH):
        print("Train file not found:", TRAIN_PATH)
        return

    kept = []
    removed = []
    total = 0

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) < 2:
                removed.append((line, "bad_format"))
                continue
            guid = parts[0].strip()
            label = parts[1].strip()
            t_ok, t_info = check_text(guid)
            i_ok, i_info = check_image(guid)
            if t_ok and i_ok:
                kept.append((guid, label))
            else:
                removed.append((guid, label, {"text": t_info, "image": i_info}))

    # write cleaned file
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("guid,tag\n")
        for g, l in kept:
            f.write(f"{g},{l}\n")

    report = {
        "total_train_rows": total,
        "kept": len(kept),
        "removed": len(removed),
        "removed_examples": removed[:50]
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Clean complete. Kept:", len(kept), "Removed:", len(removed))
    print("Cleaned file:", OUT_PATH)
    print("Report:", REPORT_PATH)

if __name__ == "__main__":
    main()


