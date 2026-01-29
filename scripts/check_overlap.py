from collections import Counter
import json

def read_train(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue
            g = parts[0].strip()
            l = parts[1].strip().lower()
            d[g] = l
    return d

def read_test(path):
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            g = line.split(",", 1)[0].strip()
            s.add(g)
    return s

if __name__ == "__main__":
    train_path = "data/train.txt"
    test_path = "data/test_without_label.txt"
    train = read_train(train_path)
    test = read_test(test_path)
    train_set = set(train.keys())
    overlap = train_set & test
    out = {
        "train_count": len(train_set),
        "test_count": len(test),
        "overlap_count": len(overlap),
        "overlap_examples": list(overlap)[:20],
        "label_distribution": dict(Counter(train.values()))
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

