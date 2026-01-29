#!/usr/bin/env python3
import re
import json
import random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    model_dir = Path('models/t5-small')
    print('using model dir', model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), local_files_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train = Path('data/train.txt')
    data_dir = Path('data/dataset')
    out_dir = Path('data/dataset_neutral_augmented'); out_dir.mkdir(exist_ok=True)
    out_index = Path('data/train_neutral_augmented.txt')

    lines = [l.strip() for l in train.read_text(encoding='utf-8', errors='ignore').splitlines() if l.strip()]
    if lines and lines[0].lower().startswith('guid'):
        lines = lines[1:]

    def strength(txt):
        toks = re.findall(r"\\w+", txt.lower())
        pos = sum(1 for w in toks if w in {'good','great','nice','love','happy','amazing','awesome'})
        neg = sum(1 for w in toks if w in {'bad','sad','hate','awful','terrible'})
        ex = len(re.findall(r'!+', txt))
        raw = (pos - neg) + 0.5*ex
        mag = abs(raw / max(1, len(toks)))
        if mag < 0.15:
            return 'weak'
        if mag < 0.4:
            return 'medium'
        return 'strong'

    candidates = []
    for ln in lines:
        parts = ln.split(',')
        if len(parts) < 2:
            continue
        guid, tag = parts[0].strip(), parts[1].strip()
        if tag != 'neutral':
            continue
        p = data_dir / f"{guid}.txt"
        if not p.exists():
            continue
        txt = p.read_text(encoding='utf-8', errors='ignore').strip()
        st = strength(txt)
        if st in ('weak', 'medium'):
            candidates.append((guid, tag, txt))

    print('neutral candidates found', len(candidates))
    sample = random.sample(candidates, min(500, len(candidates)))

    out_lines = []
    for (guid, tag, txt) in sample:
        prompt = f'paraphrase: {txt} </s>'
        inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        out_ids = model.generate(**inputs, max_length=256, num_beams=4, num_return_sequences=1)
        para = tok.decode(out_ids[0], skip_special_tokens=True)
        new_guid = f"{guid}_neutral_bt"
        (out_dir / f"{new_guid}.txt").write_text(para, encoding='utf-8')
        out_lines.append(f"{new_guid},{tag}\n")

    out_index.write_text('guid,tag\n' + ''.join(out_lines), encoding='utf-8')
    print('wrote', len(out_lines), 'neutral backtranslations to', out_dir, 'and index', out_index)

if __name__ == '__main__':
    main()


