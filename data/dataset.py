import os
import random
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset
import re
import html
import unicodedata

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from torchvision import transforms as T
except Exception:
    T = None


class MultimodalDataset(Dataset):
    """
    Minimal, robust MultimodalDataset implementation to match project's expected batch keys.

    - Supports train_list: List[Tuple[guid, label_str]]
    - Supports test_list: List[guid]
    - Returns dict with keys: input_ids, attention_mask, image, label (if is_train), guid
    - Tokenization: tries to use HuggingFace tokenizer if available; otherwise simple whitespace tokenizer.
    - Image: tries to load via PIL + torchvision transforms; otherwise returns zero tensor.
    """

    def __init__(self, data_list: List[Union[Tuple[str, str], str]], config, is_train: bool = True):
        self.data_list = data_list
        self.config = config
        self.is_train = is_train
        self.data_dir = getattr(config, "data_dir", "data")
        self.max_len = getattr(config, "max_seq_length", 128)
        self.image_size = getattr(config, "image_size", 224)

        # Try HF tokenizer if available and config points to a model name
        self.tokenizer = None
        try:
            from transformers import AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(getattr(config, "text_model_name", "bert-base-uncased"))
            except Exception:
                self.tokenizer = None
        except Exception:
            self.tokenizer = None

        # Simple vocab for fallback tokenizer
        self._word2id = {"[PAD]": 0, "[UNK]": 1}
        self._next_id = 2

        # Image transform
        # Image transform (train / eval)
        if T is not None:
            # base transforms for eval (can be overridden to use train transforms)
            self.img_transform_eval = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor()
            ])

            # train transforms: default uses config flags; can be disabled by config
            if getattr(self.config, "image_augmentation_enable", False):
                aug_list = [
                    T.RandomResizedCrop((self.image_size, self.image_size),
                                        scale=getattr(self.config, "image_aug_random_resized_crop_scale", (0.8, 1.0))),
                    T.RandomHorizontalFlip(getattr(self.config, "image_aug_hflip_prob", 0.5)),
                ]
                cj = tuple(getattr(self.config, "image_aug_color_jitter", (0.2, 0.2, 0.2, 0.02)))
                if any(cj):
                    aug_list.append(T.ColorJitter(*cj))
                aug_list.append(T.ToTensor())
                # RandomErasing applied as transform if probability > 0
                re_prob = getattr(self.config, "image_aug_random_erasing_prob", 0.0)
                if re_prob > 0:
                    aug_list.append(T.RandomErasing(p=re_prob))
                self.img_transform_train = T.Compose(aug_list)
            else:
                self.img_transform_train = self.img_transform_eval
            # Optionally force validation/test to use the same train augmentation (user-controlled)
            if getattr(self.config, "image_augmentation_eval_enable", False):
                self.img_transform_eval = self.img_transform_train
        else:
            self.img_transform_eval = None
            self.img_transform_train = None

        # Text augmentation flags
        self.text_augmentation_enable = getattr(self.config, "text_augmentation_enable", False)
        self.text_aug_syn_prob = getattr(self.config, "text_aug_synonym_replace_prob", 0.0)
        self.text_aug_del_prob = getattr(self.config, "text_aug_random_deletion_prob", 0.0)

        # Text cleaning flag (basic cleaning: lowercase, remove URLs, control chars, extra spaces)
        self.text_cleaning_enable = getattr(self.config, "text_cleaning_enable", False)
        # optional: additional cleaning rules could be added to config later

        # Try to import wordnet for synonym replacement if available
        try:
            from nltk.corpus import wordnet as _wn
            self._wordnet = _wn
        except Exception:
            self._wordnet = None

    def __len__(self):
        return len(self.data_list)

    def _load_text(self, guid: str) -> str:
        """Attempt to load text from several likely locations; return empty string if not found."""
        candidates = [
            os.path.join(self.data_dir, "dataset", f"{guid}.txt"),
            os.path.join(self.data_dir, f"{guid}.txt"),
            os.path.join(self.data_dir, "dataset", f"{guid}.text"),
        ]
        for p in candidates:
            if p and os.path.exists(p):
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
                        try:
                            text = data.decode(enc).strip()
                            if text:
                                return " ".join(text.split())
                        except Exception:
                            continue
                except Exception:
                    continue
        return ""

    def _tokenize_fallback(self, text: str):
        tokens = text.split()
        ids = []
        for t in tokens[: self.max_len]:
            if t not in self._word2id:
                self._word2id[t] = self._next_id
                self._next_id += 1
            ids.append(self._word2id.get(t, 1))
        # pad
        if len(ids) < self.max_len:
            padding = [0] * (self.max_len - len(ids))
            attention = [1] * len(ids) + [0] * len(padding)
            ids = ids + padding
        else:
            attention = [1] * self.max_len
        return torch.LongTensor(ids), torch.LongTensor(attention)

    def _load_image(self, guid: str):
        candidates = [
            os.path.join(self.data_dir, "dataset", f"{guid}.jpg"),
            os.path.join(self.data_dir, f"{guid}.jpg"),
            os.path.join(self.data_dir, "dataset", f"{guid}.png"),
            os.path.join(self.data_dir, f"{guid}.png"),
        ]
        for p in candidates:
            if p and os.path.exists(p):
                if Image is None:
                    break
                try:
                    img = Image.open(p).convert("RGB")
                    # choose train vs eval transform at runtime
                    if self.is_train and hasattr(self, "img_transform_train") and self.img_transform_train is not None:
                        return self.img_transform_train(img)
                    elif hasattr(self, "img_transform_eval") and self.img_transform_eval is not None:
                        return self.img_transform_eval(img)
                    else:
                        arr = torch.ByteTensor(bytearray(img.tobytes())).float() / 255.0
                        # best-effort reshape fallback; return zeros if unknown
                        return torch.zeros(3, self.image_size, self.image_size)
                except Exception:
                    continue
        # fallback: zero image
        return torch.zeros(3, self.image_size, self.image_size)

    def __getitem__(self, idx: int):
        item = self.data_list[idx]
        if self.is_train:
            guid, label_str = item
        else:
            guid = item
            label_str = None

        text = self._load_text(guid)
        # Basic text cleaning (optional)
        if text and self.text_cleaning_enable:
            text = self._clean_text(text)
        # Apply text augmentation if enabled (training only)
        if self.is_train and self.text_augmentation_enable and text:
            text = self._apply_text_augment(text)

        if self.tokenizer is not None and text:
            try:
                enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
                input_ids = enc["input_ids"].squeeze(0)
                attention_mask = enc["attention_mask"].squeeze(0)
            except Exception:
                input_ids, attention_mask = self._tokenize_fallback(text)
        else:
            input_ids, attention_mask = self._tokenize_fallback(text)

        image = self._load_image(guid)

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "guid": guid
        }

        if self.is_train:
            label_id = self.config.get_label_id(label_str) if label_str is not None else 1
            out["label"] = torch.LongTensor([label_id]).squeeze(0)

        return out

    def _apply_text_augment(self, text: str) -> str:
        """Simple text augmentation: synonym replacement (if available) and random deletion."""
        tokens = text.split()
        # synonym replacement
        if self._wordnet is not None and random.random() < self.text_aug_syn_prob:
            # find candidate words with synonyms
            candidates = [i for i, t in enumerate(tokens) if len(t) > 2]
            if candidates:
                idx = random.choice(candidates)
                syns = set()
                try:
                    for syn in self._wordnet.synsets(tokens[idx]):
                        for lem in syn.lemmas():
                            name = lem.name().replace('_', ' ')
                            if name.lower() != tokens[idx].lower():
                                syns.add(name)
                except Exception:
                    syns = set()
                if syns:
                    tokens[idx] = random.choice(list(syns))
        # random deletion
        if len(tokens) > 1 and random.random() < self.text_aug_del_prob:
            new_tokens = [t for t in tokens if random.random() > self.text_aug_del_prob]
            if len(new_tokens) == 0:
                new_tokens = [tokens[random.randrange(len(tokens))]]
            tokens = new_tokens
        return " ".join(tokens)

    def _clean_text(self, text: str) -> str:
        """Basic, conservative text cleaning to improve tokenizer stability.

        - Unescape HTML entities
        - Lowercase
        - Remove URLs
        - Remove control/non-printable characters
        - Normalize whitespace
        """
        try:
            t = html.unescape(text)
        except Exception:
            t = text or ""
        # normalize unicode to NFKC, lowercase
        try:
            t = unicodedata.normalize("NFKC", t)
        except Exception:
            pass
        t = t.lower()
        # remove urls
        t = re.sub(r'http\S+|www\.\S+', ' ', t)
        # remove mentions like @username
        t = re.sub(r'@\w+', ' ', t)
        # convert hashtags to words (remove leading #)
        t = re.sub(r'#([^\s#@]+)', r'\\1', t)
        # remove emojis and symbols: keep word chars, whitespace, and CJK unified ideographs
        t = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', t)
        # remove control chars (newline/tab) and collapse whitespace
        t = re.sub(r'[\r\n\t]+', ' ', t)
        t = t.replace('_', ' ')
        t = " ".join(t.split())
        return t


