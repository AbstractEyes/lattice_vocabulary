# geovocab2/data/prompt/booru_synthesizer.py

"""
Booru Tag Synthesizer
=====================

Generates captions in booru tag style for training image models.
Supports loading from Danbooru, Gelbooru, e621, Rule34xxx CSVs.

Author: AbstractPhil
"""

import os
import csv
import re
import random
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Set, Union, Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

from geovocab2.data.prompt.caption_base import CaptionBase

try:
    from rapidfuzz import process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    from langdetect import detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ============================================================================
# CONSTANTS
# ============================================================================

DANBOORU_CATEGORIES = {
    0: "general", 1: "artist", 2: "deprecated",
    3: "copyright", 4: "character", 5: "metadata"
}
E621_CATEGORIES = {
    0: "general", 1: "artist", 2: "deprecated", 3: "copyright",
    4: "character", 5: "species", 6: "invalid", 7: "metadata", 8: "lore"
}
GELBOORU_CATEGORIES = DANBOORU_CATEGORIES.copy()
R34X_CATEGORIES = DANBOORU_CATEGORIES.copy()

LANGUAGE_TAG_CODES = {
    "en": "english", "ja": "japanese", "fr": "french", "la": "latin",
    "ru": "russian", "uk": "ukrainian", "zh-cn": "chinese",
    "zh": "chinese", "ar": "arabic"
}

TAG_AESTHETIC_TYPES = [
    "masterpiece", "aesthetic", "very aesthetic", "most aesthetic",
    "displeasing", "very displeasing", "most displeasing",
    "normal aesthetic", "good aesthetic", "beautiful", "gorgeous",
    "stunning", "ugly", "unattractive", "unpleasant", "unappealing",
    "very ugly", "disgusting", "best quality",
    "score_9", "score_8", "score_7", "score_6", "score_5",
    "score_4", "score_3", "score_2", "score_1",
    "score_8_up", "score_7_up", "score_6_up", "score_5_up",
    "score_4_up", "score_3_up", "score_2_up", "score_1_up", "very awa",
]

TAG_GENDER = [
    "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys",
    "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls",
    "ambiguous", "intersex", "herm", "futanari", "femboy", "trap",
    "girly", "otoko no ko", "transgender", "shemale", "trans",
    "boy", "man", "male", "mature male",
    "girl", "woman", "mature female",
    "1futa", "2futas", "3futas", "4futas", "5futas", "6+futas",
    "1ambiguous", "2ambiguous", "3ambiguous", "4ambiguous",
    "5ambiguous", "6+ambiguous",
]

BOORU_TEMPLATES = {
    "solo_male": "{quality:1} {gender:1} {character:0,1} {general:6,8}",
    "solo_girl": "{quality:1} {gender:1} {character:0,1} {general:6,8}",
    "duo_male": "{quality:1} {gender:1} {character:0,2} {general:8,12}",
    "duo_girl": "{quality:1} {gender:1} {character:0,2} {general:8,12}",
    "trio_male": "{quality:1} {gender:1} {character:0,3} {general:8,12}",
    "trio_girl": "{quality:1} {gender:1} {character:0,3} {general:8,12}",
    "quad_male": "{quality:1} {gender:1} {character:0,4} {general:8,12}",
    "quad_girl": "{quality:1} {gender:1} {character:0,4} {general:8,12}",
    "quin_male": "{quality:1} {gender:1} {character:0,5} {general:6,18}",
    "quin_girl": "{quality:1} {gender:1} {character:0,5} {general:6,18}",
    "multi_male": "{quality:1} {gender:1} {character:0,6} {general:6,18}",
    "multi_girl": "{quality:1} {gender:1} {character:0,6} {general:6,18}",
    "template_000": "{copyright:0,2} {gender:1,2} {quality:1,4} {general:2,2} {species:1,2}",
    "template_001": "{general:1,4} {character:1} {copyright:0,2} {gender:1,2}",
    "template_002": "{species:1,2} {general:2,6} {quality:2,4} {gender:1,2} {character:2,3}",
    "template_003": "{character:2,5} {species:1,2} {general:1,2}",
    "template_004": "{general:1,4} {copyright:0,2} {character:1} {species:1,2} {quality:1,6} {gender:1,2}",
    "template_005": "{species:1,2} {gender:1,2} {general:2,6} {quality:1,2} {character:2,3} {copyright:0,2}",
    "template_006": "{species:1,2} {character:2} {copyright:0,2} {general:1,4}",
    "template_007": "{quality:2,4} {gender:1,2} {copyright:0,2} {species:1,2}",
    "template_008": "{character:1,2} {general:1,2} {quality:2,2} {copyright:0,2} {gender:1,2}",
    "template_009": "{copyright:0,2} {general:1,2} {character:1,4} {quality:1,3} {gender:1,2}",
    "template_010": "{general:1,3} {character:1,2} {gender:1,2} {quality:1,6} {species:1,2}",
    "template_011": "{species:1,2} {quality:2,4} {character:2,3} {copyright:0,2} {gender:1,2} {general:1,1}",
    "template_012": "{copyright:0,2} {quality:2,5} {character:1,5}",
    "template_013": "{character:0,3} {species:1,2} {gender:1,2} {copyright:0,2}",
    "template_014": "{species:1,2} {copyright:0,2} {general:1,1}",
    "template_015": "{gender:1,2} {species:1,2} {general:2,2}",
    "template_016": "{general:2,6} {species:1,2} {quality:2,2} {gender:1,2} {copyright:0,2} {character:1,6}",
    "template_017": "{quality:1,5} {gender:1,2} {species:1,2} {copyright:0,2} {character:2,2}",
    "template_018": "{general:1,5} {character:1,1} {quality:2,3} {gender:1,2} {species:1,2} {copyright:0,2}",
    "template_019": "{quality:1,5} {species:1,2} {general:1,4} {character:2,6} {copyright:0,2} {gender:1,2}",
    "template_020": "{general:2,4} {copyright:0,2} {gender:1,2} {character:2,5}",
    "template_021": "{copyright:0,2} {species:1,2} {general:1,4} {quality:2,4} {character:1,4} {gender:1,2}",
    "template_022": "{gender:1,2} {copyright:0,2} {character:2,4} {quality:2,2} {general:1,4}",
    "template_023": "{species:1,2} {copyright:0,2} {quality:2,3} {character:1,3} {gender:1,2}",
    "template_024": "{quality:1,5} {gender:1,2} {copyright:0,2}",
    "template_025": "{gender:1,2} {quality:1,3} {copyright:0,2} {character:2,4}",
    "template_026": "{quality:2,3} {general:2,6} {species:1,2} {gender:1,2} {character:1,6} {copyright:0,2}",
    "template_027": "{gender:1,2} {character:1,1} {copyright:0,2}",
    "template_028": "{quality:2,4} {copyright:0,2} {character:2,4} {gender:1,2} {species:1,2} {general:2,3}",
    "template_029": "{copyright:0,2} {quality:2,3} {gender:1,2}",
    "template_030": "{quality:2,5} {species:1,2} {character:2,2} {general:1,6} {copyright:0,2}",
    "template_031": "{quality:2,5} {gender:1,2} {character:1,2}",
    "template_032": "{quality:2,3} {general:1,4} {gender:1,2}",
    "template_033": "{copyright:0,2} {gender:1,2} {character:2,6} {quality:2,5} {general:2,6} {species:1,2}",
    "template_034": "{general:2,2} {species:1,2} {quality:2,6} {copyright:0,2}",
    "template_035": "{character:2,6} {gender:1,2} {copyright:0,2} {general:1,3}",
    "template_036": "{copyright:0,2} {character:2,6} {general:2,6}",
    "template_037": "{species:1,2} {copyright:0,2} {character:2,6}",
    "template_038": "{general:1,1} {gender:1,2} {copyright:0,2} {quality:2,6}",
    "template_039": "{quality:2,6} {general:2,2} {species:1,2} {character:2,6}",
    "template_040": "{general:1,5} {quality:1,4} {gender:1,2} {copyright:0,2}",
    "template_041": "{general:2,4} {character:2,3} {gender:1,2}",
    "template_042": "{general:1,5} {species:1,2} {character:2,3}",
    "template_043": "{species:1,2} {copyright:0,2} {gender:1,2} {character:2,5}",
    "template_044": "{gender:1,2} {species:1,2} {copyright:0,2} {quality:2,5}",
    "template_045": "{species:1,2} {character:1,2} {gender:1,2}",
    "template_046": "{general:1,4} {species:1,2} {quality:1,2} {character:2,5}",
    "template_047": "{gender:1,2} {species:1,2} {character:2,4} {copyright:0,2} {quality:1,5}",
    "template_048": "{copyright:0,2} {quality:1,4} {general:1,1}",
    "template_049": "{character:1,1} {copyright:0,2} {quality:2,3}",
    "template_050": "{general:2,2} {species:1,2} {character:1,1} {copyright:0,2} {gender:1,2}",
    "template_051": "{gender:1,2} {general:1,5} {character:2,2}",
    "template_052": "{general:1,4} {species:1,2} {quality:1,5} {gender:1,2} {copyright:0,2} {character:0,5}",
    "template_053": "{species:1,2} {gender:1,2} {quality:2,5} {copyright:0,2} {general:2,2} {character:0,2}",
    "template_054": "{gender:1,2} {character:0,1} {species:1,2}",
    "template_055": "{gender:1,2} {species:1,2} {character:0,2}",
    "template_056": "{quality:2,6} {copyright:0,2} {species:1,2} {gender:1,2}",
    "template_057": "{gender:1,2} {quality:2,6} {copyright:0,2} {character:2,6}",
    "template_058": "{copyright:0,2} {gender:1,2} {character:1,2}",
    "template_059": "{general:2,2} {character:2,5} {gender:1,2} {quality:2,2} {copyright:0,2}",
    "template_060": "{character:1,6} {copyright:0,2} {quality:2,2} {gender:1,2}",
    "template_061": "{character:1,2} {copyright:0,2} {quality:1,5} {gender:1,2} {species:1,2}",
    "template_062": "{quality:2,6} {copyright:0,2} {general:2,6}",
    "template_063": "{species:1,2} {gender:1,2} {general:1,6} {copyright:0,2} {quality:1,4} {character:0,2}",
    "template_064": "{quality:2,3} {copyright:0,2} {general:2,3}",
    "template_065": "{character:1,2} {quality:2,2} {species:1,2} {general:2,7} {gender:1,2} {copyright:0,2}",
    "template_066": "{species:1,2} {gender:1,2} {quality:2,2} {copyright:0,2} {character:0,5} {general:2,4}",
    "template_067": "{gender:1,2} {character:1,4} {copyright:0,2}",
    "template_068": "{character:1,3} {quality:2,4} {general:2,4} {copyright:0,2}",
    "template_069": "{species:1,2} {quality:1,2} {character:1,1} {gender:1,2} {copyright:0,2}",
    "template_070": "{quality:1,5} {copyright:0,2} {species:1,2} {general:1,4}",
    "template_071": "{copyright:0,2} {quality:2,3} {general:1,4} {gender:1,2}",
    "template_072": "{character:1,1} {copyright:0,2} {species:1,2} {quality:2,5} {general:1,6} {gender:1,2}",
    "template_073": "{gender:1,2} {general:2,6} {character:1,4} {species:1,2} {copyright:0,2}",
    "template_074": "{gender:1,2} {species:1,2} {character:2,6}",
    "template_075": "{quality:2,2} {species:1,2} {copyright:0,2}",
    "template_076": "{quality:2,6} {character:1,5} {species:1,2}",
    "template_077": "{character:1,5} {gender:1,2} {species:1,2}",
    "template_078": "{copyright:0,2} {quality:2,2} {general:1,6} {species:1,2}",
    "template_079": "{copyright:0,2} {gender:1,2} {general:1,3} {character:1,5} {species:1,2} {quality:2,3}",
    "template_080": "{species:1,2} {copyright:0,2} {character:1,2}",
    "template_081": "{copyright:0,2} {species:1,2} {character:1,2} {gender:1,2} {general:1,4}",
    "template_082": "{quality:2,6} {species:1,2} {character:2,4} {copyright:0,2} {gender:1,2}",
    "template_083": "{gender:1,2} {character:1,3} {copyright:0,2} {quality:1,4}",
    "template_084": "{species:1,2} {general:2,4} {character:1,1} {copyright:0,2} {quality:2,4}",
    "template_085": "{character:2,2} {copyright:0,2} {quality:2,4} {species:1,2} {gender:1,2} {general:1,5}",
    "template_086": "{gender:1,2} {copyright:0,2} {species:1,2}",
    "template_087": "{character:1,4} {quality:2,5} {general:1,1} {species:1,2} {copyright:0,2}",
    "template_088": "{character:1,3} {general:1,4} {quality:1,1}",
    "template_089": "{general:1,4} {copyright:0,2} {character:1,2}",
    "template_090": "{character:2,5} {gender:1,2} {quality:2,6} {species:1,2} {copyright:0,2}",
    "template_091": "{species:1,2} {character:2,5} {quality:1,1} {general:2,2} {copyright:0,2}",
    "template_092": "{general:1,6} {copyright:0,2} {character:1,2}",
    "template_093": "{species:1,2} {general:2,4} {quality:2,6} {copyright:0,2}",
    "template_094": "{copyright:0,2} {character:1,2} {quality:2,2}",
    "template_095": "{quality:1,1} {species:1,2} {copyright:0,2} {general:2,3} {character:1,2}",
    "template_096": "{species:1,2} {quality:2,5} {copyright:0,2} {general:2,6}",
    "template_097": "{species:1,2} {character:1,2} {copyright:0,2} {general:2,6}",
    "template_098": "{gender:1,2} {character:1,2} {quality:1,3} {general:2,3} {copyright:0,2} {species:1,2}",
    "template_099": "{quality:2,3} {species:1,2} {general:2,4} {gender:1,2}",
}


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class BooruConfig:
    """Configuration for BooruSynthesizer."""
    # CSV paths (None = skip loading)
    danbooru_csv: Optional[str] = None
    gelbooru_csv: Optional[str] = None
    e621_csv: Optional[str] = None
    rule34x_csv: Optional[str] = None

    # Generation settings
    deduplicate: bool = True
    replace_underscores: bool = True
    prune_languages: List[str] = field(default_factory=lambda: [
        "japanese", "chinese", "russian", "arabic", "french", "ukrainian", "latin"
    ])

    # Fuzzy matching
    apply_fuzzy: bool = False
    fuzzy_threshold: int = 90

    # Templates
    custom_templates: Optional[Dict[str, str]] = None

    # Weights for biasing tag selection
    weight_by_post_count: bool = False

    seed: Optional[int] = None


# ============================================================================
# TAG INDEX
# ============================================================================

class TagIndex:
    """Manages loaded booru tags and categories."""

    def __init__(self):
        self.tag_dict: Dict[str, dict] = {}
        self.alias_dict: Dict[str, Set[str]] = {}
        self.category_dict: Dict[str, Set[str]] = {}
        self.source_tags: Dict[str, Set[str]] = {}

    def add_tag(self, tag_name: str, post_count: int, category: str,
                aliases: List[str], source: str):
        """Add or update a tag entry."""
        tag_data = self.tag_dict.setdefault(tag_name, {
            "post_count": 0,
            "category": category,
            "aliases": set(),
            "sources": set()
        })
        tag_data["post_count"] += post_count
        tag_data["aliases"].update(aliases)
        tag_data["sources"].add(source)

        self.category_dict.setdefault(category, set()).add(tag_name)
        self.source_tags.setdefault(source, set()).add(tag_name)

    def rebuild_alias_dict(self):
        """Rebuild alias->tag mapping, removing overlaps with real tags."""
        self.alias_dict.clear()
        all_tags = set(self.tag_dict.keys())

        for tag, data in self.tag_dict.items():
            data["aliases"] -= all_tags
            for alias in data["aliases"]:
                self.alias_dict.setdefault(alias, set()).add(tag)

    def get_category_tags(self, category: str) -> List[str]:
        """Get all tags in a category."""
        return list(self.category_dict.get(category, []))

    def get_weighted_tags(self, category: str) -> List[str]:
        """Get tags weighted by post count (more popular = more likely)."""
        tags = self.category_dict.get(category, set())
        weighted = []
        for tag in tags:
            count = self.tag_dict.get(tag, {}).get("post_count", 1)
            # Log scale to prevent extreme skew
            weight = max(1, int(count ** 0.3))
            weighted.extend([tag] * weight)
        return weighted

    def __len__(self):
        return len(self.tag_dict)

    def __contains__(self, tag: str):
        return tag in self.tag_dict or tag in self.alias_dict


# ============================================================================
# CSV LOADERS
# ============================================================================

def load_danbooru_csv(path: str, index: TagIndex) -> int:
    """Load Danbooru-format CSV. Returns count of loaded tags."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                tag_name = row[0].strip()
                category = DANBOORU_CATEGORIES.get(int(row[1]), "unknown")
                post_count = int(row[2])
                aliases = [a.strip() for a in row[3].split(",")] if len(row) > 3 and row[3] else []
                index.add_tag(tag_name, post_count, category, aliases, "danbooru")
                count += 1
            except (ValueError, IndexError):
                continue
    return count


def load_gelbooru_csv(path: str, index: TagIndex) -> int:
    """Load Gelbooru-format CSV."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3 or row[0] == "name":
                continue
            try:
                tag_name = row[0].strip()
                post_count = int(row[1])
                category = row[2].strip()
                index.add_tag(tag_name, post_count, category, [], "gelbooru")
                count += 1
            except (ValueError, IndexError):
                continue
    return count


def load_e621_csv(path: str, index: TagIndex) -> int:
    """Load e621-format CSV."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                tag_name = row[0].strip()
                category = E621_CATEGORIES.get(int(row[1]), "unknown")
                post_count = int(row[2])
                aliases = [a.strip() for a in row[3].split(",")] if len(row) > 3 and row[3] else []
                index.add_tag(tag_name, post_count, category, aliases, "e621")
                count += 1
            except (ValueError, IndexError):
                continue
    return count


def load_rule34x_csv(path: str, index: TagIndex) -> int:
    """Load Rule34xxx-format CSV."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3 or row[0] == "type":
                continue
            try:
                category = R34X_CATEGORIES.get(int(row[0]), "unknown")
                post_count = int(row[1])
                tag_name = row[2].strip()
                aliases = [a.strip() for a in row[3].split(",")] if len(row) > 3 and row[3] else []
                index.add_tag(tag_name, post_count, category, aliases, "rule34x")
                count += 1
            except (ValueError, IndexError):
                continue
    return count


# ============================================================================
# UTILITIES
# ============================================================================

def detect_language(text: str) -> str:
    """Detect language of text."""
    if not HAS_LANGDETECT:
        return "unknown"
    try:
        return detect(text)
    except:
        return "unknown"


def is_language(tag: str, target: str) -> bool:
    """Check if tag appears to be in target language."""
    return LANGUAGE_TAG_CODES.get(detect_language(tag), "unknown") == target


class TagFuzzyMatcher:
    """Fuzzy matching for tag correction."""

    def __init__(self, known_tags: Set[str], min_score: int = 90):
        if not HAS_RAPIDFUZZ:
            raise ImportError("rapidfuzz required for fuzzy matching: pip install rapidfuzz")
        self.known_tags = list(known_tags)
        self.min_score = min_score

    def correct(self, query: str) -> Optional[str]:
        result = process.extractOne(query, self.known_tags, score_cutoff=self.min_score)
        return result[0] if result else None

    def correct_batch(self, tags: List[str]) -> List[str]:
        return [self.correct(t.strip()) or t for t in tags]


# ============================================================================
# TEMPLATE RENDERER
# ============================================================================

class TemplateRenderer:
    """Fast template rendering with pre-compiled patterns."""

    PATTERN = re.compile(r"\{(\w+):(\d+),?(\d+)?\}")

    def __init__(self, tag_index: TagIndex, templates: Dict[str, str],
                 preserve_tags: Optional[Set[str]] = None,
                 weight_by_post_count: bool = False):
        self.tag_index = tag_index
        self.templates = templates
        self.template_names = list(templates.keys())
        self.preserve_tags = preserve_tags or (set(TAG_AESTHETIC_TYPES) | set(TAG_GENDER))
        self.weight_by_post_count = weight_by_post_count

        # Pre-build tag pools
        self.pools: Dict[str, List[str]] = {
            "quality": TAG_AESTHETIC_TYPES.copy(),
            "gender": TAG_GENDER.copy(),
        }
        for category, tags in tag_index.category_dict.items():
            if weight_by_post_count:
                self.pools[category] = tag_index.get_weighted_tags(category)
            else:
                self.pools[category] = list(tags)

    def render(self, template_name: Optional[str] = None,
               dedupe: bool = True, replace_underscores: bool = True) -> str:
        """Render a single template."""
        if template_name is None:
            template_name = random.choice(self.template_names)

        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        final_tags = []

        for match in self.PATTERN.finditer(template):
            cat, min_ct, max_ct = match.groups()
            min_ct = int(min_ct)
            max_ct = int(max_ct or min_ct)

            pool = self.pools.get(cat, [])
            if not pool:
                continue

            count = random.randint(min_ct, max_ct)
            if count > 0:
                sampled = random.sample(pool, min(count, len(pool)))
                final_tags.extend(sampled)

        # Post-process
        processed = []
        for tag in final_tags:
            if replace_underscores and tag not in self.preserve_tags:
                tag = tag.replace("_", " ").strip()
            processed.append(tag)

        if dedupe:
            seen = set()
            deduped = []
            for tag in processed:
                if tag not in seen:
                    seen.add(tag)
                    deduped.append(tag)
            processed = deduped

        return ", ".join(processed)


# ============================================================================
# MAIN SYNTHESIZER
# ============================================================================

class BooruSynthesizer(CaptionBase):
    """
    Generates captions in booru tag style.

    Supports loading tags from CSV files (Danbooru, Gelbooru, e621, Rule34xxx)
    or from custom files (JSONL, TXT).
    """

    def __init__(self, config: Optional[BooruConfig] = None):
        super().__init__(name="BooruSynthesizer", uid="prompt.booru_synthesizer")
        self.config = config or BooruConfig()
        self.tag_index = TagIndex()
        self.renderer: Optional[TemplateRenderer] = None
        self.hash_cache: Set[str] = set()

        if self.config.seed is not None:
            random.seed(self.config.seed)

        self._load_csvs()
        self._init_renderer()

    def _load_csvs(self):
        """Load tag CSVs based on config."""
        loaders = [
            (self.config.danbooru_csv, load_danbooru_csv, "Danbooru"),
            (self.config.gelbooru_csv, load_gelbooru_csv, "Gelbooru"),
            (self.config.e621_csv, load_e621_csv, "e621"),
            (self.config.rule34x_csv, load_rule34x_csv, "Rule34xxx"),
        ]

        for path, loader, name in loaders:
            if path and os.path.exists(path):
                count = loader(path, self.tag_index)
                print(f"  ✓ Loaded {count:,} tags from {name}")

        self.tag_index.rebuild_alias_dict()

    def _init_renderer(self):
        """Initialize template renderer."""
        templates = BOORU_TEMPLATES.copy()
        if self.config.custom_templates:
            templates.update(self.config.custom_templates)

        self.renderer = TemplateRenderer(
            self.tag_index, templates,
            weight_by_post_count=self.config.weight_by_post_count
        )

    def load_from_file(self, path: str, format: str = "auto"):
        """
        Load additional tags/prompts from a file.

        Args:
            path: Path to file
            format: 'csv', 'jsonl', 'txt', or 'auto' (detect from extension)
        """
        path = Path(path)

        if format == "auto":
            format = path.suffix.lstrip(".").lower()

        if format == "csv":
            load_danbooru_csv(str(path), self.tag_index)
        elif format == "jsonl":
            self._load_jsonl(path)
        elif format == "txt":
            self._load_txt(path)
        else:
            raise ValueError(f"Unknown format: {format}")

        self.tag_index.rebuild_alias_dict()
        self._init_renderer()

    def _load_jsonl(self, path: Path):
        """Load from JSONL file."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "tags" in data:
                        for tag in data["tags"]:
                            self.tag_index.add_tag(tag, 1, "general", [], "custom")
                    elif "text" in data:
                        tags = [t.strip() for t in data["text"].split(",")]
                        for tag in tags:
                            if tag:
                                self.tag_index.add_tag(tag, 1, "general", [], "custom")
                except json.JSONDecodeError:
                    continue

    def _load_txt(self, path: Path):
        """Load from text file."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tags = [t.strip() for t in line.split(",")]
                for tag in tags:
                    if tag:
                        self.tag_index.add_tag(tag, 1, "general", [], "custom")

    def clean_tags(self, tags: List[str]) -> List[str]:
        """Clean tags by removing non-English and optionally fuzzy matching."""
        if self.config.prune_languages:
            tags = [t for t in tags
                    if not any(is_language(t, lang) for lang in self.config.prune_languages)]

        if self.config.apply_fuzzy and HAS_RAPIDFUZZ:
            known = set(self.tag_index.tag_dict.keys()) | set(self.tag_index.alias_dict.keys())
            matcher = TagFuzzyMatcher(known, self.config.fuzzy_threshold)
            tags = matcher.correct_batch(tags)

        return tags

    def generate(self, template_name: Optional[str] = None) -> str:
        """Generate a single booru-style caption."""
        if self.renderer is None:
            raise RuntimeError("Renderer not initialized. Load tags first.")

        return self.renderer.render(
            template_name=template_name,
            dedupe=self.config.deduplicate,
            replace_underscores=self.config.replace_underscores
        )

    def generate_batch(self, count: int, deduplicate: bool = True,
                       template_name: Optional[str] = None) -> List[str]:
        """Generate a batch of captions."""
        results = []

        while len(results) < count:
            caption = self.generate(template_name=template_name)

            if deduplicate:
                h = hashlib.md5(caption.encode()).hexdigest()
                if h in self.hash_cache:
                    continue
                self.hash_cache.add(h)

            results.append(caption)

        return results

    def generate_iterator(self, count: int, deduplicate: bool = True) -> Iterator[str]:
        """Generate captions as an iterator (memory efficient)."""
        generated = 0
        while generated < count:
            caption = self.generate()

            if deduplicate:
                h = hashlib.md5(caption.encode()).hexdigest()
                if h in self.hash_cache:
                    continue
                self.hash_cache.add(h)

            yield caption
            generated += 1

    def generate_to_file(self, path: str, count: int, format: str = "jsonl",
                         num_threads: int = 1, batch_size: int = 1000,
                         show_progress: bool = True):
        """
        Generate captions and save to file.

        Args:
            path: Output file path
            count: Number of captions to generate
            format: 'jsonl', 'txt', or 'csv'
            num_threads: Number of threads for generation
            batch_size: Batch size for threaded generation
            show_progress: Show progress bar
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if num_threads > 1:
            self._generate_threaded(path, count, format, num_threads, batch_size, show_progress)
        else:
            self._generate_single(path, count, format, show_progress)

    def _generate_single(self, path: Path, count: int, format: str, show_progress: bool):
        """Single-threaded generation."""
        iterator = range(count)
        if show_progress and HAS_TQDM:
            iterator = tqdm(iterator, desc="Generating", unit=" samples")

        with open(path, "w", encoding="utf-8") as f:
            for i in iterator:
                caption = self.generate()

                if format == "jsonl":
                    f.write(json.dumps({"text": caption, "index": i}) + "\n")
                elif format == "txt":
                    f.write(caption + "\n")
                elif format == "csv":
                    f.write(caption.replace(",", ";") + "\n")

    def _generate_threaded(self, path: Path, count: int, format: str,
                           num_threads: int, batch_size: int, show_progress: bool):
        """Multi-threaded generation."""
        lock = threading.Lock()
        sample_index = [0]

        def generate_batch_worker(size: int) -> List[str]:
            return [self.generate() for _ in range(size)]

        num_batches = (count + batch_size - 1) // batch_size

        iterator = range(0, num_batches, num_threads)
        if show_progress and HAS_TQDM:
            iterator = tqdm(iterator, desc="Generating", unit=" batches")

        with open(path, "w", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                for chunk_start in iterator:
                    chunk_end = min(chunk_start + num_threads, num_batches)

                    futures = []
                    for batch_idx in range(chunk_start, chunk_end):
                        current_size = min(batch_size, count - batch_idx * batch_size)
                        if current_size > 0:
                            futures.append(executor.submit(generate_batch_worker, current_size))

                    for future in futures:
                        batch = future.result()
                        with lock:
                            for caption in batch:
                                if format == "jsonl":
                                    f.write(json.dumps({"text": caption, "index": sample_index[0]}) + "\n")
                                elif format == "txt":
                                    f.write(caption + "\n")
                                elif format == "csv":
                                    f.write(caption.replace(",", ";") + "\n")
                                sample_index[0] += 1

    def clear_cache(self):
        """Clear deduplication cache."""
        self.hash_cache.clear()

    def get_template_names(self) -> List[str]:
        """Get available template names."""
        return self.renderer.template_names if self.renderer else []

    def get_categories(self) -> List[str]:
        """Get available tag categories."""
        return list(self.tag_index.category_dict.keys())

    def get_category_count(self, category: str) -> int:
        """Get number of tags in a category."""
        return len(self.tag_index.category_dict.get(category, []))

    def stats(self) -> Dict:
        """Get synthesizer statistics."""
        return {
            "total_tags": len(self.tag_index),
            "categories": {cat: len(tags) for cat, tags in self.tag_index.category_dict.items()},
            "sources": {src: len(tags) for src, tags in self.tag_index.source_tags.items()},
            "templates": len(self.get_template_names()),
            "cache_size": len(self.hash_cache),
        }

    def __getitem__(self, key: str) -> str:
        """Generate using specific template."""
        return self.generate(template_name=key)

    def __len__(self) -> int:
        """Return number of loaded tags."""
        return len(self.tag_index)

    def __repr__(self):
        return (f"BooruSynthesizer(tags={len(self.tag_index)}, "
                f"categories={len(self.tag_index.category_dict)}, "
                f"templates={len(self.get_template_names())})")


# ============================================================================
# MAIN / CLI
# ============================================================================

def demo_no_csvs():
    """Demo without CSV files - uses built-in quality/gender tags only."""
    print("=" * 60)
    print("Demo: BooruSynthesizer (no CSV files)")
    print("=" * 60)

    synth = BooruSynthesizer()
    print(f"\n{synth}")
    print(f"\nStats: {synth.stats()}")

    print("\n--- Sample Generations ---")
    for i in range(5):
        print(f"  {i + 1}. {synth.generate()}")

    print("\n--- Using Specific Templates ---")
    for template in ["solo_girl", "duo_male", "template_001"]:
        print(f"  [{template}] {synth[template]}")

    return synth


def demo_with_csvs(danbooru_path: str = None, gelbooru_path: str = None,
                   e621_path: str = None, rule34x_path: str = None):
    """Demo with CSV files."""
    print("=" * 60)
    print("Demo: BooruSynthesizer (with CSV files)")
    print("=" * 60)

    config = BooruConfig(
        danbooru_csv=danbooru_path,
        gelbooru_csv=gelbooru_path,
        e621_csv=e621_path,
        rule34x_csv=rule34x_path,
        seed=42
    )

    print("\nLoading tags...")
    synth = BooruSynthesizer(config)
    print(f"\n{synth}")

    stats = synth.stats()
    print(f"\nTotal tags: {stats['total_tags']:,}")
    print("Categories:")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:,}")

    print("\n--- Sample Generations ---")
    for i in range(10):
        print(f"  {i + 1}. {synth.generate()}")

    return synth


def benchmark(iterations: int = 10000):
    """Benchmark generation speed."""
    print("=" * 60)
    print(f"Benchmark: {iterations:,} iterations")
    print("=" * 60)

    synth = BooruSynthesizer(BooruConfig(seed=42))

    # Warm up
    for _ in range(100):
        synth.generate()

    start = time.time()
    for _ in range(iterations):
        synth.generate()
    elapsed = time.time() - start

    print(f"Time: {elapsed:.3f}s")
    print(f"Rate: {iterations / elapsed:,.0f} samples/sec")

    return elapsed


def generate_dataset(output_path: str, count: int = 100000,
                     num_threads: int = 4, format: str = "jsonl",
                     danbooru_csv: str = None):
    """Generate a dataset file."""
    print("=" * 60)
    print(f"Generate Dataset: {count:,} samples -> {output_path}")
    print("=" * 60)

    config = BooruConfig(
        danbooru_csv=danbooru_csv,
        seed=42
    )

    synth = BooruSynthesizer(config)
    print(f"\n{synth}")

    print(f"\nGenerating with {num_threads} threads...")
    start = time.time()

    synth.generate_to_file(
        path=output_path,
        count=count,
        format=format,
        num_threads=num_threads,
        batch_size=1000,
        show_progress=True
    )

    elapsed = time.time() - start
    print(f"\n✓ Generated {count:,} samples in {elapsed:.1f}s")
    print(f"  Rate: {count / elapsed:,.0f} samples/sec")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Booru Tag Synthesizer")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--benchmark", type=int, default=0, help="Run benchmark with N iterations")
    parser.add_argument("--generate", type=str, help="Generate dataset to path")
    parser.add_argument("--count", type=int, default=100000, help="Number of samples to generate")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--format", type=str, default="jsonl", choices=["jsonl", "txt", "csv"])
    parser.add_argument("--danbooru", type=str, help="Path to danbooru.csv")
    parser.add_argument("--gelbooru", type=str, help="Path to gelbooru.csv")
    parser.add_argument("--e621", type=str, help="Path to e621.csv")
    parser.add_argument("--rule34x", type=str, help="Path to rule34_xxx.csv")

    args = parser.parse_args()

    if args.demo:
        if any([args.danbooru, args.gelbooru, args.e621, args.rule34x]):
            demo_with_csvs(args.danbooru, args.gelbooru, args.e621, args.rule34x)
        else:
            demo_no_csvs()

    elif args.benchmark > 0:
        benchmark(args.benchmark)

    elif args.generate:
        generate_dataset(
            output_path=args.generate,
            count=args.count,
            num_threads=args.threads,
            format=args.format,
            danbooru_csv=args.danbooru
        )

    else:
        # Default: run demo
        demo_no_csvs()
