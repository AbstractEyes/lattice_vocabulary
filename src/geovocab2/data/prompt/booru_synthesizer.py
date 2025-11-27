# geovocab2/data/prompt/booru_synthesizer.py

"""
Booru Tag Synthesizer
=====================

Generates captions in booru tag style for training image models.
Supports loading from Danbooru, Gelbooru, e621, Rule34xxx CSVs.

Features:
- Template-based generation
- Conduit system: injects random top-N tags for generalization
- Multi-threaded batch generation

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
from typing import Optional, Dict, List, Set, Union, Iterator, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    from geovocab2.data.prompt.caption_base import CaptionBase

    HAS_CAPTION_BASE = True
except ImportError:
    HAS_CAPTION_BASE = False


    class CaptionBase:
        def __init__(self, name: str = "", uid: str = ""):
            self.name = name
            self.uid = uid

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
]

TAG_GENDER = [
    "1boy", "1boy", "1boy", "1boy", "1boy", "1boy", "1boy", "1boy", "1boy", "1boy",
    "2boys", "2boys", "2boys",
    "3boys",
    "4boys",
    "5boys",
    "6+boys",
    "1girl", "1girl", "1girl", "1girl", "1girl", "1girl", "1girl", "1girl", "1girl", "1girl",
    "2girls", "2girls", "2girls",
    "3girls",
    "4girls",
    "5girls",
    "6+girls",
    "ambiguous", "intersex", "herm", "futanari", "femboy", "trap",
    "girly", "otoko no ko", "transgender", "shemale", "trans",
    "boy", "man", "male", "mature male",
    "girl", "woman", "mature female",
    "1futa", "2futas", "3futas", "4futas", "5futas", "6+futas",
    "1ambiguous", "2ambiguous", "3ambiguous", "4ambiguous",
    "5ambiguous", "6+ambiguous",
    "1other", "2other", "3other", "4other", "5other", "6+other",
]

# ============================================================================
# PEOPLE COUNT SYSTEM
# ============================================================================

# Structured gender tags with counts
GENDER_COUNT_TAGS = {
    # (tag, count, category)
    "boys": [
        ("1boy", 1), ("2boys", 2), ("3boys", 3), ("4boys", 4), ("5boys", 5), ("6+boys", 6),
    ],
    "girls": [
        ("1girl", 1), ("2girls", 2), ("3girls", 3), ("4girls", 4), ("5girls", 5), ("6+girls", 6),
    ],
    "futas": [
        ("1futa", 1), ("2futas", 2), ("3futas", 3), ("4futas", 4), ("5futas", 5), ("6+futas", 6),
    ],
    "others": [
        ("1other", 1), ("2others", 2), ("3others", 3), ("4others", 4), ("5others", 5), ("6+others", 6),
    ],
    "ambiguous": [
        ("1ambiguous", 1), ("2ambiguous", 2), ("3ambiguous", 3), ("4ambiguous", 4), ("5ambiguous", 5),
        ("6+ambiguous", 6),
    ],
}

# Weights for solo vs group (heavily biased toward solo/duo)
GENDER_SCENARIO_WEIGHTS = {
    "solo_girl": 30,
    "solo_boy": 20,
    "duo_girls": 10,
    "duo_boys": 5,
    "mixed_duo": 8,
    "trio": 5,
    "quad": 3,
    "group": 2,
    "solo_other": 3,
}

# Number words for T5
NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve",
}

# T5 people count prefix templates
PEOPLE_COUNT_TEMPLATES = [
    "{n}people",
    "{n} people",
    "{word} people",
    "{n}persons",
    "{n} persons",
    "{word} persons",
    "there are {word} people",
    "there is {word} person" if 1 else "there are {word} people",
    "{word} characters",
    "{n} characters",
    "scene with {word} people",
    "group of {word}",
    "{word} figures",
    "{n} figures",
]

SOLO_TEMPLATES = [
    "1person",
    "1 person",
    "one person",
    "solo",
    "single person",
    "single character",
    "one character",
    "alone",
    "single figure",
]


def sample_coherent_gender() -> Tuple[List[str], int]:
    """
    Sample a coherent set of gender tags (no conflicting counts).

    Returns:
        Tuple of (gender_tags, total_people_count)
    """
    # Weighted scenario selection
    scenarios = list(GENDER_SCENARIO_WEIGHTS.keys())
    weights = list(GENDER_SCENARIO_WEIGHTS.values())
    scenario = random.choices(scenarios, weights=weights, k=1)[0]

    tags = []
    total = 0

    if scenario == "solo_girl":
        tags = ["1girl"]
        total = 1
    elif scenario == "solo_boy":
        tags = ["1boy"]
        total = 1
    elif scenario == "solo_other":
        # Pick one from other categories
        other_tags = ["1futa", "1other", "1ambiguous"]
        tags = [random.choice(other_tags)]
        total = 1
    elif scenario == "duo_girls":
        tags = ["2girls"]
        total = 2
    elif scenario == "duo_boys":
        tags = ["2boys"]
        total = 2
    elif scenario == "mixed_duo":
        tags = ["1girl", "1boy"]
        total = 2
    elif scenario == "trio":
        # Various trio combinations
        trio_options = [
            (["3girls"], 3),
            (["3boys"], 3),
            (["2girls", "1boy"], 3),
            (["1girl", "2boys"], 3),
        ]
        tags, total = random.choice(trio_options)
    elif scenario == "quad":
        quad_options = [
            (["4girls"], 4),
            (["4boys"], 4),
            (["2girls", "2boys"], 4),
            (["3girls", "1boy"], 4),
            (["1girl", "3boys"], 4),
        ]
        tags, total = random.choice(quad_options)
    elif scenario == "group":
        # 5+ people
        group_options = [
            (["5girls"], 5),
            (["5boys"], 5),
            (["6+girls"], 6),
            (["6+boys"], 6),
            (["3girls", "2boys"], 5),
            (["2girls", "3boys"], 5),
            (["3girls", "3boys"], 6),
            (["4girls", "2boys"], 6),
            (["2girls", "4boys"], 6),
        ]
        tags, total = random.choice(group_options)

    return tags, total


def generate_people_count_prefix(count: int) -> str:
    """
    Generate a T5-friendly people count prefix.

    Args:
        count: Number of people

    Returns:
        String like "3people", "three people", "there are three people", etc.
    """
    if count == 1:
        return random.choice(SOLO_TEMPLATES)

    word = NUMBER_WORDS.get(count, str(count))

    # Pick a random template
    template = random.choice(PEOPLE_COUNT_TEMPLATES)

    # Handle singular/plural edge case
    if count == 1 and "are" in template:
        template = template.replace("are", "is").replace("people", "person")

    return template.format(n=count, word=word)


def parse_gender_tags_count(tags: List[str]) -> int:
    """
    Parse gender tags and return total people count.

    Args:
        tags: List of tags that may include gender tags

    Returns:
        Total count of people
    """
    total = 0

    for tag in tags:
        tag_lower = tag.lower().strip()

        # Check numbered tags
        for category, entries in GENDER_COUNT_TAGS.items():
            for tag_name, count in entries:
                if tag_lower == tag_name.lower():
                    total += count
                    break

        # Check singular non-numbered tags
        if tag_lower in ["boy", "man", "male", "mature male"]:
            total += 1
        elif tag_lower in ["girl", "woman", "female", "mature female"]:
            total += 1
        elif tag_lower in ["futanari", "futa", "herm", "intersex"]:
            total += 1
        elif tag_lower in ["femboy", "trap", "otoko no ko", "girly"]:
            total += 1

    return total


class CoherentGenderSampler:
    """
    Samples coherent gender tag combinations without conflicts.
    Tracks people count for T5 prefix generation.
    """

    def __init__(self, include_count_prefix: bool = True):
        self.include_count_prefix = include_count_prefix
        self.last_count = 0
        self.last_tags = []

    def sample(self) -> Tuple[List[str], int, Optional[str]]:
        """
        Sample gender tags coherently.

        Returns:
            Tuple of (gender_tags, people_count, count_prefix_for_t5)
        """
        tags, count = sample_coherent_gender()
        self.last_tags = tags
        self.last_count = count

        prefix = None
        if self.include_count_prefix:
            prefix = generate_people_count_prefix(count)

        return tags, count, prefix

    def get_t5_prefix(self, count: Optional[int] = None) -> str:
        """Get a T5-friendly count prefix."""
        if count is None:
            count = self.last_count
        return generate_people_count_prefix(count)


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
class ConduitConfig:
    """Configuration for the conduit tag injection system."""
    enabled: bool = False
    top_n: int = 1000  # Pull from top N tags by post count
    sample_k: int = 10  # Sample K random tags per prompt
    sample_k_min: int = 5  # Minimum tags to sample (for variance)
    sample_k_max: int = 15  # Maximum tags to sample
    position: str = "prepend"  # "prepend", "append", or "random"
    category_filter: Optional[List[str]] = None  # Only include these categories (None = all)
    exclude_categories: Optional[List[str]] = None  # Exclude these categories
    weight_by_rank: bool = False  # Weight sampling by rank (higher rank = more likely)


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

    # Coherent gender sampling (prevents 2girls + 3girls conflicts)
    use_coherent_gender: bool = True

    # Generate T5 people count prefix ("3people", "there are three people", etc.)
    generate_t5_prefix: bool = True

    # Conduit system
    conduit: Optional[ConduitConfig] = None

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

        # Sorted cache for conduit
        self._sorted_by_count: Optional[List[Tuple[str, int, str]]] = None

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

        # Invalidate sorted cache
        self._sorted_by_count = None

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

    def get_top_tags(
            self,
            n: int = 1000,
            category_filter: Optional[List[str]] = None,
            exclude_categories: Optional[List[str]] = None
    ) -> List[Tuple[str, int, str]]:
        """
        Get top N tags sorted by post count.

        Returns:
            List of (tag_name, post_count, category) tuples
        """
        # Build sorted list if not cached
        if self._sorted_by_count is None:
            self._sorted_by_count = [
                (tag, data["post_count"], data["category"])
                for tag, data in self.tag_dict.items()
            ]
            self._sorted_by_count.sort(key=lambda x: -x[1])  # Descending by count

        # Filter by category
        result = self._sorted_by_count

        if category_filter:
            result = [t for t in result if t[2] in category_filter]

        if exclude_categories:
            result = [t for t in result if t[2] not in exclude_categories]

        return result[:n]

    def __len__(self):
        return len(self.tag_dict)

    def __contains__(self, tag: str):
        return tag in self.tag_dict or tag in self.alias_dict


# ============================================================================
# CSV LOADERS
# ============================================================================

def load_danbooru_csv(path: str, index: TagIndex) -> int:
    """
    Load Danbooru-format CSV.
    Format: tag, category, count, aliases
    Returns count of loaded tags.
    """
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
# CONDUIT SYSTEM
# ============================================================================

class TagConduit:
    """
    Conduit system for injecting random top-N tags into prompts.

    This helps with generalization by exposing the model to common tags
    in random combinations during training.
    """

    def __init__(self, tag_index: TagIndex, config: ConduitConfig):
        self.config = config
        self.tag_index = tag_index

        # Build conduit pool from top N tags
        top_tags = tag_index.get_top_tags(
            n=config.top_n,
            category_filter=config.category_filter,
            exclude_categories=config.exclude_categories
        )

        self.pool: List[str] = [t[0] for t in top_tags]  # Just tag names
        self.pool_with_counts: List[Tuple[str, int]] = [(t[0], t[1]) for t in top_tags]

        # Build weighted pool if needed
        if config.weight_by_rank:
            # Higher rank (lower index) = more weight
            self.weighted_pool: List[str] = []
            for i, (tag, count) in enumerate(self.pool_with_counts):
                # Inverse rank weighting: rank 1 gets N weight, rank N gets 1 weight
                weight = max(1, len(self.pool) - i)
                self.weighted_pool.extend([tag] * int(weight ** 0.5))  # sqrt to reduce skew
        else:
            self.weighted_pool = self.pool

        print(f"[Conduit] Initialized with {len(self.pool)} tags from top {config.top_n}")
        if config.category_filter:
            print(f"[Conduit] Category filter: {config.category_filter}")
        if config.exclude_categories:
            print(f"[Conduit] Excluding: {config.exclude_categories}")

    def sample(self) -> List[str]:
        """Sample K random tags from the conduit pool."""
        if not self.pool:
            return []

        # Variable K for more variance
        if self.config.sample_k_min != self.config.sample_k_max:
            k = random.randint(self.config.sample_k_min, self.config.sample_k_max)
        else:
            k = self.config.sample_k

        k = min(k, len(self.weighted_pool))

        if self.config.weight_by_rank:
            # Sample from weighted pool (allows duplicates from weighting, then dedupe)
            sampled = random.sample(self.weighted_pool, min(k * 2, len(self.weighted_pool)))
            seen = set()
            result = []
            for tag in sampled:
                if tag not in seen:
                    seen.add(tag)
                    result.append(tag)
                    if len(result) >= k:
                        break
            return result
        else:
            return random.sample(self.pool, k)

    def inject(self, prompt: str, replace_underscores: bool = True) -> str:
        """Inject sampled tags into a prompt."""
        tags = self.sample()

        if replace_underscores:
            tags = [t.replace("_", " ") for t in tags]

        tag_str = ", ".join(tags)

        if self.config.position == "prepend":
            return f"{tag_str}, {prompt}"
        elif self.config.position == "append":
            return f"{prompt}, {tag_str}"
        elif self.config.position == "random":
            # Insert at random position
            parts = prompt.split(", ")
            insert_pos = random.randint(0, len(parts))
            parts.insert(insert_pos, tag_str)
            return ", ".join(parts)
        else:
            return f"{tag_str}, {prompt}"

    def get_pool_stats(self) -> Dict:
        """Get statistics about the conduit pool."""
        if not self.pool_with_counts:
            return {"size": 0}

        counts = [c for _, c in self.pool_with_counts]

        # Category breakdown
        categories = {}
        for tag in self.pool:
            data = self.tag_index.tag_dict.get(tag, {})
            cat = data.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "size": len(self.pool),
            "top_tag": self.pool[0] if self.pool else None,
            "top_count": counts[0] if counts else 0,
            "bottom_tag": self.pool[-1] if self.pool else None,
            "bottom_count": counts[-1] if counts else 0,
            "median_count": counts[len(counts) // 2] if counts else 0,
            "categories": categories,
        }


# ============================================================================
# TEMPLATE RENDERER
# ============================================================================

class TemplateRenderer:
    """Fast template rendering with pre-compiled patterns."""

    PATTERN = re.compile(r"\{(\w+):(\d+),?(\d+)?\}")

    def __init__(self, tag_index: TagIndex, templates: Dict[str, str],
                 preserve_tags: Optional[Set[str]] = None,
                 weight_by_post_count: bool = False,
                 use_coherent_gender: bool = True,
                 generate_t5_prefix: bool = True):
        self.tag_index = tag_index
        self.templates = templates
        self.template_names = list(templates.keys())
        self.preserve_tags = preserve_tags or (set(TAG_AESTHETIC_TYPES) | set(TAG_GENDER))
        self.weight_by_post_count = weight_by_post_count
        self.use_coherent_gender = use_coherent_gender
        self.generate_t5_prefix = generate_t5_prefix

        # Gender sampler
        self.gender_sampler = CoherentGenderSampler(include_count_prefix=generate_t5_prefix)

        # Pre-build tag pools
        self.pools: Dict[str, List[str]] = {
            "quality": TAG_AESTHETIC_TYPES.copy(),
            "gender": TAG_GENDER.copy(),  # Fallback pool
        }
        for category, tags in tag_index.category_dict.items():
            if weight_by_post_count:
                self.pools[category] = tag_index.get_weighted_tags(category)
            else:
                self.pools[category] = list(tags)

    def render(self, template_name: Optional[str] = None,
               dedupe: bool = True, replace_underscores: bool = True) -> Tuple[str, Optional[str], int]:
        """
        Render a single template.

        Returns:
            Tuple of (prompt, t5_prefix, people_count)
        """
        if template_name is None:
            template_name = random.choice(self.template_names)

        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        final_tags = []
        people_count = 0
        t5_prefix = None
        gender_already_sampled = False

        for match in self.PATTERN.finditer(template):
            cat, min_ct, max_ct = match.groups()
            min_ct = int(min_ct)
            max_ct = int(max_ct or min_ct)

            # Handle gender specially for coherent sampling
            if cat == "gender" and self.use_coherent_gender and not gender_already_sampled:
                gender_tags, people_count, t5_prefix = self.gender_sampler.sample()
                final_tags.extend(gender_tags)
                gender_already_sampled = True
                continue

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

        return ", ".join(processed), t5_prefix, people_count


# ============================================================================
# MAIN SYNTHESIZER
# ============================================================================

class BooruSynthesizer(CaptionBase):
    """
    Generates captions in booru tag style.

    Supports loading tags from CSV files (Danbooru, Gelbooru, e621, Rule34xxx)
    or from custom files (JSONL, TXT).

    Features:
    - Template-based generation
    - Conduit system for random top-N tag injection
    - Multi-threaded batch generation
    """

    def __init__(self, config: Optional[BooruConfig] = None):
        super().__init__(name="BooruSynthesizer", uid="prompt.booru_synthesizer")
        self.config = config or BooruConfig()
        self.tag_index = TagIndex()
        self.renderer: Optional[TemplateRenderer] = None
        self.conduit: Optional[TagConduit] = None
        self.hash_cache: Set[str] = set()

        if self.config.seed is not None:
            random.seed(self.config.seed)

        self._load_csvs()
        self._init_renderer()
        self._init_conduit()

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
            weight_by_post_count=self.config.weight_by_post_count,
            use_coherent_gender=self.config.use_coherent_gender,
            generate_t5_prefix=self.config.generate_t5_prefix,
        )

    def _init_conduit(self):
        """Initialize conduit system if configured."""
        if self.config.conduit and self.config.conduit.enabled:
            self.conduit = TagConduit(self.tag_index, self.config.conduit)
            print(f"[Conduit] Stats: {self.conduit.get_pool_stats()}")

    def enable_conduit(
            self,
            top_n: int = 1000,
            sample_k: int = 10,
            sample_k_min: int = 5,
            sample_k_max: int = 15,
            position: str = "prepend",
            category_filter: Optional[List[str]] = None,
            exclude_categories: Optional[List[str]] = None,
            weight_by_rank: bool = False
    ):
        """
        Enable or reconfigure the conduit system.

        Args:
            top_n: Pull from top N tags by post count
            sample_k: Sample K random tags per prompt (fixed)
            sample_k_min: Minimum K when using variable sampling
            sample_k_max: Maximum K when using variable sampling
            position: Where to inject tags ("prepend", "append", "random")
            category_filter: Only include these categories (None = all)
            exclude_categories: Exclude these categories
            weight_by_rank: Weight sampling by rank (top tags more likely)
        """
        self.config.conduit = ConduitConfig(
            enabled=True,
            top_n=top_n,
            sample_k=sample_k,
            sample_k_min=sample_k_min,
            sample_k_max=sample_k_max,
            position=position,
            category_filter=category_filter,
            exclude_categories=exclude_categories,
            weight_by_rank=weight_by_rank
        )
        self._init_conduit()

    def disable_conduit(self):
        """Disable the conduit system."""
        self.conduit = None
        if self.config.conduit:
            self.config.conduit.enabled = False

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

        # Rebuild conduit if enabled
        if self.conduit:
            self._init_conduit()

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

    def generate(self, template_name: Optional[str] = None,
                 apply_conduit: bool = True,
                 include_t5_prefix: bool = False) -> Union[str, Tuple[str, str, int]]:
        """
        Generate a single booru-style caption.

        Args:
            template_name: Specific template to use (None = random)
            apply_conduit: Whether to apply conduit injection (if enabled)
            include_t5_prefix: If True, returns (prompt, t5_prefix, people_count)

        Returns:
            If include_t5_prefix=False: prompt string
            If include_t5_prefix=True: (prompt, t5_prefix, people_count)
        """
        if self.renderer is None:
            raise RuntimeError("Renderer not initialized. Load tags first.")

        prompt, t5_prefix, people_count = self.renderer.render(
            template_name=template_name,
            dedupe=self.config.deduplicate,
            replace_underscores=self.config.replace_underscores
        )

        # Apply conduit if enabled
        if apply_conduit and self.conduit:
            prompt = self.conduit.inject(prompt, self.config.replace_underscores)

        if include_t5_prefix:
            return prompt, t5_prefix, people_count
        return prompt

    def generate_with_t5(self, template_name: Optional[str] = None,
                         apply_conduit: bool = True) -> Dict[str, any]:
        """
        Generate a caption with T5 prefix for training.

        Returns:
            Dict with keys:
                - 'clip': The prompt for CLIP (tags only)
                - 't5': The prompt for T5 (count prefix + tags)
                - 't5_prefix': Just the count prefix
                - 'people_count': Integer count
        """
        prompt, t5_prefix, people_count = self.generate(
            template_name=template_name,
            apply_conduit=apply_conduit,
            include_t5_prefix=True
        )

        # Build T5 prompt with count prefix
        t5_prompt = f"{t5_prefix}, {prompt}" if t5_prefix else prompt

        return {
            'clip': prompt,
            't5': t5_prompt,
            't5_prefix': t5_prefix,
            'people_count': people_count,
        }

    def generate_batch(self, count: int, deduplicate: bool = True,
                       template_name: Optional[str] = None,
                       apply_conduit: bool = True,
                       include_t5: bool = False) -> Union[List[str], List[Dict]]:
        """
        Generate a batch of captions.

        Args:
            count: Number of captions to generate
            deduplicate: Skip duplicate captions
            template_name: Specific template (None = random)
            apply_conduit: Apply conduit injection
            include_t5: Return dicts with CLIP/T5 versions

        Returns:
            List of strings (if include_t5=False) or List of dicts (if include_t5=True)
        """
        results = []

        while len(results) < count:
            if include_t5:
                result = self.generate_with_t5(
                    template_name=template_name,
                    apply_conduit=apply_conduit
                )
                caption = result['clip']  # Use CLIP prompt for deduplication
            else:
                caption = self.generate(template_name=template_name,
                                        apply_conduit=apply_conduit)
                result = caption

            if deduplicate:
                h = hashlib.md5(caption.encode()).hexdigest()
                if h in self.hash_cache:
                    continue
                self.hash_cache.add(h)

            results.append(result)

        return results

    def generate_iterator(self, count: int, deduplicate: bool = True,
                          apply_conduit: bool = True) -> Iterator[str]:
        """Generate captions as an iterator (memory efficient)."""
        generated = 0
        while generated < count:
            caption = self.generate(apply_conduit=apply_conduit)

            if deduplicate:
                h = hashlib.md5(caption.encode()).hexdigest()
                if h in self.hash_cache:
                    continue
                self.hash_cache.add(h)

            yield caption
            generated += 1

    def generate_to_file(self, path: str, count: int, format: str = "jsonl",
                         num_threads: int = 1, batch_size: int = 1000,
                         show_progress: bool = True, apply_conduit: bool = True,
                         include_t5: bool = False):
        """
        Generate captions and save to file.

        Args:
            path: Output file path
            count: Number of captions to generate
            format: 'jsonl', 'txt', or 'csv'
            num_threads: Number of threads for generation
            batch_size: Batch size for threaded generation
            show_progress: Show progress bar
            apply_conduit: Whether to apply conduit injection
            include_t5: Include T5 prefix in output (JSONL: separate fields)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if num_threads > 1:
            self._generate_threaded(path, count, format, num_threads,
                                    batch_size, show_progress, apply_conduit, include_t5)
        else:
            self._generate_single(path, count, format, show_progress, apply_conduit, include_t5)

    def _generate_single(self, path: Path, count: int, format: str,
                         show_progress: bool, apply_conduit: bool,
                         include_t5: bool = False):
        """Single-threaded generation."""
        iterator = range(count)
        if show_progress and HAS_TQDM:
            iterator = tqdm(iterator, desc="Generating", unit=" samples")

        with open(path, "w", encoding="utf-8") as f:
            for i in iterator:
                if include_t5:
                    result = self.generate_with_t5(apply_conduit=apply_conduit)
                    if format == "jsonl":
                        f.write(json.dumps({
                            "clip": result['clip'],
                            "t5": result['t5'],
                            "t5_prefix": result['t5_prefix'],
                            "people_count": result['people_count'],
                            "index": i
                        }) + "\n")
                    elif format == "txt":
                        # Write T5 version for txt
                        f.write(result['t5'] + "\n")
                    elif format == "csv":
                        f.write(f"{result['clip'].replace(',', ';')}\t{result['t5'].replace(',', ';')}\n")
                else:
                    caption = self.generate(apply_conduit=apply_conduit)
                    if format == "jsonl":
                        f.write(json.dumps({"text": caption, "index": i}) + "\n")
                    elif format == "txt":
                        f.write(caption + "\n")
                    elif format == "csv":
                        f.write(caption.replace(",", ";") + "\n")

    def _generate_threaded(self, path: Path, count: int, format: str,
                           num_threads: int, batch_size: int,
                           show_progress: bool, apply_conduit: bool,
                           include_t5: bool = False):
        """Multi-threaded generation."""
        lock = threading.Lock()
        sample_index = [0]

        def generate_batch_worker(size: int) -> List:
            if include_t5:
                return [self.generate_with_t5(apply_conduit=apply_conduit) for _ in range(size)]
            else:
                return [self.generate(apply_conduit=apply_conduit) for _ in range(size)]

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
                            for item in batch:
                                if include_t5:
                                    if format == "jsonl":
                                        f.write(json.dumps({
                                            "clip": item['clip'],
                                            "t5": item['t5'],
                                            "t5_prefix": item['t5_prefix'],
                                            "people_count": item['people_count'],
                                            "index": sample_index[0]
                                        }) + "\n")
                                    elif format == "txt":
                                        f.write(item['t5'] + "\n")
                                    elif format == "csv":
                                        f.write(f"{item['clip'].replace(',', ';')}\t{item['t5'].replace(',', ';')}\n")
                                else:
                                    if format == "jsonl":
                                        f.write(json.dumps({"text": item, "index": sample_index[0]}) + "\n")
                                    elif format == "txt":
                                        f.write(item + "\n")
                                    elif format == "csv":
                                        f.write(item.replace(",", ";") + "\n")
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

    def get_top_tags(self, n: int = 100, category: Optional[str] = None) -> List[Tuple[str, int]]:
        """
        Get top N tags by post count.

        Args:
            n: Number of tags to return
            category: Filter by category (None = all)

        Returns:
            List of (tag_name, post_count) tuples
        """
        cat_filter = [category] if category else None
        top = self.tag_index.get_top_tags(n=n, category_filter=cat_filter)
        return [(t[0], t[1]) for t in top]

    def stats(self) -> Dict:
        """Get synthesizer statistics."""
        result = {
            "total_tags": len(self.tag_index),
            "categories": {cat: len(tags) for cat, tags in self.tag_index.category_dict.items()},
            "sources": {src: len(tags) for src, tags in self.tag_index.source_tags.items()},
            "templates": len(self.get_template_names()),
            "cache_size": len(self.hash_cache),
        }

        if self.conduit:
            result["conduit"] = self.conduit.get_pool_stats()

        return result

    def __getitem__(self, key: str) -> str:
        """Generate using specific template."""
        return self.generate(template_name=key)

    def __len__(self) -> int:
        """Return number of loaded tags."""
        return len(self.tag_index)

    def __repr__(self):
        conduit_str = f", conduit={len(self.conduit.pool)}" if self.conduit else ""
        return (f"BooruSynthesizer(tags={len(self.tag_index)}, "
                f"categories={len(self.tag_index.category_dict)}, "
                f"templates={len(self.get_template_names())}{conduit_str})")


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


def demo_with_conduit(danbooru_path: str):
    """Demo with conduit system enabled."""
    print("=" * 60)
    print("Demo: BooruSynthesizer with Conduit")
    print("=" * 60)

    config = BooruConfig(
        danbooru_csv=danbooru_path,
        conduit=ConduitConfig(
            enabled=True,
            top_n=1000,
            sample_k=10,
            sample_k_min=5,
            sample_k_max=15,
            position="prepend",
            exclude_categories=["artist", "copyright", "character", "metadata"],
        ),
        seed=42
    )

    print("\nLoading tags...")
    synth = BooruSynthesizer(config)
    print(f"\n{synth}")

    print("\n--- Top 20 Tags in Conduit Pool ---")
    if synth.conduit:
        for i, (tag, count) in enumerate(synth.conduit.pool_with_counts[:20]):
            print(f"  {i + 1:2d}. {tag}: {count:,}")

    print("\n--- Sample Generations (with conduit) ---")
    for i in range(10):
        print(f"  {i + 1}. {synth.generate()}")

    print("\n--- Sample Generations (without conduit) ---")
    for i in range(5):
        print(f"  {i + 1}. {synth.generate(apply_conduit=False)}")

    return synth


def demo_t5_output(danbooru_path: str = None):
    """Demo T5 prefix generation with coherent gender sampling."""
    print("=" * 60)
    print("Demo: T5 People Count Prefixes")
    print("=" * 60)

    config = BooruConfig(
        danbooru_csv=danbooru_path,
        use_coherent_gender=True,
        generate_t5_prefix=True,
        seed=42
    )

    synth = BooruSynthesizer(config)
    print(f"\n{synth}")

    print("\n--- Sample Generations with T5 Prefixes ---")
    for i in range(15):
        result = synth.generate_with_t5()
        print(f"\n  {i + 1}.")
        print(f"     CLIP: {result['clip']}")
        print(f"     T5:   {result['t5']}")
        print(f"     Count: {result['people_count']} ({result['t5_prefix']})")

    print("\n--- Gender Scenario Distribution (100 samples) ---")
    counts = {}
    for _ in range(100):
        result = synth.generate_with_t5()
        n = result['people_count']
        counts[n] = counts.get(n, 0) + 1

    for n in sorted(counts.keys()):
        bar = "█" * (counts[n] // 2)
        print(f"  {n} people: {counts[n]:3d} {bar}")

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

    print("\n--- Top 20 Tags ---")
    for i, (tag, count) in enumerate(synth.get_top_tags(20)):
        print(f"  {i + 1:2d}. {tag}: {count:,}")

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
                     danbooru_csv: str = None,
                     enable_conduit: bool = False,
                     conduit_top_n: int = 1000,
                     conduit_sample_k: int = 10,
                     include_t5: bool = False):
    """Generate a dataset file."""
    print("=" * 60)
    print(f"Generate Dataset: {count:,} samples -> {output_path}")
    if include_t5:
        print("  T5 prefixes: ENABLED")
    print("=" * 60)

    conduit_config = None
    if enable_conduit:
        conduit_config = ConduitConfig(
            enabled=True,
            top_n=conduit_top_n,
            sample_k=conduit_sample_k,
            sample_k_min=5,
            sample_k_max=15,
            position="prepend",
            exclude_categories=["artist", "copyright", "character", "metadata"],
        )

    config = BooruConfig(
        danbooru_csv=danbooru_csv,
        conduit=conduit_config,
        use_coherent_gender=True,
        generate_t5_prefix=include_t5,
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
        show_progress=True,
        apply_conduit=enable_conduit,
        include_t5=include_t5
    )

    elapsed = time.time() - start
    print(f"\n✓ Generated {count:,} samples in {elapsed:.1f}s")
    print(f"  Rate: {count / elapsed:,.0f} samples/sec")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Booru Tag Synthesizer")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--demo-t5", action="store_true", help="Run T5 prefix demo")
    parser.add_argument("--benchmark", type=int, default=0, help="Run benchmark with N iterations")
    parser.add_argument("--generate", type=str, help="Generate dataset to path")
    parser.add_argument("--count", type=int, default=100000, help="Number of samples to generate")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--format", type=str, default="jsonl", choices=["jsonl", "txt", "csv"])
    parser.add_argument("--danbooru", type=str, help="Path to danbooru.csv")
    parser.add_argument("--gelbooru", type=str, help="Path to gelbooru.csv")
    parser.add_argument("--e621", type=str, help="Path to e621.csv")
    parser.add_argument("--rule34x", type=str, help="Path to rule34_xxx.csv")

    # Conduit options
    parser.add_argument("--conduit", action="store_true", help="Enable conduit system")
    parser.add_argument("--conduit-top-n", type=int, default=1000, help="Conduit: top N tags")
    parser.add_argument("--conduit-sample-k", type=int, default=10, help="Conduit: sample K tags")

    # T5 options
    parser.add_argument("--t5", action="store_true", help="Include T5 people count prefixes")

    args = parser.parse_args()

    if args.demo_t5:
        demo_t5_output(args.danbooru)

    elif args.demo:
        if args.conduit and args.danbooru:
            demo_with_conduit(args.danbooru)
        elif any([args.danbooru, args.gelbooru, args.e621, args.rule34x]):
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
            danbooru_csv=args.danbooru,
            enable_conduit=args.conduit,
            conduit_top_n=args.conduit_top_n,
            conduit_sample_k=args.conduit_sample_k,
            include_t5=args.t5
        )

    else:
        # Default: run demo
        demo_no_csvs()