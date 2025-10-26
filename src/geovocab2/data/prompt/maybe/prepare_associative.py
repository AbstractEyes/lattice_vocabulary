"""
Template Extraction - FULL_ASSOCIATIVE → FULL_GENERIC
======================================================

Run once to generate hierarchical templates from FULL_ASSOCIATIVE.
Handles special cases: gender terms, multi-word phrases, compound tags.

Usage:
    python extract_templates.py

Output:
    FULL_GENERIC list ready to paste into BulkCaptions.py

Author: Phi + Claude
Date: 2025-10-26
"""

import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

from geovocab2.data.prompt.bulk_caption_data import BulkCaptions


# ============================================================================
# SPECIAL CASE DEFINITIONS
# ============================================================================

GENDER_TERMS = {
    'man', 'woman', 'male', 'female', 'boy', 'girl',
    'men', 'women', 'males', 'females', 'boys', 'girls'
}

# Multi-word phrases (longest first for greedy matching)
MULTI_WORD_PHRASES = [
    # Composition (3-4 words)
    "diagonal upper-left to bottom-right", "diagonal upper-right to bottom-left",
    "horizontal slice upper", "horizontal slice center", "horizontal slice lower",
    "vertical slice left", "vertical slice center", "vertical slice right",
    "rule of thirds", "golden ratio", "center point",
    "diagonal composition", "symmetrical layout", "asymmetrical balance",
    "triangle apex top", "triangle base left", "triangle base right",
    "cone axis center", "cone wide spread", "cone narrow spread",
    "spotlight center", "spotlight upper", "spotlight lower",

    # Grids/quadrants (2-3 words)
    "5x5 grid", "6x6 grid", "7x7 grid", "8x8 grid", "9x9 grid",
    "frame top", "frame bottom", "frame left", "frame right",
    "quadrant 1", "quadrant 2", "quadrant 3", "quadrant 4",
    "hex sector 1", "hex sector 2", "hex sector 3", "hex sector 4", "hex sector 5", "hex sector 6",

    # Rules (3 words)
    "rule of 3", "rule of 5", "rule of 7", "rule of 9",
    "rule of 11", "rule of 13", "rule of 15", "rule of 17", "rule of 19",
    "rule of 21", "rule of 23", "rule of 25", "rule of 27", "rule of 29",

    # Focal regions (2-3 words)
    "top left", "top center", "top right", "middle left", "middle right",
    "bottom left", "bottom center", "bottom right",
    "upper third", "middle third", "lower third",
    "left edge", "right edge", "top edge", "bottom edge",
    "inner ring", "middle ring", "outer ring",

    # Size/verbs (2 words)
    "extra small", "extra large", "put on", "take off",

    # Logic (2-3 words)
    "for all", "there exists", "set union", "set intersection", "set difference",
    "set complement", "set power set", "set subset", "set superset",
]

COMPOUND_TAG_PATTERNS = [
    r'focal_[a-z_]+',
    r'grid_[a-z0-9]+',
    r'[0-9]+x[0-9]+_grid',
]

STRUCTURAL_WORDS = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
    'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'and', 'or', 'but', 'if', 'then', 'than', 'such', 'which', 'that',
    'this', 'these', 'those', 'it', 'its', "it's", 'their', 'there',
    'over', 'under', 'beneath', 'above', 'below', 'beside', 'between',
    'through', 'during', 'before', 'after', 'while', 'until', 'since',
    'into', 'onto', 'upon', 'off', 'out', 'up', 'down', 'around', 'about',
    'across', 'along', 'among', 'behind', 'beyond', 'near', 'next',
    'towards', 'toward', 'within', 'without', 'against', 'inside', 'outside',
}

CATEGORY_TO_TREE_PATH = {
    # subject
    'ANIMAL_TYPES': 'subject.animal',
    'HUMANOID_TYPES': 'subject.humanoid',
    'SUBJECT_TYPES': 'subject.entity',
    'FRUIT_AND_VEGETABLE': 'subject.produce',
    'PLANT_CATEGORY_TAGS': 'subject.plant',
    'OBJECT_TYPES': 'subject.object',
    'CLOTHING_TYPES': 'subject.clothing',

    # human.anatomy
    'HUMAN_POSES': 'human.anatomy.pose',
    'HAIRSTYLES_TYPES': 'human.anatomy.hairstyle',
    'HAIR_LENGTH_TYPES': 'human.anatomy.hair_length',
    'GENDER_TYPES': 'human.anatomy.gender',
    'MALE_TAGS': 'human.anatomy.male',
    'FEMALE_TAGS': 'human.anatomy.female',
    'AMBIG_TAGS': 'human.anatomy.ambiguous',

    # human.attire
    'UPPER_BODY_CLOTHES_TYPES': 'human.attire.upper_clothing',
    'LOWER_BODY_CLOTHES_TYPES': 'human.attire.lower_clothing',
    'FOOTWEAR_TYPES': 'human.attire.footwear',
    'SOCK_TYPES': 'human.attire.socks',
    'ACCESSORY_TYPES': 'human.attire.accessory',
    'JEWELRY_TYPES': 'human.attire.jewelry',
    'HEADWEAR_TYPES': 'human.attire.headwear',
    'FABRIC_TYPES': 'human.attire.fabric',

    # human.expression
    'EMOTION_TYPES': 'human.expression.emotion',
    'HUMAN_ACTIONS': 'human.expression.action',
    'HUMAN_INTERACTIONS': 'human.expression.interaction',
    'HUMAN_EXPRESSIONS': 'human.expression.facial',

    # context.environment
    'BACKGROUND_TYPES': 'context.environment.background',
    'DECORATION_TYPES': 'context.environment.decoration',
    'CHAIR_TYPES': 'context.environment.furniture',
    'HUMAN_SURFACES': 'context.environment.surface',
    'CLOTHING_SURFACE_LINKERS': 'context.environment.surface_linker',

    # context.materiality
    'MATERIAL_TYPES': 'context.materiality.material',
    'TEXTURE_TAGS': 'context.materiality.texture',
    'PATTERN_TYPES': 'context.materiality.pattern',
    'LIQUID_TYPES': 'context.materiality.liquid',

    # context.depiction
    'GRID_TAGS': 'context.depiction.grid',
    'SUBJECT_PHOTOGRAPH_ANGLE': 'context.depiction.angle',
    'FOCAL_REGION_TAGS': 'context.depiction.region',
    'ZONE_TAGS': 'context.depiction.zone',
    'OFFSET_TAGS': 'context.depiction.offset',
    'SHAPE_TYPES': 'context.depiction.shape',
    'STYLE_TYPES': 'context.depiction.style',
    'LIGHTING_TYPES': 'context.depiction.lighting',
    'INTENT_TYPES': 'context.depiction.intent',

    # shared
    'PREFIXES': 'shared.modifier.prefix',
    'SUFFIX_TAGS': 'shared.modifier.suffix',
    'COLORS': 'shared.descriptor.color',
    'SIZE': 'shared.descriptor.size',
    'SCOPE': 'shared.descriptor.scope',
    'QUALITY_IMPROVERS': 'shared.descriptor.quality_plus',
    'QUALITY_REDUCERS': 'shared.descriptor.quality_minus',
    'ADJECTIVES': 'shared.descriptor.adjective',
    'ADVERBS': 'shared.descriptor.adverb',
    'VERBS': 'shared.descriptor.verb',

    # symbolic.logic
    'RELATION_TAGS': 'symbolic.logic.relation',
    'SYMBOLIC_CONJUNCTION_TAGS': 'symbolic.logic.conjunction',
    'ASSOCIATIVE_LOGICAL_TAGS': 'symbolic.logic.associative',
    'SYMBOLIC_LOGIC_TAGS': 'symbolic.logic.operator',
}


# ============================================================================
# TEMPLATE EXTRACTOR
# ============================================================================

class TemplateExtractor:

    def __init__(self, bulk_data: Dict[str, List[str]]):
        self.bulk_data = bulk_data
        self.multi_word_phrases = sorted(MULTI_WORD_PHRASES, key=lambda x: -len(x.split()))
        self.compound_patterns = [re.compile(p) for p in COMPOUND_TAG_PATTERNS]

        # Build indices
        self.word_to_categories = self._build_word_index()
        self.phrase_to_categories = self._build_phrase_index()

        # Stats
        self.stats = {
            'total_processed': 0,
            'total_replacements': 0,
            'gender_replacements': 0,
            'multi_word_replacements': 0,
            'compound_tag_replacements': 0,
            'categories_used': Counter(),
            'tree_paths_used': Counter(),
            'unmapped_words': Counter(),
        }

    def _build_word_index(self) -> Dict[str, List[str]]:
        word_index = defaultdict(list)
        for category_name, items in self.bulk_data.items():
            for item in items:
                words = self._normalize(item).split()
                for word in words:
                    if word not in STRUCTURAL_WORDS:
                        word_index[word].append(category_name)
                        stem = self._stem(word)
                        if stem != word:
                            word_index[stem].append(category_name)
        return word_index

    def _build_phrase_index(self) -> Dict[str, List[str]]:
        phrase_index = defaultdict(list)
        for category_name, items in self.bulk_data.items():
            for item in items:
                phrase_index[self._normalize(item)].append(category_name)
        return phrase_index

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _stem(self, word: str) -> str:
        for suffix in ['ing', 'ed', 'es', 's', 'ly', 'er', 'est']:
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                return word[:-len(suffix)]
        return word

    def _find_multi_word(self, words: List[str], idx: int) -> Optional[Tuple[str, int]]:
        for phrase in self.multi_word_phrases:
            phrase_words = phrase.split()
            phrase_len = len(phrase_words)
            if idx + phrase_len <= len(words):
                candidate = ' '.join(words[idx:idx + phrase_len])
                if candidate == phrase:
                    return (phrase, phrase_len)
        return None

    def _find_match(self, text: str) -> Optional[Tuple[str, str]]:
        # Try phrase match
        if text in self.phrase_to_categories:
            candidates = self.phrase_to_categories[text]
            if candidates:
                cat = candidates[0]
                path = CATEGORY_TO_TREE_PATH.get(cat)
                if path:
                    return (cat, path)

        # Try word match
        candidates = self.word_to_categories.get(text, [])
        if not candidates:
            stem = self._stem(text)
            candidates = self.word_to_categories.get(stem, [])

        if not candidates:
            return None

        # Priority order
        priority = [
            'UPPER_BODY_CLOTHES_TYPES', 'LOWER_BODY_CLOTHES_TYPES',
            'FOOTWEAR_TYPES', 'HEADWEAR_TYPES', 'HAIRSTYLES_TYPES',
            'HUMAN_POSES', 'HUMAN_ACTIONS', 'FOCAL_REGION_TAGS',
            'ANIMAL_TYPES', 'HUMANOID_TYPES',
        ]

        for priority_cat in priority:
            if priority_cat in candidates:
                path = CATEGORY_TO_TREE_PATH.get(priority_cat)
                if path:
                    return (priority_cat, path)

        for cat in candidates:
            path = CATEGORY_TO_TREE_PATH.get(cat)
            if path:
                return (cat, path)

        return None

    def extract(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        self.stats['total_processed'] += 1

        words_lower = [self._normalize(w) for w in text.split()]
        result_tokens = []
        replacements = defaultdict(list)

        i = 0
        original_words = text.split()

        while i < len(original_words):
            word = original_words[i]
            word_lower = self._normalize(word)

            # Skip structural
            if word_lower in STRUCTURAL_WORDS or not word_lower.strip():
                result_tokens.append(word)
                i += 1
                continue

            # Check gender
            if word_lower in GENDER_TERMS:
                placeholder = "{GENDER}"
                result_tokens.append(placeholder)
                replacements[placeholder].append(word)
                self.stats['gender_replacements'] += 1
                self.stats['total_replacements'] += 1
                i += 1
                continue

            # Check compound tags
            if any(p.match(word_lower) for p in self.compound_patterns):
                match = self._find_match(word_lower)
                if match:
                    cat, path = match
                    placeholder = f"{{{path}}}"
                    result_tokens.append(placeholder)
                    replacements[placeholder].append(word)
                    self.stats['compound_tag_replacements'] += 1
                    self.stats['total_replacements'] += 1
                    self.stats['categories_used'][cat] += 1
                    self.stats['tree_paths_used'][path] += 1
                else:
                    result_tokens.append(word)
                i += 1
                continue

            # Check multi-word
            phrase_match = self._find_multi_word(words_lower, i)
            if phrase_match:
                phrase, num_words = phrase_match
                original_phrase = ' '.join(original_words[i:i+num_words])
                match = self._find_match(phrase)

                if match:
                    cat, path = match
                    placeholder = f"{{{path}}}"
                    result_tokens.append(placeholder)
                    replacements[placeholder].append(original_phrase)
                    self.stats['multi_word_replacements'] += 1
                    self.stats['total_replacements'] += 1
                    self.stats['categories_used'][cat] += 1
                    self.stats['tree_paths_used'][path] += 1
                else:
                    result_tokens.append(original_phrase)

                i += num_words
                continue

            # Regular match
            match = self._find_match(word_lower)
            if match:
                cat, path = match
                placeholder = f"{{{path}}}"
                result_tokens.append(placeholder)
                replacements[placeholder].append(word)
                self.stats['total_replacements'] += 1
                self.stats['categories_used'][cat] += 1
                self.stats['tree_paths_used'][path] += 1
            else:
                result_tokens.append(word)
                self.stats['unmapped_words'][word_lower] += 1

            i += 1

        template = ' '.join(result_tokens)
        template = re.sub(r'\s+', ' ', template).strip()

        return template, dict(replacements)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("TEMPLATE EXTRACTION: FULL_ASSOCIATIVE → FULL_GENERIC")
    print("=" * 80)
    print()

    # Load BulkCaptions
    bulk_data = {}
    for attr_name in dir(BulkCaptions):
        if attr_name.isupper() and not attr_name.startswith('_'):
            attr = getattr(BulkCaptions, attr_name)
            if isinstance(attr, (list, tuple)):
                if attr_name not in ('FULL_GENERIC', 'FULL_ASSOCIATIVE'):
                    bulk_data[attr_name] = list(attr)

    print(f"Loaded {len(bulk_data)} BulkCaptions categories")

    # Extract
    FULL_ASSOCIATIVE = BulkCaptions.FULL_ASSOCIATIVE
    print(f"Processing {len(FULL_ASSOCIATIVE)} FULL_ASSOCIATIVE items")
    print()

    extractor = TemplateExtractor(bulk_data)

    templates = []
    seen = set()

    for text in FULL_ASSOCIATIVE:
        template, _ = extractor.extract(text)
        if template not in seen:
            templates.append(template)
            seen.add(template)

    # Stats
    stats = extractor.stats
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Original items:           {len(FULL_ASSOCIATIVE)}")
    print(f"Unique templates:         {len(templates)}")
    print(f"Duplicates removed:       {len(FULL_ASSOCIATIVE) - len(templates)}")
    print(f"Total replacements:       {stats['total_replacements']}")
    print(f"  Gender:                 {stats['gender_replacements']}")
    print(f"  Multi-word phrases:     {stats['multi_word_replacements']}")
    print(f"  Compound tags:          {stats['compound_tag_replacements']}")
    print()

    print("Top 10 tree paths used:")
    for path, count in stats['tree_paths_used'].most_common(10):
        print(f"  {path:40} {count:4d}")
    print()

    # Output
    print("=" * 80)
    print("FULL_GENERIC (paste into BulkCaptions.py)")
    print("=" * 80)
    print()
    print("FULL_GENERIC = [")
    for template in templates:
        escaped = template.replace('"', '\\"')
        print(f'    "{escaped}",')
    print("]")
    print()

    # Save
    with open("./FULL_GENERIC.py", 'w') as f:
        f.write("# Generated FULL_GENERIC templates\n")
        f.write("# Paste into BulkCaptions class\n\n")
        f.write("FULL_GENERIC = [\n")
        for template in templates:
            escaped = template.replace('"', '\\"')
            f.write(f'    "{escaped}",\n')
        f.write("]\n")

    print("Saved to: /FULL_GENERIC.py")


if __name__ == "__main__":
    main()