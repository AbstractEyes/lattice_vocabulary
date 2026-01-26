#!/usr/bin/env python3
"""
object_association_dsl.py
=========================
Simple DSL for generating object association captions.
Designed to be clean, scriptable, and LLM-batch-friendly.

NO humans, NO complex rule trees, just objects and their relationships.

Grammar:
  {object}              -> random object
  {object:category}     -> object from specific category (animal, produce, plant, item)
  {relation}            -> spatial/logical relation
  {color}               -> color descriptor
  {size}                -> size descriptor
  {material}            -> material type
  {texture}             -> texture descriptor
  {pattern}             -> pattern type
  {background}          -> background/setting
  {lighting}            -> lighting condition
  {style}               -> artistic style
  {angle}               -> camera angle
  {zone}                -> spatial zone
  {quality+}            -> quality enhancer
  {quality-}            -> quality reducer

Operators:
  |                     -> OR (pick one)
  ?                     -> optional (50% chance)
  *N                    -> repeat N times

Examples:
  "{color} {object} {relation} {object}"
  -> "red apple next to wooden chair"

  "{size?} {color?} {object:animal} on {material} {background}"
  -> "small orange cat on marble kitchen counter"

Author: Phil + Claude
"""

import sys

sys.path.insert(0, '/mnt/user-data/uploads')

import random
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import the data
from bulk_caption_data import BulkCaptions

# ============================================================================
# CATEGORY MAPPING - Clean subset for objects only
# ============================================================================

CATEGORIES = {
    # Primary objects
    "object": "OBJECT_TYPES",
    "item": "OBJECT_TYPES",
    "animal": "ANIMAL_TYPES",
    "produce": "FRUIT_AND_VEGETABLE",
    "fruit": "FRUIT_AND_VEGETABLE",
    "vegetable": "FRUIT_AND_VEGETABLE",
    "plant": "PLANT_CATEGORY_TAGS",

    # Relations (THE KEY FOR ASSOCIATIONS)
    "relation": "RELATION_TAGS",
    "rel": "RELATION_TAGS",

    # Descriptors
    "color": "COLORS",
    "size": "SIZE",
    "prefix": "PREFIXES",
    "suffix": "SUFFIX_TAGS",
    "quality+": "QUALITY_IMPROVERS",
    "quality-": "QUALITY_REDUCERS",
    "scope": "SCOPE",

    # Materials & textures
    "material": "MATERIAL_TYPES",
    "mat": "MATERIAL_TYPES",
    "texture": "TEXTURE_TAGS",
    "tex": "TEXTURE_TAGS",
    "pattern": "PATTERN_TYPES",
    "fabric": "FABRIC_TYPES",
    "liquid": "LIQUID_TYPES",

    # Environment
    "background": "BACKGROUND_TYPES",
    "bg": "BACKGROUND_TYPES",
    "decoration": "DECORATION_TYPES",
    "decor": "DECORATION_TYPES",
    "surface": "HUMAN_SURFACES",  # Actually general surfaces
    "chair": "CHAIR_TYPES",  # Furniture

    # Composition
    "lighting": "LIGHTING_TYPES",
    "light": "LIGHTING_TYPES",
    "style": "STYLE_TYPES",
    "angle": "SUBJECT_PHOTOGRAPH_ANGLE",
    "grid": "GRID_TAGS",
    "zone": "ZONE_TAGS",
    "offset": "OFFSET_TAGS",
    "shape": "SHAPE_TYPES",
    "focal": "FOCAL_REGION_TAGS",

    # Logic/symbolic (for complex associations)
    "logic": "SYMBOLIC_LOGIC_TAGS",
    "intent": "INTENT_TYPES",

    # Adjectives/verbs (for natural language)
    "adj": "ADJECTIVES",
    "adv": "ADVERBS",
    "verb": "VERBS",
}

# Aliases for common compound categories
CATEGORY_ALIASES = {
    "thing": ["object", "animal", "produce", "plant"],
    "entity": ["object", "animal", "produce", "plant"],
    "descriptor": ["color", "size", "texture"],
    "attr": ["color", "size", "material", "texture"],
}


# ============================================================================
# DATA CACHE
# ============================================================================

class DataCache:
    """Cached access to BulkCaptions lists"""
    _cache: Dict[str, List[str]] = {}

    @classmethod
    def get(cls, list_name: str) -> List[str]:
        if list_name not in cls._cache:
            data = getattr(BulkCaptions, list_name, [])
            cls._cache[list_name] = data if isinstance(data, list) else []
        return cls._cache[list_name]

    @classmethod
    def sample(cls, list_name: str, n: int = 1) -> List[str]:
        data = cls.get(list_name)
        if not data:
            return [f"<{list_name}>"]
        return random.choices(data, k=n)

    @classmethod
    def get_one(cls, list_name: str) -> str:
        return cls.sample(list_name, 1)[0]


# ============================================================================
# DSL PARSER
# ============================================================================

@dataclass
class Token:
    """Parsed token from template"""
    raw: str
    category: str
    subcategory: Optional[str] = None
    optional: bool = False
    repeat: int = 1


def parse_token(token_str: str) -> Token:
    """
    Parse a {token} into its components.

    Formats:
      {category}           -> basic token
      {category:sub}       -> category with subcategory filter
      {category?}          -> optional (50% chance)
      {category*3}         -> repeat 3 times
      {category?*3}        -> optional, if included repeat 3 times
    """
    # Strip braces
    inner = token_str.strip("{}")

    optional = "?" in inner
    inner = inner.replace("?", "")

    repeat = 1
    if "*" in inner:
        parts = inner.split("*")
        inner = parts[0]
        repeat = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1

    subcategory = None
    if ":" in inner:
        category, subcategory = inner.split(":", 1)
    else:
        category = inner

    return Token(
        raw=token_str,
        category=category.lower(),
        subcategory=subcategory.lower() if subcategory else None,
        optional=optional,
        repeat=repeat,
    )


def resolve_category(category: str, subcategory: Optional[str] = None) -> str:
    """Resolve category name to BulkCaptions list name"""

    # Handle aliases (pick random from list)
    if category in CATEGORY_ALIASES:
        category = random.choice(CATEGORY_ALIASES[category])

    # Handle subcategory filter
    if subcategory:
        # e.g., {object:animal} -> use ANIMAL_TYPES
        if subcategory in CATEGORIES:
            return CATEGORIES[subcategory]

    # Direct lookup
    if category in CATEGORIES:
        return CATEGORIES[category]

    # Fallback - try as direct list name
    if hasattr(BulkCaptions, category.upper()):
        return category.upper()

    return "OBJECT_TYPES"  # Default fallback


def render_token(token: Token) -> str:
    """Render a single token to text"""

    # Handle optional
    if token.optional and random.random() < 0.5:
        return ""

    list_name = resolve_category(token.category, token.subcategory)

    # Handle repeat
    items = DataCache.sample(list_name, token.repeat)

    return " ".join(items)


# ============================================================================
# TEMPLATE ENGINE
# ============================================================================

TOKEN_PATTERN = re.compile(r'\{[^}]+\}')


def render_template(template: str, seed: Optional[int] = None) -> str:
    """
    Render a template string into a caption.

    Args:
        template: Template with {tokens}
        seed: Optional random seed for reproducibility

    Returns:
        Rendered caption string
    """
    if seed is not None:
        random.seed(seed)

    def replace_token(match):
        token_str = match.group(0)
        token = parse_token(token_str)
        result = render_token(token)
        return result

    # First pass: resolve all tokens
    result = TOKEN_PATTERN.sub(replace_token, template)

    # Clean up: remove double spaces, trim
    result = re.sub(r'\s+', ' ', result).strip()

    # Clean up articles: "a apple" -> "an apple"
    result = re.sub(r'\ba ([aeiou])', r'an \1', result)

    return result


def render_batch(template: str, n: int, seed: Optional[int] = None) -> List[str]:
    """Render a template N times with unique outputs"""
    results = []
    seen = set()

    if seed is not None:
        random.seed(seed)

    attempts = 0
    max_attempts = n * 10

    while len(results) < n and attempts < max_attempts:
        caption = render_template(template)
        if caption not in seen:
            seen.add(caption)
            results.append(caption)
        attempts += 1

    return results


# ============================================================================
# TEMPLATE LIBRARY - Object Association Patterns
# ============================================================================

OBJECT_ASSOCIATION_TEMPLATES = {
    # Basic two-object relations
    "basic_relation": [
        "{object} {relation} {object}",
        "a {object} {relation} a {object}",
        "{color} {object} {relation} {object}",
        "{object} {relation} {color} {object}",
    ],

    # Attributed objects
    "attributed": [
        "{size} {color} {object} {relation} {object}",
        "{material} {object} {relation} {texture} {object}",
        "{adj} {object} {relation} {adj} {object}",
    ],

    # With context/background
    "contextual": [
        "{object} {relation} {object} on {surface}",
        "{object} {relation} {object} in {background}",
        "{object} and {object} on {material} {surface}",
        "{object} {relation} {object}, {lighting}",
    ],

    # Multi-object scenes
    "multi_object": [
        "{object} {relation} {object} {relation} {object}",
        "{object}, {object}, and {object} on {surface}",
        "group of {object}*3 arranged on {surface}",
    ],

    # Category-specific
    "animal_scene": [
        "{animal} {relation} {object}",
        "{color} {animal} on {surface}",
        "{animal} and {animal} {relation} each other",
    ],

    "produce_still_life": [
        "{produce} {relation} {produce}",
        "{color} {produce} on {material} {surface}",
        "arrangement of {produce}*3",
    ],

    "plant_scene": [
        "{plant} in {object}",
        "{plant} {relation} {object}",
        "{color} {plant} on {surface}",
    ],

    # Styled/artistic
    "stylized": [
        "{style} {object} {relation} {object}",
        "{object} {relation} {object}, {style}, {lighting}",
        "{angle}, {object} {relation} {object}",
    ],

    # Complex compositions
    "complex": [
        "{focal}, {size} {object} {relation} {object} on {material} {background}, {lighting}, {style}",
        "{angle}, {color} {object} {relation} {color} {object}, {pattern} {background}",
    ],
}


def get_random_template(category: Optional[str] = None) -> str:
    """Get a random template, optionally from a specific category"""
    if category and category in OBJECT_ASSOCIATION_TEMPLATES:
        return random.choice(OBJECT_ASSOCIATION_TEMPLATES[category])

    # Pick from all categories
    all_templates = []
    for templates in OBJECT_ASSOCIATION_TEMPLATES.values():
        all_templates.extend(templates)
    return random.choice(all_templates)


# ============================================================================
# BATCH GENERATION FOR LLM PROCESSING
# ============================================================================

def generate_skeleton_batch(n: int, categories: Optional[List[str]] = None) -> List[str]:
    """
    Generate N skeleton captions (with placeholders).
    These are meant to be sent to an LLM for refinement.
    """
    skeletons = []

    for _ in range(n):
        if categories:
            cat = random.choice(categories)
            template = get_random_template(cat)
        else:
            template = get_random_template()

        skeletons.append(render_template(template))

    return skeletons


def generate_llm_batch_prompt(captions: List[str], task: str = "refine") -> str:
    """
    Generate a prompt for batch LLM processing.

    Args:
        captions: List of skeleton captions
        task: "refine", "expand", "vary", or "validate"
    """

    tasks = {
        "refine": """Refine these object association captions to be more natural and descriptive.
Keep them focused on objects and their spatial relationships (no humans).
Return one refined caption per line.""",

        "expand": """Expand these captions with more detail about materials, textures, and lighting.
Keep them focused on objects only (no humans or characters).
Return one expanded caption per line.""",

        "vary": """Generate 3 variations of each caption, varying the specific objects while keeping the relationship structure.
Keep them focused on objects only (no humans).
Return all variations, one per line.""",

        "validate": """Check these captions for:
1. No human/character references
2. Grammatical correctness
3. Clear spatial relationships
Mark each as VALID or INVALID with reason.""",
    }

    prompt = f"""{tasks.get(task, tasks["refine"])}

CAPTIONS:
{chr(10).join(f"{i + 1}. {c}" for i, c in enumerate(captions))}
"""
    return prompt


# ============================================================================
# MAIN - Demo and testing
# ============================================================================

def main():
    print("=" * 70)
    print("OBJECT ASSOCIATION DSL - Demo")
    print("=" * 70)

    # Show available categories
    print("\nAVAILABLE CATEGORIES:")
    print("-" * 70)
    for cat, list_name in sorted(CATEGORIES.items()):
        count = len(DataCache.get(list_name))
        print(f"  {{{cat}:25s}}  -> {list_name:30s} ({count:,} items)")

    # Demo templates
    print("\n" + "=" * 70)
    print("TEMPLATE CATEGORIES:")
    print("-" * 70)
    for cat_name, templates in OBJECT_ASSOCIATION_TEMPLATES.items():
        print(f"\n  {cat_name}:")
        for t in templates[:2]:
            print(f"    {t}")

    # Generate samples
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS:")
    print("-" * 70)

    for cat_name in OBJECT_ASSOCIATION_TEMPLATES.keys():
        print(f"\n  {cat_name}:")
        for _ in range(3):
            template = get_random_template(cat_name)
            caption = render_template(template)
            print(f"    - {caption}")

    # LLM batch example
    print("\n" + "=" * 70)
    print("LLM BATCH EXAMPLE:")
    print("-" * 70)

    skeletons = generate_skeleton_batch(5)
    prompt = generate_llm_batch_prompt(skeletons, "refine")
    print(prompt)

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()