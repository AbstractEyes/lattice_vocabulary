#!/usr/bin/env python3
"""
inspect_bulk_captions.py
========================
Quick introspection of BulkCaptions data structures.
Shows all lists, dicts, their sizes, and keys.

Usage: python inspect_bulk_captions.py
"""

import sys

#sys.path.insert(0, '/mnt/user-data/uploads')

from bulk_caption_data import BulkCaptions, BEATRIX_V2_SYMBOLIC_CATEGORY_TREE


def main():
    print("=" * 70)
    print("BULK CAPTIONS DATA INTROSPECTION")
    print("=" * 70)

    # Get all class attributes
    attrs = {k: v for k, v in vars(BulkCaptions).items()
             if not k.startswith('_') and not callable(v)}

    lists = {}
    dicts = {}
    other = {}

    for name, val in attrs.items():
        if isinstance(val, list):
            lists[name] = val
        elif isinstance(val, dict):
            dicts[name] = val
        else:
            other[name] = val

    # Summary
    total_items = sum(len(v) for v in lists.values())
    total_dict_items = sum(len(v) for v in dicts.values())

    print(f"\nSUMMARY")
    print("-" * 70)
    print(f"  Lists:       {len(lists):>4}  ({total_items:,} total items)")
    print(f"  Dicts:       {len(dicts):>4}  ({total_dict_items:,} total keys)")
    print(f"  Other:       {len(other):>4}")
    print()

    # Lists sorted by size
    print("LISTS (sorted by count)")
    print("-" * 70)
    for name, val in sorted(lists.items(), key=lambda x: -len(x[1])):
        print(f"  {name:45s} {len(val):>6,} items")
    print()

    # Dicts with keys
    print("DICTS (with keys)")
    print("-" * 70)
    for name, val in sorted(dicts.items()):
        print(f"  {name} ({len(val)} keys):")
        keys = list(val.keys())[:10]
        for k in keys:
            v = val[k]
            if isinstance(v, list):
                print(f"    - {k}: [{len(v)} items]")
            elif isinstance(v, dict):
                print(f"    - {k}: {{{len(v)} keys}}")
            else:
                preview = str(v)[:40] + "..." if len(str(v)) > 40 else str(v)
                print(f"    - {k}: {preview}")
        if len(val) > 10:
            print(f"    ... and {len(val) - 10} more keys")
        print()

    # Category tree
    print("BEATRIX_V2_SYMBOLIC_CATEGORY_TREE")
    print("-" * 70)

    def print_tree(d, indent=0):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{'  ' * indent}{k}/")
                print_tree(v, indent + 1)
            elif isinstance(v, list):
                print(f"{'  ' * indent}{k}: [{', '.join(v)}]")
            else:
                print(f"{'  ' * indent}{k}: {v}")

    print_tree(BEATRIX_V2_SYMBOLIC_CATEGORY_TREE)
    print()

    # Quick lookup dict for the symbolic tree
    print("QUICK LOOKUP: List name -> Category path")
    print("-" * 70)
    mapping = {
        'ANIMAL_TYPES': 'subject.entities.animal',
        'FRUIT_AND_VEGETABLE': 'subject.entities.produce',
        'PLANT_CATEGORY_TAGS': 'subject.entities.plant',
        'OBJECT_TYPES': 'subject.identity.object',
        'CLOTHING_TYPES': 'subject.identity.clothing',
        'HUMAN_SURFACES': 'subject.identity.surface',
        'GENDER_TYPES': 'subject.identity.gender',
        'HUMAN_POSES': 'human.anatomy.pose',
        'HAIRSTYLES_TYPES': 'human.anatomy.hair_style',
        'HAIR_LENGTH_TYPES': 'human.anatomy.hair_length',
        'UPPER_BODY_CLOTHES_TYPES': 'human.attire.upper_clothing',
        'LOWER_BODY_CLOTHES_TYPES': 'human.attire.lower_clothing',
        'FOOTWEAR_TYPES': 'human.attire.footwear',
        'ACCESSORY_TYPES': 'human.attire.accessory',
        'JEWELRY_TYPES': 'human.attire.jewelry',
        'HEADWEAR_TYPES': 'human.attire.headwear',
        'HUMAN_EXPRESSIONS': 'human.expression.human_expression',
        'HUMAN_ACTIONS': 'human.expression.human_action',
        'HUMAN_INTERACTIONS': 'human.expression.human_interaction',
        'BACKGROUND_TYPES': 'context.environment.background',
        'DECORATION_TYPES': 'context.environment.decoration',
        'CHAIR_TYPES': 'context.environment.chair',
        'MATERIAL_TYPES': 'context.materiality.material',
        'FABRIC_TYPES': 'context.materiality.fabric',
        'TEXTURE_TAGS': 'context.materiality.texture',
        'PATTERN_TYPES': 'context.materiality.pattern',
        'LIQUID_TYPES': 'context.materiality.liquid',
        'GRID_TAGS': 'context.depiction.viewer.grid',
        'SUBJECT_PHOTOGRAPH_ANGLE': 'context.depiction.viewer.subject_angle',
        'FOCAL_REGION_TAGS': 'context.depiction.viewer.focal_region',
        'ZONE_TAGS': 'context.depiction.object_view.zone',
        'OFFSET_TAGS': 'context.depiction.composition.offset',
        'SHAPE_TYPES': 'context.depiction.composition.shape',
        'STYLE_TYPES': 'context.depiction.composition.style',
        'LIGHTING_TYPES': 'context.depiction.composition.lighting',
        'EMOTION_TYPES': 'context.depiction.composition.emotion',
        'PREFIXES': 'shared.descriptors.prefix',
        'SUFFIX_TAGS': 'shared.descriptors.suffix',
        'COLORS': 'shared.descriptors.color',
        'SIZE': 'shared.descriptors.size',
        'SCOPE': 'shared.descriptors.scope',
        'QUALITY_IMPROVERS': 'shared.descriptors.quality',
        'QUALITY_REDUCERS': 'shared.descriptors.degradation',
    }
    for list_name, path in sorted(mapping.items()):
        count = len(getattr(BulkCaptions, list_name, []))
        print(f"  {list_name:35s} -> {path:45s} ({count:,})")
    print()

    # Template counts
    print("TEMPLATE COLLECTIONS")
    print("-" * 70)
    templates = [
        ('SYMBOLIC_TEMPLATE', 'Basic templates'),
        ('SYMBOLIC_TEMPLATE_ADVANCED', 'Advanced multi-subject'),
        ('FULL_ASSOCIATIVE', 'Associative patterns'),
        ('FULL_GENERIC', 'Generic scene templates'),
        ('MATHEMATICAL_FORMULA_TEMPLATES', 'Math formulas'),
        ('SYMBOLIC_CONJUNCTION_TEMPLATES', 'Conjunction patterns'),
    ]
    for name, desc in templates:
        val = getattr(BulkCaptions, name, [])
        print(f"  {name:40s} {len(val):>6,}  # {desc}")
    print()

    # Categorical templates
    print("CATEGORICAL_TEMPLATES keys:")
    print("-" * 70)
    ct = BulkCaptions.CATEGORICAL_TEMPLATES
    for key in sorted(ct.keys()):
        print(f"  {key:30s} {len(ct[key]):>4} templates")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()