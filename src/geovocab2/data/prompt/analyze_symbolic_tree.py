#!/usr/bin/env python3
"""
analyze_symbolic_tree.py
========================
Deep analysis of SYMBOLIC_TREE to find:
- Dead-ends (lists that don't exist)
- Unused lists (exist but not referenced)
- Missing routes (referenced in compatible/requires but no path)
- Circular dependencies
- Human-dependent paths (for filtering out later)
- Object-only viable paths
"""

import sys

sys.path.insert(0, '/mnt/user-data/uploads')

from bulk_caption_data import BulkCaptions
from collections import defaultdict
import re

# Extract the SYMBOLIC_TREE inline (since importing has dependencies)
SYMBOLIC_TREE = {
    "subject": {
        "_rules": {
            "activation": "primary_entity",
            "max_depth": 2,
            "min_depth": 1,
            "compatible_with": ["human", "context.environment", "context.depiction"],
            "required_count": 1,
            "sampling_strategy": "weighted_random",
            "noise_chance": 0.05,
        },
        "entities": {
            "_rules": {
                "activation": "natural_subjects",
                "sampling": "weighted_random",
                "nesting": ["context.environment.background", "context.depiction.composition"],
                "max_combined": 3,
                "exclusivity": True,
                "blacklist": [
                    ("animal", "human.attire.*"),
                    ("animal", "subject.identity.gender"),
                    ("produce", "human.expression.*"),
                    ("plant", "human.anatomy.pose"),
                ],
            },
            "animal": {
                "list": "ANIMAL_TYPES",
                "weight": 0.5,
                "compatible": ["human.expression.human_action", "context.environment.on_surface"],
                "avoid_with": ["human.attire", "human.anatomy"],
                "preferred_linkages": [
                    ("animal", "context.environment.background", 0.5),
                    ("animal", "context.depiction.composition.lighting", 0.7),
                    ("animal", "context.materiality.texture", 0.6),
                    ("animal", "shared.descriptors.color", 0.4),
                ],
            },
            "produce": {
                "list": "FRUIT_AND_VEGETABLE",
                "weight": 0.15,
                "compatible": ["context.environment.on_surface", "context.materiality.texture"],
                "boost_when_underused": True,
            },
            "plant": {
                "list": "PLANT_CATEGORY_TAGS",
                "weight": 0.15,
                "compatible": ["context.environment.background", "context.materiality.texture"],
                "boost_when_underused": True,
            },
        },
        "identity": {
            "_rules": {
                "activation": "humanoid_subjects",
                "sampling": "uniform_random",
                "nesting": ["human.anatomy", "human.attire", "human.expression"],
                "max_combined": 2,
                "requires_any": ["human"],
                "blacklist": [
                    ("object", "human.anatomy.*"),
                    ("object", "human.attire.*"),
                ],
            },
            "object": {
                "list": "OBJECT_TYPES",
                "weight": 0.3,
                "compatible": ["context.environment", "context.materiality"],
                "avoid_with": ["human.anatomy", "human.attire"],
            },
            "clothing": {
                "list": "CLOTHING_TYPES",
                "weight": 0.2,
                "compatible": ["human.attire", "context.materiality.fabric"],
            },
            "surface": {
                "list": "HUMAN_SURFACES",
                "weight": 0.3,
                "compatible": ["human.anatomy.pose", "context.environment"],
                "requires": ["subject.identity.gender"],
            },
            "gender": {
                "list": "GENDER_TYPES",
                "weight": 0.2,
                "compatible": ["human.*"],
                "required_for": ["human.anatomy", "human.attire", "human.expression"],
                "primary": True,
                "solidification": "anchor",
            },
        },
    },
    "human": {
        "_rules": {
            "activation": "requires_subject.identity.gender",
            "max_depth": 4,
            "min_depth": 1,
            "compatible_with": ["subject.identity", "context"],
            "incompatible_with": ["subject.entities.animal", "subject.entities.produce", "subject.entities.plant"],
            "sampling_strategy": "coherent_outfit",
            "noise_chance": 0.03,
        },
        "anatomy": {
            "_rules": {
                "activation": "human_present",
                "sampling": "context_aware",
                "nesting": ["human.attire", "human.expression"],
                "max_combined": 5,
                "blacklist": [
                    ("hair_style", "hair_style"),
                    ("pose", "pose"),
                ],
            },
            "pose": {"list": "HUMAN_POSES", "weight": 0.6},
            "hair_style": {"list": "HAIRSTYLES_TYPES", "weight": 0.35},
            "hair_length": {"list": "HAIR_LENGTH_TYPES", "weight": 0.50, "requires": ["human.anatomy.hair_style"]},
        },
        "attire": {
            "_rules": {"activation": "human_present", "max_combined": 4},
            "upper_clothing": {"list": "UPPER_BODY_CLOTHES_TYPES", "weight": 0.35},
            "lower_clothing": {"list": "LOWER_BODY_CLOTHES_TYPES", "weight": 0.25},
            "footwear": {"list": "FOOTWEAR_TYPES", "weight": 0.2},
            "accessory": {"list": "ACCESSORY_TYPES", "weight": 0.15},
            "jewelry": {"list": "JEWELRY_TYPES", "weight": 0.075},
            "headwear": {"list": "HEADWEAR_TYPES", "weight": 0.10},
        },
        "expression": {
            "_rules": {"activation": "human_present", "max_combined": 2},
            "human_expression": {"list": "HUMAN_EXPRESSIONS", "weight": 0.4},
            "human_action": {"list": "HUMAN_ACTIONS", "weight": 0.5},
            "human_interaction": {"list": "HUMAN_INTERACTIONS", "weight": 0.15},
        },
    },
    "context": {
        "_rules": {
            "activation": "always_available",
            "max_depth": 4,
            "compatible_with": ["subject", "human"],
            "sampling_strategy": "scene_coherent",
            "noise_chance": 0.1,
        },
        "environment": {
            "_rules": {"activation": "spatial_context", "max_combined": 3},
            "background": {"list": "BACKGROUND_TYPES", "weight": 0.2},
            "decoration": {"list": "DECORATION_TYPES", "weight": 0.2},
            "chair": {"list": "CHAIR_TYPES", "weight": 0.2, "requires": ["human.anatomy.pose"]},
            "on_surface": {"list": "HUMAN_SURFACES", "weight": 0.2},
        },
        "materiality": {
            "_rules": {"activation": "descriptive_detail", "max_combined": 3},
            "material": {"list": "MATERIAL_TYPES", "weight": 0.30},
            "fabric": {"list": "FABRIC_TYPES", "weight": 0.30},
            "texture": {"list": "TEXTURE_TAGS", "weight": 0.20},
            "pattern": {"list": "PATTERN_TYPES", "weight": 0.10},
            "liquid": {"list": "LIQUID_TYPES", "weight": 0.075},
        },
        "depiction": {
            "_rules": {"activation": "compositional_control", "max_depth": 2},
            "viewer": {
                "_rules": {"activation": "camera_perspective", "max_combined": 2},
                "grid": {"list": "GRID_TAGS", "weight": 0.4},
                "subject_angle": {"list": "SUBJECT_PHOTOGRAPH_ANGLE", "weight": 0.6},
                "focal_region": {"list": "FOCAL_REGION_TAGS", "weight": 0.3},
            },
            "object_view": {
                "_rules": {"activation": "spatial_relationships", "max_combined": 2},
                "zone": {"list": "ZONE_TAGS", "weight": 0.4},
                "object_left": {"list": "LEFT_OBJECT_TYPES", "weight": 0.3},
                "object_right": {"list": "RIGHT_OBJECT_TYPES", "weight": 0.3},
            },
            "composition": {
                "_rules": {"activation": "aesthetic_control", "max_combined": 3},
                "offset": {"list": "OFFSET_TAGS", "weight": 0.2},
                "shape": {"list": "SHAPE_TYPES", "weight": 0.15},
                "style": {"list": "STYLE_TYPES", "weight": 0.25},
                "lighting": {"list": "LIGHTING_TYPES", "weight": 0.3},
                "emotion": {"list": "EMOTION_TYPES", "weight": 0.1, "requires": ["human.expression"]},
            },
        },
    },
    "shared": {
        "_rules": {"activation": "always_available", "max_depth": 1},
        "descriptors": {
            "_rules": {"activation": "detail_enhancement", "max_combined": 3},
            "prefix": {"list": "PREFIXES", "weight": 0.2},
            "suffix": {"list": "SUFFIX_TAGS", "weight": 0.15},
            "color": {"list": "COLORS", "weight": 0.25},
            "size": {"list": "SIZE", "weight": 0.15},
            "scope": {"list": "SCOPE", "weight": 0.1},
            "quality": {"list": "QUALITY_IMPROVERS", "weight": 0.1},
            "degradation": {"list": "QUALITY_REDUCERS", "weight": 0.05},
        },
    },
}


def get_all_bulk_caption_lists():
    """Get all list names from BulkCaptions"""
    return {k for k, v in vars(BulkCaptions).items()
            if isinstance(v, list) and not k.startswith('_')}


def extract_tree_info(tree, path=""):
    """Extract all paths, list references, requires, compatible from tree"""
    paths = {}
    list_refs = {}
    requires = {}
    compatible = {}
    requires_human = set()

    def traverse(subtree, current_path):
        if not isinstance(subtree, dict):
            return

        # Check if this is a leaf with a list
        if "list" in subtree:
            list_refs[current_path] = subtree["list"]
            paths[current_path] = {
                "list": subtree.get("list"),
                "weight": subtree.get("weight", 1.0),
                "requires": subtree.get("requires", []),
                "compatible": subtree.get("compatible", []),
            }

            # Check for human dependencies
            req = subtree.get("requires", [])
            comp = subtree.get("compatible", [])
            for item in req + comp:
                if "human" in item or "gender" in item:
                    requires_human.add(current_path)

            if req:
                requires[current_path] = req
            if comp:
                compatible[current_path] = comp

        # Check _rules for dependencies
        if "_rules" in subtree:
            rules = subtree["_rules"]
            if "requires_any" in rules or "activation" in rules:
                act = rules.get("activation", "")
                if "human" in act or "gender" in act:
                    requires_human.add(current_path)

        # Recurse
        for key, value in subtree.items():
            if key.startswith("_"):
                continue
            if not isinstance(value, dict):
                continue
            # Skip metadata keys
            if key in ["list", "weight", "compatible", "requires", "preferred_linkages",
                       "grammar_role", "solidification", "primary", "avoid_with",
                       "avoid_simultaneous", "boost_when_underused", "order", "required_for",
                       "prepositional"]:
                continue
            new_path = f"{current_path}.{key}" if current_path else key
            traverse(value, new_path)

    traverse(tree, "")
    return paths, list_refs, requires, compatible, requires_human


def analyze():
    print("=" * 80)
    print("SYMBOLIC TREE ANALYSIS")
    print("=" * 80)

    bulk_lists = get_all_bulk_caption_lists()
    paths, list_refs, requires, compatible, requires_human = extract_tree_info(SYMBOLIC_TREE)

    # 1. Dead-ends: lists referenced but don't exist
    print("\n" + "=" * 80)
    print("1. DEAD-ENDS (lists referenced in tree but don't exist in BulkCaptions)")
    print("-" * 80)
    dead_ends = []
    for path, list_name in list_refs.items():
        if list_name not in bulk_lists:
            dead_ends.append((path, list_name))
            print(f"  ❌ {path}")
            print(f"     References: {list_name} (MISSING)")
    if not dead_ends:
        print("  ✓ No dead-ends found")

    # 2. Unused lists: exist but not referenced
    print("\n" + "=" * 80)
    print("2. UNUSED LISTS (exist in BulkCaptions but not referenced in tree)")
    print("-" * 80)
    referenced_lists = set(list_refs.values())
    unused = bulk_lists - referenced_lists
    # Filter out obvious non-data lists
    ignore = {'BEATRIX_SPECIAL_TOKENS', 'BEATRIX_SHUNTS', 'BEATRIX_SPECIAL_TOKENS_AND_SHUNTS',
              'SYMBOLIC_TEMPLATE', 'SYMBOLIC_TEMPLATE_ADVANCED', 'FULL_ASSOCIATIVE',
              'FULL_GENERIC', 'MATHEMATICAL_FORMULA_TEMPLATES', 'SYMBOLIC_CONJUNCTION_TEMPLATES',
              'SYMBOLIC_CONJUNCTION_TAGS', 'SYMBOLIC_CONJUNCTION_EXTENSION_TAGS',
              'CATEGORICAL_TEMPLATES', 'IRREGULARS', 'BLACKLIST'}
    unused = unused - ignore
    for lst in sorted(unused):
        count = len(getattr(BulkCaptions, lst, []))
        print(f"  ⚠ {lst:40s} ({count:,} items)")
    if not unused:
        print("  ✓ All lists are referenced")

    # 3. Human-dependent paths
    print("\n" + "=" * 80)
    print("3. HUMAN-DEPENDENT PATHS (require human/gender)")
    print("-" * 80)

    # Manually identify human-dependent paths
    human_paths = set()
    for path in paths.keys():
        if path.startswith("human."):
            human_paths.add(path)
        elif path.startswith("subject.identity.gender"):
            human_paths.add(path)
        elif path.startswith("subject.identity.surface"):  # requires gender
            human_paths.add(path)
        elif path.startswith("subject.identity.clothing"):  # human clothing
            human_paths.add(path)

    # Check requires dependencies
    for path, req_list in requires.items():
        for req in req_list:
            if "human" in req or "gender" in req:
                human_paths.add(path)

    for path in sorted(human_paths):
        print(f"  👤 {path}")

    # 4. Object-only viable paths (no human dependency)
    print("\n" + "=" * 80)
    print("4. OBJECT-ONLY VIABLE PATHS (can be used without humans)")
    print("-" * 80)
    object_paths = set(paths.keys()) - human_paths
    for path in sorted(object_paths):
        list_name = list_refs.get(path, "?")
        count = len(getattr(BulkCaptions, list_name, []))
        print(f"  📦 {path:50s} -> {list_name} ({count:,})")

    # 5. Broken references in compatible/requires
    print("\n" + "=" * 80)
    print("5. BROKEN REFERENCES (paths in compatible/requires that don't exist)")
    print("-" * 80)
    all_paths_set = set(paths.keys())
    broken = []

    for path, comp_list in compatible.items():
        for comp in comp_list:
            # Handle wildcards
            if "*" in comp:
                pattern = comp.replace(".", r"\.").replace("*", ".*")
                if not any(re.match(pattern, p) for p in all_paths_set):
                    broken.append((path, "compatible", comp))
            else:
                # Check if it's a prefix match
                if not any(p.startswith(comp) for p in all_paths_set):
                    broken.append((path, "compatible", comp))

    for path, req_list in requires.items():
        for req in req_list:
            if "*" in req:
                pattern = req.replace(".", r"\.").replace("*", ".*")
                if not any(re.match(pattern, p) for p in all_paths_set):
                    broken.append((path, "requires", req))
            else:
                if not any(p.startswith(req) for p in all_paths_set):
                    broken.append((path, "requires", req))

    for path, ref_type, ref_value in broken:
        print(f"  ⚠ {path}")
        print(f"     {ref_type}: {ref_value} (NOT FOUND)")
    if not broken:
        print("  ✓ All references are valid")

    # 6. Summary for object association dataset
    print("\n" + "=" * 80)
    print("6. SUMMARY FOR OBJECT ASSOCIATION DATASET")
    print("-" * 80)

    # Categories useful for objects
    object_categories = {
        "subject.entities.animal": "Animals",
        "subject.entities.produce": "Fruits/Vegetables",
        "subject.entities.plant": "Plants",
        "subject.identity.object": "Objects",
        "context.environment.background": "Backgrounds",
        "context.environment.decoration": "Decorations",
        "context.environment.on_surface": "Surfaces",
        "context.materiality.material": "Materials",
        "context.materiality.fabric": "Fabrics",
        "context.materiality.texture": "Textures",
        "context.materiality.pattern": "Patterns",
        "context.materiality.liquid": "Liquids",
        "context.depiction.viewer.grid": "Grid positions",
        "context.depiction.viewer.subject_angle": "Camera angles",
        "context.depiction.viewer.focal_region": "Focal regions",
        "context.depiction.object_view.zone": "Zones",
        "context.depiction.composition.offset": "Offsets",
        "context.depiction.composition.shape": "Shapes",
        "context.depiction.composition.style": "Styles",
        "context.depiction.composition.lighting": "Lighting",
        "shared.descriptors.prefix": "Prefixes",
        "shared.descriptors.suffix": "Suffixes",
        "shared.descriptors.color": "Colors",
        "shared.descriptors.size": "Sizes",
        "shared.descriptors.scope": "Scope",
        "shared.descriptors.quality": "Quality+",
        "shared.descriptors.degradation": "Quality-",
    }

    total_items = 0
    print("\n  AVAILABLE FOR OBJECT CAPTIONS:")
    for path, desc in object_categories.items():
        list_name = list_refs.get(path)
        if list_name and list_name in bulk_lists:
            count = len(getattr(BulkCaptions, list_name, []))
            total_items += count
            print(f"    {desc:25s} {path:45s} {count:>6,}")
        elif list_name:
            print(f"    {desc:25s} {path:45s} MISSING: {list_name}")

    print(f"\n  Total items available: {total_items:,}")

    # 7. Recommended fixes
    print("\n" + "=" * 80)
    print("7. RECOMMENDED FIXES")
    print("-" * 80)

    print("""
  CRITICAL:
    1. Create LEFT_OBJECT_TYPES list (or alias to OBJECT_TYPES)
    2. Create RIGHT_OBJECT_TYPES list (or alias to OBJECT_TYPES)

  UNUSED LISTS TO CONSIDER:
    - RELATION_TAGS: Could be useful for object associations
    - SYMBOLIC_LOGIC_TAGS: Relational operators
    - ASSOCIATIVE_LOGICAL_TAGS: More relations
    - INTENT_TYPES: Scene intent
    - SUBJECT_TYPES: Generic subjects
    - HUMANOID_TYPES: Could filter for non-human entities

  FOR OBJECT ASSOCIATION DATASET:
    - Strip out entire "human" branch
    - Strip out subject.identity.gender, surface, clothing
    - Keep context.environment.chair but remove human.pose requirement
    - Remove emotion (requires human.expression)
    - Add RELATION_TAGS for "X next_to Y", "X on Y", etc.
    """)

    return {
        "dead_ends": dead_ends,
        "unused": unused,
        "human_paths": human_paths,
        "object_paths": object_paths,
        "broken": broken,
    }


if __name__ == "__main__":
    results = analyze()