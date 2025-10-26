"""
Convert Legacy Placeholders to Tree-Based Hierarchical System
==============================================================

This script converts all legacy flat placeholders like {gender}, {pose}, {upper_clothing}
to the new tree-based hierarchical system like {shared.gender}, {human.anatomy.pose},
{human.attire.upper_clothing}.

Usage:
    python convert_legacy_placeholders.py

Output:
    - Mapping of legacy → tree paths
    - Converted template lists ready to paste into bulk_caption_data.py
    - Statistics on conversions

Author: Phi + Claude
Date: 2025-10-26
"""

import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

from bulk_caption_data import BulkCaptions

# ============================================================================
# LOAD SYMBOLIC TREE STRUCTURE
# ============================================================================

# This is the BEATRIX_V2_SYMBOLIC_CATEGORY_TREE from symbolic_tree.py
SYMBOLIC_TREE = {
    'subject': {
        'animal': 'ANIMAL_TYPES',
        'humanoid': 'HUMANOID_TYPES',
        'produce': 'FRUIT_AND_VEGETABLE',
        'plant': 'PLANT_CATEGORY_TAGS',
        'subject': 'SUBJECT_TYPES',
        'object': 'OBJECT_TYPES',
        'clothing': 'CLOTHING_TYPES',
    },
    'human': {
        'anatomy': {
            'pose': 'HUMAN_POSES',
            'body_part': 'HUMAN_BODY_PARTS',
            'eye': 'EYE_TYPES',
            'hair_style': 'HAIRSTYLES_TYPES',
            'hair_length': 'HAIR_LENGTH_TYPES',
        },
        'attire': {
            'upper_clothing': 'UPPER_BODY_CLOTHES_TYPES',
            'lower_clothing': 'LOWER_BODY_CLOTHES_TYPES',
            'full_body': 'FULL_BODY_CLOTHES_TYPES',
            'footwear': 'FOOTWEAR_TYPES',
            'hosiery': 'HOSIERY_TYPES',
            'sock': 'SOCK_TYPES',
            'undergarment': 'UNDERGARMENT_TYPES',
            'accessory': 'ACCESSORY_TYPES',
            'jewelry': 'JEWELRY_TYPES',
            'headwear': 'HEADWEAR_TYPES',
        },
        'expression': {
            'emotion': 'EMOTION_TYPES',
            'action': 'HUMAN_ACTIONS',
            'interaction': 'HUMAN_INTERACTIONS',
            'expression': 'HUMAN_EXPRESSIONS',
        },
    },
    'context': {
        'environment': {
            'background': 'BACKGROUND_TYPES',
            'decoration': 'DECORATION_TYPES',
            'furniture': 'CHAIR_TYPES',
            'surface': 'HUMAN_SURFACES',
            'object_left': 'LEFT_OBJECT_TYPES',
            'object_right': 'RIGHT_OBJECT_TYPES',
        },
        'materiality': {
            'material': 'MATERIAL_TYPES',
            'fabric': 'FABRIC_TYPES',
            'texture': 'TEXTURE_TAGS',
            'pattern': 'PATTERN_TYPES',
            'liquid': 'LIQUID_TYPES',
        },
        'depiction': {
            'grid': 'GRID_TAGS',
            'angle': 'SUBJECT_PHOTOGRAPH_ANGLE',
            'region': 'FOCAL_REGION_TAGS',
            'zone': 'ZONE_TAGS',
            'offset': 'OFFSET_TAGS',
            'shape': 'SHAPE_TYPES',
            'style': 'STYLE_TYPES',
            'lighting': 'LIGHTING_TYPES',
            'intent': 'INTENT_TYPES',
        },
    },
    'shared': {
        'descriptors': {
            'prefix': 'PREFIXES',
            'suffix': 'SUFFIX_TAGS',
            'color': 'COLORS',
            'size': 'SIZE',
            'scope': 'SCOPE',
            'quality_plus': 'QUALITY_IMPROVERS',
            'quality_minus': 'QUALITY_REDUCERS',
            'adjective': 'ADJECTIVES',
            'adverb': 'ADVERBS',
            'verb': 'VERBS',
        },
        'gender': {
            'male': 'MALE_TAGS',
            'female': 'FEMALE_TAGS',
            'ambiguous': 'AMBIG_TAGS',
            'gender': 'GENDER_TYPES',
        },
    },
    'symbolic': {
        'operators': {
            'logic': 'LOGIC_TAGS',
            'conjunction': 'CONJUNCTION_TAGS',
            'smooth': 'SMOOTH_TAGS',
        },
    },
}


# ============================================================================
# BUILD MAPPING: BulkCaptions List Name → Tree Path
# ============================================================================

def build_list_to_path_mapping() -> Dict[str, str]:
    """
    Build mapping from BulkCaptions list names to their tree paths.

    Returns:
        Dict mapping list name (e.g., 'HUMAN_POSES') to tree path (e.g., 'human.anatomy.pose')
    """
    mapping = {}

    def traverse(tree, path=""):
        for key, value in tree.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, str):
                # Leaf node - this is a list name
                mapping[value] = current_path
            elif isinstance(value, dict):
                # Continue traversing
                traverse(value, current_path)

    traverse(SYMBOLIC_TREE)
    return mapping


# ============================================================================
# BUILD REVERSE MAPPING: Legacy Placeholder → Tree Path
# ============================================================================

def build_legacy_to_tree_mapping(list_to_path: Dict[str, str]) -> Dict[str, str]:
    """
    Build mapping from legacy placeholder names to tree paths.

    Example:
        'pose' → 'human.anatomy.pose'
        'upper_clothing' → 'human.attire.upper_clothing'
        'lighting' → 'context.depiction.lighting'

    Returns:
        Dict mapping legacy name to tree path
    """
    legacy_mapping = {}

    for list_name, tree_path in list_to_path.items():
        # Extract the leaf name from the tree path
        leaf_name = tree_path.split('.')[-1]
        legacy_mapping[leaf_name] = tree_path

        # Also map common variations
        # e.g., HUMAN_POSES → 'pose' or 'poses'
        if list_name.startswith('HUMAN_'):
            short_name = list_name.replace('HUMAN_', '').lower()
            if short_name.endswith('s'):
                legacy_mapping[short_name[:-1]] = tree_path  # singular
            legacy_mapping[short_name] = tree_path

        # Handle other naming patterns
        common_name = list_name.replace('_TYPES', '').replace('_TAGS', '').lower()
        if common_name != leaf_name:
            legacy_mapping[common_name] = tree_path

    # Add special cases that don't follow patterns
    special_cases = {
        'hair_style': 'human.anatomy.hair_style',
        'hairstyle': 'human.anatomy.hair_style',
        'hair_length': 'human.anatomy.hair_length',
        'object_left': 'context.environment.object_left',
        'object_right': 'context.environment.object_right',
    }
    legacy_mapping.update(special_cases)

    return legacy_mapping


# ============================================================================
# EXTRACT PLACEHOLDERS FROM TEMPLATE
# ============================================================================

def extract_placeholders(template: str) -> Set[str]:
    """
    Extract all {placeholder} names from a template string.

    Example:
        "a {gender} wearing {upper_clothing}" → {'gender', 'upper_clothing'}
    """
    pattern = r'\{([a-zA-Z_]+)\}'
    matches = re.findall(pattern, template)
    return set(matches)


# ============================================================================
# CONVERT TEMPLATE TO TREE-BASED
# ============================================================================

def convert_template(template: str, legacy_to_tree: Dict[str, str]) -> Tuple[str, List[str]]:
    """
    Convert a template from legacy placeholders to tree-based.

    Args:
        template: Template string with legacy placeholders
        legacy_to_tree: Mapping of legacy names to tree paths

    Returns:
        Tuple of (converted_template, list_of_unmapped_placeholders)
    """
    placeholders = extract_placeholders(template)
    unmapped = []
    converted = template

    for placeholder in placeholders:
        if placeholder in legacy_to_tree:
            tree_path = legacy_to_tree[placeholder]
            # Replace {placeholder} with {tree.path.placeholder}
            converted = converted.replace(
                f'{{{placeholder}}}',
                f'{{{tree_path}}}'
            )
        else:
            unmapped.append(placeholder)

    return converted, unmapped


# ============================================================================
# CONVERT TEMPLATE LISTS
# ============================================================================

def convert_template_list(templates: List[str],
                          legacy_to_tree: Dict[str, str],
                          list_name: str = "TEMPLATES") -> Dict:
    """
    Convert an entire list of templates.

    Returns:
        Dict with:
            - converted_templates: List of converted templates
            - unmapped_placeholders: Counter of unmapped placeholders
            - conversion_rate: Percentage of successfully mapped placeholders
    """
    converted_templates = []
    all_unmapped = Counter()
    total_placeholders = 0
    mapped_placeholders = 0

    for template in templates:
        converted, unmapped = convert_template(template, legacy_to_tree)
        converted_templates.append(converted)

        # Track statistics
        placeholders = extract_placeholders(template)
        total_placeholders += len(placeholders)
        mapped_placeholders += len(placeholders) - len(unmapped)

        for u in unmapped:
            all_unmapped[u] += 1

    conversion_rate = (mapped_placeholders / total_placeholders * 100) if total_placeholders > 0 else 0

    return {
        'list_name': list_name,
        'converted_templates': converted_templates,
        'unmapped_placeholders': dict(all_unmapped),
        'total_placeholders': total_placeholders,
        'mapped_placeholders': mapped_placeholders,
        'conversion_rate': conversion_rate,
    }


# ============================================================================
# FORMAT OUTPUT
# ============================================================================

def format_as_python_list(templates: List[str], name: str = "CONVERTED_TEMPLATES") -> str:
    """Format templates as Python list ready to paste."""
    lines = [f"{name} = ["]
    for template in templates:
        escaped = template.replace('"', '\\"')
        lines.append(f'    "{escaped}",')
    lines.append("]")
    return '\n'.join(lines)


def format_statistics(results: List[Dict]) -> str:
    """Format conversion statistics."""
    lines = ["=" * 80]
    lines.append("CONVERSION STATISTICS")
    lines.append("=" * 80)
    lines.append("")

    total_templates = sum(len(r['converted_templates']) for r in results)
    total_placeholders = sum(r['total_placeholders'] for r in results)
    total_mapped = sum(r['mapped_placeholders'] for r in results)
    overall_rate = (total_mapped / total_placeholders * 100) if total_placeholders > 0 else 0

    lines.append(f"Total template lists processed: {len(results)}")
    lines.append(f"Total templates converted:      {total_templates}")
    lines.append(f"Total placeholders found:       {total_placeholders}")
    lines.append(f"Successfully mapped:            {total_mapped} ({overall_rate:.1f}%)")
    lines.append("")

    lines.append("Per-list breakdown:")
    for r in results:
        lines.append(
            f"  {r['list_name']:30} {r['conversion_rate']:5.1f}% ({r['mapped_placeholders']}/{r['total_placeholders']})")

    lines.append("")
    lines.append("Unmapped placeholders across all lists:")
    all_unmapped = Counter()
    for r in results:
        all_unmapped.update(r['unmapped_placeholders'])

    if all_unmapped:
        for placeholder, count in all_unmapped.most_common(20):
            lines.append(f"  {placeholder:30} {count:4d} occurrences")
    else:
        lines.append("  (none - 100% success!)")

    return '\n'.join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main conversion script."""

    print("Building tree mappings...")
    list_to_path = build_list_to_path_mapping()
    legacy_to_tree = build_legacy_to_tree_mapping(list_to_path)

    print(f"Found {len(list_to_path)} BulkCaptions lists mapped to tree paths")
    print(f"Created {len(legacy_to_tree)} legacy → tree mappings")
    print()

    # Print some example mappings
    print("Example mappings:")
    example_keys = ['pose', 'upper_clothing', 'lighting', 'gender', 'emotion', 'surface']
    for key in example_keys:
        if key in legacy_to_tree:
            print(f"  {{{key}}} → {{{legacy_to_tree[key]}}}")
    print()

    # Example template lists to convert
    # (In production, these would be loaded from bulk_caption_data.py)

    SYMBOLIC_TEMPLATE_ADVANCED = [
        "a {gender} wearing {upper_clothing}, {footwear}, and {headwear}, standing {offset} on a {surface} under {lighting}",
        "a {gender} {pose} in a {zone} with {texture} walls and a {material} bench beside them",
        "a {gender} seen from {offset}, wearing {upper_clothing} and {accessory}, standing near a {object_right} with a {pattern} backdrop",
        "a {gender} resting on a {surface} patterned with {pattern}, framed by {lighting} and {hair_style} hair",
        "a {gender} with {hair_length} {hair_style} hair, wearing {upper_clothing}, a {jewelry}, and {footwear}, facing a {material} sculpture",
        "a {gender} positioned in the {zone}, framed by {texture} panels and {grid} layout, adjusting their {headwear}",
        "a {gender} in a {pattern} dress with {footwear}, standing beside a {object_right} on a {surface} under {lighting}",
        "a {gender} crouched near a {grid} edge, wearing {upper_clothing}, {jewelry}, and clutching a {material} {object_right}",
        "a {gender} seen through {lighting} mist, resting against a {fabric} draped chair while holding a {accessory}",
    ]

    SYMBOLIC_TEMPLATE_BASIC = [
        "a {gender} {pose}",
        "a {gender} wearing {upper_clothing}",
        "a {gender} standing on a {surface}",
        "{lighting} {style} photograph",
        "a {gender} with {emotion} expression",
    ]

    # Convert all template lists
    print("=" * 80)
    print("CONVERTING TEMPLATE LISTS")
    print("=" * 80)
    print()

    results = []

    # Convert SYMBOLIC_TEMPLATE_ADVANCED
    print("Converting SYMBOLIC_TEMPLATE_ADVANCED...")
    result_advanced = convert_template_list(
        SYMBOLIC_TEMPLATE_ADVANCED,
        legacy_to_tree,
        "SYMBOLIC_TEMPLATE_ADVANCED"
    )
    results.append(result_advanced)

    # Convert SYMBOLIC_TEMPLATE_BASIC
    print("Converting SYMBOLIC_TEMPLATE_BASIC...")
    result_basic = convert_template_list(
        SYMBOLIC_TEMPLATE_BASIC,
        legacy_to_tree,
        "SYMBOLIC_TEMPLATE_BASIC"
    )
    results.append(result_basic)

    # Print statistics
    print()
    print(format_statistics(results))
    print()

    # Output converted templates
    print("=" * 80)
    print("CONVERTED TEMPLATES (ready to paste into bulk_caption_data.py)")
    print("=" * 80)
    print()

    for result in results:
        print(format_as_python_list(result['converted_templates'], result['list_name']))
        print()

    # Save to file
    output_file = "./converted_templates.py"
    with open(output_file, 'w') as f:
        f.write("# Converted Templates - Tree-Based Hierarchical Placeholders\n")
        f.write("# Generated by convert_legacy_placeholders.py\n")
        f.write("# Date: 2025-10-26\n\n")

        for result in results:
            f.write(format_as_python_list(result['converted_templates'], result['list_name']))
            f.write("\n\n")

        f.write("# Statistics:\n")
        for line in format_statistics(results).split('\n'):
            f.write(f"# {line}\n")

    print(f"Saved to: {output_file}")
    print()

    # Save mapping for reference
    mapping_file = "./legacy_to_tree_mapping.py"
    with open(mapping_file, 'w') as f:
        f.write("# Legacy Placeholder → Tree Path Mapping\n")
        f.write("# Generated by convert_legacy_placeholders.py\n\n")
        f.write("LEGACY_TO_TREE_MAPPING = {\n")
        for legacy, tree in sorted(legacy_to_tree.items()):
            f.write(f"    '{legacy}': '{tree}',\n")
        f.write("}\n")

    print(f"Saved mapping to: {mapping_file}")


if __name__ == "__main__":
    main()