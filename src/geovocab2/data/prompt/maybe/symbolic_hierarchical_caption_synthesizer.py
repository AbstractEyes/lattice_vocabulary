"""
Enhanced Hierarchical Template Synthesizer with Special Cases
==============================================================

Handles special placeholders:
- {GENDER} - Maps to MALE_TAGS + FEMALE_TAGS combined
- Multi-word phrases preserved during synthesis
- Compound tags handled correctly

Author: Phi + Claude
Date: 2025-10-26
"""

from __future__ import annotations

import random
import re
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass

# ============================================================================
# TREE PATH TO CATEGORY MAPPING (with special GENDER handling)
# ============================================================================

TREE_PATH_TO_CATEGORY = {
    # subject
    'subject.animal': 'ANIMAL_TYPES',
    'subject.humanoid': 'HUMANOID_TYPES',
    'subject.entity': 'SUBJECT_TYPES',
    'subject.produce': 'FRUIT_AND_VEGETABLE',
    'subject.plant': 'PLANT_CATEGORY_TAGS',
    'subject.object': 'OBJECT_TYPES',
    'subject.clothing': 'CLOTHING_TYPES',

    # human.anatomy
    'human.anatomy.pose': 'HUMAN_POSES',
    'human.anatomy.hairstyle': 'HAIRSTYLES_TYPES',
    'human.anatomy.hair_length': 'HAIR_LENGTH_TYPES',
    'human.anatomy.gender': 'GENDER_TYPES',
    'human.anatomy.male': 'MALE_TAGS',
    'human.anatomy.female': 'FEMALE_TAGS',
    'human.anatomy.ambiguous': 'AMBIG_TAGS',

    # human.attire
    'human.attire.upper_clothing': 'UPPER_BODY_CLOTHES_TYPES',
    'human.attire.lower_clothing': 'LOWER_BODY_CLOTHES_TYPES',
    'human.attire.footwear': 'FOOTWEAR_TYPES',
    'human.attire.socks': 'SOCK_TYPES',
    'human.attire.accessory': 'ACCESSORY_TYPES',
    'human.attire.jewelry': 'JEWELRY_TYPES',
    'human.attire.headwear': 'HEADWEAR_TYPES',
    'human.attire.fabric': 'FABRIC_TYPES',

    # human.expression
    'human.expression.emotion': 'EMOTION_TYPES',
    'human.expression.action': 'HUMAN_ACTIONS',
    'human.expression.interaction': 'HUMAN_INTERACTIONS',
    'human.expression.facial': 'HUMAN_EXPRESSIONS',

    # context.environment
    'context.environment.background': 'BACKGROUND_TYPES',
    'context.environment.decoration': 'DECORATION_TYPES',
    'context.environment.furniture': 'CHAIR_TYPES',
    'context.environment.surface': 'HUMAN_SURFACES',
    'context.environment.surface_linker': 'CLOTHING_SURFACE_LINKERS',

    # context.materiality
    'context.materiality.material': 'MATERIAL_TYPES',
    'context.materiality.texture': 'TEXTURE_TAGS',
    'context.materiality.pattern': 'PATTERN_TYPES',
    'context.materiality.liquid': 'LIQUID_TYPES',

    # context.depiction
    'context.depiction.grid': 'GRID_TAGS',
    'context.depiction.angle': 'SUBJECT_PHOTOGRAPH_ANGLE',
    'context.depiction.region': 'FOCAL_REGION_TAGS',
    'context.depiction.zone': 'ZONE_TAGS',
    'context.depiction.offset': 'OFFSET_TAGS',
    'context.depiction.shape': 'SHAPE_TYPES',
    'context.depiction.style': 'STYLE_TYPES',
    'context.depiction.lighting': 'LIGHTING_TYPES',
    'context.depiction.intent': 'INTENT_TYPES',

    # shared
    'shared.modifier.prefix': 'PREFIXES',
    'shared.modifier.suffix': 'SUFFIX_TAGS',
    'shared.descriptor.color': 'COLORS',
    'shared.descriptor.size': 'SIZE',
    'shared.descriptor.scope': 'SCOPE',
    'shared.descriptor.quality_plus': 'QUALITY_IMPROVERS',
    'shared.descriptor.quality_minus': 'QUALITY_REDUCERS',
    'shared.descriptor.adjective': 'ADJECTIVES',
    'shared.descriptor.adverb': 'ADVERBS',
    'shared.descriptor.verb': 'VERBS',

    # symbolic.logic
    'symbolic.logic.relation': 'RELATION_TAGS',
    'symbolic.logic.conjunction': 'SYMBOLIC_CONJUNCTION_TAGS',
    'symbolic.logic.associative': 'ASSOCIATIVE_LOGICAL_TAGS',
    'symbolic.logic.operator': 'SYMBOLIC_LOGIC_TAGS',
}


# ============================================================================
# USAGE ACCUMULATOR
# ============================================================================

class UsageAccumulator:
    """Tracks usage to prevent oversaturation"""

    def __init__(self, threshold: float = 0.1):
        self.usage_counts = Counter()
        self.total_uses = 0
        self.threshold = threshold

    def record_use(self, item: str, category: str = None):
        key = f"{category}:{item}" if category else item
        self.usage_counts[key] += 1
        self.total_uses += 1

    def get_weight(self, item: str, category: str = None) -> float:
        key = f"{category}:{item}" if category else item

        if self.total_uses == 0:
            return 1.0

        usage_freq = self.usage_counts[key] / self.total_uses

        if usage_freq > self.threshold:
            excess = (usage_freq - self.threshold) / self.threshold
            weight = max(0.1, 1.0 - (excess * 0.5))
            return weight

        return 1.0

    def get_weighted_choice(self, items: List[str], category: str = None) -> str:
        if not items:
            return ""

        if len(items) == 1:
            item = items[0]
            self.record_use(item, category)
            return item

        weights = [self.get_weight(item, category) for item in items]
        selected = random.choices(items, weights=weights, k=1)[0]
        self.record_use(selected, category)
        return selected

    def reset(self):
        self.usage_counts.clear()
        self.total_uses = 0


# ============================================================================
# ENHANCED HIERARCHICAL TEMPLATE SYNTHESIZER
# ============================================================================

@dataclass
class SynthesisResult:
    """Result of template synthesis"""
    text: str
    template: str
    filled_paths: Dict[str, str]
    category_usage: Dict[str, str]


class EnhancedHierarchicalTemplateSynthesizer:
    """
    Enhanced synthesizer with special case handling.
    """

    def __init__(
            self,
            bulk_captions_data: Dict[str, List[str]],
            full_generic_templates: List[str],
            prevent_oversaturation: bool = True,
            oversaturation_threshold: float = 0.05
    ):
        self.data = bulk_captions_data
        self.templates = full_generic_templates

        # Create combined gender list
        self.gender_items = self._build_gender_list()

        # Usage tracking
        self.usage = UsageAccumulator(oversaturation_threshold) if prevent_oversaturation else None

        # Statistics
        self.stats = {
            'total_synthesized': 0,
            'templates_used': Counter(),
            'paths_filled': Counter(),
            'categories_accessed': Counter(),
            'failed_fills': Counter(),
            'gender_fills': 0,
        }

    def _build_gender_list(self) -> List[str]:
        """Build combined gender list from MALE_TAGS + FEMALE_TAGS"""
        gender_list = []

        # Add male tags
        if 'MALE_TAGS' in self.data:
            gender_list.extend(self.data['MALE_TAGS'])

        # Add female tags
        if 'FEMALE_TAGS' in self.data:
            gender_list.extend(self.data['FEMALE_TAGS'])

        # Add from GENDER_TYPES if available
        if 'GENDER_TYPES' in self.data:
            gender_list.extend(self.data['GENDER_TYPES'])

        # Deduplicate
        return list(set(gender_list))

    def _extract_placeholders(self, template: str) -> List[str]:
        """
        Extract all placeholders from template.

        Matches both:
        - {tree.path} format
        - {GENDER} special format
        """
        pattern = r'\{([A-Z_a-z.]+)\}'
        matches = re.findall(pattern, template)
        return matches

    def _get_items_for_path(self, placeholder: str) -> Optional[List[str]]:
        """
        Get items for a placeholder.

        Handles special cases:
        - {GENDER} → combined MALE_TAGS + FEMALE_TAGS
        - {tree.path} → normal lookup
        """
        # Special case: GENDER
        if placeholder == 'GENDER':
            self.stats['gender_fills'] += 1
            return self.gender_items if self.gender_items else None

        # Normal tree path lookup
        category_name = TREE_PATH_TO_CATEGORY.get(placeholder)

        if not category_name:
            return None

        items = self.data.get(category_name)

        if items:
            self.stats['categories_accessed'][category_name] += 1

        return items

    def _fill_placeholder(self, placeholder: str) -> Tuple[str, Optional[str]]:
        """
        Fill a single placeholder.

        Args:
            placeholder: Either 'GENDER' or 'tree.path' format

        Returns:
            Tuple of (filled_text, category_name or None)
        """
        items = self._get_items_for_path(placeholder)

        if not items:
            self.stats['failed_fills'][placeholder] += 1
            return f"{{{placeholder}}}", None

        # Get category name for tracking
        if placeholder == 'GENDER':
            category_name = 'GENDER_COMBINED'
        else:
            category_name = TREE_PATH_TO_CATEGORY.get(placeholder)

        # Select item
        if self.usage:
            selected = self.usage.get_weighted_choice(items, category_name)
        else:
            selected = random.choice(items)

        self.stats['paths_filled'][placeholder] += 1

        return selected, category_name

    def synthesize(self, template: Optional[str] = None) -> SynthesisResult:
        """
        Synthesize a caption from template.

        Args:
            template: Specific template or None for random

        Returns:
            SynthesisResult
        """
        # Select template
        if template is None:
            template = random.choice(self.templates)

        self.stats['total_synthesized'] += 1
        self.stats['templates_used'][template] += 1

        # Extract placeholders
        placeholders = self._extract_placeholders(template)

        # Fill each placeholder
        result_text = template
        filled_paths = {}
        category_usage = {}

        for placeholder in placeholders:
            filled_item, category_name = self._fill_placeholder(placeholder)

            # Replace first occurrence
            result_text = result_text.replace(f"{{{placeholder}}}", filled_item, 1)

            filled_paths[placeholder] = filled_item
            if category_name:
                category_usage[placeholder] = category_name

        return SynthesisResult(
            text=result_text,
            template=template,
            filled_paths=filled_paths,
            category_usage=category_usage
        )

    def synthesize_batch(self, n: int) -> List[SynthesisResult]:
        """Synthesize multiple captions"""
        return [self.synthesize() for _ in range(n)]

    def get_stats(self) -> Dict:
        """Get synthesis statistics"""
        return {
            'total_synthesized': self.stats['total_synthesized'],
            'unique_templates_used': len(self.stats['templates_used']),
            'templates_used': dict(self.stats['templates_used'].most_common(10)),
            'paths_filled': dict(self.stats['paths_filled'].most_common(20)),
            'categories_accessed': dict(self.stats['categories_accessed'].most_common(20)),
            'failed_fills': dict(self.stats['failed_fills']) if self.stats['failed_fills'] else {},
            'gender_fills': self.stats['gender_fills'],
            'usage_stats': {
                'total_uses': self.usage.total_uses if self.usage else 0,
                'unique_items_used': len(self.usage.usage_counts) if self.usage else 0,
                'top_used_items': dict(self.usage.usage_counts.most_common(20)) if self.usage else {},
            } if self.usage else None,
        }

    def reset_usage(self):
        """Reset usage tracking"""
        if self.usage:
            self.usage.reset()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_bulk_captions_data() -> Dict[str, List[str]]:
    """Load BulkCaptions data"""
    try:
        from geovocab2.data.prompt.bulk_caption_data import BulkCaptions

        data = {}
        for attr_name in dir(BulkCaptions):
            if attr_name.isupper() and not attr_name.startswith('_'):
                attr = getattr(BulkCaptions, attr_name)
                if isinstance(attr, (list, tuple)):
                    if attr_name not in ('FULL_ASSOCIATIVE',):
                        data[attr_name] = list(attr)
        return data
    except ImportError:
        print("Error: Could not import BulkCaptions")
        return {}


def create_enhanced_synthesizer(
        prevent_oversaturation: bool = True,
        oversaturation_threshold: float = 0.05
) -> EnhancedHierarchicalTemplateSynthesizer:
    """
    Create enhanced synthesizer with special case support.
    """
    # Load data
    bulk_data = load_bulk_captions_data()

    # Load FULL_GENERIC
    try:
        from geovocab2.data.prompt.bulk_caption_data import BulkCaptions
        full_generic = BulkCaptions.FULL_GENERIC
    except (ImportError, AttributeError):
        print("Error: Could not load FULL_GENERIC")
        full_generic = []

    return EnhancedHierarchicalTemplateSynthesizer(
        bulk_captions_data=bulk_data,
        full_generic_templates=full_generic,
        prevent_oversaturation=prevent_oversaturation,
        oversaturation_threshold=oversaturation_threshold
    )


# ============================================================================
# TESTING
# ============================================================================

def test_enhanced_synthesis():
    """Test enhanced synthesis with special cases"""
    print("=" * 80)
    print("ENHANCED TEMPLATE SYNTHESIS TEST")
    print("=" * 80)
    print()

    # Create synthesizer
    print("Creating enhanced synthesizer...")
    synthesizer = create_enhanced_synthesizer()
    print(f"Loaded {len(synthesizer.data)} categories")
    print(f"Loaded {len(synthesizer.templates)} templates")
    print(f"Gender items: {len(synthesizer.gender_items)}")
    print()

    # Test with manual template containing {GENDER}
    test_template = "a {GENDER} {human.expression.action} {context.depiction.lighting}"

    print("=" * 80)
    print("TESTING {GENDER} SPECIAL PLACEHOLDER")
    print("=" * 80)
    print()
    print(f"Template: {test_template}")
    print()

    for i in range(5):
        result = synthesizer.synthesize(template=test_template)
        print(f"[{i + 1}] {result.text}")
        print(f"    Filled: {result.filled_paths}")
        print()

    # Test with random templates
    print("=" * 80)
    print("RANDOM TEMPLATE SYNTHESIS")
    print("=" * 80)
    print()

    for i in range(10):
        result = synthesizer.synthesize()
        print(f"[{i + 1}] {result.text}")
    print()

    # Statistics
    stats = synthesizer.get_stats()
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print()
    print(f"Total synthesized: {stats['total_synthesized']}")
    print(f"Gender fills: {stats['gender_fills']}")
    print(f"Unique templates: {stats['unique_templates_used']}")
    print()
    print("Top paths filled:")
    for path, count in list(stats['paths_filled'].items())[:10]:
        print(f"  {path:40} : {count:3d}")
    print()


if __name__ == "__main__":
    test_enhanced_synthesis()