"""
Template Synthesizer
====================

Fills FULL_GENERIC templates with BulkCaptions data.
Handles special placeholders and prevents oversaturation.

Author: Phi + Claude
Date: 2025-10-26
"""

import random
import re
from typing import List, Dict, Optional
from collections import Counter
from dataclasses import dataclass

from geovocab2.data.prompt.bulk_caption_data import BulkCaptions

# ============================================================================
# TREE PATH TO CATEGORY MAPPING
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
    """Prevents oversaturation by tracking usage and applying penalties."""

    def __init__(self, threshold: float = 0.05):
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

        freq = self.usage_counts[key] / self.total_uses

        if freq > self.threshold:
            excess = (freq - self.threshold) / self.threshold
            return max(0.1, 1.0 - (excess * 0.5))

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
# TEMPLATE SYNTHESIZER
# ============================================================================

@dataclass
class SynthesisResult:
    text: str
    template: str
    filled_paths: Dict[str, str]


class TemplateSynthesizer:
    """Fills templates with BulkCaptions data."""

    def __init__(
            self,
            templates: List[str],
            prevent_oversaturation: bool = True,
            oversaturation_threshold: float = 0.05
    ):
        self.templates = templates

        # Load BulkCaptions data
        self.data = {}
        for attr_name in dir(BulkCaptions):
            if attr_name.isupper() and not attr_name.startswith('_'):
                attr = getattr(BulkCaptions, attr_name)
                if isinstance(attr, (list, tuple)):
                    if attr_name != 'FULL_ASSOCIATIVE':
                        self.data[attr_name] = list(attr)

        # Build gender list
        self.gender_items = self._build_gender_list()

        # Usage tracking
        self.usage = UsageAccumulator(oversaturation_threshold) if prevent_oversaturation else None

        # Stats
        self.stats = {
            'total_synthesized': 0,
            'gender_fills': 0,
            'paths_filled': Counter(),
            'categories_accessed': Counter(),
        }

    def _build_gender_list(self) -> List[str]:
        gender = []
        for key in ['MALE_TAGS', 'FEMALE_TAGS', 'GENDER_TYPES']:
            if key in self.data:
                gender.extend(self.data[key])
        return list(set(gender))

    def _extract_placeholders(self, template: str) -> List[str]:
        return re.findall(r'\{([A-Z_a-z.]+)\}', template)

    def _get_items(self, placeholder: str) -> Optional[List[str]]:
        if placeholder == 'GENDER':
            self.stats['gender_fills'] += 1
            return self.gender_items if self.gender_items else None

        category = TREE_PATH_TO_CATEGORY.get(placeholder)
        if not category:
            return None

        items = self.data.get(category)
        if items:
            self.stats['categories_accessed'][category] += 1

        return items

    def _fill_placeholder(self, placeholder: str) -> str:
        items = self._get_items(placeholder)

        if not items:
            return f"{{{placeholder}}}"

        category = 'GENDER_COMBINED' if placeholder == 'GENDER' else TREE_PATH_TO_CATEGORY.get(placeholder)

        if self.usage:
            selected = self.usage.get_weighted_choice(items, category)
        else:
            selected = random.choice(items)

        self.stats['paths_filled'][placeholder] += 1
        return selected

    def synthesize(self, template: Optional[str] = None) -> SynthesisResult:
        if template is None:
            template = random.choice(self.templates)

        self.stats['total_synthesized'] += 1

        placeholders = self._extract_placeholders(template)
        result_text = template
        filled_paths = {}

        for placeholder in placeholders:
            filled = self._fill_placeholder(placeholder)
            result_text = result_text.replace(f"{{{placeholder}}}", filled, 1)
            filled_paths[placeholder] = filled

        return SynthesisResult(
            text=result_text,
            template=template,
            filled_paths=filled_paths
        )

    def synthesize_batch(self, n: int) -> List[SynthesisResult]:
        return [self.synthesize() for _ in range(n)]

    def get_stats(self) -> Dict:
        return {
            'total_synthesized': self.stats['total_synthesized'],
            'gender_fills': self.stats['gender_fills'],
            'paths_filled': dict(self.stats['paths_filled'].most_common(20)),
            'categories_accessed': dict(self.stats['categories_accessed'].most_common(20)),
            'usage_stats': {
                'total_uses': self.usage.total_uses if self.usage else 0,
                'unique_items': len(self.usage.usage_counts) if self.usage else 0,
                'top_used': dict(self.usage.usage_counts.most_common(20)) if self.usage else {},
            } if self.usage else None,
        }

    def reset_usage(self):
        if self.usage:
            self.usage.reset()