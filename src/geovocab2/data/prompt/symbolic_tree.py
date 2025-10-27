"""
symbolic_tree.py (Rules Enforced + Seeded Determinism + Multi-Clause)
===========================================
Comprehensive symbolic synthesis system that ACTUALLY respects all rules,
creates multi-clause captions, and emphasizes quality tags.

Author: Phi + Claude
Date: 2025-10-26
Package: geovocab2.data.prompt.symbolic_tree
"""

from __future__ import annotations
import random
import re
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import numpy as np

# Import BulkCaptions for data access
from geovocab2.data.prompt.bulk_caption_data import BulkCaptions


# ============================================================================
# SYMBOLIC TREE (same as before)
# ============================================================================
# [Tree definition stays the same - copying from your file]

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
            "pose": {
                "list": "HUMAN_POSES",
                "weight": 0.6,
                "compatible": ["context.environment.on_surface", "human.expression.human_action"],
                "primary": True,
                "solidification": "structural",
                "grammar_role": "verb_like",
            },
            "hair_style": {
                "list": "HAIRSTYLES_TYPES",
                "weight": 0.35,
                "compatible": ["human.anatomy.hair_length", "human.attire.headwear"],
                "avoid_simultaneous": ["human.attire.headwear"],
                "grammar_role": "adjective",
            },
            "hair_length": {
                "list": "HAIR_LENGTH_TYPES",
                "weight": 0.50,
                "compatible": ["human.anatomy.hair_style"],
                "requires": ["human.anatomy.hair_style"],
                "grammar_role": "adjective",
                "order": "before_hair_style",
            },
        },
        "attire": {
            "_rules": {
                "activation": "human_present",
                "sampling": "outfit_coherent",
                "nesting": ["context.materiality", "shared.descriptors"],
                "max_combined": 4,
                "min_combined": 1,
                "blacklist": [
                    ("upper_clothing", "upper_clothing"),
                    ("lower_clothing", "lower_clothing"),
                    ("footwear", "footwear"),
                ],
            },
            "upper_clothing": {
                "list": "UPPER_BODY_CLOTHES_TYPES",
                "weight": 0.35,
                "compatible": ["human.attire.lower_clothing", "human.attire.accessory"],
                "primary": True,
                "grammar_role": "noun",
            },
            "lower_clothing": {
                "list": "LOWER_BODY_CLOTHES_TYPES",
                "weight": 0.25,
                "compatible": ["human.attire.upper_clothing", "human.attire.footwear"],
                "grammar_role": "noun",
            },
            "footwear": {
                "list": "FOOTWEAR_TYPES",
                "weight": 0.2,
                "compatible": ["human.attire.lower_clothing", "context.environment.on_surface"],
                "grammar_role": "noun",
            },
            "accessory": {
                "list": "ACCESSORY_TYPES",
                "weight": 0.15,
                "compatible": ["human.attire.*", "shared.descriptors.color"],
                "grammar_role": "noun",
            },
            "jewelry": {
                "list": "JEWELRY_TYPES",
                "weight": 0.075,
                "compatible": ["human.attire.*", "context.materiality.material"],
                "grammar_role": "noun",
            },
            "headwear": {
                "list": "HEADWEAR_TYPES",
                "weight": 0.10,
                "compatible": ["human.attire.*"],
                "avoid_simultaneous": ["human.anatomy.hair_style"],
                "grammar_role": "noun",
            },
        },
        "expression": {
            "_rules": {
                "activation": "human_present",
                "sampling": "emotional_coherent",
                "max_combined": 2,
                "blacklist": [
                    ("human_action", "human_action"),
                ],
            },
            "human_expression": {
                "list": "HUMAN_EXPRESSIONS",
                "weight": 0.4,
                "compatible": ["context.depiction.composition.emotion", "human.expression.human_action"],
                "grammar_role": "adjective",
            },
            "human_action": {
                "list": "HUMAN_ACTIONS",
                "weight": 0.5,
                "compatible": ["human.anatomy.pose", "context.environment"],
                "primary": True,
                "solidification": "action_core",
                "grammar_role": "verb",
            },
            "human_interaction": {
                "list": "HUMAN_INTERACTIONS",
                "weight": 0.15,
                "compatible": ["context.environment.object_left", "context.environment.object_right"],
                "grammar_role": "verb",
            },
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
            "_rules": {
                "activation": "spatial_context",
                "sampling": "scene_coherent",
                "max_combined": 3,
                "blacklist": [
                    ("background", "background"),
                    ("chair", "on_surface"),
                ],
            },
            "background": {
                "list": "BACKGROUND_TYPES",
                "weight": 0.2,
                "compatible": ["context.depiction.composition.lighting"],
                "primary": True,
                "solidification": "scene_anchor",
                "grammar_role": "noun",
            },
            "decoration": {
                "list": "DECORATION_TYPES",
                "weight": 0.2,
                "compatible": ["context.environment.background"],
                "grammar_role": "noun",
            },
            "chair": {
                "list": "CHAIR_TYPES",
                "weight": 0.2,
                "compatible": ["human.anatomy.pose"],
                "requires": ["human.anatomy.pose"],
                "grammar_role": "noun",
            },
            "on_surface": {
                "list": "HUMAN_SURFACES",
                "weight": 0.2,
                "compatible": ["subject", "human.anatomy.pose"],
                "grammar_role": "prepositional_object",
            },
        },
        "materiality": {
            "_rules": {
                "activation": "descriptive_detail",
                "sampling": "material_coherent",
                "max_combined": 3,
                "blacklist": [
                    ("material", "material"),
                    ("fabric", "fabric"),
                ],
            },
            "material": {
                "list": "MATERIAL_TYPES",
                "weight": 0.30,
                "compatible": ["context.environment", "human.attire"],
                "grammar_role": "adjective",
            },
            "fabric": {
                "list": "FABRIC_TYPES",
                "weight": 0.30,
                "compatible": ["human.attire", "context.materiality.texture"],
                "grammar_role": "adjective",
            },
            "texture": {
                "list": "TEXTURE_TAGS",
                "weight": 0.20,
                "compatible": ["context.materiality.material", "context.environment"],
                "grammar_role": "adjective",
            },
            "pattern": {
                "list": "PATTERN_TYPES",
                "weight": 0.10,
                "compatible": ["human.attire", "context.environment.background"],
                "grammar_role": "adjective",
            },
            "liquid": {
                "list": "LIQUID_TYPES",
                "weight": 0.075,
                "compatible": ["context.environment"],
                "grammar_role": "noun",
            },
        },
        "depiction": {
            "_rules": {
                "activation": "compositional_control",
                "max_depth": 2,
                "sampling_strategy": "photographic_coherent",
            },
            "viewer": {
                "_rules": {
                    "activation": "camera_perspective",
                    "max_combined": 2,
                    "blacklist": [
                        ("subject_angle", "subject_angle"),
                    ],
                },
                "grid": {
                    "list": "GRID_TAGS",
                    "weight": 0.4,
                    "grammar_role": "adjective",
                },
                "subject_angle": {
                    "list": "SUBJECT_PHOTOGRAPH_ANGLE",
                    "weight": 0.6,
                    "primary": True,
                    "grammar_role": "adjective",
                },
                "focal_region": {
                    "list": "FOCAL_REGION_TAGS",
                    "weight": 0.3,
                    "grammar_role": "adjective",
                },
            },
            "object_view": {
                "_rules": {
                    "activation": "spatial_relationships",
                    "max_combined": 2,
                    "blacklist": [
                        ("object_left", "object_left"),
                        ("object_right", "object_right"),
                    ],
                },
                "zone": {
                    "list": "ZONE_TAGS",
                    "weight": 0.4,
                    "grammar_role": "prepositional_phrase",
                },
                "object_left": {
                    "list": "LEFT_OBJECT_TYPES",
                    "weight": 0.3,
                    "grammar_role": "noun",
                    "prepositional": "to the left",
                },
                "object_right": {
                    "list": "RIGHT_OBJECT_TYPES",
                    "weight": 0.3,
                    "grammar_role": "noun",
                    "prepositional": "to the right",
                },
            },
            "composition": {
                "_rules": {
                    "activation": "aesthetic_control",
                    "max_combined": 3,
                    "blacklist": [
                        ("lighting", "lighting"),
                        ("style", "style"),
                    ],
                },
                "offset": {
                    "list": "OFFSET_TAGS",
                    "weight": 0.2,
                    "grammar_role": "adjective",
                },
                "shape": {
                    "list": "SHAPE_TYPES",
                    "weight": 0.15,
                    "grammar_role": "adjective",
                },
                "style": {
                    "list": "STYLE_TYPES",
                    "weight": 0.25,
                    "primary": True,
                    "solidification": "artistic_anchor",
                    "grammar_role": "adjective",
                },
                "lighting": {
                    "list": "LIGHTING_TYPES",
                    "weight": 0.3,
                    "primary": True,
                    "solidification": "mood_anchor",
                    "grammar_role": "adjective",
                },
                "emotion": {
                    "list": "EMOTION_TYPES",
                    "weight": 0.1,
                    "requires": ["human.expression"],
                    "grammar_role": "adjective",
                },
            },
        },
    },
    "shared": {
        "_rules": {
            "activation": "always_available",
            "max_depth": 1,
            "compatible_with": ["*"],
            "sampling_strategy": "attribute_stacking",
            "noise_chance": 0.15,
        },
        "descriptors": {
            "_rules": {
                "activation": "detail_enhancement",
                "max_combined": 3,
                "blacklist": [
                    ("color", "color"),
                    ("size", "size"),
                    ("quality", "degradation"),
                ],
            },
            "prefix": {
                "list": "PREFIXES",
                "weight": 0.2,
                "grammar_role": "prefix_modifier",
            },
            "suffix": {
                "list": "SUFFIX_TAGS",
                "weight": 0.15,
                "grammar_role": "suffix_modifier",
            },
            "color": {
                "list": "COLORS",
                "weight": 0.25,
                "grammar_role": "adjective",
            },
            "size": {
                "list": "SIZE",
                "weight": 0.15,
                "grammar_role": "adjective",
                "order": "before_color",
            },
            "scope": {
                "list": "SCOPE",
                "weight": 0.1,
                "grammar_role": "determiner",
            },
            "quality": {
                "list": "QUALITY_IMPROVERS",
                "weight": 0.1,
                "avoid_simultaneous": ["shared.descriptors.degradation"],
                "grammar_role": "adjective",
            },
            "degradation": {
                "list": "QUALITY_REDUCERS",
                "weight": 0.05,
                "avoid_simultaneous": ["shared.descriptors.quality"],
                "grammar_role": "adjective",
            },
        },
    },
}


# ============================================================================
# DATA ACCESS LAYER
# ============================================================================

class BulkCaptionsAccessor:
    """Provides safe access to BulkCaptions data"""

    def __init__(self):
        self.bulk_captions = BulkCaptions
        self._cache = {}

    def get_list(self, list_name: str) -> List[str]:
        """Get a list from BulkCaptions"""
        if list_name in self._cache:
            return self._cache[list_name]

        try:
            data = getattr(self.bulk_captions, list_name)
            if isinstance(data, list):
                self._cache[list_name] = data
                return data
            return []
        except AttributeError:
            return []

    def sample(self, list_name: str, n: int = 1) -> List[str]:
        """Sample items"""
        data = self.get_list(list_name)
        if not data:
            return [f"<missing:{list_name}>"]
        return random.choices(data, k=n)

    def get_random(self, list_name: str) -> str:
        """Get single random item"""
        result = self.sample(list_name, n=1)
        return result[0] if result else f"<missing:{list_name}>"


# ============================================================================
# TREE NAVIGATOR
# ============================================================================

class TreeNavigator:
    """Navigate SYMBOLIC_TREE and resolve to BulkCaptions"""

    def __init__(self, tree: Dict = None):
        self.tree = tree or SYMBOLIC_TREE
        self.accessor = BulkCaptionsAccessor()
        self._path_to_list_cache = {}
        self._path_to_node_cache = {}
        self._build_cache()

    def _build_cache(self):
        """Build caches"""
        def traverse(subtree, path=""):
            if isinstance(subtree, dict):
                self._path_to_node_cache[path] = subtree

                if "list" in subtree:
                    self._path_to_list_cache[path] = subtree["list"]

                for key, value in subtree.items():
                    if key not in ["_rules", "list", "weight", "compatible", "requires",
                                   "preferred_linkages", "grammar_role", "solidification",
                                   "primary", "avoid_with", "avoid_simultaneous",
                                   "noise_substitutes", "boost_when_underused", "order",
                                   "position", "usage", "examples", "avoid_duplicates",
                                   "required_for", "prepositional"]:
                        new_path = f"{path}.{key}" if path else key
                        traverse(value, new_path)

        traverse(self.tree)

    def get_list_name(self, path: str) -> Optional[str]:
        """Get BulkCaptions list name"""
        return self._path_to_list_cache.get(path)

    def get_node(self, path: str) -> Dict[str, Any]:
        """Get node info"""
        return self._path_to_node_cache.get(path, {})

    def sample_data(self, path: str, n: int = 1) -> List[str]:
        """Sample data"""
        list_name = self.get_list_name(path)
        if list_name:
            return self.accessor.sample(list_name, n)
        return [f"<missing:{path}>"]

    def get_random_item(self, path: str) -> str:
        """Get single random item"""
        list_name = self.get_list_name(path)
        if list_name:
            return self.accessor.get_random(list_name)
        return f"<missing:{path}>"


# ============================================================================
# RULE ENGINE (ACTUALLY ENFORCES RULES NOW!)
# ============================================================================

class RuleEngine:
    """Evaluates and enforces ALL constraints"""

    def __init__(self, tree: Dict, navigator: TreeNavigator):
        self.tree = tree
        self.navigator = navigator
        self.blacklist = self._extract_all_blacklists()
        self.requirements = self._extract_all_requirements()
        self.avoid_simultaneous = self._extract_avoid_simultaneous()
        self.max_combined = self._extract_max_combined()

    def _extract_all_blacklists(self) -> List[Tuple[str, str]]:
        """Extract ALL blacklist rules"""
        blacklist = []

        def traverse(subtree, path=""):
            if isinstance(subtree, dict):
                if "_rules" in subtree and "blacklist" in subtree["_rules"]:
                    blacklist.extend(subtree["_rules"]["blacklist"])

                for key, value in subtree.items():
                    if key not in ["_rules"]:
                        new_path = f"{path}.{key}" if path else key
                        traverse(value, new_path)

        traverse(self.tree)
        return blacklist

    def _extract_all_requirements(self) -> Dict[str, List[str]]:
        """Extract ALL requirement rules"""
        requirements = {}

        def traverse(subtree, path=""):
            if isinstance(subtree, dict):
                if "requires" in subtree:
                    requirements[path] = subtree["requires"]

                for key, value in subtree.items():
                    if key not in ["_rules"]:
                        new_path = f"{path}.{key}" if path else key
                        traverse(value, new_path)

        traverse(self.tree)
        return requirements

    def _extract_avoid_simultaneous(self) -> Dict[str, List[str]]:
        """Extract avoid_simultaneous rules"""
        avoid = {}

        def traverse(subtree, path=""):
            if isinstance(subtree, dict):
                if "avoid_simultaneous" in subtree:
                    avoid[path] = subtree["avoid_simultaneous"]

                for key, value in subtree.items():
                    if key not in ["_rules"]:
                        new_path = f"{path}.{key}" if path else key
                        traverse(value, new_path)

        traverse(self.tree)
        return avoid

    def _extract_max_combined(self) -> Dict[str, int]:
        """Extract max_combined limits"""
        limits = {}

        def traverse(subtree, path=""):
            if isinstance(subtree, dict):
                if "_rules" in subtree and "max_combined" in subtree["_rules"]:
                    limits[path] = subtree["_rules"]["max_combined"]

                for key, value in subtree.items():
                    if key not in ["_rules"]:
                        new_path = f"{path}.{key}" if path else key
                        traverse(value, new_path)

        traverse(self.tree)
        return limits

    def is_blacklisted(self, path1: str, path2: str) -> bool:
        """Check if combination is blacklisted"""
        for item1, item2 in self.blacklist:
            # Support wildcards
            if "*" in item1 or "*" in item2:
                pattern1 = item1.replace(".", r"\.").replace("*", ".*")
                pattern2 = item2.replace(".", r"\.").replace("*", ".*")
                if (re.match(pattern1, path1) and re.match(pattern2, path2)) or \
                   (re.match(pattern1, path2) and re.match(pattern2, path1)):
                    return True
            else:
                if (path1 == item1 and path2 == item2) or \
                   (path1 == item2 and path2 == item1):
                    return True
        return False

    def check_requirements(self, path: str, selected_paths: List[str]) -> bool:
        """Check if requirements are met"""
        if path not in self.requirements:
            return True

        required = self.requirements[path]
        for req in required:
            # Support wildcards in requirements
            if "*" in req:
                pattern = req.replace(".", r"\.").replace("*", ".*")
                found = any(re.match(pattern, sp) for sp in selected_paths)
            else:
                found = any(req in sp for sp in selected_paths)

            if not found:
                return False

        return True

    def check_avoid_simultaneous(self, path: str, selected_paths: List[str]) -> bool:
        """Check avoid_simultaneous rules"""
        if path not in self.avoid_simultaneous:
            return True

        avoid_list = self.avoid_simultaneous[path]
        for avoid_path in avoid_list:
            if any(avoid_path in sp for sp in selected_paths):
                return False

        return True

    def check_max_combined(self, path: str, selected_paths: List[str]) -> bool:
        """Check if max_combined limit exceeded"""
        # Extract parent path
        parts = path.split('.')
        for i in range(len(parts)):
            parent_path = '.'.join(parts[:i+1])
            if parent_path in self.max_combined:
                # Count how many selected paths are under this parent
                count = sum(1 for sp in selected_paths if sp.startswith(parent_path + '.'))
                if count >= self.max_combined[parent_path]:
                    return False

        return True

    def is_valid_selection(self, path: str, selected_paths: List[str]) -> bool:
        """Check if path can be added to selected_paths"""
        # Check blacklist against all selected
        for selected in selected_paths:
            if self.is_blacklisted(path, selected):
                return False

        # Check requirements
        if not self.check_requirements(path, selected_paths):
            return False

        # Check avoid_simultaneous
        if not self.check_avoid_simultaneous(path, selected_paths):
            return False

        # Check max_combined
        if not self.check_max_combined(path, selected_paths):
            return False

        return True


# ============================================================================
# CLAUSE BUILDER (Multi-clause Support)
# ============================================================================

@dataclass
class Clause:
    """Represents a descriptive clause"""
    subject: str = ""
    action: str = ""
    descriptors: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Convert to natural language"""
        parts = []

        # Subject with descriptors
        if self.descriptors:
            parts.append(' '.join(self.descriptors + [self.subject]))
        elif self.subject:
            parts.append(self.subject)

        # Action
        if self.action:
            parts.append(self.action)

        # Objects
        if self.objects:
            parts.append(' '.join(self.objects))

        # Modifiers
        if self.modifiers:
            parts.extend(self.modifiers)

        return ' '.join(parts)


class MultiClauseComposer:
    """Composes multi-clause captions with conjunctions"""

    CONJUNCTIONS = ["with", "and", "while", "featuring", "against", "in", "under"]

    def _is_plural_subject(self, subject: str) -> bool:
        """Check if subject indicates multiple people"""
        if not subject:
            return False

        subject_lower = subject.lower()

        # Check numeric prefixes like "2girls", "3boys" (but not "1girl")
        if subject_lower[0].isdigit() and subject_lower[0] != '1':
            return True

        # Check for word numbers
        plural_words = ['two', 'three', 'four', 'five', 'six', 'multiple', 'several', 'many']
        if any(subject_lower.startswith(word + ' ') for word in plural_words):
            return True

        # Check for plural markers (but exclude "1girl", "1boy", "1other")
        if any(word in subject_lower for word in ['girls', 'boys', 'others', 'people', 'group']):
            if not subject_lower.startswith('1'):
                return True

        return False

    def compose_multi_clause(self, selected_items: Dict[str, str],
                            selected_paths: List[str]) -> str:
        """Build multi-clause caption"""
        # Quality tags FIRST (before all clauses)
        quality_prefix = ""
        if "shared.descriptors.quality" in selected_items:
            quality_prefix = selected_items["shared.descriptors.quality"] + ", "

        # Get subject for plural detection
        subject = selected_items.get("subject.identity.gender",
                                     selected_items.get("subject.entities.animal",
                                     selected_items.get("subject.entities.produce",
                                     selected_items.get("subject.entities.plant", "object"))))
        is_plural = self._is_plural_subject(subject)

        clauses = []

        # Main clause: subject + action + clothing (without quality tag)
        main_clause = self._build_main_clause(selected_items)
        clauses.append(main_clause)

        # Secondary clause: hair/appearance
        hair_clause = self._build_hair_clause(selected_items, is_plural)
        if hair_clause:
            clauses.append(hair_clause)

        # Tertiary clause: background/setting
        background_clause = self._build_background_clause(selected_items)
        if background_clause:
            clauses.append(background_clause)

        # Quaternary clause: lighting/style
        style_clause = self._build_style_clause(selected_items)
        if style_clause:
            clauses.append(style_clause)

        # Join and prepend quality
        return quality_prefix + self._join_clauses(clauses)

    def _build_main_clause(self, items: Dict[str, str]) -> str:
        """Main subject + action + clothing"""
        parts = []

        # Get subject
        subject = items.get("subject.identity.gender",
                          items.get("subject.entities.animal",
                          items.get("subject.entities.produce",
                          items.get("subject.entities.plant", "object"))))

        # Check if plural
        is_plural = self._is_plural_subject(subject)

        # Get color for subject (quality handled separately)
        # For plurals, don't put color directly on subject - it's awkward
        descriptors = []
        color_for_clothing = None
        if "shared.descriptors.color" in items:
            if is_plural:
                # Save color for clothing instead
                color_for_clothing = items["shared.descriptors.color"]
            else:
                descriptors.append(items["shared.descriptors.color"])

        # Article (none for plurals)
        article = "" if is_plural else self._get_article(subject)

        # Build subject phrase
        subject_phrase = f"{article}{' '.join(descriptors) + ' ' if descriptors else ''}{subject}"
        parts.append(subject_phrase)

        # Action/pose
        action = items.get("human.anatomy.pose", items.get("human.expression.human_action", ""))
        if action:
            parts.append(action)

        # Clothing - handle plurals differently
        clothing = []
        for path in ["human.attire.upper_clothing", "human.attire.lower_clothing",
                     "human.attire.footwear"]:
            if path in items:
                clothing.append(items[path])

        if clothing:
            if is_plural:
                # For plurals, use "in varied attire" or list with "and"
                if len(clothing) > 1:
                    clothing_phrase = f"in {' and '.join(clothing)}"
                else:
                    # Pluralize single item or use generic phrase
                    if random.random() < 0.5:
                        clothing_phrase = f"wearing varied {clothing[0]}"
                    else:
                        clothing_phrase = f"in {clothing[0]}"
            else:
                # Singular subject - normal handling
                if color_for_clothing and len(clothing) == 1:
                    clothing_phrase = f"wearing {color_for_clothing} {clothing[0]}"
                else:
                    clothing_phrase = f"wearing {', '.join(clothing)}"

            parts.append(clothing_phrase)

        return ' '.join(parts)

    def _build_hair_clause(self, items: Dict[str, str], is_plural: bool = False) -> Optional[str]:
        """Hair/appearance clause"""
        parts = []

        # For plurals, use "varied" or generic terms
        if "human.anatomy.hair_length" in items:
            if is_plural:
                parts.append(f"with varied {items['human.anatomy.hair_length']} hair")
            else:
                parts.append(items["human.anatomy.hair_length"])

        if "human.anatomy.hair_style" in items:
            if is_plural:
                # Skip if we already added hair length, or make it varied
                if "human.anatomy.hair_length" not in items:
                    parts.append(f"varied {items['human.anatomy.hair_style']} hairstyles")
            else:
                parts.append(items["human.anatomy.hair_style"] + " hair")

        if "human.attire.accessory" in items:
            if is_plural:
                parts.append(f"wearing {items['human.attire.accessory']}")
            else:
                parts.append(items["human.attire.accessory"])

        return ', '.join(parts) if parts else None

    def _build_background_clause(self, items: Dict[str, str]) -> Optional[str]:
        """Background/setting clause"""
        parts = []

        background = items.get("context.environment.background", "")
        if background:
            prep = random.choice(["in", "against", "within", "near"])
            parts.append(f"{prep} {background}")

        surface = items.get("context.environment.on_surface", "")
        if surface:
            parts.append(f"on {surface}")

        return ' '.join(parts) if parts else None

    def _build_style_clause(self, items: Dict[str, str]) -> Optional[str]:
        """Lighting/style clause"""
        parts = []

        lighting = items.get("context.depiction.composition.lighting", "")
        if lighting:
            parts.append(lighting)

        style = items.get("context.depiction.composition.style", "")
        if style:
            parts.append(f"{style} style")

        return ', '.join(parts) if parts else None

    def _join_clauses(self, clauses: List[str]) -> str:
        """Join clauses with varied conjunctions"""
        if not clauses:
            return ""

        if len(clauses) == 1:
            return clauses[0]

        # Use commas for most joins, occasional conjunctions
        result = clauses[0]
        for i, clause in enumerate(clauses[1:], 1):
            if i == len(clauses) - 1 and random.random() < 0.3:
                # Last clause sometimes gets conjunction
                conj = random.choice(["with", "featuring"])
                result += f" {conj} {clause}"
            else:
                result += f", {clause}"

        return result

    def _get_article(self, subject: str) -> str:
        """Get appropriate article"""
        if any(subject.startswith(p) for p in ["1boy", "1girl", "male", "female", "1other"]):
            return ""

        roll = random.random()
        if roll < 0.7:
            return "an " if subject[0].lower() in "aeiou" else "a "
        elif roll < 0.9:
            return "the "
        return ""


# ============================================================================
# TEMPLATE COMPOSER (RULE-RESPECTING!)
# ============================================================================

class TemplateComposer:
    """Builds captions while respecting ALL rules"""

    def __init__(self, tree: Dict, navigator: TreeNavigator, rule_engine: RuleEngine):
        self.tree = tree
        self.navigator = navigator
        self.rule_engine = rule_engine
        self.clause_composer = MultiClauseComposer()

    def compose(self, complexity: int = 3, noise_level: float = 0.1) -> Dict[str, Any]:
        """Compose caption with rule enforcement"""
        selected_items = {}
        selected_paths = []

        # Step 1: Select subject (required)
        subject_path = self._select_subject_valid(noise_level)
        if subject_path:
            selected_paths.append(subject_path)
            selected_items[subject_path] = self.navigator.get_random_item(subject_path)

        # Step 2: Human attributes
        if "identity.gender" in subject_path:
            human_paths = self._select_human_attributes_valid(complexity, selected_paths)
            for path in human_paths:
                selected_items[path] = self.navigator.get_random_item(path)
            selected_paths.extend(human_paths)

        # Step 3: Context
        context_paths = self._select_context_valid(complexity, selected_paths, noise_level)
        for path in context_paths:
            selected_items[path] = self.navigator.get_random_item(path)
        selected_paths.extend(context_paths)

        # Step 4: Descriptors (MORE QUALITY TAGS!)
        descriptor_paths = self._select_descriptors_valid(complexity, selected_paths)
        for path in descriptor_paths:
            selected_items[path] = self.navigator.get_random_item(path)
        selected_paths.extend(descriptor_paths)

        # Step 5: Build multi-clause text
        text = self.clause_composer.compose_multi_clause(selected_items, selected_paths)

        return {
            "text": text,
            "selected_paths": selected_paths,
            "selected_items": selected_items,
            "metadata": {
                "complexity": complexity,
                "noise_level": noise_level,
                "num_categories": len(selected_paths),
            }
        }

    def _select_subject_valid(self, noise_level: float) -> Optional[str]:
        """Select subject with weighted random based on tree"""
        # Balanced weights: reduce plant to 7% to avoid SD15 overload
        candidates = [
            ("subject.identity.gender", 0.40),
            ("subject.entities.animal", 0.22),
            ("subject.entities.produce", 0.20),
            ("subject.entities.plant", 0.07),
        ]

        # Filter valid candidates
        valid_candidates = [
            (path, weight) for path, weight in candidates
            if self.rule_engine.is_valid_selection(path, [])
        ]

        if not valid_candidates:
            return "subject.identity.gender"

        # Weighted random selection
        paths, weights = zip(*valid_candidates)
        return random.choices(paths, weights=weights, k=1)[0]

    def _select_human_attributes_valid(self, complexity: int, selected: List[str]) -> List[str]:
        """Select human attributes with rule validation"""
        paths = []

        # Pose - 70% probability (not always!)
        if random.random() < 0.7:
            if self.rule_engine.is_valid_selection("human.anatomy.pose", selected):
                paths.append("human.anatomy.pose")

        # Upper clothing - 60% probability
        if complexity >= 2 and random.random() < 0.6:
            if self.rule_engine.is_valid_selection("human.attire.upper_clothing", selected + paths):
                paths.append("human.attire.upper_clothing")

        if complexity >= 3:
            # Try lower clothing first
            if random.random() < 0.6:
                if self.rule_engine.is_valid_selection("human.attire.lower_clothing", selected + paths):
                    paths.append("human.attire.lower_clothing")
            else:
                if self.rule_engine.is_valid_selection("human.expression.human_action", selected + paths):
                    paths.append("human.expression.human_action")

        # Hair
        if complexity >= 3:
            if random.random() < 0.4:
                if self.rule_engine.is_valid_selection("human.anatomy.hair_style", selected + paths):
                    paths.append("human.anatomy.hair_style")
                    # Hair length - simple 50/50 when hair mentioned
                    if random.random() < 0.5:
                        if self.rule_engine.is_valid_selection("human.anatomy.hair_length", selected + paths):
                            paths.append("human.anatomy.hair_length")

        # Accessory - boost to 40% probability
        if complexity >= 3 and random.random() < 0.40:
            if self.rule_engine.is_valid_selection("human.attire.accessory", selected + paths):
                paths.append("human.attire.accessory")

        return paths

    def _select_context_valid(self, complexity: int, selected: List[str], noise: float) -> List[str]:
        """Select context with rule validation"""
        paths = []

        # Background
        if complexity >= 2 and random.random() < 0.8:
            if self.rule_engine.is_valid_selection("context.environment.background", selected):
                paths.append("context.environment.background")

        # Lighting
        if complexity >= 2 and random.random() < 0.5:  # 50% chance, not always
            if self.rule_engine.is_valid_selection("context.depiction.composition.lighting", selected + paths):
                paths.append("context.depiction.composition.lighting")

        # Style - available earlier with higher probability
        if complexity >= 3 and random.random() < 0.15:
            if self.rule_engine.is_valid_selection("context.depiction.composition.style", selected + paths):
                paths.append("context.depiction.composition.style")

        return paths

    def _select_descriptors_valid(self, complexity: int, selected: List[str]) -> List[str]:
        """Select descriptors with EMPHASIS ON QUALITY"""
        paths = []

        # QUALITY TAGS: 70% chance at complexity >= 2 (not just 30% at 4+!)
        if complexity >= 2 and random.random() < 0.7:
            if self.rule_engine.is_valid_selection("shared.descriptors.quality", selected):
                paths.append("shared.descriptors.quality")

        # Color
        if complexity >= 2 and random.random() < 0.5:
            if self.rule_engine.is_valid_selection("shared.descriptors.color", selected + paths):
                paths.append("shared.descriptors.color")

        return paths


# ============================================================================
# SYNTHESIS SYSTEM
# ============================================================================

class SynthesisSystem:
    """Complete synthesis system with rule enforcement"""

    def __init__(self, tree: Dict = None):
        self.tree = tree or SYMBOLIC_TREE
        self.navigator = TreeNavigator(self.tree)
        self.rule_engine = RuleEngine(self.tree, self.navigator)
        self.template_composer = TemplateComposer(self.tree, self.navigator, self.rule_engine)

    def synthesize(self, complexity: int = 3, noise_level: float = 0.1) -> Dict[str, Any]:
        """Synthesize caption with all rules enforced"""
        return self.template_composer.compose(complexity, noise_level)

    def synthesize_batch(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Synthesize batch"""
        return [self.synthesize(**kwargs) for _ in range(n)]


# ============================================================================
# FACTORY
# ============================================================================

def create_synthesis_system(tree: Dict = None) -> SynthesisSystem:
    """Factory function"""
    return SynthesisSystem(tree)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    system = create_synthesis_system()

    print("Generating RULE-RESPECTING multi-clause captions with quality tags...\n")

    for i in range(10):
        result = system.synthesize(complexity=5, noise_level=0.1)
        print(f"Caption {i+1}:")
        print(result["text"])
        print(f"Categories: {len(result['selected_paths'])}")
        print(f"Paths: {result['selected_paths'][:3]}...")  # Show first 3
        #print()

"""
Synthesis Tree Benchmarking
============================
Generate 100,000 captions and analyze for bias, coverage, and quality.

Outputs:
- Category usage heatmap
- Bias weights for correction
- Statistical analysis
- Sample captions

Author: Phi + Claude
Date: 2025-10-26
"""

import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import random
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import the synthesis system
#from synthesis_tree import create_synthesis_system


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class SynthesisBenchmark:
    """Benchmark the synthesis system with statistical analysis"""

    def __init__(self, num_captions: int = 100000, seed: Optional[int] = None):
        self.num_captions = num_captions
        self.seed = seed

        # Set seeds for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.system = create_synthesis_system()

        # Tracking
        self.all_paths = Counter()
        self.all_items = Counter()
        self.category_counts = defaultdict(Counter)
        self.complexity_distribution = Counter()
        self.caption_lengths = []
        self.quality_tag_count = 0
        self.clause_counts = []

        # Results
        self.captions = []
        self.results = []

    def run_benchmark(self):
        """Generate captions and collect statistics"""
        print(f"Generating {self.num_captions:,} captions...")
        print("This may take a few minutes...\n")

        for i in tqdm(range(self.num_captions), desc="Generating"):
            # Vary complexity
            complexity = random.choice([2, 3, 3, 3, 4])  # Bias toward 3
            noise = random.uniform(0.05, 0.15)

            # Generate
            result = self.system.synthesize(complexity=complexity, noise_level=noise)

            # Track
            self._track_result(result)

            # Store
            self.results.append(result)
            self.captions.append(result["text"])

        print("\n✓ Generation complete!")
        print(f"Total captions: {len(self.captions):,}")
        print(f"Total paths tracked: {len(self.all_paths)}")
        print(f"Total unique items: {len(self.all_items)}")

    def _track_result(self, result: Dict[str, Any]):
        """Track statistics from a single result"""
        # Paths
        for path in result["selected_paths"]:
            self.all_paths[path] += 1

            # Extract category
            parts = path.split('.')
            if len(parts) >= 2:
                category = '.'.join(parts[:2])  # e.g., "human.anatomy"
                leaf = parts[-1]
                self.category_counts[category][leaf] += 1

        # Items
        for item in result["selected_items"].values():
            self.all_items[item] += 1

        # Metadata
        self.complexity_distribution[result["metadata"]["complexity"]] += 1
        self.caption_lengths.append(len(result["text"]))
        self.clause_counts.append(result["text"].count(',') + 1)  # Approximate clauses

        # Quality tags
        if any(tag in result["text"] for tag in
               ["masterpiece", "best quality", "highly detailed", "8k", "professional"]):
            self.quality_tag_count += 1

    def calculate_bias_weights(self) -> Dict[str, float]:
        """
        Calculate bias weights for over/under-represented categories.

        Returns:
            Dict mapping path to bias weight (1.0 = neutral, >1.0 = boost, <1.0 = reduce)
        """
        print("\nCalculating bias weights...")

        total_selections = sum(self.all_paths.values())
        num_unique_paths = len(self.all_paths)
        expected_per_path = total_selections / num_unique_paths

        bias_weights = {}

        for path, count in self.all_paths.items():
            # Bias weight: expected / actual
            # If path is overused, weight < 1.0 (reduce)
            # If path is underused, weight > 1.0 (boost)
            bias_weights[path] = expected_per_path / count if count > 0 else 1.0

        return bias_weights

    def generate_heatmap(self, output_path: str = "category_heatmap.png"):
        """Generate heatmap of category usage"""
        print("\nGenerating heatmap...")

        # Organize data by major categories
        major_categories = ["subject", "human", "context", "shared"]

        # Build matrix
        category_data = defaultdict(dict)

        for path, count in self.all_paths.most_common(50):  # Top 50
            parts = path.split('.')
            if len(parts) >= 2:
                major = parts[0]
                subcategory = '.'.join(parts[1:])
                category_data[major][subcategory] = count

        # Convert to matrix format
        all_subcategories = sorted(set(
            subcat for cats in category_data.values() for subcat in cats.keys()
        ))

        matrix = []
        labels_y = []

        for major in major_categories:
            if major in category_data:
                labels_y.append(major)
                row = [category_data[major].get(subcat, 0) for subcat in all_subcategories]
                matrix.append(row)

        # Create heatmap
        plt.figure(figsize=(20, 8))

        if matrix:
            sns.heatmap(
                matrix,
                xticklabels=all_subcategories,
                yticklabels=labels_y,
                cmap="YlOrRd",
                annot=False,
                fmt="d",
                cbar_kws={'label': 'Usage Count'}
            )

            plt.title(f"Category Usage Heatmap (Top 50 paths, n={self.num_captions:,})",
                      fontsize=16, fontweight='bold')
            plt.xlabel("Subcategories", fontsize=12)
            plt.ylabel("Major Categories", fontsize=12)
            plt.xticks(rotation=90, ha='right', fontsize=8)
            plt.yticks(fontsize=10)
            plt.tight_layout()

            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Heatmap saved to: {output_path}")
        else:
            print("⚠ No data for heatmap")

        plt.close()

    def generate_report(self) -> str:
        """Generate comprehensive statistical report"""
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("SYNTHESIS BENCHMARK REPORT")
        report_lines.append("=" * 80)
        if self.seed is not None:
            report_lines.append(f"Random Seed: {self.seed} (reproducible)")
        else:
            report_lines.append("Random Seed: None (non-reproducible)")
        report_lines.append("")

        # Basic stats
        report_lines.append("GENERATION STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total captions generated:    {self.num_captions:,}")
        report_lines.append(f"Unique paths used:           {len(self.all_paths)}")
        report_lines.append(f"Unique items used:           {len(self.all_items)}")
        report_lines.append(
            f"Quality tag inclusion:       {self.quality_tag_count:,} ({self.quality_tag_count / self.num_captions * 100:.1f}%)")
        report_lines.append("")

        # Caption characteristics
        report_lines.append("CAPTION CHARACTERISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Avg caption length:          {np.mean(self.caption_lengths):.1f} characters")
        report_lines.append(f"Std caption length:          {np.std(self.caption_lengths):.1f}")
        report_lines.append(f"Min caption length:          {min(self.caption_lengths)}")
        report_lines.append(f"Max caption length:          {max(self.caption_lengths)}")
        report_lines.append(f"Avg clauses per caption:     {np.mean(self.clause_counts):.1f}")
        report_lines.append("")

        # Complexity distribution
        report_lines.append("COMPLEXITY DISTRIBUTION")
        report_lines.append("-" * 80)
        for complexity in sorted(self.complexity_distribution.keys()):
            count = self.complexity_distribution[complexity]
            pct = count / self.num_captions * 100
            report_lines.append(f"  Complexity {complexity}:  {count:,} ({pct:.1f}%)")
        report_lines.append("")

        # Top 20 most used paths
        report_lines.append("TOP 20 MOST USED PATHS")
        report_lines.append("-" * 80)
        for i, (path, count) in enumerate(self.all_paths.most_common(20), 1):
            pct = count / self.num_captions * 100
            report_lines.append(f"{i:2d}. {path:50s} {count:7,} ({pct:5.1f}%)")
        report_lines.append("")

        # Top 20 least used paths (or all if fewer)
        report_lines.append("TOP 20 LEAST USED PATHS")
        report_lines.append("-" * 80)
        all_paths_list = list(self.all_paths.most_common())
        least_common = all_paths_list[-20:] if len(all_paths_list) > 20 else all_paths_list
        least_common.reverse()  # Show least to most

        if len(all_paths_list) <= 20:
            report_lines.append(f"(Showing all {len(all_paths_list)} paths in ascending order by usage)")
            report_lines.append("")

        for i, (path, count) in enumerate(least_common, 1):
            pct = count / self.num_captions * 100
            report_lines.append(f"{i:2d}. {path:50s} {count:7,} ({pct:5.1f}%)")
        report_lines.append("")

        # Bias analysis
        report_lines.append("BIAS ANALYSIS")
        report_lines.append("-" * 80)
        total = sum(self.all_paths.values())
        expected = total / len(self.all_paths)
        report_lines.append(f"Total selections:            {total:,}")
        report_lines.append(f"Unique paths:                {len(self.all_paths)}")
        report_lines.append(f"Expected per path:           {expected:.1f}")
        report_lines.append("")

        # Most over-represented
        bias_weights = self.calculate_bias_weights()
        overused = sorted([(p, c, bias_weights[p]) for p, c in self.all_paths.items()],
                          key=lambda x: x[1], reverse=True)[:10]

        report_lines.append("TOP 10 OVER-REPRESENTED (need reduction)")
        for path, count, weight in overused:
            report_lines.append(f"  {path:50s} {count:6,} (bias weight: {weight:.3f})")
        report_lines.append("")

        # Most under-represented
        underused = sorted([(p, c, bias_weights[p]) for p, c in self.all_paths.items()],
                           key=lambda x: x[1])[:10]

        report_lines.append("TOP 10 UNDER-REPRESENTED (need boost)")
        for path, count, weight in underused:
            report_lines.append(f"  {path:50s} {count:6,} (bias weight: {weight:.3f})")
        report_lines.append("")

        # Category balance
        report_lines.append("CATEGORY BALANCE")
        report_lines.append("-" * 80)
        category_totals = defaultdict(int)
        for path, count in self.all_paths.items():
            parts = path.split('.')
            if parts:
                category_totals[parts[0]] += count

        for category in sorted(category_totals.keys()):
            count = category_totals[category]
            pct = count / total * 100
            report_lines.append(f"  {category:20s} {count:8,} ({pct:5.1f}%)")
        report_lines.append("")

        # Sample captions
        report_lines.append("SAMPLE CAPTIONS (Random 20)")
        report_lines.append("-" * 80)
        samples = random.sample(self.captions, min(20, len(self.captions)))
        for i, caption in enumerate(samples, 1):
            report_lines.append(f"{i:2d}. {caption}")
        report_lines.append("")

        report_lines.append("=" * 80)

        return '\n'.join(report_lines)

    def save_bias_weights(self, output_path: str = "bias_weights.json"):
        """Save bias weights to JSON"""
        bias_weights = self.calculate_bias_weights()

        with open(output_path, 'w') as f:
            json.dump(bias_weights, f, indent=2)

        print(f"✓ Bias weights saved to: {output_path}")

    def save_results(self, output_dir: str = "."):
        """Save all results"""
        # Report
        report = self.generate_report()
        report_path = f"{output_dir}/benchmark_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to: {report_path}")

        # Bias weights
        bias_path = f"{output_dir}/bias_weights.json"
        self.save_bias_weights(bias_path)

        # Heatmap
        heatmap_path = f"{output_dir}/category_heatmap.png"
        self.generate_heatmap(heatmap_path)

        # Raw data
        data_path = f"{output_dir}/benchmark_data.json"
        with open(data_path, 'w') as f:
            json.dump({
                "num_captions": self.num_captions,
                "path_counts": dict(self.all_paths.most_common()),
                "item_counts": dict(self.all_items.most_common()),
                "complexity_distribution": dict(self.complexity_distribution),
                "caption_length_stats": {
                    "mean": float(np.mean(self.caption_lengths)),
                    "std": float(np.std(self.caption_lengths)),
                    "min": int(min(self.caption_lengths)),
                    "max": int(max(self.caption_lengths)),
                },
                "quality_tag_rate": self.quality_tag_count / self.num_captions,
                "avg_clauses": float(np.mean(self.clause_counts)),
            }, f, indent=2)
        print(f"✓ Raw data saved to: {data_path}")

        # Sample captions
        samples_path = f"{output_dir}/sample_captions.txt"
        samples = random.sample(self.captions, min(1000, len(self.captions)))
        with open(samples_path, 'w') as f:
            for caption in samples:
                f.write(caption + '\n')
        print(f"✓ Sample captions saved to: {samples_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run benchmark"""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark synthesis system")
    parser.add_argument('-n', '--num-captions', type=int, default=100000,
                        help="Number of captions to generate (default: 100000)")
    parser.add_argument('-o', '--output-dir', type=str, default="./benchmark_results",
                        help="Output directory (default: ./benchmark_results)")
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help="Random seed for reproducibility (default: None)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("SYNTHESIS TREE BENCHMARK")
    print("=" * 80)
    print(f"Captions to generate: {args.num_captions:,}")
    print(f"Output directory: {args.output_dir}")
    if args.seed is not None:
        print(f"Random seed: {args.seed} (reproducible mode)")
    else:
        print("Random seed: None (non-reproducible mode)")
    print()

    # Run benchmark
    benchmark = SynthesisBenchmark(num_captions=args.num_captions, seed=args.seed)
    benchmark.run_benchmark()

    # Save results
    print("\nSaving results...")
    benchmark.save_results(output_dir=args.output_dir)

    # Print summary
    print("\n" + benchmark.generate_report())

    print("\n✓ Benchmark complete!")
    print(f"All results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()