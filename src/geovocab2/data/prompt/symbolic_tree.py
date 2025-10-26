"""
symbolic_tree.py
================
Hierarchical symbolic category tree with relational mappings to BulkCaptions.

This module provides:
- SYMBOLIC_TREE: Complete hierarchical structure
- CategoryTreeNavigator: Intelligence for tree traversal and compatibility
- Direct data access from BulkCaptions

Part of: geovocab2.data.prompt.symbolic_tree

Author: Phi + Claude
Date: 2025-10-26
License: MIT
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import random
import numpy as np

from geovocab2.data.prompt.bulk_caption_data import BulkCaptions


# ============================================================================
# SYMBOLIC CATEGORY TREE - HIERARCHICAL STRUCTURE
# ============================================================================

SYMBOLIC_TREE = {
    "subject": {
        "entities": {
            "animal": "ANIMAL_TYPES",              # 83 items
            "humanoid": "HUMANOID_TYPES",          # 61 items
            "produce": "FRUIT_AND_VEGETABLE",      # 131 items
            "plant": "PLANT_CATEGORY_TAGS",        # 46 items
        },
        "identity": {
            "subject": "SUBJECT_TYPES",            # 125 items
            "object": "OBJECT_TYPES",              # 421 items
            "clothing": "CLOTHING_TYPES",          # 154 items
            "surface": "HUMAN_SURFACES",           # 236 items
            "gender": {
                "male": "MALE_TAGS",               # 10 items
                "female": "FEMALE_TAGS",           # 99 items
                "ambiguous": "AMBIG_TAGS",         # 17 items
                "all": "GENDER_TYPES",             # 119 items
            },
        },
    },
    "human": {
        "anatomy": {
            "pose": "HUMAN_POSES",                 # 128 items
            "hair_style": "HAIRSTYLES_TYPES",      # 449 items
            "hair_length": "HAIR_LENGTH_TYPES",    # 96 items
        },
        "attire": {
            "upper_clothing": "UPPER_BODY_CLOTHES_TYPES",  # 2646 items
            "lower_clothing": "LOWER_BODY_CLOTHES_TYPES",  # 894 items
            "footwear": "FOOTWEAR_TYPES",          # 166 items
            "socks": "SOCK_TYPES",                 # 112 items
            "accessory": "ACCESSORY_TYPES",        # 221 items
            "jewelry": "JEWELRY_TYPES",            # 229 items
            "headwear": "HEADWEAR_TYPES",          # 1220 items
        },
        "expression": {
            "emotion": "EMOTION_TYPES",            # 148 items
            "action": "HUMAN_ACTIONS",             # 281 items
            "interaction": "HUMAN_INTERACTIONS",   # 129 items
            "expression": "HUMAN_EXPRESSIONS",     # 91 items
        },
    },
    "context": {
        "environment": {
            "background": "BACKGROUND_TYPES",      # 1418 items
            "decoration": "DECORATION_TYPES",      # 175 items
            "chair": "CHAIR_TYPES",                # 90 items
            "on_surface": "HUMAN_SURFACES",        # 236 items (shared with identity)
        },
        "materiality": {
            "material": "MATERIAL_TYPES",          # 567 items
            "fabric": "FABRIC_TYPES",              # 70 items
            "texture": "TEXTURE_TAGS",             # 51 items
            "pattern": "PATTERN_TYPES",            # 292 items
            "liquid": "LIQUID_TYPES",              # 40 items
        },
        "depiction": {
            "viewer": {
                "grid": "GRID_TAGS",               # 652 items
                "subject_angle": "SUBJECT_PHOTOGRAPH_ANGLE",  # 86 items
                "focal_region": "FOCAL_REGION_TAGS",  # 100 items
            },
            "object_view": {
                "zone": "ZONE_TAGS",               # 80 items
                "offset": "OFFSET_TAGS",           # 95 items
            },
            "composition": {
                "shape": "SHAPE_TYPES",            # 112 items
                "style": "STYLE_TYPES",            # 349 items
                "lighting": "LIGHTING_TYPES",      # 126 items
                "intent": "INTENT_TYPES",          # 245 items
            },
        },
    },
    "shared": {
        "descriptors": {
            "prefix": "PREFIXES",                  # 201 items
            "suffix": "SUFFIX_TAGS",               # 95 items
            "color": "COLORS",                     # 126 items
            "size": "SIZE",                        # 122 items
            "scope": "SCOPE",                      # 199 items
            "quality": {
                "improver": "QUALITY_IMPROVERS",   # 50 items
                "reducer": "QUALITY_REDUCERS",     # 50 items
            },
        },
        "modifiers": {
            "adjective": "ADJECTIVES",             # 95 items
            "adverb": "ADVERBS",                   # 90 items
            "verb": "VERBS",                       # 105 items
        },
    },
    "symbolic": {
        "logic": {
            "conjunction": "SYMBOLIC_CONJUNCTION_TAGS",      # 26 items
            "relation": "RELATION_TAGS",                     # 122 items
            "logic_operators": "SYMBOLIC_LOGIC_TAGS",        # 301 items
            "associative": "ASSOCIATIVE_LOGICAL_TAGS",       # 166 items
            "conjunction_extension": "SYMBOLIC_CONJUNCTION_EXTENSION_TAGS",  # 20 items
        },
        "semantic": {
            "associations": "FULL_ASSOCIATIVE",    # 3147 pre-composed triples
        },
    },
}


# ============================================================================
# SPECIAL TOKEN MAPPINGS
# ============================================================================

# Map special tokens to their tree categories
SPECIAL_TOKEN_TO_CATEGORY = {
    "<subject>": "subject",
    "<subject1>": "subject",
    "<subject2>": "subject",
    "<pose>": "pose",
    "<emotion>": "emotion",
    "<upper_clothing>": "upper_clothing",
    "<lower_clothing>": "lower_clothing",
    "<footwear>": "footwear",
    "<accessory>": "accessory",
    "<jewelry>": "jewelry",
    "<headwear>": "headwear",
    "<hair_style>": "hair_style",
    "<hair_length>": "hair_length",
    "<background>": "background",
    "<material>": "material",
    "<fabric>": "fabric",
    "<texture>": "texture",
    "<pattern>": "pattern",
    "<lighting>": "lighting",
    "<style>": "style",
    "<intent>": "intent",
    "<color>": "color",
    "<size>": "size",
    "<grid>": "grid",
    "<zone>": "zone",
}


# ============================================================================
# CATEGORY TREE NAVIGATOR
# ============================================================================

class CategoryTreeNavigator:
    """
    Intelligent navigation through the symbolic category tree.

    Provides:
    - Hierarchical path finding
    - Sibling/parent/child relationships
    - Semantic compatibility scoring
    - Composition partner suggestions
    - Domain classification
    """

    def __init__(self, tree: Dict = None):
        self.tree = tree or SYMBOLIC_TREE
        self._path_cache: Dict[str, List[str]] = {}
        self._sibling_cache: Dict[str, List[str]] = {}
        self._list_name_cache: Dict[str, str] = {}
        self._build_caches()

    def _build_caches(self):
        """Build lookup caches for fast navigation"""
        def traverse(node, path=[]):
            if isinstance(node, dict):
                for key, value in node.items():
                    current_path = path + [key]
                    self._path_cache[key] = current_path

                    # If value is a string, it's a list name mapping
                    if isinstance(value, str):
                        self._list_name_cache[key] = value
                    else:
                        traverse(value, current_path)
            elif isinstance(node, str):
                # Leaf node - store the list name
                pass

        traverse(self.tree)

        # Build sibling relationships
        def find_siblings(node, path=[]):
            if isinstance(node, dict):
                keys = [k for k, v in node.items() if not isinstance(v, str)]
                for i, key in enumerate(keys):
                    siblings = [k for j, k in enumerate(keys) if j != i]
                    self._sibling_cache[key] = siblings

                    value = node[key]
                    if isinstance(value, dict):
                        find_siblings(value, path + [key])

        find_siblings(self.tree)

    def get_path(self, category: str) -> List[str]:
        """Get hierarchical path to a category"""
        return self._path_cache.get(category, [])

    def get_list_name(self, category: str) -> Optional[str]:
        """Get the BulkCaptions list name for a category"""
        return self._list_name_cache.get(category)

    def get_data(self, category: str) -> Optional[List[Any]]:
        """
        Get the actual data list from BulkCaptions for a category.
        Returns None if category doesn't exist or has no data.
        """
        list_name = self.get_list_name(category)
        if not list_name:
            return None

        return getattr(BulkCaptions, list_name, None)

    def sample_data(self, category: str, n: int = 1) -> List[Any]:
        """
        Sample n random items from a category's data.
        Returns empty list if category doesn't exist.
        """
        data = self.get_data(category)
        if not data:
            return []

        n = min(n, len(data))
        return random.sample(data, n) if n > 1 else [random.choice(data)]

    def get_random_item(self, category: str) -> Optional[Any]:
        """
        Get one random item from a category's data.
        Returns None if category doesn't exist.
        """
        data = self.get_data(category)
        if not data:
            return None
        return random.choice(data)

    def get_data_size(self, category: str) -> int:
        """Get the number of items in a category's data list"""
        data = self.get_data(category)
        return len(data) if data else 0

    def get_parent(self, category: str) -> Optional[str]:
        """Get parent category"""
        path = self.get_path(category)
        return path[-2] if len(path) >= 2 else None

    def get_siblings(self, category: str) -> List[str]:
        """Get sibling categories (same parent)"""
        return self._sibling_cache.get(category, [])

    def get_children(self, category: str) -> List[str]:
        """Get child categories"""
        def find_children(node, target, path=[]):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == target:
                        if isinstance(value, dict):
                            return [k for k in value.keys() if not isinstance(value[k], str)]
                        return []
                    if isinstance(value, dict):
                        result = find_children(value, target, path + [key])
                        if result:
                            return result
            return []

        return find_children(self.tree, category)

    def get_root_domain(self, category: str) -> Optional[str]:
        """Get top-level domain (subject, human, context, shared, symbolic)"""
        path = self.get_path(category)
        return path[0] if path else None

    def are_compatible(self, cat1: str, cat2: str) -> float:
        """
        Calculate semantic compatibility score between two categories.
        Returns 0.0-1.0 based on tree relationship.

        Scoring:
        - Same category: 0.3 (redundant)
        - Siblings: 0.7 (related but distinct)
        - Same parent: 0.8
        - Same domain: 0.8
        - Cross-domain pairs have specific scores
        """
        if cat1 == cat2:
            return 0.3  # Same category is somewhat redundant

        path1 = self.get_path(cat1)
        path2 = self.get_path(cat2)

        if not path1 or not path2:
            return 0.5  # Unknown relationship

        # Check if siblings
        if cat2 in self.get_siblings(cat1):
            return 0.7

        # Check if same parent
        parent1 = self.get_parent(cat1)
        parent2 = self.get_parent(cat2)
        if parent1 and parent1 == parent2:
            return 0.8

        # Check domain compatibility
        domain1 = path1[0]
        domain2 = path2[0]

        if domain1 == domain2:
            return 0.8  # Same domain

        # Cross-domain compatibility rules
        domain_compatibility = {
            ('subject', 'human'): 0.9,
            ('subject', 'context'): 0.85,
            ('subject', 'shared'): 0.8,
            ('human', 'context'): 0.9,
            ('human', 'shared'): 0.8,
            ('context', 'shared'): 0.85,
            ('symbolic', 'subject'): 0.6,
            ('symbolic', 'human'): 0.6,
            ('symbolic', 'context'): 0.65,
            ('symbolic', 'shared'): 0.7,
        }

        key = (domain1, domain2)
        reverse_key = (domain2, domain1)

        return domain_compatibility.get(key, domain_compatibility.get(reverse_key, 0.5))

    def suggest_composition_partners(self, category: str, n: int = 3) -> List[Tuple[str, float]]:
        """
        Suggest compatible categories for composition.
        Returns list of (category, compatibility_score) tuples.
        """
        candidates = []

        # Add siblings
        siblings = self.get_siblings(category)
        for sib in siblings[:3]:
            score = self.are_compatible(category, sib)
            candidates.append((sib, score))

        # Add from compatible domains
        domain = self.get_root_domain(category)
        compatible_domains = {
            'subject': ['human', 'context', 'shared'],
            'human': ['subject', 'context', 'shared'],
            'context': ['subject', 'human', 'shared'],
            'shared': ['subject', 'human', 'context'],
            'symbolic': ['subject', 'human', 'context'],
        }

        for comp_domain in compatible_domains.get(domain, []):
            # Get some categories from that domain
            domain_cats = self._get_categories_in_domain(comp_domain)
            for cat in random.sample(domain_cats, min(2, len(domain_cats))):
                score = self.are_compatible(category, cat)
                candidates.append((cat, score))

        # Sort by score and return top n
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]

    def _get_categories_in_domain(self, domain: str) -> List[str]:
        """Get all leaf categories in a domain"""
        categories = []
        for cat, path in self._path_cache.items():
            if path and path[0] == domain:
                # Only include categories that have list mappings
                if cat in self._list_name_cache:
                    categories.append(cat)
        return categories

    def get_composition_chain(self, start_category: str, depth: int = 3) -> List[str]:
        """
        Generate a composition chain starting from a category.
        Uses tree relationships to build semantically coherent chains.
        """
        chain = [start_category]
        current = start_category

        for _ in range(depth - 1):
            partners = self.suggest_composition_partners(current, n=3)
            if not partners:
                break

            # Weight selection by compatibility
            categories, scores = zip(*partners)
            probs = np.array(scores) / sum(scores)
            next_cat = np.random.choice(categories, p=probs)

            chain.append(next_cat)
            current = next_cat

        return chain

    def compose_from_chain(self, chain: List[str]) -> Dict[str, Any]:
        """
        Get actual data items for each category in a composition chain.
        Returns dict mapping category -> sampled item.
        """
        composition = {}
        for category in chain:
            item = self.get_random_item(category)
            if item:
                composition[category] = item
        return composition

    def compose_semantic(self, primary_category: str, depth: int = 3) -> Dict[str, Any]:
        """
        Generate a semantically coherent composition with actual data.
        Returns dict with category chain and sampled items.
        """
        chain = self.get_composition_chain(primary_category, depth)
        items = self.compose_from_chain(chain)

        return {
            'chain': chain,
            'items': items,
            'compatibility_scores': [
                self.are_compatible(chain[i], chain[i+1])
                for i in range(len(chain)-1)
            ] if len(chain) > 1 else []
        }

    def get_multi_category_data(self, categories: List[str], samples_per_category: int = 1) -> Dict[str, List[Any]]:
        """
        Get data from multiple categories at once.
        Returns dict mapping category -> list of sampled items.
        """
        result = {}
        for category in categories:
            data = self.sample_data(category, n=samples_per_category)
            if data:
                result[category] = data
        return result

    def get_all_leaf_categories(self) -> List[str]:
        """Get all categories that map to actual lists"""
        return list(self._list_name_cache.keys())

    def get_categories_by_domain(self, domain: str) -> List[str]:
        """Get all categories in a specific domain"""
        return self._get_categories_in_domain(domain)

    def validate_tree(self) -> Dict[str, Any]:
        """
        Validate that all tree mappings point to valid list names in BulkCaptions.
        Returns validation report with data sizes.
        """
        report = {
            'total_categories': len(self._list_name_cache),
            'total_domains': len(self.tree),
            'categories_per_domain': {},
            'mappings': {},
            'data_sizes': {},
            'missing_data': [],
        }

        # Check each mapping
        for category, list_name in self._list_name_cache.items():
            report['mappings'][category] = list_name

            # Try to get actual data
            data = self.get_data(category)
            if data is not None:
                report['data_sizes'][category] = len(data)
            else:
                report['missing_data'].append((category, list_name))

        # Count by domain
        for domain in self.tree.keys():
            cats = self.get_categories_by_domain(domain)
            total_items = sum(self.get_data_size(cat) for cat in cats)
            report['categories_per_domain'][domain] = {
                'count': len(cats),
                'total_items': total_items
            }

        return report


# ============================================================================
# DOMAIN CLASSIFICATION
# ============================================================================

def classify_category(category: str, navigator: CategoryTreeNavigator = None) -> Optional[str]:
    """Classify a category into its root domain"""
    nav = navigator or CategoryTreeNavigator()
    return nav.get_root_domain(category)


def get_cross_domain_pairs(navigator: CategoryTreeNavigator = None) -> List[Tuple[str, str, float]]:
    """
    Get all high-compatibility cross-domain category pairs.
    Returns list of (cat1, cat2, compatibility) tuples.
    """
    nav = navigator or CategoryTreeNavigator()
    pairs = []

    all_cats = nav.get_all_leaf_categories()

    for i, cat1 in enumerate(all_cats):
        for cat2 in all_cats[i+1:]:
            domain1 = nav.get_root_domain(cat1)
            domain2 = nav.get_root_domain(cat2)

            # Only include cross-domain pairs
            if domain1 != domain2:
                compat = nav.are_compatible(cat1, cat2)
                if compat >= 0.7:  # High compatibility threshold
                    pairs.append((cat1, cat2, compat))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_tree(tree: Dict = None, indent: int = 0, navigator: CategoryTreeNavigator = None):
    """Pretty print the tree structure"""
    tree = tree or SYMBOLIC_TREE
    nav = navigator or CategoryTreeNavigator()

    for key, value in tree.items():
        list_name = nav.get_list_name(key)

        if isinstance(value, str):
            print("  " * indent + f"├─ {key} → {value}")
        elif isinstance(value, dict):
            if list_name:
                print("  " * indent + f"├─ {key}/ → {list_name}")
            else:
                print("  " * indent + f"├─ {key}/")
            print_tree(value, indent + 1, nav)


def get_tree_stats(navigator: CategoryTreeNavigator = None) -> Dict[str, Any]:
    """Get comprehensive tree statistics"""
    nav = navigator or CategoryTreeNavigator()

    stats = {
        'total_leaf_categories': len(nav.get_all_leaf_categories()),
        'domains': {},
        'depth': {},
    }

    for domain in ['subject', 'human', 'context', 'shared', 'symbolic']:
        cats = nav.get_categories_by_domain(domain)
        stats['domains'][domain] = {
            'count': len(cats),
            'categories': cats[:5] + ['...'] if len(cats) > 5 else cats
        }

    # Calculate depth distribution
    for cat in nav.get_all_leaf_categories():
        path = nav.get_path(cat)
        depth = len(path)
        stats['depth'][depth] = stats['depth'].get(depth, 0) + 1

    return stats


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'SYMBOLIC_TREE',
    'SPECIAL_TOKEN_TO_CATEGORY',
    'CategoryTreeNavigator',
    'classify_category',
    'get_cross_domain_pairs',
    'print_tree',
    'get_tree_stats',
]


if __name__ == '__main__':
    # Quick test
    nav = CategoryTreeNavigator()

    print("=" * 80)
    print("SYMBOLIC TREE VALIDATION")
    print("=" * 80)

    report = nav.validate_tree()
    print(f"\nTotal Categories: {report['total_categories']}")
    print(f"Total Domains: {report['total_domains']}")
    print("\nCategories per Domain:")
    for domain, info in report['categories_per_domain'].items():
        print(f"  {domain}: {info['count']} categories, {info['total_items']} items")

    if report['missing_data']:
        print("\n⚠️  Missing Data:")
        for cat, list_name in report['missing_data']:
            print(f"  {cat} → {list_name} (not found in BulkCaptions)")

    print("\n" + "=" * 80)
    print("TREE STRUCTURE")
    print("=" * 80)
    print_tree(navigator=nav)

    print("\n" + "=" * 80)
    print("EXAMPLE USAGE")
    print("=" * 80)

    # Test navigation
    test_cat = "pose"
    print(f"\nCategory: {test_cat}")
    print(f"  Path: {' → '.join(nav.get_path(test_cat))}")
    print(f"  List Name: {nav.get_list_name(test_cat)}")
    print(f"  Domain: {nav.get_root_domain(test_cat)}")
    print(f"  Data Size: {nav.get_data_size(test_cat)} items")

    # Test data access
    print(f"\n  Sample Data (3 items):")
    samples = nav.sample_data(test_cat, n=3)
    for item in samples:
        print(f"    - {item}")

    # Test compatibility
    print(f"\n  Composition Partners:")
    partners = nav.suggest_composition_partners(test_cat, n=5)
    for partner, score in partners:
        data_size = nav.get_data_size(partner)
        print(f"    {partner}: {score:.2f} ({data_size} items)")

    # Test semantic composition
    print(f"\n  Semantic Composition:")
    composition = nav.compose_semantic(test_cat, depth=4)
    print(f"    Chain: {' → '.join(composition['chain'])}")
    print(f"    Items:")
    for cat, item in composition['items'].items():
        print(f"      {cat}: {item}")
    if composition['compatibility_scores']:
        avg_compat = sum(composition['compatibility_scores']) / len(composition['compatibility_scores'])
        print(f"    Avg Compatibility: {avg_compat:.2f}")