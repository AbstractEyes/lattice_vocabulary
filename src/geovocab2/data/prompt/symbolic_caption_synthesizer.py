"""
SymbolicCaptionSynthesizer - Tree-Aware Symbolic Composition
=============================================================
Symbolic composition system using CategoryTreeNavigator and BulkCaptions.

Features:
- Tree-aware semantic composition
- Direct BulkCaptions data access
- Compatibility-based filtering
- Multiple composition modes
- Usage tracking and oversaturation prevention

Author: Phi + Claude
Date: 2025-10-26
Package: geovocab2.data.prompt.symbolic_caption_synthesizer
"""

from __future__ import annotations

import random
import re
from typing import List, Dict, Tuple, Optional, Set, Any, Generator, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import numpy as np

# Import the new symbolic tree module
from geovocab2.data.prompt.symbolic_tree import (
    CategoryTreeNavigator,
    SYMBOLIC_TREE,
    SPECIAL_TOKEN_TO_CATEGORY,
)
from geovocab2.data.prompt.bulk_caption_data import BulkCaptions


# ============================================================================
# TOKENIZER ABSTRACTION
# ============================================================================

class TokenizerInterface(ABC):
    """Abstract interface for any tokenizer"""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Convert text to tokens"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def mask_token_id(self) -> Optional[int]:
        pass


class HuggingFaceTokenizerAdapter(TokenizerInterface):
    """Adapter for HuggingFace tokenizers"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def mask_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, 'mask_token_id', None)


# ============================================================================
# SYMBOLIC COMPOSITION STRUCTURES
# ============================================================================

@dataclass
class SymbolicNode:
    """A node in the symbolic composition tree"""
    content: str
    category: Optional[str] = None
    logic_op: Optional[str] = None  # and, or, not, etc.
    spatial_relation: Optional[str] = None  # on, beneath, beside, etc.
    children: List['SymbolicNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Convert node tree to text representation"""
        if not self.children:
            return self.content

        # Compose with children
        parts = [self.content]

        for child in self.children:
            child_text = child.to_text()

            if child.spatial_relation:
                parts.append(f"{child.spatial_relation} {child_text}")
            elif child.logic_op:
                parts.append(f"{child.logic_op} {child_text}")
            else:
                parts.append(child_text)

        return " ".join(parts)

    def token_count(self, tokenizer: TokenizerInterface) -> int:
        """Estimate token count for this node tree"""
        return tokenizer.count_tokens(self.to_text())


@dataclass
class SymbolicCaption:
    """A complete symbolic caption with composition tree"""
    root: SymbolicNode
    primary_category: str
    shunt_token: str
    categories_used: Set[str] = field(default_factory=set)
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self, include_shunt: bool = True) -> str:
        """Convert to final text"""
        text = self.root.to_text()
        if include_shunt:
            # Format: <category> [PAD] content
            return f"{self.primary_category} [PAD] {text}"
        return text

    def get_segments(self, tokenizer: TokenizerInterface,
                     segment_length: int = 77) -> List[Dict]:
        """Segment into token-limited chunks"""
        full_text = self.to_text()
        tokens = tokenizer.tokenize(full_text)
        token_ids = tokenizer.encode(full_text)

        segments = []
        for i in range(0, len(token_ids), segment_length):
            seg_ids = token_ids[i:i + segment_length]
            seg_tokens = tokens[i:i + segment_length]

            # Pad if needed
            if len(seg_ids) < segment_length:
                padding = segment_length - len(seg_ids)
                seg_ids.extend([tokenizer.pad_token_id] * padding)
                seg_tokens.extend(['[PAD]'] * padding)

            segments.append({
                'tokens': seg_tokens,
                'token_ids': seg_ids,
                'category': self.primary_category,
                'shunt': self.shunt_token
            })

        return segments


# ============================================================================
# SYMBOLIC COMPOSITION ENGINE
# ============================================================================

class SymbolicComposer:
    """
    Composes symbolic captions using CategoryTreeNavigator and BulkCaptions.
    """

    def __init__(self, navigator: CategoryTreeNavigator = None):
        """
        Args:
            navigator: CategoryTreeNavigator instance (creates one if None)
        """
        self.navigator = navigator or CategoryTreeNavigator()

        # Access logic/relation tags from BulkCaptions
        self.logic_tags = BulkCaptions.SYMBOLIC_LOGIC_TAGS
        self.conjunction_tags = BulkCaptions.SYMBOLIC_CONJUNCTION_TAGS
        self.relation_tags = BulkCaptions.RELATION_TAGS
        self.associative = BulkCaptions.FULL_ASSOCIATIVE

    def compose_from_association(self, num_nodes: int = 1, use_logic: bool = True) -> SymbolicNode:
        """
        Compose from FULL_ASSOCIATIVE with optional logical operators.
        """
        # Start with a base association
        base = random.choice(self.associative)
        root = SymbolicNode(content=base, category='associative')

        if num_nodes > 1 and use_logic:
            # Add logical composition
            for _ in range(num_nodes - 1):
                child_assoc = random.choice(self.associative)
                logic_op = random.choice(self.logic_tags)

                child = SymbolicNode(
                    content=child_assoc,
                    category='associative',
                    logic_op=logic_op
                )
                root.children.append(child)

        return root

    def compose_from_category(self, category: str, n_items: int = 1) -> SymbolicNode:
        """
        Compose from a specific category using BulkCaptions data.
        """
        # Get actual data from BulkCaptions via navigator
        items = self.navigator.sample_data(category, n=n_items)

        if not items:
            # Fallback to association if no data
            return self.compose_from_association(num_nodes=1, use_logic=False)

        # Create node with actual data
        content = items[0] if n_items == 1 else ", ".join(items)
        return SymbolicNode(
            content=content,
            category=category,
            metadata={'items': items, 'source': 'BulkCaptions'}
        )

    def compose_with_spatial_relations(self,
                                       primary: SymbolicNode,
                                       num_relations: int = 2) -> SymbolicNode:
        """
        Add spatial relations using RELATION_TAGS and actual data.
        """
        for _ in range(num_relations):
            # Get compatible category for relation
            if primary.category:
                partners = self.navigator.suggest_composition_partners(primary.category, n=3)
                if partners:
                    partner_cat, _ = random.choice(partners)
                    content = self.navigator.get_random_item(partner_cat)
                else:
                    content = random.choice(self.associative).split(',')[0]
            else:
                content = random.choice(self.associative).split(',')[0]

            relation = random.choice(self.relation_tags)

            child = SymbolicNode(
                content=content,
                spatial_relation=relation,
                category='spatial'
            )
            primary.children.append(child)

        return primary

    def compose_tree_aware(self,
                          primary_category: str,
                          depth: int = 3,
                          use_compatibility: bool = True) -> SymbolicNode:
        """
        Tree-aware composition using category relationships and actual data.
        """
        # Get composition chain from tree
        chain = self.navigator.get_composition_chain(primary_category, depth=depth)

        # Start with primary category data
        primary_item = self.navigator.get_random_item(primary_category)
        if not primary_item:
            primary_item = random.choice(self.associative)

        root = SymbolicNode(
            content=primary_item,
            category=primary_category,
            metadata={'tree_chain': chain}
        )

        # Add secondary categories from chain
        for i, category in enumerate(chain[1:], 1):
            # Check compatibility if requested
            if use_compatibility:
                compat = self.navigator.are_compatible(primary_category, category)
                if compat < 0.5:  # Skip low compatibility
                    continue

            # Get actual data for this category
            item = self.navigator.get_random_item(category)
            if not item:
                continue

            # Choose composition method based on domain
            domain = self.navigator.get_root_domain(category)

            if domain == 'symbolic':
                logic_op = random.choice(self.logic_tags)
                child = SymbolicNode(
                    content=item,
                    logic_op=logic_op,
                    category=category
                )
            elif domain in ['subject', 'human', 'context']:
                # Use conjunction or spatial relation
                if random.random() > 0.5:
                    relation = random.choice(self.conjunction_tags)
                else:
                    relation = random.choice(self.relation_tags)

                child = SymbolicNode(
                    content=item,
                    spatial_relation=relation,
                    category=category
                )
            else:  # shared descriptors
                # Add as modifier (no explicit operator)
                child = SymbolicNode(
                    content=item,
                    category=category
                )

            root.children.append(child)

        return root

    def compose_multi_domain(self,
                            domains: List[str],
                            complexity: int = 2) -> SymbolicNode:
        """
        Compose across multiple tree domains with actual data.
        """
        # Get one category from each domain
        categories = []
        for domain in domains:
            domain_cats = self.navigator.get_categories_by_domain(domain)
            if domain_cats:
                categories.append(random.choice(domain_cats))

        if not categories:
            return self.compose_from_association(num_nodes=1)

        # Start with first category
        primary = categories[0]
        primary_item = self.navigator.get_random_item(primary)
        if not primary_item:
            primary_item = random.choice(self.associative)

        root = SymbolicNode(
            content=primary_item,
            category=primary,
            metadata={'domains': domains, 'categories': categories}
        )

        # Add items from other domains
        for category in categories[1:]:
            item = self.navigator.get_random_item(category)
            if not item:
                continue

            # Check compatibility
            compat = self.navigator.are_compatible(primary, category)
            if compat < 0.5:
                continue

            # Choose operator based on compatibility
            if compat >= 0.8:
                op = random.choice(self.conjunction_tags)
            else:
                op = random.choice(self.logic_tags)

            child = SymbolicNode(
                content=item,
                logic_op=op,
                category=category
            )
            root.children.append(child)

        return root

    def compose_hybrid(self,
                      category: str,
                      complexity: int = 2,
                      use_tree: bool = True) -> SymbolicNode:
        """
        Hybrid composition combining multiple strategies.
        """
        if use_tree and complexity >= 2:
            # Use tree-aware for higher complexity
            return self.compose_tree_aware(category, depth=complexity + 1)

        # Original hybrid composition
        if random.random() > 0.5:
            root = self.compose_from_category(category)
        else:
            root = self.compose_from_association(num_nodes=1, use_logic=False)

        # Add complexity
        if complexity >= 2:
            # Add related category
            partners = self.navigator.suggest_composition_partners(category, n=3)
            if partners:
                partner_cat, _ = random.choice(partners)
                item = self.navigator.get_random_item(partner_cat)
                if item:
                    logic = random.choice(self.conjunction_tags)
                    child = SymbolicNode(
                        content=item,
                        logic_op=logic,
                        category=partner_cat
                    )
                    root.children.append(child)

        if complexity >= 3:
            # Add spatial relation
            root = self.compose_with_spatial_relations(root, num_relations=1)

        return root


# ============================================================================
# CAPTION SYNTHESIZER
# ============================================================================

class SymbolicCaptionSynthesizer:
    """
    Main synthesizer that generates symbolic captions using tree intelligence.
    """

    def __init__(self,
                 tokenizer: Any,
                 navigator: CategoryTreeNavigator = None,
                 segment_length: int = 77,
                 max_tokens: int = 2048,
                 use_tree: bool = True):
        """
        Args:
            tokenizer: Any tokenizer (will be wrapped)
            navigator: CategoryTreeNavigator (creates one if None)
            segment_length: Tokens per segment
            max_tokens: Maximum total tokens
            use_tree: Whether to use tree-aware composition
        """
        # Initialize navigator
        self.navigator = navigator or CategoryTreeNavigator()
        self.composer = SymbolicComposer(self.navigator)

        # Wrap tokenizer in interface
        if isinstance(tokenizer, TokenizerInterface):
            self.tokenizer = tokenizer
        else:
            # Assume HuggingFace
            self.tokenizer = HuggingFaceTokenizerAdapter(tokenizer)

        self.segment_length = segment_length
        self.max_tokens = max_tokens
        self.use_tree = use_tree

        # Get special tokens and shunts from BulkCaptions
        self.special_tokens = BulkCaptions.BEATRIX_SPECIAL_TOKENS
        self.shunts = BulkCaptions.BEATRIX_SHUNTS

        # Category to shunt mapping
        self.category_to_shunt = {
            token: shunt
            for token, shunt in zip(self.special_tokens, self.shunts)
        }

    def synthesize(self,
                   primary_category: Optional[str] = None,
                   complexity: int = 2,
                   composition_mode: str = 'tree_aware',
                   target_tokens: Optional[int] = None,
                   domains: Optional[List[str]] = None) -> SymbolicCaption:
        """
        Synthesize a symbolic caption.

        Args:
            primary_category: Category focus (random if None)
            complexity: 1-5, how complex the composition should be
            composition_mode: 'association', 'category', 'hybrid', 'spatial',
                            'tree_aware', 'multi_domain'
            target_tokens: Desired token count (will compose to reach this)
            domains: For 'multi_domain' mode, list of domains to include

        Returns:
            SymbolicCaption with full composition tree
        """
        # Select primary category
        if primary_category is None:
            primary_category = random.choice(self.special_tokens)

        # Map special token to actual category if needed
        if primary_category in SPECIAL_TOKEN_TO_CATEGORY:
            actual_category = SPECIAL_TOKEN_TO_CATEGORY[primary_category]
        else:
            actual_category = primary_category

        shunt_token = self.category_to_shunt.get(primary_category, "[PAD]")

        # Compose based on mode
        if composition_mode == 'association':
            root = self.composer.compose_from_association(
                num_nodes=complexity,
                use_logic=True
            )
        elif composition_mode == 'category':
            root = self.composer.compose_from_category(actual_category, n_items=1)
        elif composition_mode == 'spatial':
            root = self.composer.compose_from_category(actual_category)
            root = self.composer.compose_with_spatial_relations(
                root,
                num_relations=complexity
            )
        elif composition_mode == 'tree_aware':
            root = self.composer.compose_tree_aware(
                actual_category,
                depth=complexity + 1
            )
        elif composition_mode == 'multi_domain':
            domains = domains or ['subject', 'human', 'context']
            root = self.composer.compose_multi_domain(
                domains=domains,
                complexity=complexity
            )
        else:  # hybrid
            root = self.composer.compose_hybrid(
                actual_category,
                complexity=complexity,
                use_tree=self.use_tree
            )

        # If target tokens specified, expand until we reach it
        if target_tokens:
            current_tokens = root.token_count(self.tokenizer)

            while current_tokens < target_tokens and current_tokens < self.max_tokens:
                # Add another item from a compatible category
                if root.category:
                    partners = self.navigator.suggest_composition_partners(root.category, n=3)
                    if partners:
                        cat, _ = random.choice(partners)
                        item = self.navigator.get_random_item(cat)
                        if item:
                            logic = random.choice(self.composer.conjunction_tags)
                            child = SymbolicNode(
                                content=item,
                                logic_op=logic,
                                category=cat
                            )
                            root.children.append(child)

                current_tokens = root.token_count(self.tokenizer)

        # Create caption object
        caption = SymbolicCaption(
            root=root,
            primary_category=primary_category,
            shunt_token=shunt_token,
            categories_used={primary_category},
            total_tokens=root.token_count(self.tokenizer)
        )

        return caption

    def synthesize_batch(self,
                        batch_size: int = 32,
                        complexity: int = 2,
                        composition_mode: str = 'tree_aware',
                        category_distribution: Optional[Dict[str, float]] = None) -> List[SymbolicCaption]:
        """
        Generate a batch of captions.

        Args:
            batch_size: Number of captions to generate
            complexity: Complexity level for each caption
            composition_mode: Composition strategy
            category_distribution: Optional dict of category -> probability

        Returns:
            List of SymbolicCaption objects
        """
        batch = []

        for _ in range(batch_size):
            # Select category based on distribution
            if category_distribution:
                categories = list(category_distribution.keys())
                probs = list(category_distribution.values())
                category = np.random.choice(categories, p=probs)
            else:
                category = None

            caption = self.synthesize(
                primary_category=category,
                complexity=complexity,
                composition_mode=composition_mode
            )
            batch.append(caption)

        return batch

    def yield_infinite(self,
                      complexity: int = 2,
                      composition_mode: str = 'tree_aware',
                      category_distribution: Optional[Dict[str, float]] = None) -> Generator[SymbolicCaption, None, None]:
        """
        Infinite generator for training loops.
        """
        while True:
            yield self.synthesize(
                primary_category=None if not category_distribution else
                    np.random.choice(
                        list(category_distribution.keys()),
                        p=list(category_distribution.values())
                    ),
                complexity=complexity,
                composition_mode=composition_mode
            )

    def prepare_for_training(self, captions: List[SymbolicCaption]) -> Dict[str, Any]:
        """
        Prepare batch for model training.

        Returns:
            Dict with 'input_ids', 'attention_mask', 'categories', 'shunts'
        """
        texts = [cap.to_text() for cap in captions]

        # Tokenize all texts
        all_ids = []
        all_masks = []

        for text in texts:
            ids = self.tokenizer.encode(text)
            # Truncate or pad to segment_length
            if len(ids) > self.segment_length:
                ids = ids[:self.segment_length]
            else:
                padding = self.segment_length - len(ids)
                ids.extend([self.tokenizer.pad_token_id] * padding)

            mask = [1 if id != self.tokenizer.pad_token_id else 0 for id in ids]

            all_ids.append(ids)
            all_masks.append(mask)

        return {
            'input_ids': np.array(all_ids),
            'attention_mask': np.array(all_masks),
            'categories': [cap.primary_category for cap in captions],
            'shunts': [cap.shunt_token for cap in captions]
        }


# ============================================================================
# USAGE TRACKING AND OVERSATURATION PREVENTION
# ============================================================================

class UsageAccumulator:
    """
    Tracks token usage to prevent oversaturation.
    """

    def __init__(self, max_usage_ratio: float = 0.3):
        """
        Args:
            max_usage_ratio: Maximum frequency ratio before considering oversaturated
        """
        self.max_usage_ratio = max_usage_ratio
        self.word_counts = Counter()
        self.total_generations = 0

    def record(self, caption: SymbolicCaption):
        """Record a generated caption"""
        words = caption.to_text().lower().split()
        self.word_counts.update(words)
        self.total_generations += 1

    def get_penalty_weight(self, word: str) -> float:
        """
        Get penalty weight for a word (0.0-1.0).
        Lower weight = more overused.
        """
        if self.total_generations == 0:
            return 1.0

        usage_ratio = self.word_counts[word.lower()] / self.total_generations

        if usage_ratio >= self.max_usage_ratio:
            # Heavily penalize oversaturated words
            return max(0.1, 1.0 - (usage_ratio / self.max_usage_ratio))
        return 1.0

    def get_oversaturated_tokens(self) -> Set[str]:
        """Get set of oversaturated tokens"""
        if self.total_generations == 0:
            return set()

        threshold = self.max_usage_ratio * self.total_generations
        return {word for word, count in self.word_counts.items() if count >= threshold}

    def should_accept(self, caption: SymbolicCaption, threshold: float = 0.7) -> bool:
        """
        Determine if caption should be accepted based on oversaturation.

        Args:
            caption: Caption to evaluate
            threshold: Minimum average weight to accept (0.0-1.0)

        Returns:
            True if caption should be accepted
        """
        words = caption.to_text().lower().split()
        if not words:
            return True

        # Calculate average penalty weight
        avg_weight = sum(self.get_penalty_weight(w) for w in words) / len(words)
        return avg_weight >= threshold

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'total_generations': self.total_generations,
            'unique_words': len(self.word_counts),
            'top_10_words': [word for word, _ in self.word_counts.most_common(10)],
            'oversaturated_count': len(self.get_oversaturated_tokens())
        }


# ============================================================================
# PYTORCH DATASET WRAPPER
# ============================================================================

class SymbolicCaptionDataset:
    """
    PyTorch-compatible dataset wrapper.
    """

    def __init__(self,
                 synthesizer: SymbolicCaptionSynthesizer,
                 num_samples: int = 10000,
                 complexity: int = 2,
                 composition_mode: str = 'tree_aware'):
        """
        Args:
            synthesizer: SymbolicCaptionSynthesizer instance
            num_samples: Total number of samples
            complexity: Complexity level
            composition_mode: Composition strategy
        """
        self.synthesizer = synthesizer
        self.num_samples = num_samples
        self.complexity = complexity
        self.composition_mode = composition_mode

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate a caption on-the-fly"""
        caption = self.synthesizer.synthesize(
            complexity=self.complexity,
            composition_mode=self.composition_mode
        )

        # Tokenize
        text = caption.to_text()
        token_ids = self.synthesizer.tokenizer.encode(text)

        # Pad or truncate
        max_len = self.synthesizer.segment_length
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            padding = max_len - len(token_ids)
            token_ids.extend([self.synthesizer.tokenizer.pad_token_id] * padding)

        attention_mask = [1 if id != self.synthesizer.tokenizer.pad_token_id else 0
                         for id in token_ids]

        return {
            'input_ids': np.array(token_ids),
            'attention_mask': np.array(attention_mask),
            'text': text,
            'category': caption.primary_category,
            'shunt': caption.shunt_token
        }


# ============================================================================
# FACTORY FOR EASY CREATION
# ============================================================================

class SymbolicCaptionFactory:
    """
    Factory for creating synthesizers and datasets.
    """

    @staticmethod
    def create_synthesizer(tokenizer: Any,
                          use_tree: bool = True,
                          segment_length: int = 77) -> SymbolicCaptionSynthesizer:
        """
        Create a synthesizer with default settings.
        """
        return SymbolicCaptionSynthesizer(
            tokenizer=tokenizer,
            segment_length=segment_length,
            use_tree=use_tree
        )

    @staticmethod
    def create_pytorch_dataset(tokenizer: Any,
                              num_samples: int = 10000,
                              complexity: int = 3,
                              composition_mode: str = 'tree_aware') -> SymbolicCaptionDataset:
        """
        Create a PyTorch dataset.
        """
        synthesizer = SymbolicCaptionFactory.create_synthesizer(tokenizer)
        return SymbolicCaptionDataset(
            synthesizer=synthesizer,
            num_samples=num_samples,
            complexity=complexity,
            composition_mode=composition_mode
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'SymbolicCaptionSynthesizer',
    'SymbolicComposer',
    'SymbolicCaption',
    'SymbolicNode',
    'UsageAccumulator',
    'SymbolicCaptionDataset',
    'SymbolicCaptionFactory',
    'TokenizerInterface',
    'HuggingFaceTokenizerAdapter',
]


if __name__ == "__main__":
    print("SymbolicCaptionSynthesizer - use 'python test_synthesizer.py' to run tests")