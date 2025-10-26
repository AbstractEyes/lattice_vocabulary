"""
Template Synthesis Integration
================================

Integrates hierarchical template synthesis into SymbolicCaptionSynthesizer.
Adds 'template' composition mode that uses FULL_GENERIC templates.

Author: Phi + Claude
Date: 2025-10-26
"""

from __future__ import annotations

import random
from typing import List, Dict, Optional

# Import existing synthesizer
from geovocab2.data.prompt.maybe.symbolic_caption_synthesizer import (
    SymbolicCaptionSynthesizer,
    SymbolicCaption,
    TokenizerInterface,
    HuggingFaceTokenizerAdapter,
    UsageAccumulator,
)

# Import hierarchical template synthesizer
from geovocab2.data.prompt.hierarchical_template_synthesizer import (
    HierarchicalTemplateSynthesizer,
    SynthesisResult,
    load_bulk_captions_data,
)

# Import tree navigator
from geovocab2.data.prompt.symbolic_tree import CategoryTreeNavigator


# ============================================================================
# ENHANCED COMPOSER WITH TEMPLATE MODE
# ============================================================================

class EnhancedSymbolicComposer:
    """
    Enhanced composer that includes template-based synthesis.
    """

    def __init__(
        self,
        tree_navigator: CategoryTreeNavigator,
        template_synthesizer: Optional[HierarchicalTemplateSynthesizer] = None,
        usage_tracker: Optional[UsageAccumulator] = None
    ):
        """
        Args:
            tree_navigator: CategoryTreeNavigator instance
            template_synthesizer: HierarchicalTemplateSynthesizer instance
            usage_tracker: Optional UsageAccumulator for oversaturation prevention
        """
        self.tree = tree_navigator
        self.template_synth = template_synthesizer
        self.usage = usage_tracker

    def compose_template(self) -> List[str]:
        """
        Compose using hierarchical template synthesis.

        Returns:
            List of tokens forming the caption
        """
        if not self.template_synth:
            raise ValueError("Template synthesizer not configured")

        # Synthesize from template
        result = self.template_synth.synthesize()

        # Split into tokens
        tokens = result.text.split()

        return tokens

    def compose_template_batch(self, n: int) -> List[List[str]]:
        """
        Compose multiple captions using templates.

        Args:
            n: Number of captions to generate

        Returns:
            List of token lists
        """
        if not self.template_synth:
            raise ValueError("Template synthesizer not configured")

        results = self.template_synth.synthesize_batch(n)

        return [result.text.split() for result in results]

    def compose_hybrid_template(self, complexity: int = 3) -> List[str]:
        """
        Hybrid mode: use template as base, add tree-aware extensions.

        Args:
            complexity: Number of additional elements to add

        Returns:
            List of tokens
        """
        if not self.template_synth:
            raise ValueError("Template synthesizer not configured")

        # Start with template
        result = self.template_synth.synthesize()
        tokens = result.text.split()

        # Analyze what's already in the template
        used_paths = set(result.filled_paths.keys())

        # Add complementary elements from tree
        # Find unused but compatible categories
        available_categories = []

        for path in self.tree.get_all_leaf_paths():
            if path not in used_paths:
                available_categories.append(path)

        # Add a few random compatible elements
        num_additions = min(complexity, len(available_categories))

        for _ in range(num_additions):
            if not available_categories:
                break

            path = random.choice(available_categories)
            available_categories.remove(path)

            items = self.tree.get_items_at_path(path)
            if items:
                if self.usage:
                    item = self.usage.get_weighted_choice(items, path)
                else:
                    item = random.choice(items)

                tokens.append(item)

        return tokens


# ============================================================================
# ENHANCED SYNTHESIZER WITH TEMPLATE MODE
# ============================================================================

class EnhancedSymbolicCaptionSynthesizer(SymbolicCaptionSynthesizer):
    """
    Enhanced synthesizer that supports template-based composition.

    Adds new composition modes:
        - 'template': Pure template synthesis
        - 'hybrid_template': Template base + tree extensions
    """

    def __init__(
        self,
        tokenizer: TokenizerInterface,
        tree_navigator: CategoryTreeNavigator,
        template_synthesizer: Optional[HierarchicalTemplateSynthesizer] = None,
        prevent_oversaturation: bool = True,
        oversaturation_threshold: float = 0.05
    ):
        """
        Args:
            tokenizer: Tokenizer interface
            tree_navigator: CategoryTreeNavigator instance
            template_synthesizer: Optional HierarchicalTemplateSynthesizer
            prevent_oversaturation: Whether to prevent token oversaturation
            oversaturation_threshold: Max usage frequency before deprioritizing
        """
        # Initialize parent
        super().__init__(
            tokenizer=tokenizer,
            tree_navigator=tree_navigator,
            prevent_oversaturation=prevent_oversaturation,
            oversaturation_threshold=oversaturation_threshold
        )

        # Add template synthesizer
        self.template_synth = template_synthesizer

        # Create enhanced composer
        self.enhanced_composer = EnhancedSymbolicComposer(
            tree_navigator=tree_navigator,
            template_synthesizer=template_synthesizer,
            usage_tracker=self.usage
        )

    def synthesize(
        self,
        complexity: int = 3,
        composition_mode: str = 'tree_aware',
        **kwargs
    ) -> SymbolicCaption:
        """
        Synthesize a caption with enhanced template support.

        Args:
            complexity: Complexity level (1-5)
            composition_mode: One of:
                - 'template': Pure template synthesis
                - 'hybrid_template': Template + tree extensions
                - 'tree_aware': Existing tree-aware (default)
                - 'multi_domain': Existing multi-domain
                - 'association': Existing association
                - 'category': Existing category
                - 'spatial': Existing spatial
                - 'hybrid': Existing adaptive hybrid

        Returns:
            SymbolicCaption
        """
        # Handle new template modes
        if composition_mode == 'template':
            tokens = self.enhanced_composer.compose_template()
            return self._tokens_to_caption(tokens)

        elif composition_mode == 'hybrid_template':
            tokens = self.enhanced_composer.compose_hybrid_template(complexity)
            return self._tokens_to_caption(tokens)

        # Fallback to parent implementation for existing modes
        return super().synthesize(complexity, composition_mode, **kwargs)

    def get_stats(self) -> Dict:
        """Get enhanced statistics including template usage"""
        stats = super().get_stats()

        # Add template statistics if available
        if self.template_synth:
            template_stats = self.template_synth.get_stats()
            stats['template_synthesis'] = {
                'total_synthesized': template_stats['total_synthesized'],
                'unique_templates_used': template_stats['unique_templates_used'],
                'top_templates': template_stats['templates_used'],
                'top_paths_filled': template_stats['paths_filled'],
            }

        return stats


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_enhanced_synthesizer(
    tokenizer,
    prevent_oversaturation: bool = True,
    oversaturation_threshold: float = 0.05,
    template_oversaturation_threshold: float = 0.05
) -> EnhancedSymbolicCaptionSynthesizer:
    """
    Create an enhanced synthesizer with full template support.

    Args:
        tokenizer: HuggingFace tokenizer or TokenizerInterface
        prevent_oversaturation: Whether to prevent token oversaturation
        oversaturation_threshold: Threshold for tree-based synthesis
        template_oversaturation_threshold: Threshold for template synthesis

    Returns:
        Configured EnhancedSymbolicCaptionSynthesizer
    """
    # Wrap tokenizer if needed
    if not isinstance(tokenizer, TokenizerInterface):
        tokenizer = HuggingFaceTokenizerAdapter(tokenizer)

    # Create tree navigator
    tree = CategoryTreeNavigator()

    # Load bulk data
    bulk_data = load_bulk_captions_data()

    # Load FULL_GENERIC
    try:
        from geovocab2.data.prompt.bulk_caption_data import BulkCaptions
        full_generic = BulkCaptions.FULL_GENERIC
    except (ImportError, AttributeError):
        print("Warning: Could not load FULL_GENERIC, template mode will be unavailable")
        full_generic = []

    # Create template synthesizer
    template_synth = None
    if full_generic:
        template_synth = HierarchicalTemplateSynthesizer(
            bulk_captions_data=bulk_data,
            full_generic_templates=full_generic,
            prevent_oversaturation=prevent_oversaturation,
            oversaturation_threshold=template_oversaturation_threshold
        )

    # Create enhanced synthesizer
    synthesizer = EnhancedSymbolicCaptionSynthesizer(
        tokenizer=tokenizer,
        tree_navigator=tree,
        template_synthesizer=template_synth,
        prevent_oversaturation=prevent_oversaturation,
        oversaturation_threshold=oversaturation_threshold
    )

    return synthesizer


# ============================================================================
# TESTING
# ============================================================================

def test_template_integration():
    """Test the integrated template synthesis"""
    print("=" * 80)
    print("TEMPLATE SYNTHESIS INTEGRATION TEST")
    print("=" * 80)
    print()

    # Create synthesizer
    print("Creating enhanced synthesizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    synthesizer = create_enhanced_synthesizer(tokenizer)
    print("✓ Synthesizer created")
    print()

    # Test all modes
    modes = [
        'template',
        'hybrid_template',
        'tree_aware',
        'multi_domain',
    ]

    print("=" * 80)
    print("TESTING COMPOSITION MODES")
    print("=" * 80)
    print()

    for mode in modes:
        print(f"Mode: {mode}")
        print("-" * 80)

        try:
            for i in range(3):
                caption = synthesizer.synthesize(complexity=3, composition_mode=mode)
                print(f"  [{i+1}] {caption.to_text()}")
            print()
        except Exception as e:
            print(f"  Error: {e}")
            print()

    # Print statistics
    stats = synthesizer.get_stats()
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print()
    print(f"Total synthesized: {stats['total_synthesized']}")
    print(f"Composition modes used: {stats['composition_modes_used']}")
    print()

    if 'template_synthesis' in stats:
        template_stats = stats['template_synthesis']
        print("Template synthesis:")
        print(f"  Total: {template_stats['total_synthesized']}")
        print(f"  Unique templates: {template_stats['unique_templates_used']}")
        print()


if __name__ == "__main__":
    test_template_integration()