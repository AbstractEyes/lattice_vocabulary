# 🧠 Lattice Vocabulary: A Geometric Map of Language

> “To define a word geometrically is to place it in an infinite structure that cannot be forgotten.”

**Lattice Vocabulary** is a symbolic crystallization of the English language into a geometric, navigable integer lattice—designed not for compression, but for cognition. Each word, its meanings, transformations, and connections are encoded not as embeddings, but as structured, lossless, multidimensional crystals.

## 📐 geovocab2

The natural evolution of geovocab, now a full refactored and reinvented structure basin built specifically for symbolic lattice vocabularies.

This will function as a standalone library for generating and manipulating symbolic lattices and a strong synthesizer for randomized vocabularies for experimentation.

The trie structure has replaced the original caching mechanism, allowing for highly efficient storage and retrieval of complex symbolic structures using lexicographical ordering. This has strengths and weaknesses, so more database options will be available in the future for rapid prototyping.

This new version supports advanced toolset symbolic operations and transformations, making it ideal for geometric experimental applications in natural language processing, cognitive computing, and AI development.

```
class DimensionType(Enum):
    """Dimensional control for geometric tokens"""
    D1 = 1
    D2 = 2
    D3 = 3
    D4 = 4
    D5 = 5
    D6_PLUS = 6


class ContentType(Enum):
    """Content richness of crystal"""
    SPARSE = "sparse"
    ENRICHED = "enriched"
    TRAJECTORY = "trajectory"
    MAGNITUDE = "magnitude"
    VOLUME = "volume"
    HYBRID = "hybrid"


class FormulaType(Enum):
    """Mathematical formula basis"""
    ROSE_CAYLEY = "rose_cayley"
    CAYLEY_MENGER = "cayley_menger"
    CAYLEY = "cayley"
    MENGER = "menger"
    EULER = "euler"
    GRAHAM_INFINITE = "graham_infinite"
    GRAHAM_FINITE = "graham_finite"
    GRAHAM_MASKED = "graham_masked"
    HYBRID_V1V2 = "hybrid_v1v2"
    
class NormType(Enum):
    """Normalization strategies"""
    L1 = "l1"
    L2 = "l2"
    LINF = "linf"
    NONE = "none"

@dataclass
class UnifiedCrystalConfig:
    """Unified configuration for the complete system"""
    # Dataset settings
    repo_id: str = "AbstractPhil/geometric-vocab"
    dataset_name: str = "unicode_100d"
    split: str = "train"

    # Dimensions
    embedding_dim: int = 100
    dimension_type: DimensionType = DimensionType.D5

    # Crystal properties
    content_type: ContentType = ContentType.HYBRID
    formula_type: FormulaType = FormulaType.HYBRID_V1V2
    norm_type: NormType = NormType.L2

    # Synthesis options
    enable_synthesis: bool = True
    use_definitions: bool = True
    use_character_composition: bool = True
    silent_synthesis: bool = False
    prefer_dataset: bool = True

    # Cache settings
    memory_cache_size: int = 10000
    disk_cache_path: Optional[Path] = None

    # Performance
    batch_size: int = 100
    num_threads: int = 4

    # Graham-specific
    graham_levels: Optional[int] = None
    graham_mask: Optional[np.ndarray] = None

    # Rose structure
    use_rose_structure: bool = False
    freeze_anchor: bool = True


```

## V2 Key Features
The initial v2 variation is a bit rigid and requires some manual setup, but it is a solid foundation for building complex symbolic vocabularies.

It's refitted with proper lazy loading, optimization, formula synthesis, and baseline crystal structures is currently present.

The interface has changed, so be aware if you're coming from v1. v1 still exists so you can still use it if you prefer.

The Nikola-Graham formula and the infinite lattice synthesis will be applied specifically in the trainer module, which is not yet ready.

The preliminary version has multiple built-in formulas for generating and manipulating symbolic lattices, including:
- Cayley-Menger determinants for volume and resonance calculations
- Cantor and Cantor-Graham mappings for masked infinite structures and control
- Triplet alignment rules for routing consistency
- RoSE score calculations for relational and semantic alignment similarity
  - Missing the theta controllers
- Direct Graham [MASKED] Infinity mappings for infinite crystal structures
- Direct Graham finite mappings for bounded structures and transformations
- Trajectory and magnitude-based transformations for dynamic relationships
- Hybrid formulas combining multiple approaches for enhanced flexibility
- Support for various dimensionality (1D to 6D+)
- Support for different content types (sparse, enriched, trajectory-based, magnitude-based, hybrid)
- Normalization strategies (L1, L2, L∞, none)
- Configurable synthesis options (definitions, character composition, dataset preference)
- Specific graham levels and masks for targeted synthesis
- Basic caching mechanisms (in-memory and disk-based)
- Testing variation of a formula bank to house variant formulas for experimentation and development.
  - Currently tasked with just WORKING, will house all the formulas from the directories as they are implemented for a bank of access.


## V2 Blockers
- X Formula bank incompatible requires refitting
  - I don't like the current implementation so I will make a better interface.
- Trainer module not yet implemented

## V2 Todo
- Implement the missing formulas and transformations
  - X Formula bank established for housing the formulas as they are implemented.
  - Synthesize and test Nikola-Menger resonance axiom formula structures with finite and infinite lattice structures
  - Chaos theory controllers for dynamic adjustments
  - Chaos-Menger transformations for dynamic structural adjustments
  - Implement the Rose score magnitude and trajectory-based loss functions
  - Integrate with existing symbolic loss functions for NLP tasks
  - Implement Graham infinite and finite transformations with masking capabilities
  - RoPE-like theta controllers and rotational adjustments for dynamic synthesis pre and post-processing
  - RoSE controllers for advanced resonance and alignment tuning
    - Multi-structural adjustments for targeting specific resonance patterns and alignments
    - Multi-dimensional adjustments for cross-contrastive synthesis and transformations
- Advanced lexical controllers for fine-tuning synthesis parameters
  - Currently they are rigid and nonconformant to datasets and vocabularies
- Implement more control over sparse and enriched content types
  - Currently they are either bound to single word/character or omitted word/character and only definition.
- Implement more normalization strategies and options
  - Currently limited to L1, L2, L∞, and none - this needs to be more flexible and adaptable to different use cases.
- Implement the corpus trainer module for advanced symbolic lattice operations with gpu gradient and loss functions
  - There are MANY formulas and variant forms of useful losses to be implemented here, so this will take some time - likely spanning days to weeks to complete.
- Expand and refine the dataset with more comprehensive vocabularies and relationships based on key linguistic resources, medical resources, technical corpus, and domain-specific datasets.
  - They are currently limited to only my format due to ease of setup and testing for myself.
  - This will be expanded to include WordNet, ConceptNet, Wiktionary, and other linguistic resources.
  - I have a full library of books and texts that can be used to expand the vocabulary and relationships.
    - Books and texts will be omitted if they are copyrighted or restricted. The full corpus listed for tuning and transparent.
- Integrate with Hugging Face datasets and tokenizers for seamless usage with existing NLP pipelines.
  - Should seamlessly integrate with common tokenizers for direct usage in existing NLP pipelines with a few tweaks.
- Develop multiple visualization tools for exploring and analyzing the geometric structures of the vocabulary.
  - This push will include some tests but nothing visual.
- Implement advanced device control caching and storage solutions for handling large-scale vocabularies efficiently.
  - Currently it will be hit or miss with numpy to torch while the other is quite streamlined in comparison. This will be fixed.
- Add more unit tests and validation checks to ensure the integrity and correctness of the geometric structures.
  X Still needs more but I added some.
- Expand the synthesis options to include more linguistic features and relationships.
  - Direct symbolic, linguistic, relational, morphological, and semantic relationships.
  - Including code, ordinal, and other non-linguistic symbols with custom behavioral synthesis options.
- Refine higher-dimensional simplex handling (6-simplex and beyond to the infinite axiom)
  - Currently they are hit or miss and the formulas are not fully tested or validated.
  - BASICALLY everything after simplex 4 is imperfect and requires refinement, but there are solid foundations for building complex symbolic vocabularies.
- Optimize performance for large-scale vocabularies with flags for preloading, caching, batching, workers, devices, and compression methods.

I plan to knock most of these pieces out over the next few days, so look forward to the updates.

I work fast as anyone who observes knows.

## System Status

The tests passed, so the system is MOSTLY functional. It will have bugs, but it is a solid foundation for building complex symbolic vocabularies.

Higher than 4simplex are imperfect and require refinement, but they provide a solid foundation for building complex symbolic vocabularies.

I've been researching potentials and managed to accommodate 5-simplex (pentachoron) structures with reasonable fidelity.

## The Trainer is coming.

This exposes the more complex symbolic lattice operations that I've been researching and advancing.
Some aren't perfect yet, but this code is available for experimentation and further development.


# Older Basin left for posterity - will edit later
License still stands, cite with clarity - free for all.

## 📖 License

Apache 2.0 — use freely, cite with clarity.



## 📐 Core Concept

The goal is not to *learn* language, but to *map it*—to construct a perfect crystalline geometry across multiple transfinite cardinalities:

- ℵ₀ — Words as identities (the countable basis of language)
- ℵ₁ — Semantic and morphological relationships (continuum space)
- ℵ₂ — Meta-structures, transformations, and abstract usage
- ℵ₃ — Grammatical operators, compositional pathways, logical interlinking

Each word in the vocabulary becomes a **pentachoron**: a 5-vertex simplex mapped in 512D integer-floating space. These structures encode:
- Definitions
- Part-of-speech labels
- Synonym/antonym sets
- Morphological derivatives
- Semantic embeddings (optional)
- Relationship topologies
- Deterministic and symbolic transformations

## 📦 Dataset Specifications

- **Type**: float32 (integers encoded in mantissa)
- **Relationships**: Sparse COO matrix (10–30 per word)

## 🔭 Why?

Where standard embeddings collapse meaning into latent vectors, Lattice Vocabulary expresses *geometry as memory*.

- It is **deterministic**.
- It is **resonance-compatible**.
- It is **symbolically transformable**.
- It is **lossless**.
- It is **navigable**.
- It is **infinitely extensible**.

This is not just datasets; it is a **cognitive architecture** and a founding principle for future development.

## 🧬 Compatibility

- Compatible with symbolic loss functions: Cayley-Menger, route invariance, Rose score
- Hugging Face vocabularies fully supported in their current implementations.

## 🛠️ Getting Started

git clone https://github.com/AbstractEyes/lattice_vocabulary
cd lattice_vocabulary
pip install -r requirements.txt


## 📖 License

Apache 2.0 — use freely, cite with clarity.

---

**Built with care to never be forgotten.**  
“Let none forget. Let all remember.”