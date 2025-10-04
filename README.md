# 🧠 Lattice Vocabulary: A Geometric Map of Language

> “To define a word geometrically is to place it in an infinite structure that cannot be forgotten.”

**Lattice Vocabulary** is a symbolic crystallization of representative multimodal language into a geometric, navigable systemic-driven lattice. This is designed not for compression, but for cognition. Each word, its meanings, transformations, and connections are encoded not as just embeddings, but as structured, lossless, multidimensional shapes.

## geovocab: The Original
The original geovocab structure is still present in the `geovocab` directory for legacy support and reference.
If you have problems you can revert to an earlier commit or use the original code as needed.

# 📐 geovocab2

* The natural evolution of geovocab, now a full refactored and reinvented structure basin built specifically for symbolic lattice vocabularies.
* This will function as a standalone library for generating and manipulating symbolic lattices while fulfilling the role for a strong synthesizer for randomized vocabularies for experimentation.
* The trie structure will properly replace the original caching mechanism for large vocabulary ngram structures, allowing for highly efficient storage and retrieval of complex symbolic structures using lexicographical ordering. This has strengths and weaknesses, so more database options will be available in the future for rapid prototyping.
* This new version supports advanced toolset symbolic operations and transformations, making it ideal for geometric experimental applications in natural language processing, cognitive computing, and AI development.
* Equipped with pyi for autocomplete and a standardized package structure for ease of use.
* Formatted specifically for easy expansion with IDE by utilizing autocomplete pyi capacity while retaining that core functionality for imported projects.


# Resolved immediate issues 10/4/2025
- Importing from geovocab2 may be problematic due to the change in structure and organization.
- I'm looking into solutions for relative import to ensure individual systems function correctly, likely an __init__.py file in each directory to ensure proper module recognition and import resolution.
- - This should be tested and ready by tomorrow at some point and the README updated accordingly.
- - This has been resolved with __init__.py and __init__.pyi files in each directory, with a baseline generate_stubs.py included in src/geovocab2/tools meant to be run as needed to regenerate the stubs for type checking and IDE support.

# Current Access Import Structure:
```
from geovocab2.shapes.factory import SimplexFactory
from geovocab2.shapes.formula import CayleyMengerFromSimplex
from geovocab2.shapes.fusion import LexicalSimplexSynthesizer
```

## I've made the dev branch the main branch now.

This is primarily due to AI not seeing the dev branch at all so they aren't helping me directly edit the repo and haven't been able to for days at this point.

The original code is all still present in the geovocab directory - so you can access the legacy code without interference... hopefully.

Let me know if there are any critical issues and I'll attempt to resolve it until v2 is fully operational with a matching interface as v1.


## V2 Key Features
The initial v2 variation is a bit rigid and requires some manual setup, but it is a solid foundation for building complex symbolic vocabularies.

The interface has changed, so be aware if you're coming from v1. v1 still exists so you can still use it if you prefer.

# Explanation for changes

geovocab2 is a complete refit and rewrite of the original geovocab structure.

It uses a series of principles that the original did not have;

1. simple objects with hierarchical inheritance by 1 or 2 levels at most.
2. overload operators with self-documenting methods and properties.
3. reusable formulas and transformations each standalone capable of being tested and validated independently.
4. factory structures meant to efficiently generate and manage complex symbolic lattices with minimal overhead.
5. synthesis options for generating complex symbolic lattices with various configurations and parameters.

# Core directive:

1. torch and numpy compatible.
2. everything compiles or translates directly to torch tensors for gpu acceleration.
3. everything is independently testable and usable with at most one file dependency.
4. everything is inherited from a base class with common methods and properties.
5. import footprint is small and efficient, useful for rapid prototyping and experimentation.
6. everything is human-readable and AI friendly, so it can be easily understood and modified by both humans and AI systems.

These principles allow for a more modular and maintainable codebase that is easier to extend and adapt to new requirements.

To ensure these structures align; I've specifically designed the inheritance hierarchy to be shallow and focused. 
The wording is concise and meaningful. The structures are designed to be easily understandable and modifiable.
The hierarchy follows a clear pattern. The torch and numpy compatibility is maintained throughout the codebase with careful attention to detail.

Some things may not compile at first, but it will be fixed as I go through and test everything with direct scrutiny.

# Structure Hierarchy

```
BaseFormula > OverriddenFormula > SpecificFormula
    Example:
        BaseFactory -> SimplexFactory -> Produces simplex shaped structures 
        BaseFormula -> CayleyMengerFromSimplex -> Produces Cayley-Menger determinants from simplex structures
        CompositionBase -> LexicalSimplexSynthesizer -> Synthesizes lexical simplex structures from vocab data and shapes
```

The structures are formatted specifically to be expandable and utility-focused, so they can be used in a variety of contexts and applications.

They are AI friendly and human-readable, so they can be easily understood and modified by both humans and AI systems.

# Formula Playground

The current implementation has these directories and files for the formula playground:

`src/geovocab2/shapes/formula/`
- engineering: 
- - atomic.py for basic atomic operations and structures, not so useful on its own but foundational if necessary.
- - fundamental.py for more advanced basic operations like atan2, vector norms, and other basic mathematical operations.
- - geometric.py for geometric analysis and transformations like raycast, geometric point calculations, and other geometric operations.
- - projection.py for projection operations like to_homogenous, orthographic, perspective, and other projection methods.
- - simplex.py for simplex structures and operations like generating simplex shapes, calculating volumes, and other simplex-related operations.
- - wave.py for advanced wave and resonance operations like heat diffusion, reactive diffusion, wave propagation, and other wave-related operations.
- symbolic:
- - cantor.py for advanced symbolic infinity operations.
- - cayley.py unpopulated, most of the cayley operations are in cayley_menger.py
- - cayley_menger.py for Cayley-Menger determinant calculations and related operations, backbone of core.
- - einstein.py for Einstein summation operations and related tensor manipulations.
- - euler.py for Euler characteristic calculations and related topological operations.
- - graham.py unpopulated
- - hawking.py for Hawking radiation and black hole related symbolic operations.
- - hooke.py for Hooke's law and related elastic deformation operations.
- - menger.py unpopulated
- - newton.py for Newtonian mechanics and related physical operations.
- - nikola.py for Nikola Tesla's resonance, frequency, and energy operations.

- specifically k-simplex supported on many of these files, but not all yet.
- The k-simplex depth is nth, so 0-simplex is a point, 1-simplex is a line, 2-simplex is a triangle, 3-simplex is a tetrahedron, 4-simplex is a pentachoron, and so on.




# Factories
The factories are designed to produce specific types of symbolic lattice structures based on the provided configuration.

`src/geovocab2/shapes/factory/`
- factory_base.py for the base factory class and common methods.
- simplex_factory.py for generating simplex-shaped symbolic lattice structures.
- legacy_factory.py for direct 1:1 compatibility with the original geovocab structures, almost ready.
- factory_dataloaders.py for loading datasets and vocabularies into the factories.

# Synthesis

These are specifically curated to synthesize complex symbolic lattice structures from vocabularies and shapes interconnected.
Meant to be independent and reusable for various synthesis tasks.


`src/geovocab2/shapes/fusion/`
- composition_base.py for the base composition class and common character embedding synthesis methods.
- lexical_simplex_synthesizer.py for synthesizing lexical simplex structures from vocab data and shapes.
- wordnet_synthesizer.py for synthesizing structures using WordNet data, doesn't do much yet. Use simplex for now.


## V2 Blockers
- X Formula bank incompatible requires refitting
  - I don't like the current implementation so I will make a better interface.
- X Structure refitting for baseline new structure established
- Trainer module not yet implemented

## V2 Todo
- Implement the missing formulas and transformations
  - X Formula bank established for housing the formulas as they are implemented.
  - X Synthesize and test Nikola-Menger resonance axiom formula structures with finite and infinite lattice structures
  - Chaos theory controllers for dynamic adjustments
  - X Chaos-Menger transformations for dynamic structural adjustments
  - Implement the Rose score magnitude and trajectory-based loss functions
  - Integrate with existing symbolic loss functions for NLP tasks
  - X Implement Graham infinite and finite transformations with masking capabilities
  - RoPE-like theta controllers and rotational adjustments for dynamic synthesis pre and post-processing
  - RoSE controllers for advanced resonance and alignment tuning
    - Multi-structural adjustments for targeting specific resonance patterns and alignments
    - Multi-dimensional adjustments for cross-contrastive synthesis and transformations
- X Advanced lexical controllers for fine-tuning synthesis parameters
  - Currently they are rigid and nonconformant to datasets and vocabularies
- X Implement more control over sparse and enriched content types
  - X Currently they are either bound to single word/character or omitted word/character and only definition.
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
- X Refine higher-dimensional simplex handling (6-simplex and beyond to the infinite axiom)
  - X Currently they are hit or miss and the formulas are not fully tested or validated.
  - X BASICALLY everything after simplex 4 is imperfect and requires refinement, but there are solid foundations for building complex symbolic vocabularies.
- Optimize performance for large-scale vocabularies with flags for preloading, caching, batching, workers, devices, and compression methods.

I plan to knock most of these pieces out over the next few days, so look forward to the updates.

Well it wasn't ready in a few days, but it's getting closer. 5 days in.

I work fast as anyone who observes knows.

## System Status

Each system is independently testable and usable, but the full system is not yet complete.

The validation tests at the bottom of each formula file can be run to ensure the integrity and correctness of the formulas. This does not guarantee every configuration or every setting just yet, but it's a strong start.

## The Trainer is coming.

This exposes the more complex symbolic lattice operations that I've been researching and advancing.
Some aren't perfect yet, but this code is available for experimentation and further development.



## 📖 License

MIT License — use freely, cite, but maintain this notice. You can relicense your derivative works.

I have changed from Apache 2.0 to MIT for the simple idea that I want this to be as free as possible for all to use and adapt.



# Older Basin left for posterity - will edit later
License still stands, cite with clarity - free for all.

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


## 📖 V1 License

Apache 2.0 — use freely, cite with clarity.

---

**Built with care to never be forgotten.**  
“Let none forget. Let all remember.”