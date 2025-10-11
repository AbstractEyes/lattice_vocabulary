# 🧠 Lattice Vocabulary: A Geometric Map of Language

> “To define a word geometrically is to place it in an infinite structure that cannot be forgotten.”

**Lattice Vocabulary** is a symbolic crystallization of representative multimodal language into a geometric, navigable systemic-driven lattice. This is designed not for compression, but for cognition. Each word, its meanings, transformations, and connections are encoded not as just embeddings, but as structured, lossless, multidimensional shapes.

# 📐 geovocab2

`pip install git+https://github.com/AbstractEyes/lattice_vocabulary.git`

* The natural evolution of geovocab, now a full refactored and reinvented structure basin built specifically for symbolic lattice vocabularies.
* This will function as a standalone library for generating and manipulating symbolic lattices while fulfilling the role for a strong synthesizer for randomized vocabularies for experimentation.
* The trie structure will properly replace the original caching mechanism for large vocabulary ngram structures, allowing for highly efficient storage and retrieval of complex symbolic structures using lexicographical ordering. This has strengths and weaknesses, so more database options will be available in the future for rapid prototyping.
* This new version supports advanced toolset symbolic operations and transformations, making it ideal for geometric experimental applications in natural language processing, cognitive computing, and AI development.
* Equipped with pyi for autocomplete and a standardized package structure for ease of use.
* Formatted specifically for easy expansion with IDE by utilizing autocomplete pyi capacity while retaining that core functionality for imported projects.


# System Status
* IN DEVELOPMENT

The package is still in active development. Many components are functional, but some advanced features and optimizations are still pending.

Each system is independently testable and usable, but the full system is not yet complete. The validation tests at the bottom of each formula file can be run to ensure the integrity and correctness of the formulas. This does not guarantee every configuration or every setting just yet, but it's a strong start.

It's messy still, but the organization will improve over time as I refine and expand the codebase.


# Proofs
```
# import geovocab2.proofs.beatrix_staircase
```
Each proof file is individually self-proving, so they require only running them to validate if they work or not.
They are primarily targeting a quick-run on colab, and require no dependencies from this repo to function, just the independent script and the correct torch and python versions.

1. Cayley-Menger determinants are validated against known values for 1-simplex, 2-simplex, and 3-simplex are available and valid to nth k-simplex with some higher dimension limitations due to numerical instability and floating-point precision limitations.
2. Cantor Stairs are validated against known values for 1-simplex and 2-simplex, with higher dimensions requiring further validation and are testing as potential PositionalEncoding replacements for large language models, classification models, and other transformer-based architectures.


## Current Access Import Structure:

```
from geovocab2.data.dataloader import FactoryDataset
# access to the dataloader factory for loading datasets and vocabularies into the factories for batched formula processing, modification, synthesis, and transformations.
```
```
from geovocab2.shapes.factory import SimplexFactory
# access to multiple factories and the system will expand
```
```
from geovocab2.shapes.formula.formula_base import FormulaBase
# access to the base formula class for creating custom formulas compatible with the factories and synthesizers
```
```
from geovocab2.shapes.formula import CayleyMengerFromSimplex # or any formula autocompleted
# access to hundreds of formulas from the geovocab2.shapes.formula
```
```
from geovocab2.shapes.fusion.composition_base import CompositionBase
# access to the base composition class for creating custom synthesizers compatible with the factories and formulas
```
```
from geovocab2.shapes.fusion import LexicalSimplexSynthesizer
# access to multiple synthesizers and fusion methods
```

The structure is built primarily with a few classes meant to be overridden and shallow for ease-of-use and understanding for both AI and humans.
# V2 Key Features
The initial v2 variation is a bit rigid and requires some manual setup, but it is a solid foundation for building complex symbolic vocabularies.

The interface has changed, so be aware if you're coming from v1. v1 still exists so you can still use it if you prefer.

## Explanation for changes

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

## Structure Hierarchy

```
BaseFormula > OverriddenFormula > SpecificFormula
    Example:
        BaseFactory -> SimplexFactory -> Produces simplex shaped structures 
        BaseFormula -> CayleyMengerFromSimplex -> Produces Cayley-Menger determinants from simplex structures
        CompositionBase -> LexicalSimplexSynthesizer -> Synthesizes lexical simplex structures from vocab data and shapes
```

The structures are formatted specifically to be expandable and utility-focused, so they can be used in a variety of contexts and applications.

They are AI friendly and human-readable, so they can be easily understood and modified by both humans and AI systems.

## Formula Playground

All formulas are now package-friendly and import at the geovocab2.shapes.formula level. The current implementation has these directories and files for the formula playground:

Formulas will be improved over time for speed, accuracy, and compile capability. I will be avoiding python loops and overhead as much as possible for efficiency, which can be a train killer if I don't.

```geovocab2.shapes.formula```
- engineering: 
- - atomic.py for basic atomic operations and structures, not so useful on its own but foundational if necessary.
- - fundamental.py for more advanced basic operations like atan2, vector norms, and other basic mathematical operations.
- - geometric.py for geometric analysis and transformations like raycast, geometric point calculations, and other geometric operations.
- - projection.py for projection operations like to_homogenous, orthographic, perspective, and other projection methods.
- - simplex.py for simplex structures and operations like generating simplex shapes, calculating volumes, and other simplex-related operations.
- - wave.py for advanced wave and resonance operations like heat diffusion, reactive diffusion, wave propagation, and other wave-related operations.
- symbolic:
- - cantor.py for advanced symbolic infinity operations.
- - cayley_menger.py for Cayley-Menger determinant calculations and related operations, backbone of core.
- - einstein.py for Einstein summation operations and related tensor manipulations.
- - euler.py for Euler characteristic calculations and related topological operations.
- - hawking.py for Hawking radiation and black hole related symbolic operations.
- - hooke.py for Hooke's law and related elastic deformation operations.
- - newton.py for Newtonian mechanics and related physical operations.
- - nikola.py for Nikola Tesla's resonance, frequency, and energy operations.

- specifically k-simplex supported on many of these files, but not all yet.
- The k-simplex depth is nth, so 0-simplex is a point, 1-simplex is a line, 2-simplex is a triangle, 3-simplex is a tetrahedron, 4-simplex is a pentachoron, and so on.


# Factories
The factories are designed to produce specific types of symbolic lattice structures based on the provided configuration.

```geovocab2.shapes.factory```
## Factory Progress
1. factory_base.py for the base factory class and common methods. 
2. simplex_factory.py for generating simplex-shaped symbolic lattice structures. 
3. legacy_factory.py for direct 1:1 compatibility with the original geovocab structures, almost ready. 
4. factory_dataloaders.py for loading datasets and vocabularies into the factories.


## Factory Todo
1. Implement more factory types for different symbolic lattice structures including cantor stairs and other complex structures.
2. Build support to load a model's prepared vocabulary from multiple types of tokenizers
3. Optimize performance for large-scale vocabularies with flags for preloading, caching, batching



# Synthesis

These are specifically curated to synthesize complex symbolic lattice structures from vocabularies and shapes interconnected.
Meant to be independent and reusable for various synthesis tasks.

## Synthesis Progress
```
# absolute import path: from geovocab2.shapes.fusion.composition_base import CompositionBase
# absolute import path: from geovocab2.shapes.fusion import LexicalSimplexSynthesizer
# absolute import path: from geovocab2.shapes.fusion import WordNetSynthes
```
### Lexical Progress
1. Package stubs generate but there aren't very many synthesizers.
2. CompositionBase is functional and serves as the base class for all synthesizers, providing common methods and properties.
2. LexicalSimplexSynthesizer is functional and can synthesize lexical simplex structures from vocab data and shapes, basically legacy with more power and a better interface.
3. WordNetSynthesizer is a placeholder for future development, can be made work with a tweak or two but might be broken.

### Lexical Todo
1. Implement more synthesis methods and options for different types of vocabularies and shapes.
2. Integrate with more datasets and vocabularies for broader applicability.
3. Optimize performance for large-scale vocabularies with flags for preloading, caching, batching

### Image Progress
1. Basically nothing implemented yet, everything can be generated using the factory structure but image + text synthesis isn't implemented yet.

### Image Todo
1. Implement image-based synthesis methods for generating symbolic lattice structures from images for advanced embedding and multimodal applications.
2. Integrate with image datasets and vocabularies for broader applicability including multiple standard types of datasets like cifar100 for ease of access.
3. Optimize performance for large-scale image datasets with flags for preloading, caching, batching, workers, devices, and compression methods.
4. Develop multiple visualization tools for exploring and analyzing the geometric structures of the vocabulary.

`geovocab2.fusion`
- composition_base.py for the base composition class and common character embedding synthesis methods.
- lexical_simplex_synthesizer.py for synthesizing lexical simplex structures from vocab data and shapes.
- wordnet_synthesizer.py for synthesizing structures using WordNet data, doesn't do much yet. Use simplex for now.


## V2 Blockers
- X Formula bank incompatible requires refitting
  - X I don't like the current implementation so I will make a better interface.
- X Structure refitting for baseline new structure established
- Trainer module not yet implemented

## V2 Lexical Todo
- Implement the missing formulas and transformations
  - X Formula bank established for housing the formulas as they are implemented.
  - X Synthesize and test Nikola-Menger resonance axiom formula structures with finite and infinite lattice structures
  - X Chaos theory controllers for dynamic adjustments
  - X Chaos-Menger transformations for dynamic structural adjustments
  - X Implement the Rose score magnitude and trajectory-based loss functions
  - Integrate with existing symbolic loss functions for NLP tasks
  - X Implement Graham infinite and finite transformations with masking capabilities
  - RoPE-like theta controllers and rotational adjustments for dynamic synthesis pre and post-processing
  - X RoSE controllers for advanced resonance and alignment tuning
    - X Multi-structural adjustments for targeting specific resonance patterns and alignments
    - X Multi-dimensional adjustments for cross-contrastive synthesis and transformations
- X Advanced lexical controllers for fine-tuning synthesis parameters
  - Currently they are rigid and nonconformant to datasets and vocabularies
- X Implement more control over sparse and enriched content types
  - X Currently they are either bound to single word/character or omitted word/character and only definition.
- X Implement more normalization strategies and options
  - X Currently limited to L1, L2, L∞, and none - this needs to be more flexible and adaptable to different use cases.
- Implement the corpus trainer module for advanced symbolic lattice operations with gpu gradient and loss functions
  - There are MANY formulas and variant forms of useful losses to be implemented here, so this will take some time - likely spanning days to weeks to complete.
- Expand and refine the dataset with more comprehensive vocabularies and relationships based on key linguistic resources, medical resources, technical corpus, and domain-specific datasets.
  - They are currently limited to only my format due to ease of setup and testing for myself.
  - This will be expanded to include WordNet, ConceptNet, Wiktionary, and other linguistic resources.
  - I have a full library of books and texts that can be used to expand the vocabulary and relationships.
    - Books and texts will be omitted if they are copyrighted or restricted. The full corpus listed for tuning and transparency.
- Integrate with Hugging Face datasets and tokenizers for seamless usage with existing NLP pipelines.
  - Should seamlessly integrate with common tokenizers for direct usage in existing NLP pipelines with a few tweaks.
- Develop multiple visualization tools for exploring and analyzing the geometric structures of the vocabulary.
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

I work fast as anyone who observes knows - I will be pushing updates frequently as I refine and expand the codebase.

## Trainer Todo
```
# absolute import path: from geovocab2.trainer.trainer_base import TrainerBase
# Currently unimplemented for hierarchy will be supported.
```
This exposes the more complex symbolic lattice operations that I've been researching and advancing.
Some aren't perfect yet, but this code is available for experimentation and further development.

### Current powerful features implemented
1. There's a ton of model architectures available for experimentation and expansion, if you feel like picking the pieces out and working with them. This will be streamlined and uniformied into an ease-of-access system like formulas are.
2. Multiple loss functions that may or may not help, with a few that guarantee full cohesive assistance with some variations of models. These aren't generic enough and require formula solidification.
3. ModelBase is partially implemented and requires expansion.
4. TrainerBase is partially implemented and requires expansion.
5. LossBase is partially implemented and requires expansion.

### Missing important features
1. Logger isn't implemented yet, but will be added later. Will likely use TensorBoard and keep a manifest in a repo shared by multiple models for easy access and comparison.
2. Checkpointing is whatever, janky and working for my current trainer. Needs proper implementation - likely discarding PT and moving directly to safetensors only.
3. AdamW and SGD optimizers have specific settings for cantor, cayley-menger, and other symbolic lattice operations. These need to be refined and expanded for more complex operations and presets for layers.
4. Schedulers don't seem to matter yet unless they are directly tied to multiple LR for multiple types of layers. This will be expanded and refined through experimentation.
5. Mixed precision training currently has little to no impact on performance, but this will require further testing and refinement.
6. Distributed training is not yet implemented, but will be added later as one of the primary goals for multi-gpu.
7. There is no GUI and likely never will be in this repo. This is meant to be used as a library and integrated into existing pipelines and workflows.

## 📖 License

MIT License — use freely, cite, but maintain this notice. You can relicense your derivative works.

I have changed from Apache 2.0 to MIT for the simple idea that I want this to be as free as possible for all to use and adapt.
