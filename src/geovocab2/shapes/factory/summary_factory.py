# geovocab2/shapes/factory/caption_factory.py

"""
CaptionFactory
--------------
Factory for generating natural language captions from booru tags using LLMs.

Supports multiple model backends:
    - Qwen2.5-1.5B-Instruct (recommended)
    - Llama-3.2-1B-Instruct
    - Llama-3.1-8B-Instruct
    - Flan-T5-Small/Base/Large/XL/XXL

Design:
    - Lazy model loading (load on first use)
    - Memory management (explicit unload)
    - Batch processing for throughput
    - Optional tensor output for embeddings

License: MIT
"""

import gc
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum
import numpy as np

try:
    from .factory_base import FactoryBase, HAS_TORCH
except ImportError:
    from factory_base import FactoryBase, HAS_TORCH

if HAS_TORCH:
    import torch

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ============================================================================
# MODEL REGISTRY
# ============================================================================

class ModelType(Enum):
    """Model architecture types."""
    CAUSAL_LM = "causal"  # Llama, Qwen
    SEQ2SEQ = "seq2seq"  # T5, Flan-T5


@dataclass
class ModelSpec:
    """Specification for a summarization model."""
    model_id: str
    model_type: ModelType
    vram_fp16_gb: float
    vram_int8_gb: float
    max_context: int
    prompt_template: str
    default_max_new_tokens: int = 64
    supports_system_prompt: bool = False

    def __post_init__(self):
        if "{tags}" not in self.prompt_template:
            raise ValueError(f"prompt_template must contain {{tags}} placeholder")


# Model registry with specs
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # Qwen family
    "qwen2.5-1.5b": ModelSpec(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_type=ModelType.CAUSAL_LM,
        vram_fp16_gb=3.0,
        vram_int8_gb=1.5,
        max_context=32768,
        prompt_template="Convert these image tags into a brief, natural English caption.\n\nTags: {tags}\nCaption:",
        default_max_new_tokens=60,
        supports_system_prompt=True
    ),

    # Llama family
    "llama-1b": ModelSpec(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        model_type=ModelType.CAUSAL_LM,
        vram_fp16_gb=2.5,
        vram_int8_gb=1.2,
        max_context=131072,
        prompt_template="Convert these image tags into a brief, natural English caption.\n\nTags: {tags}\nCaption:",
        default_max_new_tokens=60,
        supports_system_prompt=True
    ),
    "llama-8b": ModelSpec(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_type=ModelType.CAUSAL_LM,
        vram_fp16_gb=16.0,
        vram_int8_gb=8.0,
        max_context=131072,
        prompt_template="Convert these image tags into a brief, natural English caption.\n\nTags: {tags}\nCaption:",
        default_max_new_tokens=60,
        supports_system_prompt=True
    ),

    # Flan-T5 family (seq2seq)
    "flan-t5-small": ModelSpec(
        model_id="google/flan-t5-small",
        model_type=ModelType.SEQ2SEQ,
        vram_fp16_gb=0.3,
        vram_int8_gb=0.15,
        max_context=512,
        prompt_template="Describe this image: {tags}",
        default_max_new_tokens=48
    ),
    "flan-t5-base": ModelSpec(
        model_id="google/flan-t5-base",
        model_type=ModelType.SEQ2SEQ,
        vram_fp16_gb=0.9,
        vram_int8_gb=0.45,
        max_context=512,
        prompt_template="Describe this image: {tags}",
        default_max_new_tokens=48
    ),
    "flan-t5-large": ModelSpec(
        model_id="google/flan-t5-large",
        model_type=ModelType.SEQ2SEQ,
        vram_fp16_gb=3.0,
        vram_int8_gb=1.5,
        max_context=512,
        prompt_template="Describe this image: {tags}",
        default_max_new_tokens=48
    ),
    "flan-t5-xl": ModelSpec(
        model_id="google/flan-t5-xl",
        model_type=ModelType.SEQ2SEQ,
        vram_fp16_gb=11.0,
        vram_int8_gb=5.5,
        max_context=512,
        prompt_template="Describe this image: {tags}",
        default_max_new_tokens=48
    ),
    "flan-t5-xxl": ModelSpec(
        model_id="google/flan-t5-xxl",
        model_type=ModelType.SEQ2SEQ,
        vram_fp16_gb=42.0,
        vram_int8_gb=21.0,
        max_context=512,
        prompt_template="Describe this image: {tags}",
        default_max_new_tokens=48
    ),
}


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class CaptionFactoryConfig:
    """Configuration for CaptionFactory."""
    # Model selection
    model_name: str = "qwen2.5-1.5b"

    # Quantization
    use_int8: bool = False
    use_int4: bool = False

    # Generation parameters
    max_new_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = False

    # Custom prompt
    custom_prompt_template: Optional[str] = None

    # Memory management
    keep_model_loaded: bool = True

    # Device
    device: str = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

    # Output options
    return_tensors: bool = False

    # Batch processing
    batch_size: int = 8

    def __post_init__(self):
        if self.model_name not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model '{self.model_name}'. Available: {available}")


# ============================================================================
# CAPTION FACTORY
# ============================================================================

class CaptionFactory(FactoryBase):
    """
    Factory for generating natural language captions from booru tags.

    Uses LLMs to convert tag sequences into coherent descriptions suitable
    for T5 encoding in multi-modal VAE training.

    Example:
        factory = CaptionFactory(CaptionFactoryConfig(model_name="qwen2.5-1.5b"))

        # Single caption
        caption = factory.summarize("1girl, blue hair, school uniform, sitting")

        # Batch
        captions = factory.summarize_batch(["tags1", "tags2", "tags3"])

        # With tensors
        factory.config.return_tensors = True
        result = factory.build(tags="1girl, red eyes", backend="torch")
    """

    def __init__(self, config: Optional[CaptionFactoryConfig] = None):
        self.config = config or CaptionFactoryConfig()
        self.spec = MODEL_REGISTRY[self.config.model_name]

        super().__init__(
            name=f"caption_factory_{self.config.model_name}",
            uid=f"factory.caption.{self.config.model_name}"
        )

        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Model Loading
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._is_loaded:
            return

        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers library required: pip install transformers")

        print(f"Loading {self.spec.model_id}...")

        model_kwargs = {"device_map": self.config.device}

        if self.config.use_int8:
            model_kwargs["load_in_8bit"] = True
        elif self.config.use_int4:
            model_kwargs["load_in_4bit"] = True
        elif "cuda" in self.config.device:
            model_kwargs["torch_dtype"] = torch.float16

        if self.spec.model_type == ModelType.SEQ2SEQ:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.spec.model_id,
                **model_kwargs
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.spec.model_id,
                **model_kwargs
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model.eval()
        self._is_loaded = True

        vram = self.spec.vram_int8_gb if self.config.use_int8 else self.spec.vram_fp16_gb
        print(f"✓ Loaded {self.config.model_name} (~{vram:.1f}GB VRAM)")

    def unload_model(self):
        """Explicitly unload model to free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._is_loaded = False

        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"✓ Unloaded {self.config.model_name}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Prompt Formatting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _format_prompt(self, tags: str) -> str:
        """Format tags into model prompt."""
        template = self.config.custom_prompt_template or self.spec.prompt_template
        return template.format(tags=tags)

    def _format_prompts_batch(self, tags_list: List[str]) -> List[str]:
        """Format multiple tag strings into prompts."""
        return [self._format_prompt(tags) for tags in tags_list]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Generation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _generate_causal(self, prompts: List[str]) -> List[str]:
        """Generate captions using causal LM."""
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.spec.max_context
        ).to(self._model.device)

        max_new = self.config.max_new_tokens or self.spec.default_max_new_tokens

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=self.config.temperature if self.config.do_sample else None,
                top_p=self.config.top_p if self.config.do_sample else None,
                do_sample=self.config.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id
            )

        captions = []
        for i, output in enumerate(outputs):
            new_tokens = output[inputs.input_ids.shape[1]:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = text.strip()

            if text and text[-1] not in ".!?":
                last_period = text.rfind(".")
                if last_period > 0:
                    text = text[:last_period + 1]

            captions.append(text)

        return captions

    def _generate_seq2seq(self, prompts: List[str]) -> List[str]:
        """Generate captions using seq2seq model (T5)."""
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.spec.max_context
        ).to(self._model.device)

        max_new = self.config.max_new_tokens or self.spec.default_max_new_tokens

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None,
                top_p=self.config.top_p if self.config.do_sample else None
            )

        captions = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [c.strip() for c in captions]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def summarize(self, tags: str) -> str:
        """
        Convert a single tag string to natural language caption.

        Args:
            tags: Comma-separated booru tags

        Returns:
            Natural language caption
        """
        self._load_model()

        prompt = self._format_prompt(tags)

        if self.spec.model_type == ModelType.SEQ2SEQ:
            captions = self._generate_seq2seq([prompt])
        else:
            captions = self._generate_causal([prompt])

        if not self.config.keep_model_loaded:
            self.unload_model()

        return captions[0]

    def summarize_batch(self, tags_list: List[str]) -> List[str]:
        """
        Convert multiple tag strings to captions.

        Args:
            tags_list: List of comma-separated tag strings

        Returns:
            List of natural language captions
        """
        self._load_model()

        results = []

        for i in range(0, len(tags_list), self.config.batch_size):
            batch = tags_list[i:i + self.config.batch_size]
            prompts = self._format_prompts_batch(batch)

            if self.spec.model_type == ModelType.SEQ2SEQ:
                captions = self._generate_seq2seq(prompts)
            else:
                captions = self._generate_causal(prompts)

            results.extend(captions)

        if not self.config.keep_model_loaded:
            self.unload_model()

        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FactoryBase Implementation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_numpy(
            self,
            tags: Union[str, List[str]],
            *,
            dtype=np.float32,
            **kwargs
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Build caption(s) from tags.

        Args:
            tags: Single tag string or list of tag strings
            dtype: Used if return_tensors=True

        Returns:
            Dict with 'captions' key, optionally 'token_ids' if return_tensors=True
        """
        is_single = isinstance(tags, str)
        tags_list = [tags] if is_single else tags

        captions = self.summarize_batch(tags_list)

        result = {
            "captions": captions[0] if is_single else captions,
            "tags": tags
        }

        if self.config.return_tensors:
            self._load_model()

            encoded = self._tokenizer(
                captions,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512
            )

            result["token_ids"] = encoded.input_ids.astype(np.int64)
            result["attention_mask"] = encoded.attention_mask.astype(np.int64)

            if not self.config.keep_model_loaded:
                self.unload_model()

        return result

    def build_torch(
            self,
            tags: Union[str, List[str]],
            *,
            device: str = "cpu",
            dtype: Optional["torch.dtype"] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Build caption(s) with PyTorch tensors.

        Returns:
            Dict with 'captions', optionally 'token_ids', 'attention_mask'
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for build_torch")

        is_single = isinstance(tags, str)
        tags_list = [tags] if is_single else tags

        captions = self.summarize_batch(tags_list)

        result = {
            "captions": captions[0] if is_single else captions,
            "tags": tags
        }

        if self.config.return_tensors:
            self._load_model()

            encoded = self._tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            result["token_ids"] = encoded.input_ids.to(device)
            result["attention_mask"] = encoded.attention_mask.to(device)

            if not self.config.keep_model_loaded:
                self.unload_model()

        return result

    def validate(self, output: Any) -> Tuple[bool, str]:
        """Validate factory output."""
        if not isinstance(output, dict):
            return False, "Output must be a dictionary"

        if "captions" not in output:
            return False, "Output missing 'captions' key"

        captions = output["captions"]
        if isinstance(captions, str):
            if len(captions) == 0:
                return False, "Empty caption generated"
        elif isinstance(captions, list):
            if any(len(c) == 0 for c in captions):
                return False, "One or more empty captions generated"

        return True, ""

    def info(self) -> Dict[str, Any]:
        """Factory metadata."""
        base_info = super().info()
        base_info.update({
            "description": f"Caption factory using {self.config.model_name}",
            "model_id": self.spec.model_id,
            "model_type": self.spec.model_type.value,
            "vram_fp16_gb": self.spec.vram_fp16_gb,
            "vram_int8_gb": self.spec.vram_int8_gb,
            "max_context": self.spec.max_context,
            "is_loaded": self._is_loaded,
            "return_tensors": self.config.return_tensors,
            "available_models": list(MODEL_REGISTRY.keys())
        })
        return base_info

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Context Manager
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def __enter__(self):
        """Context manager entry - preload model."""
        self._load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload_model()
        return False

    def __repr__(self):
        status = "loaded" if self._is_loaded else "unloaded"
        return f"CaptionFactory(model='{self.config.model_name}', status={status})"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_caption_factory(
        model: str = "qwen2.5-1.5b",
        device: str = "cuda",
        return_tensors: bool = False,
        **kwargs
) -> CaptionFactory:
    """
    Create a CaptionFactory with common defaults.

    Args:
        model: Model name from registry
        device: Target device
        return_tensors: Whether to include tokenized output
        **kwargs: Additional CaptionFactoryConfig options

    Returns:
        Configured CaptionFactory
    """
    config = CaptionFactoryConfig(
        model_name=model,
        device=device,
        return_tensors=return_tensors,
        **kwargs
    )
    return CaptionFactory(config)


def list_models() -> Dict[str, Dict[str, Any]]:
    """List all available models with their specs."""
    return {
        name: {
            "model_id": spec.model_id,
            "type": spec.model_type.value,
            "vram_fp16": f"{spec.vram_fp16_gb:.1f}GB",
            "vram_int8": f"{spec.vram_int8_gb:.1f}GB",
            "max_context": spec.max_context
        }
        for name, spec in MODEL_REGISTRY.items()
    }


# ============================================================================
# MAIN
# ============================================================================

def demo_basic():
    """Basic demo without loading heavy models."""
    print("=" * 70)
    print("CAPTION FACTORY - BASIC DEMO")
    print("=" * 70)

    print("\n[Available Models]")
    for name, info in list_models().items():
        print(f"  {name:20s} {info['type']:8s} {info['vram_fp16']:>8s} (fp16)")

    print("\n[Factory Creation]")
    factory = create_caption_factory(model="qwen2.5-1.5b", device="cuda")
    print(f"  {factory}")
    print(f"  Info: {factory.info()['description']}")

    return factory


def demo_generation(model_name: str = "flan-t5-base"):
    """Demo with actual generation."""
    print("=" * 70)
    print(f"CAPTION FACTORY - GENERATION DEMO ({model_name})")
    print("=" * 70)

    test_tags = [
        "1girl, blue hair, red eyes, school uniform, sitting, classroom, window, sunlight",
        "masterpiece, 1boy, armor, sword, fantasy, castle, dramatic lighting",
        "2girls, holding hands, park, cherry blossoms, spring, happy",
        "score_9, 1girl, cyberpunk, neon lights, rain, night city, futuristic"
    ]

    config = CaptionFactoryConfig(
        model_name=model_name,
        device="cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu",
        do_sample=False,
        keep_model_loaded=False
    )

    factory = CaptionFactory(config)

    print(f"\n[Model: {model_name}]")
    print(f"  VRAM (fp16): {factory.spec.vram_fp16_gb:.1f}GB")

    print("\n[Tag → Caption Conversion]")
    for tags in test_tags:
        caption = factory.summarize(tags)
        print(f"\n  Tags: {tags[:60]}...")
        print(f"  Caption: {caption}")

    print("\n[Batch Processing]")
    captions = factory.summarize_batch(test_tags)
    print(f"  Generated {len(captions)} captions in batch")

    return factory, captions


def demo_with_tensors(model_name: str = "flan-t5-small"):
    """Demo with tensor output."""
    print("=" * 70)
    print(f"CAPTION FACTORY - TENSOR OUTPUT DEMO ({model_name})")
    print("=" * 70)

    config = CaptionFactoryConfig(
        model_name=model_name,
        device="cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu",
        return_tensors=True,
        keep_model_loaded=False
    )

    factory = CaptionFactory(config)

    tags = "1girl, white dress, garden, flowers, peaceful, afternoon"

    print(f"\n[Input Tags]")
    print(f"  {tags}")

    result = factory.build(tags=tags, backend="torch", validate=True)

    print(f"\n[Output]")
    print(f"  Caption: {result['captions']}")
    print(f"  Token IDs shape: {result['token_ids'].shape}")
    print(f"  Token IDs device: {result['token_ids'].device}")

    return factory, result


def demo_context_manager(model_name: str = "flan-t5-small"):
    """Demo using context manager for automatic cleanup."""
    print("=" * 70)
    print(f"CAPTION FACTORY - CONTEXT MANAGER DEMO ({model_name})")
    print("=" * 70)

    config = CaptionFactoryConfig(
        model_name=model_name,
        device="cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
    )

    tags_batch = [
        "1girl, red hair, warrior, battle",
        "landscape, mountains, sunset, peaceful"
    ]

    print("\n[Using context manager]")
    with CaptionFactory(config) as factory:
        print(f"  Model loaded: {factory._is_loaded}")
        captions = factory.summarize_batch(tags_batch)
        for t, c in zip(tags_batch, captions):
            print(f"  {t[:40]}... → {c[:50]}...")

    print(f"  After exit - Model loaded: {factory._is_loaded}")

    return captions


def demo_compare_models():
    """Compare outputs from different models."""
    print("=" * 70)
    print("CAPTION FACTORY - MODEL COMPARISON")
    print("=" * 70)

    test_tags = "1girl, silver hair, golden eyes, elegant dress, ballroom, chandelier, dancing"

    models_to_test = ["flan-t5-small", "flan-t5-base"]

    if HAS_TORCH and torch.cuda.is_available():
        # Add larger models if GPU available
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024 ** 3

        if vram_gb >= 4:
            models_to_test.append("qwen2.5-1.5b")
        if vram_gb >= 8:
            models_to_test.append("flan-t5-large")

    print(f"\n[Input Tags]")
    print(f"  {test_tags}")

    print(f"\n[Model Outputs]")
    for model_name in models_to_test:
        config = CaptionFactoryConfig(
            model_name=model_name,
            device="cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu",
            do_sample=False,
            keep_model_loaded=False
        )

        with CaptionFactory(config) as factory:
            caption = factory.summarize(test_tags)
            print(f"\n  [{model_name}]")
            print(f"    {caption}")


def benchmark(model_name: str = "flan-t5-small", n_samples: int = 100):
    """Benchmark generation speed."""
    import time

    print("=" * 70)
    print(f"CAPTION FACTORY - BENCHMARK ({model_name}, n={n_samples})")
    print("=" * 70)

    test_tags = [
        f"1girl, tag{i}, quality{i % 5}, style{i % 3}"
        for i in range(n_samples)
    ]

    config = CaptionFactoryConfig(
        model_name=model_name,
        device="cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu",
        batch_size=16,
        do_sample=False
    )

    with CaptionFactory(config) as factory:
        # Warmup
        _ = factory.summarize(test_tags[0])

        # Benchmark
        start = time.time()
        captions = factory.summarize_batch(test_tags)
        elapsed = time.time() - start

    print(f"\n[Results]")
    print(f"  Samples: {n_samples}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Rate: {n_samples / elapsed:.1f} captions/sec")
    print(f"  Avg latency: {1000 * elapsed / n_samples:.1f}ms/caption")

    return elapsed, captions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Caption Factory")
    parser.add_argument("--demo", type=str, default="generate",
                        choices=["basic", "generate", "tensors", "context", "compare", "benchmark"],
                        help="Demo to run")
    parser.add_argument("--model", type=str, default="flan-t5-small",
                        help="Model to use")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of samples for benchmark")

    args = parser.parse_args()

    if args.demo == "basic":
        demo_basic()
    elif args.demo == "generate":
        demo_generation(args.model)
    elif args.demo == "tensors":
        demo_with_tensors(args.model)
    elif args.demo == "context":
        demo_context_manager(args.model)
    elif args.demo == "compare":
        demo_compare_models()
    elif args.demo == "benchmark":
        benchmark(args.model, args.n)
    else:
        demo_basic()