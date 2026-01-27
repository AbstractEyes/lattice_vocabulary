"""
Clean Character Generator v3
============================
With guaranteed uniform ethnicity distribution.
Fashion styles + photo styles from BulkCaptions (no LAION flavors).
"""

from typing import List, Optional, Tuple

from geovocab2.data.prompt.bulk_caption_data import BulkCaptions
from geovocab2.data.prompt.symbolic_tree import SeedGen


class CleanCharacterGenerator:
    """
    Generate character prompts with:
    - GUARANTEED uniform ethnicity distribution (round-robin)
    - Fashion style integration from BulkCaptions
    - Photo style suffixes from BulkCaptions
    - No external LAION flavors
    """

    def __init__(self, seed: int = None, include_ethnicity: bool = True):
        self.sg = SeedGen(seed)
        self.bc = BulkCaptions
        self.include_ethnicity = include_ethnicity

        # Ethnicity - round-robin for uniform distribution
        self.ethnicities = self.bc.ETHNICITY
        self._ethnicity_index = 0
        self._ethnicity_shuffled = list(self.ethnicities)
        self.sg.shuffle(self._ethnicity_shuffled)

        # Fashion & photo styles from BulkCaptions
        self.fashion_styles = self.bc.FASHION_STYLES
        self.photo_styles = self.bc.PHOTO_STYLES

        # Core identity
        self.genders = self.bc.GENDER_TYPES

        # Anatomy
        self.poses = self.bc.HUMAN_POSES
        self.hair_styles = self.bc.HAIRSTYLES_TYPES
        self.hair_lengths = self.bc.HAIR_LENGTH_TYPES

        # Attire
        self.upper_clothing = self.bc.UPPER_BODY_CLOTHES_TYPES
        self.lower_clothing = self.bc.LOWER_BODY_CLOTHES_TYPES
        self.footwear = self.bc.FOOTWEAR_TYPES
        self.accessories = self.bc.ACCESSORY_TYPES
        self.jewelry = self.bc.JEWELRY_TYPES
        self.headwear = self.bc.HEADWEAR_TYPES

        # Expression
        self.expressions = self.bc.HUMAN_EXPRESSIONS
        self.actions = self.bc.HUMAN_ACTIONS

        # Context
        self.backgrounds = self.bc.BACKGROUND_TYPES
        self.lighting = self.bc.LIGHTING_TYPES
        self.styles = self.bc.STYLE_TYPES
        self.angles = self.bc.SUBJECT_PHOTOGRAPH_ANGLE
        self.simple_angles = self.bc.SIMPLE_CHARACTER_ANGLE

        # Descriptors
        self.colors = self.bc.COLORS
        self.materials = self.bc.MATERIAL_TYPES

        # Tracking
        self.ethnicity_counts = {e: 0 for e in self.ethnicities}
        self.fashion_counts = {}

    def _sample(self, lst: List[str]) -> str:
        return self.sg.choice(lst)

    def _maybe(self, lst: List[str], prob: float = 0.5) -> Optional[str]:
        if self.sg.random() < prob:
            return self.sg.choice(lst)
        return None

    def _next_ethnicity(self) -> str:
        """Round-robin ethnicity for uniform distribution."""
        eth = self._ethnicity_shuffled[self._ethnicity_index]
        self.ethnicity_counts[eth] += 1
        self._ethnicity_index = (self._ethnicity_index + 1) % len(self._ethnicity_shuffled)
        return eth

    def _sample_fashion(self, prob: float = 0.5) -> Optional[str]:
        """Sample a fashion style with given probability."""
        if self.sg.random() < prob:
            style = self._sample(self.fashion_styles)
            self.fashion_counts[style] = self.fashion_counts.get(style, 0) + 1
            return style
        return None

    def _sample_photo_style(self, prob: float = 0.4) -> Optional[str]:
        """Sample a photo style suffix."""
        if self.sg.random() < prob:
            return self._sample(self.photo_styles)
        return None

    def _build_subject(self, gender: str, ethnicity: str = None) -> str:
        if ethnicity and self.include_ethnicity:
            return f"{ethnicity} {gender}".replace(',,', ',').strip()
        return gender

    def _build_hair(self, length: str = None, style: str = None, color: str = None) -> str:
        parts = []
        if color:
            parts.append(color)
        if length:
            parts.append(length)
        if style:
            parts.append(style)
        if parts:
            return ' '.join(parts) + ' hair'
        return None

    def _build_suffix(self, fashion_prob: float = 0.5, photo_prob: float = 0.4) -> Optional[str]:
        """Build style suffix from fashion + photo styles."""
        parts = []

        fashion = self._sample_fashion(fashion_prob)
        if fashion:
            parts.append(fashion)

        photo = self._sample_photo_style(photo_prob)
        if photo:
            parts.append(photo)

        return ', '.join(parts) if parts else None

    # === GENERATORS ===

    def simple_character(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)

        hair_color = self._maybe(self.colors, 0.6)
        hair_length = self._maybe(self.hair_lengths, 0.7)
        hair = self._build_hair(hair_length, color=hair_color)

        upper_body_clothes = self._maybe(self.upper_clothing, 1.0)
        upper_body_color = self._maybe(self.colors, 0.5)

        angle = self._maybe(self.simple_angles, 0.4)
        if angle:
            subject = f"{subject} {angle}"

        parts = []
        if hair:
            parts.append(f"{subject} with {hair}")
        else:
            parts.append(subject)

        if upper_body_clothes:
            if upper_body_color:
                subject += f" wearing {upper_body_color} {upper_body_clothes}"
            else:
                subject += f" wearing {upper_body_clothes}"

        suffix = self._build_suffix()
        if suffix:
            parts.append(suffix)

        return ', '.join(parts).replace(',,', ',').strip()

    def character_with_clothing(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)

        upper = self._sample(self.upper_clothing)
        lower = self._maybe(self.lower_clothing, 0.6)
        color = self._maybe(self.colors, 0.5)
        lower_color = self._maybe(self.colors, 0.4)

        parts = [subject]

        clothing_parts = []
        if color:
            clothing_parts.append(f"{color} {upper}")
        else:
            clothing_parts.append(upper)
        if lower:
            if lower_color:
                clothing_parts.append(f"{lower_color} {lower}")
            else:
                clothing_parts.append(lower)

        if len(clothing_parts) > 1:
            parts.append(f"wearing {clothing_parts[0]} and {clothing_parts[1]}")
        else:
            parts.append(f"wearing {clothing_parts[0]}")

        suffix = self._build_suffix()
        if suffix:
            parts.append(suffix)

        return ', '.join(parts)

    def character_with_pose(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)
        pose = self._sample(self.poses)

        parts = [f"{subject} {pose}"]

        suffix = self._build_suffix()
        if suffix:
            parts.append(suffix)

        return ', '.join(parts)

    def character_with_expression(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        expression = self._sample(self.expressions)

        if self.include_ethnicity:
            subject = f"{expression} {ethnicity} {gender}"
        else:
            subject = f"{expression} {gender}"

        parts = [subject]

        suffix = self._build_suffix()
        if suffix:
            parts.append(suffix)

        return ', '.join(parts)

    def character_with_action(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)
        action = self._sample(self.actions)

        parts = [f"{subject} {action}"]

        suffix = self._build_suffix()
        if suffix:
            parts.append(suffix)

        return ', '.join(parts)

    def character_full_outfit(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)

        hair_color = self._maybe(self.colors, 0.5)
        hair_length = self._maybe(self.hair_lengths, 0.6)
        hair_style = self._maybe(self.hair_styles, 0.4)
        hair = self._build_hair(hair_length, hair_style, hair_color)

        upper = self._sample(self.upper_clothing)
        lower = self._maybe(self.lower_clothing, 0.7)
        footwear = self._maybe(self.footwear, 0.4)
        color = self._maybe(self.colors, 0.5)
        lower_color = self._maybe(self.colors, 0.5)
        shoe_color = self._maybe(self.colors, 0.5)

        parts = [subject]
        if hair:
            parts[0] = f"{subject} with {hair}"

        clothing_parts = []
        if color:
            clothing_parts.append(f"{color} {upper}")
        else:
            clothing_parts.append(upper)
        if lower:
            if lower_color:
                clothing_parts.append(f"{lower_color} {lower}")
            else:
                clothing_parts.append(lower)
        if footwear:
            if shoe_color:
                clothing_parts.append(f"{shoe_color} {footwear}")
            else:
                clothing_parts.append(footwear)

        if len(clothing_parts) > 1:
            parts.append(f"wearing {', '.join(clothing_parts[:-1])} and {clothing_parts[-1]}")
        else:
            parts.append(f"wearing {clothing_parts[0]}")

        # Higher fashion prob for outfit-focused generator
        suffix = self._build_suffix(fashion_prob=0.6, photo_prob=0.5)
        if suffix:
            parts.append(suffix)

        return ', '.join(parts)

    def character_with_accessory(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)
        accessory = self._sample(self.accessories)

        parts = [f"{subject} with {accessory}"]

        suffix = self._build_suffix()
        if suffix:
            parts.append(suffix)

        return ', '.join(parts)

    def character_with_jewelry(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)
        jewelry = self._sample(self.jewelry)
        color = self._maybe(self.colors, 0.4)
        material = self._maybe(self.materials, 0.3)

        jewelry_parts = []
        if color:
            jewelry_parts.append(color)
        if material:
            jewelry_parts.append(material)
        jewelry_parts.append(jewelry)

        parts = [f"{subject} wearing {' '.join(jewelry_parts)}"]

        suffix = self._build_suffix()
        if suffix:
            parts.append(suffix)

        return ', '.join(parts)

    def character_with_headwear(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)
        headwear = self._sample(self.headwear)
        color = self._maybe(self.colors, 0.4)

        if color:
            parts = [f"{subject} wearing {color} {headwear}"]
        else:
            parts = [f"{subject} wearing {headwear}"]

        suffix = self._build_suffix()
        if suffix:
            parts.append(suffix)

        return ', '.join(parts)

    def character_in_scene(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)
        background = self._sample(self.backgrounds)

        parts = [f"{subject} in {background}"]

        suffix = self._build_suffix()
        if suffix:
            parts.append(suffix)

        return ', '.join(parts)

    def character_with_lighting(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)
        upper = self._sample(self.upper_clothing)
        lighting = self._sample(self.lighting)

        parts = [f"{subject} wearing {upper}", lighting]

        fashion = self._sample_fashion(0.4)
        if fashion:
            parts.append(fashion)

        return ', '.join(parts)

    def character_portrait(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        angle = self._sample(self.angles)
        expression = self._maybe(self.expressions, 0.7)
        hair_color = self._maybe(self.colors, 0.6)
        hair_length = self._maybe(self.hair_lengths, 0.5)
        hair = self._build_hair(hair_length, color=hair_color)

        parts = [angle]
        if expression:
            if self.include_ethnicity:
                parts.append(f"{expression} {ethnicity} {gender}")
            else:
                parts.append(f"{expression} {gender}")
        else:
            parts.append(self._build_subject(gender, ethnicity))

        if hair:
            parts[-1] = f"{parts[-1]} with {hair}"

        # Portrait-focused photo style
        photo = self._sample_photo_style(0.6)
        if photo:
            parts.append(photo)

        return ', '.join(parts)

    def character_styled(self) -> str:
        """Fashion-forward character with guaranteed fashion style."""
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)

        upper = self._sample(self.upper_clothing)
        lower = self._maybe(self.lower_clothing, 0.6)

        # Always include fashion style
        fashion = self._sample(self.fashion_styles)
        self.fashion_counts[fashion] = self.fashion_counts.get(fashion, 0) + 1

        parts = [subject]

        if lower:
            parts.append(f"wearing {upper} and {lower}")
        else:
            parts.append(f"wearing {upper}")

        parts.append(fashion)

        photo = self._sample_photo_style(0.5)
        if photo:
            parts.append(photo)

        return ', '.join(parts)

    def full_character(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)

        hair_color = self._maybe(self.colors, 0.6)
        hair_length = self._maybe(self.hair_lengths, 0.7)
        hair_style = self._maybe(self.hair_styles, 0.5)
        hair = self._build_hair(hair_length, hair_style, hair_color)

        expression = self._maybe(self.expressions, 0.5)
        pose = self._maybe(self.poses, 0.4)

        upper = self._sample(self.upper_clothing)
        lower = self._maybe(self.lower_clothing, 0.6)
        color = self._maybe(self.colors, 0.5)
        lower_color = self._maybe(self.colors, 0.4)

        accessory = self._maybe(self.accessories, 0.3)
        accessory_material = self._maybe(self.materials, 0.3)

        background = self._maybe(self.backgrounds, 0.5)
        lighting = self._maybe(self.lighting, 0.4)

        parts = []

        if expression:
            if self.include_ethnicity:
                parts.append(f"{expression} {ethnicity} {gender}")
            else:
                parts.append(f"{expression} {gender}")
        else:
            parts.append(subject)

        if hair:
            parts[-1] = f"{parts[-1]} with {hair}"

        if pose:
            parts.append(pose)

        clothing_parts = []
        if color:
            clothing_parts.append(f"{color} {upper}")
        else:
            clothing_parts.append(upper)
        if lower:
            if lower_color:
                clothing_parts.append(f"{lower_color} {lower}")
            else:
                clothing_parts.append(lower)

        if clothing_parts:
            parts.append(f"wearing {' and '.join(clothing_parts)}")

        if accessory:
            if accessory_material:
                parts.append(f"with {accessory_material} {accessory}")
            else:
                parts.append(f"with {accessory}")

        if background:
            parts.append(f"in {background}")

        fashion = self._sample_fashion(0.5)
        if fashion:
            parts.append(fashion)

        if lighting:
            parts.append(lighting)

        return ', '.join(parts)

    def action_scene(self) -> str:
        gender = self._sample(self.genders)
        ethnicity = self._next_ethnicity()
        subject = self._build_subject(gender, ethnicity)
        action = self._sample(self.actions)
        background = self._sample(self.backgrounds)
        upper = self._maybe(self.upper_clothing, 0.5)

        parts = [subject]
        if upper:
            parts.append(f"wearing {upper}")
        parts.append(action)
        parts.append(f"in {background}")

        photo = self._sample_photo_style(0.4)
        if photo:
            parts.append(photo)

        return ', '.join(parts)

    # === BATCH ===

    def generate_batch(self, n: int) -> Tuple[List[str], List[str]]:
        generators = [
            (0.08, 'simple_character', self.simple_character),
            (0.10, 'character_with_clothing', self.character_with_clothing),
            (0.08, 'character_with_pose', self.character_with_pose),
            (0.08, 'character_with_expression', self.character_with_expression),
            (0.08, 'character_with_action', self.character_with_action),
            (0.12, 'character_full_outfit', self.character_full_outfit),
            (0.05, 'character_with_accessory', self.character_with_accessory),
            (0.04, 'character_with_jewelry', self.character_with_jewelry),
            (0.05, 'character_with_headwear', self.character_with_headwear),
            (0.06, 'character_in_scene', self.character_in_scene),
            (0.05, 'character_with_lighting', self.character_with_lighting),
            (0.07, 'character_portrait', self.character_portrait),
            (0.08, 'character_styled', self.character_styled),
            (0.04, 'full_character', self.full_character),
            (0.02, 'action_scene', self.action_scene),
        ]

        captions = []
        gen_types = []

        for _ in range(n):
            r = self.sg.random()
            cumulative = 0
            for weight, name, func in generators:
                cumulative += weight
                if r <= cumulative:
                    captions.append(func())
                    gen_types.append(name)
                    break

        return captions, gen_types

    def print_ethnicity_report(self):
        total = sum(self.ethnicity_counts.values())
        if total == 0:
            print("No samples generated yet.")
            return

        print(f"\n{'='*70}")
        print(f"ETHNICITY DISTRIBUTION ({total:,} samples)")
        print(f"{'='*70}")
        expected_pct = 100.0 / len(self.ethnicities)

        sorted_counts = sorted(self.ethnicity_counts.items(), key=lambda x: -x[1])
        for eth, count in sorted_counts:
            pct = count / total * 100
            deviation = pct - expected_pct
            print(f"  {eth:24s}: {count:5,} ({pct:5.2f}%) [{deviation:+.2f}%]")

        counts = list(self.ethnicity_counts.values())
        print(f"\n  Min: {min(counts)}, Max: {max(counts)}, Diff: {max(counts) - min(counts)}")
        print(f"{'='*70}")

    def print_fashion_report(self):
        total = sum(self.fashion_counts.values())
        if total == 0:
            print("No fashion styles used yet.")
            return

        print(f"\n{'='*70}")
        print(f"FASHION STYLE DISTRIBUTION ({total:,} uses)")
        print(f"{'='*70}")

        sorted_counts = sorted(self.fashion_counts.items(), key=lambda x: -x[1])
        for style, count in sorted_counts[:20]:
            pct = count / total * 100
            print(f"  {style:30s}: {count:5,} ({pct:5.2f}%)")

        if len(sorted_counts) > 20:
            print(f"  ... and {len(sorted_counts) - 20} more styles")
        print(f"{'='*70}")


def demo():
    gen = CleanCharacterGenerator(seed=42, include_ethnicity=True)

    print("=" * 70)
    print("CLEAN CHARACTER GENERATOR v3")
    print("(Round-Robin Ethnicity, Fashion Styles, No LAION)")
    print("=" * 70)
    print(f"Ethnicities:    {len(gen.ethnicities)}")
    print(f"Fashion styles: {len(gen.fashion_styles)}")
    print(f"Photo styles:   {len(gen.photo_styles)}")

    print("\nGENERATOR SAMPLES:")
    print("-" * 70)

    for name in ['simple_character', 'character_with_clothing', 'character_full_outfit',
                 'character_styled', 'character_portrait', 'full_character']:
        method = getattr(gen, name)
        print(f"  {name:28s}: {method()}")

    print("\nBATCH (25):")
    print("-" * 70)
    captions, types = gen.generate_batch(25)
    for cap, t in zip(captions, types):
        print(f"  [{t:24s}] {cap}")

    # Test distributions
    print("\n\nTesting distributions (1000 samples)...")
    gen2 = CleanCharacterGenerator(seed=123, include_ethnicity=True)
    gen2.generate_batch(1000)
    gen2.print_ethnicity_report()
    gen2.print_fashion_report()


if __name__ == "__main__":
    demo()