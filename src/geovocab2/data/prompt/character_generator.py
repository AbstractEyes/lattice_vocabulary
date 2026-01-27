"""
Clean Character Generator
=========================
Generates character descriptions for image synthesis.
Uses trusted human-related lists from BulkCaptions.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import random

from geovocab2.data.prompt.bulk_caption_data import BulkCaptions
from geovocab2.data.prompt.symbolic_tree import SeedGen


@dataclass
class CharacterSpec:
    """Structured character specification"""
    gender: str
    pose: Optional[str] = None
    hair_length: Optional[str] = None
    hair_style: Optional[str] = None
    hair_color: Optional[str] = None
    upper_clothing: Optional[str] = None
    lower_clothing: Optional[str] = None
    footwear: Optional[str] = None
    accessory: Optional[str] = None
    jewelry: Optional[str] = None
    headwear: Optional[str] = None
    expression: Optional[str] = None
    action: Optional[str] = None
    background: Optional[str] = None
    lighting: Optional[str] = None
    style: Optional[str] = None
    angle: Optional[str] = None
    color_theme: Optional[str] = None


class CleanCharacterGenerator:
    """
    Generate character prompts from trusted lists:
    - GENDER_TYPES
    - HUMAN_POSES
    - HAIRSTYLES_TYPES
    - HAIR_LENGTH_TYPES
    - UPPER_BODY_CLOTHES_TYPES
    - LOWER_BODY_CLOTHES_TYPES
    - FOOTWEAR_TYPES
    - ACCESSORY_TYPES
    - JEWELRY_TYPES
    - HEADWEAR_TYPES
    - HUMAN_EXPRESSIONS
    - HUMAN_ACTIONS
    - BACKGROUND_TYPES
    - LIGHTING_TYPES
    - STYLE_TYPES
    - SUBJECT_PHOTOGRAPH_ANGLE
    - COLORS
    """

    def __init__(self, seed: int = None):
        self.sg = SeedGen(seed)
        self.bc = BulkCaptions

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

        # Descriptors
        self.colors = self.bc.COLORS
        self.materials = self.bc.MATERIAL_TYPES

    def _sample(self, lst: List[str]) -> str:
        return self.sg.choice(lst)

    def _maybe(self, lst: List[str], prob: float = 0.5) -> Optional[str]:
        """Sample with probability, else None"""
        if self.sg.random() < prob:
            return self.sg.choice(lst)
        return None

    def _build_hair(self, length: str = None, style: str = None, color: str = None) -> str:
        """Build hair description"""
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

    def _build_clothing(self, upper: str = None, lower: str = None,
                        color: str = None, material: str = None) -> str:
        """Build clothing description"""
        parts = []

        if upper:
            upper_desc = []
            if color and self.sg.random() < 0.5:
                upper_desc.append(color)
            if material and self.sg.random() < 0.3:
                upper_desc.append(material)
            upper_desc.append(upper)
            parts.append(' '.join(upper_desc))

        if lower:
            lower_desc = []
            if color and self.sg.random() < 0.3 and not upper:
                lower_desc.append(color)
            lower_desc.append(lower)
            parts.append(' '.join(lower_desc))

        if len(parts) == 2:
            return f"wearing {parts[0]} and {parts[1]}"
        elif len(parts) == 1:
            return f"wearing {parts[0]}"
        return None

    # === GENERATORS ===

    def simple_character(self) -> str:
        """Basic: {gender} with {hair}"""
        gender = self._sample(self.genders)
        hair_color = self._maybe(self.colors, 0.6)
        hair_length = self._maybe(self.hair_lengths, 0.7)
        hair = self._build_hair(hair_length, color=hair_color)

        if hair:
            return f"{gender} with {hair}"
        return gender

    def character_with_clothing(self) -> str:
        """{gender} wearing {clothing}"""
        gender = self._sample(self.genders)
        upper = self._sample(self.upper_clothing)
        lower = self._maybe(self.lower_clothing, 0.6)
        color = self._maybe(self.colors, 0.5)

        clothing = self._build_clothing(upper, lower, color)
        return f"{gender} {clothing}"

    def character_with_pose(self) -> str:
        """{gender} {pose}"""
        gender = self._sample(self.genders)
        pose = self._sample(self.poses)
        return f"{gender} {pose}"

    def character_with_expression(self) -> str:
        """{expression} {gender}"""
        gender = self._sample(self.genders)
        expression = self._sample(self.expressions)
        return f"{expression} {gender}"

    def character_with_action(self) -> str:
        """{gender} {action}"""
        gender = self._sample(self.genders)
        action = self._sample(self.actions)
        return f"{gender} {action}"

    def character_full_outfit(self) -> str:
        """{gender} with {hair}, wearing {full outfit}"""
        gender = self._sample(self.genders)

        # Hair
        hair_color = self._maybe(self.colors, 0.5)
        hair_length = self._maybe(self.hair_lengths, 0.6)
        hair_style = self._maybe(self.hair_styles, 0.4)
        hair = self._build_hair(hair_length, hair_style, hair_color)

        # Clothing
        upper = self._sample(self.upper_clothing)
        lower = self._maybe(self.lower_clothing, 0.7)
        footwear = self._maybe(self.footwear, 0.4)
        color = self._maybe(self.colors, 0.5)

        parts = [gender]
        if hair:
            parts.append(f"with {hair}")

        clothing_parts = []
        if color:
            clothing_parts.append(f"{color} {upper}")
        else:
            clothing_parts.append(upper)
        if lower:
            clothing_parts.append(lower)
        if footwear:
            clothing_parts.append(footwear)

        if len(clothing_parts) > 1:
            parts.append(f"wearing {', '.join(clothing_parts[:-1])} and {clothing_parts[-1]}")
        else:
            parts.append(f"wearing {clothing_parts[0]}")

        return ', '.join(parts)

    def character_with_accessory(self) -> str:
        """{gender} with {accessory}"""
        gender = self._sample(self.genders)
        accessory = self._sample(self.accessories)
        return f"{gender} with {accessory}"

    def character_with_jewelry(self) -> str:
        """{gender} wearing {jewelry}"""
        gender = self._sample(self.genders)
        jewelry = self._sample(self.jewelry)
        color = self._maybe(self.colors, 0.4)
        material = self._maybe(self.materials, 0.3)

        parts = []
        if color:
            parts.append(color)
        if material:
            parts.append(material)
        parts.append(jewelry)

        return f"{gender} wearing {' '.join(parts)}"

    def character_with_headwear(self) -> str:
        """{gender} wearing {headwear}"""
        gender = self._sample(self.genders)
        headwear = self._sample(self.headwear)
        color = self._maybe(self.colors, 0.4)

        if color:
            return f"{gender} wearing {color} {headwear}"
        return f"{gender} wearing {headwear}"

    def character_in_scene(self) -> str:
        """{gender} in {background}"""
        gender = self._sample(self.genders)
        background = self._sample(self.backgrounds)
        return f"{gender} in {background}"

    def character_with_lighting(self) -> str:
        """{gender}, {lighting}"""
        gender = self._sample(self.genders)
        upper = self._sample(self.upper_clothing)
        lighting = self._sample(self.lighting)
        return f"{gender} wearing {upper}, {lighting}"

    def character_portrait(self) -> str:
        """Portrait-style: {angle}, {expression} {gender} with {hair}"""
        gender = self._sample(self.genders)
        angle = self._sample(self.angles)
        expression = self._maybe(self.expressions, 0.7)
        hair_color = self._maybe(self.colors, 0.6)
        hair_length = self._maybe(self.hair_lengths, 0.5)
        hair = self._build_hair(hair_length, color=hair_color)

        parts = [angle]
        if expression:
            parts.append(f"{expression} {gender}")
        else:
            parts.append(gender)
        if hair:
            parts.append(f"with {hair}")

        return ', '.join(parts)

    def character_styled(self) -> str:
        """{gender} with {hair}, {style}"""
        gender = self._sample(self.genders)
        hair_color = self._maybe(self.colors, 0.5)
        hair_length = self._maybe(self.hair_lengths, 0.6)
        hair = self._build_hair(hair_length, color=hair_color)
        style = self._sample(self.styles)

        if hair:
            return f"{gender} with {hair}, {style}"
        return f"{gender}, {style}"

    def full_character(self) -> str:
        """Complete character with all elements"""
        gender = self._sample(self.genders)

        # Hair
        hair_color = self._maybe(self.colors, 0.6)
        hair_length = self._maybe(self.hair_lengths, 0.7)
        hair_style = self._maybe(self.hair_styles, 0.5)
        hair = self._build_hair(hair_length, hair_style, hair_color)

        # Expression/pose
        expression = self._maybe(self.expressions, 0.5)
        pose = self._maybe(self.poses, 0.4)

        # Clothing
        upper = self._sample(self.upper_clothing)
        lower = self._maybe(self.lower_clothing, 0.6)
        color = self._maybe(self.colors, 0.5)

        # Accessories
        accessory = self._maybe(self.accessories, 0.3)

        # Scene
        background = self._maybe(self.backgrounds, 0.5)
        lighting = self._maybe(self.lighting, 0.4)

        # Build prompt
        parts = []

        # Subject
        if expression:
            parts.append(f"{expression} {gender}")
        else:
            parts.append(gender)

        if hair:
            parts.append(f"with {hair}")

        if pose:
            parts.append(pose)

        # Clothing
        clothing_parts = []
        if color:
            clothing_parts.append(f"{color} {upper}")
        else:
            clothing_parts.append(upper)
        if lower:
            clothing_parts.append(lower)

        if clothing_parts:
            parts.append(f"wearing {' and '.join(clothing_parts)}")

        if accessory:
            parts.append(f"with {accessory}")

        # Scene
        if background:
            parts.append(f"in {background}")

        if lighting:
            parts.append(lighting)

        return ', '.join(parts)

    def action_scene(self) -> str:
        """{gender} {action} in {background}"""
        gender = self._sample(self.genders)
        action = self._sample(self.actions)
        background = self._sample(self.backgrounds)
        upper = self._maybe(self.upper_clothing, 0.5)

        if upper:
            return f"{gender} wearing {upper}, {action}, in {background}"
        return f"{gender} {action}, in {background}"

    # === BATCH ===

    def generate_batch(self, n: int) -> List[str]:
        """Generate n character prompts with balanced mix."""
        generators = [
            (0.08, 'simple_character', self.simple_character),
            (0.12, 'character_with_clothing', self.character_with_clothing),
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
            (0.05, 'character_styled', self.character_styled),
            (0.05, 'full_character', self.full_character),
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


def demo():
    gen = CleanCharacterGenerator(seed=42)

    print("=" * 70)
    print("CLEAN CHARACTER GENERATOR")
    print("=" * 70)

    print("\nGENERATOR SAMPLES:")
    print("-" * 70)

    for name in ['simple_character', 'character_with_clothing', 'character_with_pose',
                 'character_with_expression', 'character_with_action', 'character_full_outfit',
                 'character_with_accessory', 'character_with_jewelry', 'character_with_headwear',
                 'character_in_scene', 'character_with_lighting', 'character_portrait',
                 'character_styled', 'full_character', 'action_scene']:
        method = getattr(gen, name)
        print(f"  {name:28s}: {method()}")

    print("\nBATCH (20):")
    print("-" * 70)
    captions, types = gen.generate_batch(20)
    for cap, t in zip(captions, types):
        print(f"  [{t:24s}] {cap}")


if __name__ == "__main__":
    demo()