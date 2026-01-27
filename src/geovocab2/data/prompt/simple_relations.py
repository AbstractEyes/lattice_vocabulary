"""
Simple Object Relations - Clean Version v2
==========================================
Uses trusted lists + HUMAN_SURFACES + CLOTHING_SURFACE_LINKERS
"""

from typing import List
from dataclasses import dataclass
from geovocab2.data.prompt.bulk_caption_data import BulkCaptions
from geovocab2.data.prompt.symbolic_tree import SeedGen


@dataclass
class Relation:
    phrase: str
    weight: float = 1.0
    reverse: str = None


# Object-to-object relations
OBJECT_RELATIONS = [
    Relation("on", 3.0, "under"),
    Relation("on top of", 2.0, "beneath"),
    Relation("next to", 3.0),
    Relation("beside", 2.5),
    Relation("near", 2.0),
    Relation("above", 2.0, "below"),
    Relation("left of", 2.0, "right of"),
    Relation("in front of", 2.0, "behind"),
]

# Containment (only for valid size pairings)
CONTAINMENT_RELATIONS = [
    Relation("in", 3.0, "containing"),
    Relation("inside", 2.5),
]


class CleanRelationGenerator:
    """
    Trusted lists only:
    - OBJECT_TYPES
    - FRUIT_AND_VEGETABLE
    - CHAIR_TYPES
    - CLOTHING_TYPES
    - JEWELRY_TYPES
    - COLORS
    - MATERIAL_TYPES
    - HUMAN_SURFACES
    - CLOTHING_SURFACE_LINKERS
    """

    def __init__(self, seed: int = None):
        self.sg = SeedGen(seed)
        self.bc = BulkCaptions

        # Core object lists
        self.objects = self.bc.OBJECT_TYPES
        self.produce = self.bc.FRUIT_AND_VEGETABLE
        self.chairs = self.bc.CHAIR_TYPES
        self.clothing = self.bc.CLOTHING_TYPES
        self.jewelry = self.bc.JEWELRY_TYPES

        # Descriptors
        self.colors = self.bc.COLORS
        self.materials = self.bc.MATERIAL_TYPES

        # Surfaces and linkers
        self.surfaces = self.bc.HUMAN_SURFACES
        self.clothing_linkers = self.bc.CLOTHING_SURFACE_LINKERS

        # Containers (hand-picked from objects that can hold things)
        self.containers = [
            "bowl", "cup", "mug", "glass", "jar", "vase", "pot",
            "basket", "box", "bucket", "bottle", "bag", "tray",
            "plate", "dish", "drawer", "bin", "crate", "barrel",
        ]

        # Small items (can go IN containers)
        self.small_items = self.produce[:60] + [
            "key", "coin", "ring", "pen", "pencil", "marble",
            "button", "seed", "leaf", "flower", "stone", "nut",
        ]

        self.object_relations = OBJECT_RELATIONS
        self.containment_relations = CONTAINMENT_RELATIONS

    def _sample(self, lst: List[str]) -> str:
        return self.sg.choice(lst)

    def _sample_relation(self) -> Relation:
        weights = [r.weight for r in self.object_relations]
        total = sum(weights)
        r = self.sg.random() * total
        cumulative = 0
        for rel, w in zip(self.object_relations, weights):
            cumulative += w
            if r <= cumulative:
                return rel
        return self.object_relations[-1]

    def _sample_containment(self) -> Relation:
        return self.sg.choice(self.containment_relations)

    def _sample_clothing_linker(self) -> str:
        return self.sg.choice(self.clothing_linkers)

    # === OBJECT GENERATORS ===

    def object_object(self) -> str:
        """object {rel} object"""
        obj_a = self._sample(self.objects)
        obj_b = self._sample(self.objects)
        while obj_b == obj_a:
            obj_b = self._sample(self.objects)
        rel = self._sample_relation()
        return f"{obj_a} {rel.phrase} {obj_b}"

    def object_related_object(self) -> str:
        """object {rel} object"""
        obj_a = self._sample(self.objects)
        obj_b = self._sample(self.objects)
        while obj_b == obj_a:
            obj_b = self._sample(self.objects)
        rel = self._sample_relation()
        return f"{obj_a} {rel.phrase} {obj_b}"

    def colored_objects(self) -> str:
        """{color} object {rel} {color} object"""
        obj_a = self._sample(self.objects)
        obj_b = self._sample(self.objects)
        while obj_b == obj_a:
            obj_b = self._sample(self.objects)
        col_a = self._sample(self.colors)
        col_b = self._sample(self.colors)
        rel = self._sample_relation()
        return f"{col_a} {obj_a} {rel.phrase} {col_b} {obj_b}"

    def object_on_surface(self) -> str:
        """object on surface"""
        obj = self._sample(self.objects)
        surface = self._sample(self.surfaces)
        return f"{obj} on {surface}"

    def two_on_surface(self) -> str:
        """object {rel} object on surface"""
        obj_a = self._sample(self.objects)
        obj_b = self._sample(self.objects)
        while obj_b == obj_a:
            obj_b = self._sample(self.objects)
        surface = self._sample(self.surfaces)
        rel = self._sample_relation()
        return f"{obj_a} {rel.phrase} {obj_b} on {surface}"

    def item_in_container(self) -> str:
        """small_item in container"""
        item = self._sample(self.small_items)
        container = self._sample(self.containers)
        rel = self._sample_containment()
        return f"{item} {rel.phrase} {container}"

    def material_object(self) -> str:
        """{material} object"""
        material = self._sample(self.materials)
        obj = self._sample(self.objects)
        return f"{material} {obj}"

    def material_object_relation(self) -> str:
        """{material} object {rel} object"""
        material = self._sample(self.materials)
        obj_a = self._sample(self.objects)
        obj_b = self._sample(self.objects)
        while obj_b == obj_a:
            obj_b = self._sample(self.objects)
        rel = self._sample_relation()
        return f"{material} {obj_a} {rel.phrase} {obj_b}"

    def triple_objects(self) -> str:
        """obj, obj, and obj on surface"""
        objs = []
        rels = []
        while len(objs) < 3:
            obj = self._sample(self.objects)
            if obj not in objs:
                objs.append(obj)
                rels.append(self._sample_relation())
        return f"{objs[0]} {rels[0].phrase} {objs[1]} {rels[1].phrase} {objs[2]}"

    # === PRODUCE GENERATORS ===

    def produce_pair(self) -> str:
        """fruit {rel} fruit"""
        prod_a = self._sample(self.produce)
        prod_b = self._sample(self.produce)
        while prod_b == prod_a:
            prod_b = self._sample(self.produce)
        rel = self._sample_relation()
        return f"{prod_a} {rel.phrase} {prod_b}"

    def produce_on_surface(self) -> str:
        """fruit on surface"""
        prod = self._sample(self.produce)
        surface = self._sample(self.surfaces)
        return f"{prod} on {surface}"

    def still_life(self) -> str:
        """fruit {rel} fruit on surface"""
        prod_a = self._sample(self.produce)
        prod_b = self._sample(self.produce)
        while prod_b == prod_a:
            prod_b = self._sample(self.produce)
        surface = self._sample(self.surfaces)
        rel = self._sample_relation()
        return f"{prod_a} {rel.phrase} {prod_b} on {surface}"

    def colored_produce(self) -> str:
        """{color} fruit {rel} {color} fruit"""
        prod_a = self._sample(self.produce)
        prod_b = self._sample(self.produce)
        while prod_b == prod_a:
            prod_b = self._sample(self.produce)
        col_a = self._sample(self.colors)
        col_b = self._sample(self.colors)
        rel = self._sample_relation()
        return f"{col_a} {prod_a} {rel.phrase} {col_b} {prod_b}"

    # === CLOTHING GENERATORS (uses CLOTHING_SURFACE_LINKERS) ===

    def clothing_described(self) -> str:
        """{color} {material} clothing"""
        color = self._sample(self.colors)
        material = self._sample(self.materials)
        cloth = self._sample(self.clothing)
        return f"{color} {material} {cloth}"

    def clothing_on_surface(self) -> str:
        """clothing {linker} surface"""
        cloth = self._sample(self.clothing)
        surface = self._sample(self.surfaces)
        linker = self._sample_clothing_linker()
        return f"{cloth} {linker} {surface}"

    def colored_clothing_on_surface(self) -> str:
        """{color} clothing {linker} surface"""
        color = self._sample(self.colors)
        cloth = self._sample(self.clothing)
        surface = self._sample(self.surfaces)
        linker = self._sample_clothing_linker()
        return f"{color} {cloth} {linker} {surface}"

    def material_clothing_on_surface(self) -> str:
        """{material} clothing {linker} surface"""
        material = self._sample(self.materials)
        cloth = self._sample(self.clothing)
        surface = self._sample(self.surfaces)
        linker = self._sample_clothing_linker()
        return f"{material} {cloth} {linker} {surface}"

    # === JEWELRY GENERATORS ===

    def jewelry_described(self) -> str:
        """{color} {material} jewelry"""
        color = self._sample(self.colors)
        material = self._sample(self.materials)
        jewel = self._sample(self.jewelry)
        return f"{color} {material} {jewel}"

    def jewelry_on_surface(self) -> str:
        """jewelry on surface"""
        jewel = self._sample(self.jewelry)
        surface = self._sample(self.surfaces)
        return f"{jewel} on {surface}"

    # === CHAIR GENERATORS ===

    def chair_described(self) -> str:
        """{color} {material} chair"""
        color = self._sample(self.colors)
        material = self._sample(self.materials)
        chair = self._sample(self.chairs)
        return f"{color} {material} {chair}"

    def object_on_chair(self) -> str:
        """object on chair"""
        obj = self._sample(self.objects)
        chair = self._sample(self.chairs)
        return f"{obj} on {chair}"

    def clothing_on_chair(self) -> str:
        """clothing {linker} chair"""
        cloth = self._sample(self.clothing)
        chair = self._sample(self.chairs)
        linker = self._sample_clothing_linker()
        return f"{cloth} {linker} {chair}"

    # === BATCH ===

    def generate_batch(self, n: int) -> List[str]:
        """Generate n captions with balanced mix."""
        generators = [
            # Objects (35%)
            (0.10, self.object_object),
            (0.08, self.colored_objects),
            (0.08, self.object_on_surface),
            (0.06, self.two_on_surface),
            (0.03, self.triple_objects),

            # Produce (20%)
            (0.06, self.produce_pair),
            (0.05, self.produce_on_surface),
            (0.05, self.still_life),
            (0.04, self.colored_produce),

            # Clothing (20%)
            (0.05, self.clothing_described),
            (0.06, self.clothing_on_surface),
            (0.05, self.colored_clothing_on_surface),
            (0.04, self.material_clothing_on_surface),

            # Containment (5%)
            (0.05, self.item_in_container),

            # Materials (8%)
            (0.04, self.material_object),
            (0.04, self.material_object_relation),

            # Jewelry (6%)
            (0.03, self.jewelry_described),
            (0.03, self.jewelry_on_surface),

            # Chairs (6%)
            (0.02, self.chair_described),
            (0.02, self.object_on_chair),
            (0.02, self.clothing_on_chair),
        ]

        captions = []
        for _ in range(n):
            r = self.sg.random()
            cumulative = 0
            for weight, gen in generators:
                cumulative += weight
                if r <= cumulative:
                    captions.append(gen())
                    break

        return captions


def demo():
    gen = CleanRelationGenerator(seed=42)

    print("=" * 70)
    print("CLEAN OBJECT RELATION GENERATOR v2")
    print("=" * 70)

    print("\nOBJECT SAMPLES:")
    print("-" * 70)
    for name in ['object_object', 'colored_objects', 'object_on_surface',
                 'two_on_surface', 'item_in_container', 'material_object']:
        print(f"  {name:28s}: {getattr(gen, name)()}")

    print("\nPRODUCE SAMPLES:")
    print("-" * 70)
    for name in ['produce_pair', 'produce_on_surface', 'still_life', 'colored_produce']:
        print(f"  {name:28s}: {getattr(gen, name)()}")

    print("\nCLOTHING SAMPLES (with linkers):")
    print("-" * 70)
    for name in ['clothing_described', 'clothing_on_surface',
                 'colored_clothing_on_surface', 'material_clothing_on_surface']:
        print(f"  {name:28s}: {getattr(gen, name)()}")

    print("\nJEWELRY / CHAIR SAMPLES:")
    print("-" * 70)
    for name in ['jewelry_described', 'jewelry_on_surface',
                 'chair_described', 'object_on_chair', 'clothing_on_chair']:
        print(f"  {name:28s}: {getattr(gen, name)()}")

    print("\nBATCH (25):")
    print("-" * 70)
    for cap in gen.generate_batch(50):
        print(f"  {cap}")


if __name__ == "__main__":
    demo()