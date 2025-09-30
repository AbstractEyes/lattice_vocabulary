# F.py - Ultra-concise formula interface
"""
Usage:
    from F import F
    F.mix(x, y, 0.5)
    F.sig("mix:linear:v1")  # See signature
"""

import torch
from functools import lru_cache

from shapes.formula.geo_tensor import GeoTensor


from formula_bank import LIB, GeoTensor  # Assume LIB is pre-loaded global

class F:
    """Direct formula access."""

    @staticmethod
    def run(fid: str, **kw):
        """Run any formula directly."""
        gt = GeoTensor(kw)
        return LIB.bind(fid, gt, {k: k for k in kw}).run()

    @staticmethod
    def info(fid: str):
        """Get formula constraints and info."""
        spec = LIB.get(fid)
        return {
            "inputs": spec.inputs,
            "outputs": spec.outputs,
            "required": spec.constraint.required,
            "tier": spec.constraint.tier,
            "shape_bind": spec.constraint.shape_bind,
            "description": getattr(spec, 'description', ''),
            "category": spec.category.name
        }

    @staticmethod
    def sig(fid: str):
        """Get formula signature with constraints."""
        spec = LIB.get(fid)
        c = spec.constraint

        # Build signature string
        inputs = []
        for inp in spec.inputs:
            constraint = ""

            # Add shape constraints
            if c.shape_bind and inp in c.shape_bind:
                constraint = f"(shape={c.shape_bind[inp]})"
            elif c.shape_bind:
                for k, v in c.shape_bind.items():
                    if v == inp:
                        constraint = f"(base_shape)"
                        break

            # Add type hints based on common patterns
            if inp in ["x", "y", "z", "q", "k", "v", "anchor", "target", "context", "basis", "kv"]:
                type_hint = "[B,L,D]"
            elif inp in ["alpha", "gate", "mask", "weights", "scale", "t", "w1", "w2", "momentum", "condition"]:
                type_hint = "[B,L,1]"
            elif inp == "weight":
                type_hint = "[D,D]"
            elif inp in ["temperature", "sparsity", "levels", "components"]:
                type_hint = "scalar"
            else:
                type_hint = "Tensor"

            inputs.append(f"{inp}:{type_hint}{constraint}")

        outputs = list(spec.outputs)

        return {
            "signature": f"{fid}({', '.join(inputs)}) -> ({', '.join(outputs)})",
            "inputs": inputs,
            "outputs": outputs,
            "tier": c.tier,
            "required": c.required
        }

    @staticmethod
    def check(fid: str, **tensors):
        """Check if tensors meet constraints."""
        spec = LIB.get(fid)
        c = spec.constraint

        # Check required fields
        missing = [r for r in c.required if r not in tensors]
        if missing:
            return False, f"Missing: {missing}"

        # Check shape bindings
        if c.shape_bind:
            for field, bind_to in c.shape_bind.items():
                if field in tensors and bind_to in tensors:
                    if tensors[field].shape[:-1] != tensors[bind_to].shape[:-1]:
                        return False, f"{field} shape must match {bind_to}"

        # Check dtypes
        for k, v in tensors.items():
            if v.dtype not in c.allow_dtypes:
                return False, f"{k} dtype {v.dtype} not allowed"

        return True, "OK"

    @staticmethod
    def find(pattern: str):
        """Find formulas by pattern."""
        pattern_lower = pattern.lower()
        matches = []
        for fid in LIB._formulas.keys():
            if pattern_lower in fid.lower():
                matches.append(fid)
        return matches

    @staticmethod
    def list(category: str = None):
        """List formulas by category."""
        if category:
            from formula_bank import FormulaCategory
            cat = getattr(FormulaCategory, category.upper())
            return LIB.list_by_category(cat)
        return list(LIB._formulas.keys())

    # === SHORTCUTS ===
    @classmethod
    def mix(cls, x, y, alpha, m="linear"):
        return cls.run(f"mix:{m}:v1", x=x, y=y, alpha=alpha)["out"]

    @classmethod
    def attn(cls, q, k, v, m="flash"):
        return cls.run(f"attn:{m}:v1", q=q, k=k, v=v)["out"]

    @classmethod
    def sim(cls, x, y, m="cosine"):
        return cls.run(f"sim:{m}:v1", x=x, y=y)["similarity"]

    @classmethod
    def norm(cls, x, m="rms"):
        return cls.run(f"norm:{m}:v1", x=x)["out"]

    @classmethod
    def proj(cls, x, m="unit_sphere"):
        return cls.run(f"proj:{m}:v1", x=x)["out"]

    @classmethod
    def gate(cls, x, m="sigmoid"):
        return cls.run(f"gate:{m}:v1", x=x)["gate"]


# Quick access functions
def sig(fid):
    """Display formula signature with constraints."""
    try:
        s = F.sig(fid)
        print(s["signature"])
        if s.get("tier"):
            print(f"  Tier: {s['tier']}")
        return s
    except KeyError:
        print(f"Formula '{fid}' not found")
        return None


def sigs(pattern_or_category):
    """Show signatures for multiple formulas."""
    # Check if it's a category
    try:
        from formula_bank import FormulaCategory
        cat = getattr(FormulaCategory, pattern_or_category.upper(), None)
        if cat:
            formulas = F.list(pattern_or_category)
            print(f"\n{pattern_or_category.upper()} signatures:")
        else:
            formulas = F.find(pattern_or_category)
            print(f"\nSignatures matching '{pattern_or_category}':")
    except:
        formulas = F.find(pattern_or_category)
        print(f"\nSignatures matching '{pattern_or_category}':")

    for fid in formulas:
        s = F.sig(fid)
        print(f"  {s['signature']}")

    return formulas


def info(fid):
    """Quick info about a formula."""
    try:
        i = F.info(fid)
        s = F.sig(fid)
        print(s["signature"])
        print(f"  Category: {i['category']}")
        print(f"  Tier: {i['tier']}")
        if i.get('shape_bind'):
            print(f"  Constraints: {i['shape_bind']}")
        return i
    except KeyError:
        print(f"Formula '{fid}' not found")
        return None


def check(fid, **tensors):
    """Check constraints before running."""
    try:
        # First show what's expected
        s = F.sig(fid)
        print(f"Expected: {s['signature']}")

        # Then check provided tensors
        valid, msg = F.check(fid, **tensors)
        if valid:
            print(f"✓ Provided tensors: {msg}")
        else:
            print(f"✗ Provided tensors: {msg}")
        return valid
    except KeyError:
        print(f"✗ {fid}: Formula not found")
        return False


def find(pattern):
    """Find and display matching formulas."""
    matches = F.find(pattern)
    if matches:
        print(f"Found {len(matches)} formulas matching '{pattern}':")
        for fid in matches:
            s = F.sig(fid)
            print(f"  {s['signature']}")
    else:
        print(f"No formulas found matching '{pattern}'")
    return matches


def ls(category=None):
    """List formulas with signatures."""
    if category:
        formulas = F.list(category)
        print(f"\n{category} formulas ({len(formulas)}):")
    else:
        formulas = F.list()
        print(f"\nAll formulas ({len(formulas)}):")

    for fid in formulas[:10]:  # Show first 10
        s = F.sig(fid)
        # Shorten long signatures
        sig = s['signature']
        if len(sig) > 60:
            sig = sig[:57] + "..."
        print(f"  {sig}")
    if len(formulas) > 10:
        print(f"  ... and {len(formulas) - 10} more")
    return formulas


# Usage demo
if __name__ == "__main__":
    x = torch.randn(2, 16, 64)
    y = torch.randn(2, 16, 64)
    alpha = torch.rand(2, 16, 1)

    # Get signature
    print("Signature:")
    sig("mix:linear:v1")
    print()

    # Show multiple signatures
    sigs("MIXING")
    print()

    # Full info with signature
    print("Full info:")
    info("mix:linear:v1")
    print()

    # Check with signature display
    print("Constraint check:")
    check("mix:linear:v1", x=x, y=y, alpha=alpha)
    print()
    check("mix:linear:v1", x=x, y=y)  # Missing alpha
    print()

    # Find with signatures
    print("Find results:")
    find("attention")
    print()

    # Direct execution
    result = F.run("mix:smooth:v1", x=x, y=y, t=alpha)
    print(f"Result shape: {result['out'].shape}")