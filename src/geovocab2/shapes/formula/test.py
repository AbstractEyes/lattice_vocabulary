
import torch

from shapes.formula.simple.add import AddFormula

if __name__ == "__main__":
    model = AddFormula()

    # Runtime (dev-mode) execution
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    out = model.evaluate(a, b)
    print("Eager result:", out["sum"])  # tensor([4., 6.])

    # TorchScript compile
    try:
        scripted = torch.jit.script(model)
        out_scripted = scripted(a, b)
        print("Scripted result:", out_scripted["sum"])  # tensor([4., 6.])
    except Exception as e:
        print("TorchScript not available:", e)
    # Torch compile (PyTorch 2.0+)
    try:
        compiled = torch.compile(model)
        out_compiled = compiled(a, b)
        print("Compiled result:", out_compiled["sum"])  # tensor([4., 6.])
    except Exception as e:
        print("Torch compile not available:", e)
