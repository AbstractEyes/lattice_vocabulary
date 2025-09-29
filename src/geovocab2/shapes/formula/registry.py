
# Contains the singleton global registry of formulas currently known to GeoVocab2.
class FormulaRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FormulaRegistry, cls).__new__(cls)
            cls._instance._formulas = {}
        return cls._instance

    def register_formula(self, name, formula):
        self._formulas[name] = formula

    def get_formula(self, name):
        return self._formulas.get(name)

    def list_formulas(self):
        return list(self._formulas.keys())