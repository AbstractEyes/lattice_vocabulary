import geovocab
print ("Formulas complete")
from geovocab2.shapes.formula import AngleBetweenVectors
from geovocab2.shapes.factory import SimplexFactory


print(AngleBetweenVectors)

factory = SimplexFactory(3, 4, backend="torch")
