"""Maps each type to NL descriptions of the type."""

# Maps from the type name to (singular description, singular when plural)
type_to_descs = {
    "tint": ("an integer", "integers"),
    "tbool": ("a boolean", "booleans"),
    "tstr": ("a string", "strings"),
    "arrow": ("a function. For example, arrow(tint, tint, tbool) represents a function that accepts 2 integer variables and returns a boolean.", "functions"),
    "tlist": ("a list.", "lists"),

    # clevr
    "tclevrcolor": (
        "a string representing a color (one of \"gray\", \"red\", \"blue\", \"green\", \"brown\", \"purple\", \"cyan\", \"yellow\")",
        "strings each representing a color (one of \"gray\", \"red\", \"blue\", \"green\", \"brown\", \"purple\", \"cyan\", \"yellow\")"),
    "tclevrshape": (
        "a string representing a shape (one of \"cube\", \"cylinder\", \"sphere\")",
        "strings each representing a shape (one of \"cube\", \"cylinder\", \"sphere\")"),
    "tclevrmaterial": (
        "a string representing a material (one of \"rubber\", \"metal\")",
        "strings each representing a material (one of \"rubber\", \"metal\")"),
    "tclevrsize": (
        "a string representing a size (one of \"small\", \"large\")",
        "strings each representing a size (one of \"small\", \"large\")"),
    "tclevrrelation": (
        "a string representing a relation (one of \"left\", \"right\", \"behind\", \"front\")",
        "strings each representing a relation (one of \"left\", \"right\", \"behind\", \"front\")"),
    "tclevrobject": (
        "a dictionary of attributes (color, material, shape, size, list of objects behind, list of objects in front, list of objects to the left, list of objects to the right) representing one object",
        "dictionaries of attributes (color, material, shape, size, list of objects behind, list of objects in front, list of objects to the left, list of objects to the right) each representing one object")
}
type_to_descs["int"] = type_to_descs["tint"]
type_to_descs["bool"] = type_to_descs["tbool"]
type_to_descs["str"] = type_to_descs["tstr"]

tclevrobj_str = (
    "a dictionary of attributes:"
    "\n\t- color: a string representing the color of the shape"
    "\n\t- material: a string representing the material of the shape"
    "\n\t- shape: a string representing the shape of the shape"
    "\n\t- size: a string representing the size of the shape (small, medium, or large)"
    "\n\t- behind: a list of the indices of the shapes that are behind this shape"
    "\n\t- front: a list of the indices of the shapes that are in front of this shape"
    "\n\t- left: a list of the indices of the shapes that are to the left of this shape"
    "\n\t- right: a list of the indices of the shapes that are to the right of this shape"
)
