"""
list_functions: grammar.py | Author : Sam Acquaviva.
Utility functions for loading in the DSLs for the list domain.
"""

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader
from src.models.laps_grammar import LAPSGrammar

import dreamcoder.domains.list.listPrimitives as listPrimitives
from dreamcoder.program import Program

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]

LIST_PRIMITIVE_SETS = {
    "rich": listPrimitives.primitives,
    "base": listPrimitives.basePrimitives,
    "re2_list_v0": listPrimitives.re2_list_v0,
    "mccarthy": listPrimitives.McCarthyPrimitives,
    "common": listPrimitives.bootstrapTarget_extra
}


@GrammarRegistry.register
class ListFunctionsGrammarLoader(ModelLoader):
    """Loads the list functions grammar.
    Original source: dreamcoder/domains/re2/re2Primitives.
    Semantics are only implemented in OCaml.
    """

    name = "list_functions"  # Special handler for OCaml enumeration.

    def load_model(self, experiment_state, dsl="rich"):
        primitives = LIST_PRIMITIVE_SETS[dsl]() if dsl else []
        grammar = LAPSGrammar.uniform(primitives)
        grammar.function_prefix = "_lf"
        return grammar
