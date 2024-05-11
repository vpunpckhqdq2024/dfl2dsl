"""Makes the prompts for converting Python to DSL."""

from typing import Optional
def make_convert_to_python_prompt(domain_description: Optional[str], examples: Optional[str], python_code: Optional[str], problem_description: Optional[str]) -> str:
    """Makes the prompts for converting a set of examples to a Python function."""
    
    prompt = ""
    if domain_description:
        prompt += f"# Domain description\n{domain_description}\n\n"
    if problem_description:
        prompt += f"# Problem description\nHere is a description of the task: \"{problem_description}\"\n\n"
    if examples:
        prompt += f"Here are examples of the transformation:\n{examples}\n\n"
    if python_code:
        prompt += f"```\nHere is the python code that solves the task.\n{python_code}\n```\n\n"

DECLARE_PRIMITIVES_INIT_PROGRAM = """# Domain description
{domain_description}

# Problem description
Here is a description of the task: "{problem_description}"

Here are examples of the transformation:
{examples}
```

Here is the python code that solves the task.
{python_code}
```

# DSL description
Your DSL already has the following primitive functions:
```
{existing_primitive_functions}
```

and the following primitive constants:
```
{existing_primitive_constants}
```

These are the types in your DSL:
{existing_types}

# Task description
Define all the new primitives (functions and/or constants) that you will need in order to translate the task specified above. Each primitive declaration should have the following format:
`Primitive(<primitive name>, <primitive type>, <curried primitive_function>)`. Then, write a program in De-Brujin indexed lambda calculus using these primitives.

# Final instructions
Remember, *anything* you use in the solution will have to be defined, even basic control flow functions (like "if") and constants (like "s"). Do not put quotes around primitives in the final program. Follow the exact format of previous examples."""

arithmetic_ex_q = DECLARE_PRIMITIVES_INIT_PROGRAM.format(
    domain_description="You are creating a DSL for arithmetic operations. Each task takes in 2 numbers and returns 1 number.",
    problem_description="return the sum if both numbers are even, otherwise return the product.",
    examples="(5, 4) -> 20\n(8, 4) -> 12\n(7, 7) -> 49",
    python_code="def fn(num1, num2):\n    if num1 % 2 == 0 and num2 % 2 == 0:\n        return num1 + num2\n    return num1 * num2",
    existing_primitive_functions=(
        "Primitive(\"if\", arrow(tbool, t0, t0, t0), lambda b: lambda x: lambda y: x if b else y)\n"
        "Primitive(\"+\", arrow(tint, tint, tint), lambda x: lambda y: x + y)"),
    existing_primitive_constants="Primitive(\"0\", tint, 0)",
    existing_types=(
        "`tint`: represents an integer type.\n"
        "`tbool`: represents a boolean type.\n"
        "`arrow`: represents a function. For example, arrow(tint, tint, tbool) represents a function that accepts 2 integer variables and returns a boolean.\n"
        "`t0` and `t1`: 2 type variables each representing a generic type.")
)

arithmetic_ex_a = """Here are the new primitives that the DSL needs:
```
[
    Primitive("is_even", arrow(tint, tbool), lambda x: x % 2 == 0),
    Primitive("and", arrow(tbool, tbool, tbool), lambda x: lambda y: x and y),
    Primitive("*", arrow(tint, tint, tint), lambda x: lambda y: x * y)
]
```

Here is the solution program:
```
(lambda (lambda (if (and (is_even $0) (is_even $1)) (+ $0 $1) (* $0 $1))))
```"""

list_function_ex_q = DECLARE_PRIMITIVES_INIT_PROGRAM.format(
    domain_description="You are creating a DSL for list functions. Each task takes in 1 list of integers and returns 1 list of integers.",
    problem_description="get the element at index 2",
    examples="[6, 8, 2] -> [2]\n[8, 0, 4, 1, 2] -> [4]\n[1] -> []",
    python_code="def fn(x):\n    if len(x) >= 3:\n        return x[2]\n    return []",
    existing_primitive_functions="(None)",
    existing_primitive_constants="(None)",
    existing_types=(
        "`tlist`: represents a list type. tlist(<type>) represents a list of <type>.\n"
        "`tint`: represents an integer type.\n"
        "`tbool`: represents a boolean type.\n"
        "`arrow`: represents a function. For example, arrow(tint, tint, tbool) represents a function that accepts 2 integer variables and returns a boolean.\n"
        "`t0` and `t1`: 2 type variables each representing a generic type."))


list_function_ex_a = """Here are the new primitives that the DSL needs:
```
[
    Primitive("get_item", arrow(tlist(t0), tint, tint), lambda l: lambda i: l[i]),
    Primitive("length", arrow(tlist(t0), tint), lambda x: len(x)),
    Primitive(">", arrow(tint, tint, tbool), lambda x: lambda y: x > y),
    Primitive("if", arrow(tbool, t0, t0, t0), lambda c: lambda x: lambda y: x if c else y),
    Primitive("2", tint, 2),
    Primitive("empty_list", tlist, []),
    Primitive("int->list", arrow(tint, tlist(tint)), lambda x: [x])
]
```

Here is the solution program:
```
(lambda (if (> (length $0) 2) (int->list (get_item $0 2)) empty_list))
```"""

initial_conversion_2shot = [
    {"role": "user", "content": arithmetic_ex_q},
    {"role": "assistant", "content": arithmetic_ex_a},
    {"role": "user", "content": list_function_ex_q},
    {"role": "assistant", "content": list_function_ex_a}
]
