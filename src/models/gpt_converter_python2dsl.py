"""
gpt_python_to_dsl.py | Author: Sam Acquaviva.

Queries GPT to convert solutions in python to DSL solutions in DreamCoder format.
"""

import os
import json
from typing import Dict, List, Tuple
import re
from copy import deepcopy
import numpy as np

from dreamcoder.task import Task
from dreamcoder.program import Program, Primitive, Invented, arrow, tint, tbool, t0, t1, t2, baseType, ParseFailure, tstr, tlist
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.enumeration import multicoreEnumeration
from prompts.python_to_dsl import initial_conversion_2shot, DECLARE_PRIMITIVES_INIT_PROGRAM
from prompts.type_to_description import type_to_descs
import src.models.model_loaders as model_loaders
from src.models.laps_grammar import LAPSGrammar
from src.models.sample_generator import GPTSampleGenerator
from src.query_utils import query_batch
from src.python_utils import execute_function, FailedExecutionResult

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LLM_CONVERTER_PYTHON_DSL]
CONVERT_COPY = "convert_copy"
CONVERT_LLM = "convert_llm"
COMPRESS_NONE = "compress_none"
COMPRESS_GREEDY_MERGE = "compress_greedy_merge"


# TODO: Currently brittly-dependent on primitive globals and names of primitives.
# TODO: Skip tasks that have already been solved in the DSL.


def get_lambdas(og_str: str):
    """Extracts all unique programs of form "(lambda ...)" from a string."""

    prim_def_starts = [m.start() for m in re.finditer(r"\(lambda", og_str)]
    lambda_defs = list()
    lambda_ranges = set()
    for idx in prim_def_starts:
        lambda_dec = extract_parentheses(og_str[idx:])

        # Avoid repeating sub programs.
        if idx in lambda_ranges:
            continue
        lambda_ranges |= set(range(idx, idx+len(lambda_dec)))

        # Only save unique programs.
        if lambda_dec not in lambda_defs:  # Not using sets so order is same across runs.
            lambda_defs.append(lambda_dec)
    return lambda_defs


def get_primitive_declarations(og_str: str):
    """Extracts unique strings representing Primitive declarations of form "Primitive(...)"."""
    prim_def_starts = [m.start() for m in re.finditer(r"Primitive\(", og_str)]
    primitive_defs = list()
    for idx in prim_def_starts:
        prim_dec = extract_parentheses(og_str[idx:])
        if prim_dec not in primitive_defs:  # Not using sets so order is same across runs.
            primitive_defs.append(prim_dec)
    return primitive_defs


def extract_parentheses(s):
    """Gets the substring that ends when the first parentheses is closed."""

    # Example: "Primitive(...) ..." -> "Primitive("...")"

    if '(' not in s:
        raise ValueError

    start_index = s.index('(')
    depth = 0
    for i in range(start_index, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return s[:i+1]
    raise ValueError


def get_primitive_str(prog: Program):
    """Gets the string declaring all of the primitives in the program."""

    prim_decs = []
    for _, subprog in prog.walk():
        if isinstance(subprog, Primitive):
            assert "Primitive(" in subprog.function_comment
            prim_decs.append(subprog.function_comment)

    return "[\n\t" + ",\n\t".join(prim_decs) + "\n]"


def get_primitives_in_prog(prog: Program, name_mapping: Dict[str, Primitive]):
    """Gets all the primitives used in a program."""

    prims = []
    for token in prog.left_order_tokens():
        if token in name_mapping:
            prims.append(name_mapping[token])
    return prims


@ModelRegistry.register
class GPTConverterPython2DSL(GPTSampleGenerator):    # type: ignore
    """Solves tasks in Python."""

    name = "gpt_converter_python2dsl"
    results_file = "gpt_converter_python2dsl_results.json"

    @staticmethod
    def load_model(experiment_state, engine=None, **kwargs):
        return GPTConverterPython2DSL(experiment_state=experiment_state, engine=engine)  # type: ignore

    def __init__(self, experiment_state=None, engine=None):
        # TODO: This will cause issues.
        Primitive.GLOBALS = {}  # Reset the global primitives.
        print("RESET PRIMITIVE GLOBALS")
        self.all_defined_prims = {}
        super().__init__(self, engine=engine)

    def convert_frontiers_to_dsl(
            self, experiment_state, task_split: str, task_batch_ids: list, conversion_strategy: str = CONVERT_LLM,
            compression_strategy: str = COMPRESS_GREEDY_MERGE,
            iteratively_update_grammar: bool = True, maximum_frontier: int = 5, verbose: bool = True,
            n_tasks = None, n_entries_to_try: int = 2, use_prev_primitives: bool = True,
            try_solved_only: bool = True) -> None:
        """Converts the frontiers to a DSL."""

        del n_tasks

        if not use_prev_primitives:
            Primitive.GLOBALS = {}

        old_globals = deepcopy(Primitive.GLOBALS)

        if iteratively_update_grammar or use_prev_primitives:
            raise ValueError("Broken, need to re-implement.")

        # TODO: Use local grammar throughout instead of updating the experiment state.
        if not use_prev_primitives:
            experiment_state.models[model_loaders.GRAMMAR] = LAPSGrammar.uniform([])

        conversion_strat_to_proc = {
            CONVERT_COPY: self.convert_solution_copy,
            CONVERT_LLM: self.convert_solution_llm
        }

        compression_strat_to_proc = {
            COMPRESS_NONE: self.no_compress,
            COMPRESS_GREEDY_MERGE: self.merge_duplicates
        }

        # Remove inventions from the frontiers.
        experiment_state.convert_frontiers_beta_normal()
        primitives_to_add = {}
        frontiers = {task: Frontier([], task) for task in experiment_state.get_tasks_for_ids(task_split, task_batch_ids)}
        task_to_solutions, task_to_primitives = {}, {}  # For logging.
        logs_per_task = {}

        # We only want to convert tasks that have been solved in Python but not in the DSL.
        solved_python_tasks = set([f.task.name for f in experiment_state.get_non_empty_python_frontiers_for_split(task_split)])
        solved_dsl_tasks = set([f.task.name for f in experiment_state.get_non_empty_frontiers_for_split(task_split)])
        tasks_of_interest = solved_python_tasks.union(solved_dsl_tasks)

        if try_solved_only:
            task_batch_ids = [task_id for task_id in task_batch_ids if (task_id in tasks_of_interest)]
        else:
            raise NotImplementedError("Need to implement storing / loading incorrect programs to convert.")

        for n, task_id in enumerate(task_batch_ids):

            task = experiment_state.get_tasks_for_ids(task_split, [task_id])[0]
            if verbose:
                print(f"\nConverting task {task_id} ({n+1}/{len(tasks_of_interest)})")

            # TODO: If task already in DSL, then don't try to translate it.
            # Currently, this will break it.
            # if task.name in solved_dsl_tasks:
            #     continue

            # Try to convert each solution to DSL primitives until success.
            python_frontier = experiment_state.task_frontiers_python[task_split][task]
            entries = python_frontier.entries[:n_entries_to_try]
            if task_id in solved_dsl_tasks and not python_frontier:
                entries = [FrontierEntry(program=None, logLikelihood=0.0, tokens=None, logPrior=0.0, origin=None)]
            for i, entry in enumerate(entries):

                if verbose:
                    print(f"{i+1}/{min(len(python_frontier.entries), n_entries_to_try)} entry")
                    print(f"Program:\n{entry.program}\n")

                new_primitives, new_frontier, logs = conversion_strat_to_proc[conversion_strategy](
                    entry.program, task_id, task_split, experiment_state)
                Primitive.GLOBALS = deepcopy(old_globals) # Reset the global primitives.
                logs_per_task.setdefault(task_id, []).append(logs)

                if not new_frontier:    # If couldn't convert, ignore.
                    if verbose:
                        print(f"Could not convert task {task_id} entry {i+1}")
                    continue
                if verbose:
                    print(f"New primitives: {[p.name for p in new_primitives]}")
                    print(f"Converted program: {new_frontier.entries[0].program}")

                assert not frontiers[task]
                frontiers[task].entries += new_frontier.entries
                primitives_to_add[task] = new_primitives
                task_to_solutions[task_id] = str(new_frontier.entries[0].program)
                task_to_primitives[task_id] = [str(p) for p in new_primitives]
                break   # Only need one successful conversion.

        # Compress DSL across individual task DSLs.
        new_frontiers, primitives_to_add = compression_strat_to_proc[compression_strategy](frontiers, primitives_to_add)

        # Update grammar + frontiers at the end if not iteratively updating grammar.
        # TODO: Only keep maximum_frontier.
        if not iteratively_update_grammar:
            for f in new_frontiers:
                experiment_state.task_frontiers[task_split][f.task] = f
            new_grammar = LAPSGrammar.uniform(primitives_to_add)

            # Recompute production probabilities in grammar
            non_empty_frontiers = [f for f in experiment_state.task_frontiers[task_split].values() if f]
            try:
                new_grammar = new_grammar.insideOutside(
                    non_empty_frontiers, pseudoCounts=30, iterations=1
                )
            except Exception as e:
                import pdb; pdb.set_trace()
                raise e
            grammar = LAPSGrammar.fromGrammar(new_grammar)  # Wrap in LAPSGrammar

        experiment_state.models[model_loaders.GRAMMAR] = grammar

        results = {
            "params": {
                "temperature": 0.,
                "n_samples": 1,
                "engine": self.ENGINE,
                "conversion_strategy": conversion_strategy,
            },
            "summary": {
                "n_tasks_solved": len(task_to_solutions.keys()),
                "tasks_solved": list(task_to_solutions.keys()),
            },
            "task_to_solutions": task_to_solutions,
            "task_to_primitives": task_to_primitives,
            "logs_per_task": logs_per_task,
        }

        # Save results to file.
        results_filepath = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            task_split,
            self.results_file,
        )
        os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
        with open(results_filepath, "w") as f:
            json.dump(results, f, indent=4)
        if verbose:
            print(f"Wrote results: {results_filepath}")

    # Python -> Individual task DSL strategies.

    def convert_solution_copy(self, solution: str, task_id: int, task_split: str, experiment_state) -> Tuple[List[Primitive], Frontier, None]:
        """Converts solution to DSL by just copying the solution as a new primitive.
        
        Args:
            solution (str): The solution in python.
            task (Task): The task to solve.

        Returns:
            (list[Primitive], Frontier, None): The newly defined primitives and frontier 
                the solution in these primitives.
        """

        # TODO: Name w/ LLM (or at least better parsing of task name).
        # TODO: Minimal exec checks, check name of solution is correct.
        # TODO: Check if solution already in grammar.
        task = experiment_state.get_tasks_for_ids(task_split, [task_id])[0]
        name = task.name[len("re2_train_0"):]
        loc = {}
        exec(solution, globals(), loc)  # Get the executable function.
        new_primitive = Primitive(name, task.request, loc["fn"])
        solution_prog = Program.parse("(lambda (" + name + " $0))")
        assert task.check(solution_prog, timeout=5)

        # Convert to frontier entry.
        new_frontier_entry = FrontierEntry(
            program=solution_prog, logLikelihood=0.0, tokens=None, logPrior=0.0, origin=None
        )

        return [new_primitive], Frontier([new_frontier_entry], task=task), None

    def _describe_type(self, task):
        """Generates a readable description for the type of the task."""

        task_type = task.request
        assert task_type.isArrow() and len(task_type.arguments) == 2
        tin, tout = task_type.arguments

        # Special case for strings = list(char).
        if tin.name == "list" and len(tin.arguments) == 1 and tin.arguments[0].name == "char":
            tin.name = "tstr"
            tin.arguments = []
        if tout.name == "list" and len(tout.arguments) == 1 and tout.arguments[0].name == "char":
            tout.name = "tstr"
            tout.arguments = []

        # Handle lists.
        if tin.name == "list":
            tin = "a list of " + type_to_descs[tin.arguments[0].name][1]
        else:
            tin = type_to_descs[tin.name][0]

        if tout.name == "list":
            tout = "a list of " + type_to_descs[tout.arguments[0].name][1]
        else:
            tout = type_to_descs[tout.name][0]

        return tin, tout

    def convert_solution_llm(self, solution: str, task_id: int, task_split: str, experiment_state, n_exs: int = 2):
        """Converts solution to DSL by querying LLM for primitives then searching over those primitives.
        
        Returns the new primitives + frontier + logs if the solution can be induced in the DSL or by the LLM."""

        task = experiment_state.get_tasks_for_ids(task_split, [task_id])[0]

        domain_types = experiment_state.metadata["existing_types"]
        task_in, task_out = self._describe_type(task)
        domain_desc = (f"You are creating a DSL for {experiment_state.metadata['domain_description']}. "
                    f"Each task takes in {task_in} and returns {task_out}.")
        existing_types = [f"`{t}`: {type_to_descs[t][0]}" for t in domain_types]
        existing_types.append("`t0` and `t1`: 2 type variables each representing a generic type.")
        existing_types = "\n".join(existing_types)

        # Get info to fill prompt.
        problem_descs = experiment_state.get_language_for_ids(task_split, [task_id])[0]
        if len(problem_descs) > 1:
            raise ValueError("Cannot handle multiple problem descriptions.")
        problem_desc = problem_descs[0] if problem_descs else "(None)"

        # Initial step: Define new primitives + suggest program using these primitives.
        prev_primitives = experiment_state.models[model_loaders.GRAMMAR].primitives
        proc_prims = [p for p in prev_primitives if p.tp.isArrow()]
        cons_prims = [p for p in prev_primitives if not p.tp.isArrow()]

        prompt = self.construct_initial_prompt(domain_desc, problem_desc, task.examples[:n_exs], solution, proc_prims, cons_prims, existing_types)

        # If task has solution in DSL, then don't query LLM -- just recreate string based on solution.
        task_frontier = experiment_state.task_frontiers[task_split][task]
        task_has_prog = bool(task_frontier)
        if task_has_prog:
            try:
                primitives_str = get_primitive_str(task_frontier.entries[0].program)
                program_str = str(task_frontier.entries[0].program)
            except:
                print("error")
                import pdb; pdb.set_trace()
            print(primitives_str)
            print(program_str)
            query_resp_str = (
                f"Here are the new primitives that the DSL needs:\n```\n{primitives_str}\n```"
                f"\n\nHere is the solution program:\n```\n{program_str}\n```")
            query_resp = [{"choices": [{"message": query_resp_str}]}]

        else:
            if solution is None:
                import pdb; pdb.set_trace()
            query_resp = query_batch([prompt], n_samples=1, temperature=0.0, max_tokens=1024, model_name=self.ENGINE, only_sample_once=False)
            try:
                primitives_str, program_str = self.extract_primitives_and_program(query_resp)
            except ValueError as e:  # Couldn't parse response.
                message = f"Could not parse response (error={e}): \n{query_resp}"
                print(message)
                return None, None, {"chat": prompt + [query_resp[0]["choices"][0]["message"]], "error": message}

        # Second step: Continue to re-prompt to fix common errors (e.g. not all primitives defined).
        primitive_strings = [primitives_str]  # Keep primitive declarations for all updates.
        max_redefine = min(max(len(primitives_str.split("\n")) - 2, 3), 5)  # 3-5 attempts to redefine.
        solution_prog = None
        for _ in range(max_redefine):
            solution_prog, primitives = self.check_solution(primitive_strings, program_str, task, domain_types)

            if task_has_prog:
                assert isinstance(solution_prog, Program)

            # Program compiled without error, continue to next step.
            if isinstance(solution_prog, Program) or solution_prog is None:
                break
            if solution is None:
                import pdb; pdb.set_trace()

            # Execution error -- attempt to fix well-known translation errors.
            assert isinstance(solution_prog, list) and isinstance(solution_prog[0], FailedExecutionResult)
            refine_prompt = self.construct_refine_prompt(solution_prog)
            if not refine_prompt:   # Unrecoverable error.
                message = f"Uncorrectable error for {program_str}: {solution_prog[0]}."
                print(message)
                return None, None, {"chat": prompt + [query_resp[0]["choices"][0]["message"]], "error": message}

            resp_message = query_resp[0]["choices"][0]["message"]
            prompt += [resp_message, refine_prompt]
            query_resp = query_batch([prompt], n_samples=1, temperature=0.0, max_tokens=1024, model_name=self.ENGINE, only_sample_once=False)
            try:
                new_primitives_str, program_str = self.extract_primitives_and_program(query_resp)
                primitive_strings.append(new_primitives_str)
            except ValueError as e:  # Couldn't parse response.
                message = f"Could not parse response on update (error={e}): \n{query_resp}"
                print(message)
                return None, None, {"chat": prompt + [query_resp[0]["choices"][0]["message"]], "error": message}

        else:   # Final try to induce program.
            solution_prog, primitives = self.check_solution(primitive_strings, program_str, task, domain_types)
            if isinstance(solution_prog, list) and isinstance(solution_prog[0], FailedExecutionResult):
                message = f"Could not induce valid program after {max_redefine}: {solution_prog[0]}."
                return None, None, {"chat": prompt + [query_resp[0]["choices"][0]["message"]], "error": message}

        print(f"{program_str} correct?: {solution_prog is not None}")

        if not primitives:
            return None, None, {"chat": prompt + [query_resp[0]["choices"][0]["message"]], "error": "No primitives defined."}
        # Search over primitives to try to find a shorter program (if the LLM-induced program is correct)
        # or to find a program if the LLM-induced program is incorrect.
        try:
            search_results = self.restricted_dsl_solves(primitives, task)
        except Exception as e:
            search_results = None
        search_prog = str(search_results.entries[0].program) if search_results else None
        llm_prog = str(solution_prog) if solution_prog else None
        ensure_searchable = False    # TODO: make this an argument. Makes sure primitives don't have any strange loops.

        if not search_results:
            print(f"Could not induce program in DSL.")
        else:
            print(f"Induced program {search_results.entries[0].program} in DSL.")

        # If both the LLM + search induced the program, use whichever one is shorter.
        # TODO: Determine MDL using grammar instead of # tokens.
        if solution_prog and search_results:
            assert isinstance(solution_prog, Program)
            if len(search_results.entries[0].program.left_order_tokens()) < len(solution_prog.left_order_tokens()):
                solution_prog = search_results.entries[0].program
                primitives = self.get_primitives(solution_prog)

        # Couldn't find solution that solves task.
        if not (solution_prog or search_results):
            return None, None, {"chat": prompt + [query_resp[0]["choices"][0]["message"]], "search_prog": search_prog, "llm_prog": llm_prog}

        # If LLM program is correct but couldn't search to find a solution, use the LLM program.
        if not search_results and not ensure_searchable:
            assert isinstance(solution_prog, Program)
            search_results = Frontier([FrontierEntry(program=solution_prog, logLikelihood=0.0, tokens=None, logPrior=0.0, origin=None)], task=task)

        # Make sure that types are valid.
        primitives = self.make_safe(primitives)
        g = LAPSGrammar.uniform(primitives)
        try:
            g = g.insideOutside([search_results], pseudoCounts=30, iterations=1)
        except Exception as e:
            return None, None, {"chat": prompt + [query_resp[0]["choices"][0]["message"]], "error": "Invalid grammar suggested."}

        return primitives, search_results, {"chat": prompt + [query_resp[0]["choices"][0]["message"]], "search_prog": search_prog, "llm_prog": llm_prog}

    def make_safe(self, primitives):
        """Protects against some common errors in enumeration.

        This model runs untested python code. This is dangerous from a security
        standpoint and a performance standpoint. This function addresses some
        marked-by-hand performance issues.
        
        All of the errors below can be addressed by using a memory-safe and
        controlled environment to execute primitives (like multiprocessing with
        timeout). However, since we execute **a lot** of code and speed is important,
        starting a new process is not realistic for every execution. Instead, we
        use signal to run a timeout on the execution. This is not perfect, but it
        catches the vast majority of cases where the code will hang. We try to
        catch the rest here.
        """

        # TODO: Add max memory consumption to evaluation as well -- that will
        # catch a lot of the issues here.

        # Ensure that if the primitive is range, then there is a limit on the range.
        # Since range is a built-in c implementation, it cannot be interrupted
        # via signal. Alternatively, we could replace range(n) by a for loop
        # (which is interuptable). Opted for this implementation because "range"
        # could be a variety of values (e.g. range(1, n+1) or range(n)) and parsing
        # that would be difficult.
        max_range = 1000
        def make_range_val(prim):
            prim_val = deepcopy(prim.value)
            def limited_range(n):
                if n > max_range:
                    return None
                return prim_val(n)
            return limited_range
        def is_range_prim(p):
            return (
                "range(" in p.function_comment and
                p.tp == arrow(tint, tlist(tint))
            )

        for prim in primitives:
            if is_range_prim(prim):
                prim.value = make_range_val(prim)

        return primitives

    # Individual task DSL -> domain DSL strategies.

    def no_compress(self, frontiers: Dict[Task, Frontier], primitives_to_add: Dict[Task, List[Primitive]]):
        """Does not compress the frontiers. Just returns as is."""

        for prim in [p for ps in primitives_to_add.values() for p in ps]:
            Primitive.GLOBALS[prim.name] = prim

        return list(frontiers.values()), list(Primitive.GLOBALS.values())

    def merge_duplicates(self, frontiers: Dict[Task, Frontier], primitives_to_add: Dict[Task, List[Primitive]]):
        """Merges duplicate primitives to avoid naming conflicts."""

        def rename_primitive(prim: Primitive, new_name: str):
            """Renames  a primitive."""
            # assert prim.alternate_names == [prim.name]
            # assert prim.function_comment

            new_function_comment = prim.function_comment.replace(
                f"Primitive(\"{prim.name}\"", f"Primitive(\"{new_name}\"") if prim.function_comment else ""
            return Primitive(new_name, prim.tp, prim.value, function_comment=new_function_comment)

        def rewrite_prog(program: Program, prim_mapping: Dict[Primitive, Primitive]):
            """Rewrites a program to use new primitives."""

            rewritten_prog = program
            for t_old, t_new in prim_mapping.items():
                rewritten_prog = rewritten_prog.substitute(t_old, t_new)

            return rewritten_prog

        # Give all primitives unique names, rewriting their original frontiers.
        Primitive.GLOBALS = {}
        prim_to_progs = {}
        for task, prims in primitives_to_add.items():
            for prim in prims:

                # Declare the renamed primitive.
                prim_name = prim.name + "_" + task.name.replace(" ", "_")
                assert prim not in Primitive.GLOBALS, f"Duplicate primitive name: {prim}"
                renamed_prim = rename_primitive(prim, prim_name)
                assert prim_name in Primitive.GLOBALS, f"Failed to add primitive: {renamed_prim}"

                # Rewrite the frontier to use the new primitive.
                assert renamed_prim not in prim_to_progs, f"Duplicate primitive: {renamed_prim}"
                prim_to_progs[renamed_prim] = []
                for entry in frontiers[task].entries:
                    assert task.check(entry.program, timeout=5), f"Failed to solve task: {entry.program}"
                    entry.program = entry.program.substitute(prim, renamed_prim)
                    assert task.check(entry.program, timeout=5), f"Failed to solve task: {entry.program}"
                
                    prim_to_progs[renamed_prim].append((task, entry.program))

        # Get which primitive can replace which.
        # prim1 can replace prim2 if all programs that use prim2 can be solved
        # with the same program but with prim1 instead of prim2.
        prim_can_replace_prim = {}
        for prim1 in prim_to_progs:
            prim_can_replace_prim[prim1] = []
            for prim2, progs in prim_to_progs.items():
                if prim1 == prim2:
                    continue

                can_replace = True
                for (task, prog) in progs:
                    try:
                        # TODO: Allow if type is unifiable
                        if prim1.tp != prim2.tp or not task.check(prog.substitute(prim2, prim1), timeout=5):
                            can_replace = False
                            break
                    except Exception as e:
                        can_replace = False
                        break

                if can_replace:
                    prim_can_replace_prim[prim1].append(prim2)

        # Greedily find primitive merges.
        prim_to_replacement = {prim: prim for prim in Primitive.GLOBALS.values()}
        assert len(prim_to_replacement) == len(Primitive.GLOBALS)
        max_merge_steps = len(Primitive.GLOBALS)
        for _ in range(max_merge_steps):

            # Get the primitive that can replace the most other primitives.
            dominant_prim = max(prim_can_replace_prim, key=lambda p: len(prim_can_replace_prim[p]))
            if not prim_can_replace_prim[dominant_prim]:     # Nothing left to merge.
                break

            # Merge the primitives.
            for replaced_prim in prim_can_replace_prim[dominant_prim]:
                prim_to_replacement[replaced_prim] = dominant_prim
                for prim in prim_can_replace_prim:
                    if replaced_prim in prim_can_replace_prim[prim]:
                        prim_can_replace_prim[prim].remove(replaced_prim)
                del prim_can_replace_prim[replaced_prim]
            for prim in prim_can_replace_prim:
                if dominant_prim in prim_can_replace_prim[prim]:
                    prim_can_replace_prim[prim].remove(dominant_prim)

        # Re-write frontiers to use new primitives.
        rewitten_frontiers = []
        for task, task_frontier in frontiers.items():
            entries = []
            for entry in task_frontier.entries:

                rewritten_prog = rewrite_prog(entry.program, prim_to_replacement)
                new_entry = FrontierEntry(
                    program=rewritten_prog, logLikelihood=entry.logLikelihood, tokens=entry.tokens,
                    logPrior=entry.logPrior, origin=entry.origin
                )
                assert not any([p.isInvented for _, p in entry.program.walk()])
                assert task.check(new_entry.program, timeout=5)
                entries.append(new_entry)

            rewitten_frontiers.append(Frontier(entries, task=task_frontier.task))

        # Remove replaced primitives from globals.
        for prim, replacement_prim in prim_to_replacement.items():
            if prim != replacement_prim and not isinstance(replacement_prim, Invented):
                del Primitive.GLOBALS[prim.name]

        # TODO: Rename primitives to base names.

        return rewitten_frontiers, list(Primitive.GLOBALS.values())

    # Helpers.

    def extract_primitives_and_program(self, query_resp: list) -> Tuple[str, str]:
        """Gets the primitives defined and the program in the LLM response.
        
        Expects the response to be of the form:
        yapping...
        ```
        list of primitives
        ```
        more yapping...
        ```
        program
        ```
        """

        # TODO: Also try parsing by getting Primitive(...) and (lambda ...).

        # Parse the primitives + program strings.
        if len(query_resp) != 1:
            raise NotImplementedError("Haven't implemented multiple responses yet.")
        response_str = query_resp[0]["choices"][0]["message"]["content"]

        primitive_decs = get_primitive_declarations(response_str)
        if len(primitive_decs) == 0:
            raise ValueError("Coud not find primitives in the response.")
        primitives_str = "[\n\t" + ",\n\t".join(primitive_decs) + "\n]"

        lambda_decs = get_lambdas(response_str)
        if len(lambda_decs) == 0:
            raise ValueError("Could not find a program in the response.")
        elif len(lambda_decs) > 1:
            print("WARNING: Found 2 programs in the respnse, taking the last one.")
            for d in lambda_decs:
                print(d)
        
        program_str = lambda_decs[-1]
        return primitives_str, program_str

    def construct_refine_prompt(self, failed_exec_results: List[FailedExecutionResult]):
        """Constructs a prompt to refine the solution program."""

        # Check if referencing undefined variable.
        # (Probably referencing an undefined type or library).
        name_error_outputs = [o for o in failed_exec_results if isinstance(o.exception, NameError)]
        if name_error_outputs:
            first_name_error = repr(name_error_outputs[0].exception)
            # "NameError("name 'tlist' is not defined")" -> "tlist"
            try:
                referenced_name = first_name_error.split("NameError(\"name '")[1].split("' is not defined\")")[0]
            except IndexError:
                return None
            return {"role": "user", "content": (
                f"NameError: The variable '{referenced_name}' is not defined. Please rewrite your primitives and "
                "response to not reference this variable.")}

        # Check for undefined primitives.
        parse_error_outputs = [o for o in failed_exec_results if isinstance(o.exception, ParseFailure)]
        if parse_error_outputs:
            first_parse_error = repr(parse_error_outputs[0].exception)
            # "ParseFailure((['lambda', ['if', ['and', ['is_alpha', '$0'], ['starts_with_consonant', '$0']], ['concat',
            # ['concat', 'je', ['substring', '$0', '1']], 'w'], '$0']], 'and'))" -> "and"
            undefined_prim_name = first_parse_error.split("'")[-2]

            return {"role": "user", "content": (
                f"\"{undefined_prim_name}\" is not defined in your DSL yet. Please define it. Return the new primitive "
                f"\"{undefined_prim_name}\" AND re-write your solution.")}

        return None

    def check_solution(self, primitives_strs: List[str], program_str: str, task, prim_types: List[str]):
        """Checks if the solution can be induced in the DSL.

        Args:
            primitives_str (list[str]): A list of strings in the format of a list of primitives.
            program_str (str): A string representing the program.

        Returns:
            A 2-tuple:
                Either the solution program in the DSL (solution works), None (if
                    solution executes but is incorrect), or a list of exceptions caused
                    by trying to execute the program.
                The list of primitives in the program (or the defined primitives if the
                    program executed successfully but is incorrect).
        """

        prev_globs = deepcopy(Primitive.GLOBALS)
        for p in prev_globs:
            if Primitive.GLOBALS[p].function_comment == "":
                import pdb; pdb.set_trace()
                del Primitive.GLOBALS[p]

        # Create a string with primitive declarations + program parsing to be
        # executed in contained environment.
        def to_f_string(primitives, prog_str):
            f_lines = ["def fn(inp):"]
            prim_str = "_ = " + str(primitives.strip().replace("\n", "\n\t")) + "\n"
            f_lines.append(prim_str)
            f_lines.append("p = Program.parse(\"\"\"" + prog_str + "\"\"\")")

            f_lines.append("init_ = p.evaluate([])")
            f_lines.append("return init_(inp)")

            f_lines = [f_lines[0] ]+ ["\t" + line for line in f_lines[1:]]
            return "\n".join(f_lines)

        def join_prim_strings(p_strs: List[str], prog_str):

            # Find all instances of "Primitive(...)" in each string.
            prim_decs = [get_primitive_declarations(p_str) for p_str in p_strs]
            unique_prim_decs = []
            prim_names = set()

            # Add all unique primitive declarations, giving later-defined primitives
            # higher priority. Only add primitives in the program.
            for prim_dec in reversed(prim_decs):
                for dec in prim_dec:
                    prim_name = dec.split("Primitive(\"")[1].split("\",")[0]
                    if prim_name in prim_names or prim_name not in prog_str:
                        continue
                    prim_names.add(prim_name)
                    unique_prim_decs.append(dec)

            return "[\n\t" + ",\n\t".join(unique_prim_decs) + "\n]"

        primitives_str = join_prim_strings(primitives_strs, program_str)

        # Create an environment w/ the defined types.
        str_to_type = {
            "arrow": arrow, "tint": tint, "tbool": tbool, "baseType": baseType, "tstr": tstr, "tlist": tlist}
        globs = {"Program": Program, "Primitive": Primitive, "t0": t0, "t1": t1, "t2": t2}
        for prim_type in prim_types:
            globs[prim_type] = str_to_type.get(prim_type, baseType(prim_type))  # If not a base type, create it.

        str_repr = to_f_string(primitives_str, program_str)
        # TODO: Better format inputs for execution.
        inputs = [f"\"{ex_in[0]}\"" if isinstance(ex_in[0], str) else str(ex_in[0]) for ex_in, ex_out in task.examples]

        # Execute the program + check if it works.
        # Must ignore cache to execute Primitives.
        # TODO: ^^ sketchy.
        outputs = execute_function(str_repr, inputs, globals=globs, ignore_cache=True)

        print("globals:", Primitive.GLOBALS)
        print(f"Primitives: {primitives_str}")
        print(f"Program: {program_str}")
        for ex, pred_out in zip(task.examples, outputs):
            print(f"Example: {ex[0]} -> {ex[1]}")
            print(f"Predicted output: {pred_out}")
            print()

        # Attach primitive declarations to object.
        # TODO: This is hacky.
        all_primitives = []
        for prim_line in get_primitive_declarations(primitives_str):
            prim_name = prim_line.split("Primitive(\"")[1].split("\",")[0]
            if prim_name in Primitive.GLOBALS:
                fc = prim_line.strip()
                if fc.endswith(","):
                    fc = fc[:-1]
                Primitive.GLOBALS[prim_name].function_comment = fc
                all_primitives.append(Primitive.GLOBALS[prim_name])
            else:
                assert all([o for o in outputs if isinstance(o, FailedExecutionResult)])

        for p in Primitive.GLOBALS.values():
            self.all_defined_prims.setdefault(p.name, []).append(p)

        # If the program raises the exception of a primitve not being defined, try to define it by using definitions
        # defined in other API calls.
        # This is a heuristic / cheat to avoid more API calls.
        has_undefined_prim = any([o for o in outputs if isinstance(o, FailedExecutionResult) and isinstance(o.exception, ParseFailure)])
        if has_undefined_prim:
            max_retries = 5
            new_outputs = outputs.copy()
            prev_globs = deepcopy(Primitive.GLOBALS)
            now_solves = False
            for i in range(max_retries):
                parse_error_outputs = [o for o in new_outputs if isinstance(o, FailedExecutionResult) and isinstance(o.exception, ParseFailure)]
                if not parse_error_outputs:
                    break

                # Get the name of the undefined primitive.
                first_parse_error = repr(parse_error_outputs[0].exception)
                undefined_prim_name = first_parse_error.split("'")[-2]

                # Try to define the primitive.
                if undefined_prim_name not in self.all_defined_prims:
                    break
                for prim_def in self.all_defined_prims[undefined_prim_name]:
                    primitives_strs_ = primitives_strs + [prim_def.function_comment]
                    primitives_str_ = join_prim_strings(primitives_strs_, program_str)
                    str_repr_ = to_f_string(primitives_str_, program_str)
                    new_outputs = execute_function(str_repr_, inputs, globals=globs, ignore_cache=True)

            if all([out == ex_out for out, (_, ex_out) in zip(outputs, task.examples)]):    # TODO: Implement using the built-in task check.
                now_solves = True

            # If the program does not solve the task, return to previous state.
            if not now_solves:
                Primitive.GLOBALS = prev_globs

        # If the program raises any exceptions, return those exceptions.
        failed_outputs = [o for o in outputs if isinstance(o, FailedExecutionResult)]
        if failed_outputs:
            Primitive.GLOBALS = prev_globs
            return failed_outputs, None
        
        if any(p.function_comment == "" for p in Primitive.GLOBALS.values()):
            import pdb; pdb.set_trace()

        # If the program solves the task, return that program and the primitives
        # used in the program.
        try:
            prog = Program.parse(program_str)
        except:
            import pdb; pdb.set_trace()
        prog_primitives = self.get_primitives(prog)
        if all([out == ex_out for out, (_, ex_out) in zip(outputs, task.examples)]):    # TODO: Implement using the built-in task check.
            return prog, prog_primitives

        # If the program does not solve the task, return all the primitives defined.

        return None, all_primitives

    def get_primitives(self, program: Program) -> List[Primitive]:
        """Gets the list of primitives in a program."""

        # Get unique primitives, maintaining order.
        prim_names = program.left_order_tokens()
        unique_prim_names, seen = [], set()
        for name in prim_names:
            if name not in seen:
                unique_prim_names.append(name)
                seen.add(name)
        prims = [Primitive.GLOBALS[name] for name in unique_prim_names]
        return prims

    def restricted_dsl_solves(self, primitives, task, max_timeout: float = 3.):
        """Checks if the primitives can solve the task by running enumerative search."""

        grammar = LAPSGrammar.uniform(primitives)
        # task.request = arrow(tstr, tstr)

        new_frontiers, _ = multicoreEnumeration(
            g=grammar,
            tasks=[task],
            maximumFrontier=1,
            enumerationTimeout=max_timeout,
            CPUs=1,
            solver="python",
            no_candidates_okay=True,
            evaluationTimeout=5,
        )
        return new_frontiers[0]

    def construct_initial_prompt(
            self,
            domain_description: str,
            problem_description: str,
            examples: list,
            python_code: str,
            existing_primitive_functions: list,
            existing_primitive_constants: list,
            existing_types: str):
        """Creates prompt to define new primitives + suggest solution program."""

        # TODO: Make some of the prompt details parameters (e.g. # few-shot examples).

        messages = [{"role": "system", "content": (
            "You are an expert programmer working in a language based on lambda calculus. "
            "Your goal is to convert python programs into programs in a DSL.")}]
        messages += initial_conversion_2shot

        # Remove print statements + comments (outside of function) from python code.
        # TODO: Do this when originally parsing response instead of here.
        if python_code:
            python_code_lines = []
            for line in python_code.split("\n"):
                if "print(" not in line and not line.startswith("#"):
                    python_code_lines.append(line)
            python_code = "\n".join(python_code_lines).strip()
        else:
            python_code = "(None)"

        def format_examples(examples):
            """Convert tuple of examples to string."""
            exs_string = ""
            for (inp, out) in examples:
                exs_string += f'"{inp[0]}" -> "{out}"\n'
            return exs_string
        
        def format_primitives(primitives):
            """Convert list of primitives to string."""
            if not primitives:
                return "(None)"
            return "\n".join([p.function_comment for p in primitives])
            

        request = DECLARE_PRIMITIVES_INIT_PROGRAM.format(
            domain_description=domain_description,
            problem_description=problem_description,
            examples=format_examples(examples),
            python_code=python_code,
            existing_primitive_functions=format_primitives(existing_primitive_functions),
            existing_primitive_constants=format_primitives(existing_primitive_constants),
            existing_types=existing_types
        )

        messages += [{"role": "user", "content": request}]
        return messages
