"""
gpt_python_to_dsl.py | Author: Sam Acquaviva.

Queries GPT to improve a DSL via decomposition, expansion, and compression.
"""

from copy import deepcopy
import os
import json

from dreamcoder.task import Task
from dreamcoder.type import Context
from dreamcoder.grammar import NoCandidates
from dreamcoder.program import Program, Primitive
from dreamcoder.enumeration import multicoreEnumeration
import src.models.model_loaders as model_loaders
from src.models.sample_generator import GPTSampleGenerator
from src.models.laps_grammar import LAPSGrammar


ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.DSL_IMPROVER]

def get_primitives(program):
    """Returns the set of primitives in the program."""
    primitives = set()
    for _, subprog in program.walk():
        if isinstance(subprog, Primitive):
            primitives.add(subprog)

    return primitives

@ModelRegistry.register
class GPTDSLImprover(GPTSampleGenerator):    # type: ignore
    """Solves tasks in Python."""

    name = "dsl_improver"
    results_file = "dsl_improver_results.json"

    @staticmethod
    def load_model(experiment_state, engine=None, **kwargs):
        return GPTDSLImprover(experiment_state=experiment_state, engine=engine)  # type: ignore

    def __init__(self, experiment_state=None, engine=None):
        super().__init__(self, engine=engine)
    
    def decompose_primitive(self, primitive, frontiers_with_primitive):
        """Decomposes a primitive into smaller primitives."""
        
        # Prompt LLM to generate decomposition.

        # Check if the decomposition still solves all tasks that involve the previous primitive.

        # Merge the new primitives into the DSL.
        pass

    def suggest_new_primitives(self):
        """Suggests new primitives to add to the DSL based on solution programs / existing library."""
        pass

    def _prim_to_tasks(self, experiment_state, task_split: str, task_batch_ids: list):
        """Returns a dictionary mapping each primitive to the (task/program index) it is used in."""
        task_frontiers = experiment_state.get_frontiers_for_ids(task_split, task_batch_ids)
        prim_to_tasks = {}
        for frontier in task_frontiers:
            for i, entry in enumerate(frontier.entries):
                # Get program in beta normal form to remove the invented primitives.
                for prim in get_primitives(entry.program):
                    prim_to_tasks.setdefault(prim, set()).add((frontier.task, i))

        return prim_to_tasks
    
    def make_replacement_tasks(
            self, task_prog_pairs, experiment_state, task_split, primitive_to_replace):
        """Creates ReplacementTasks for each task/program pair."""
        replacement_tasks = []
        for task, program_i in task_prog_pairs:
            task_frontier = experiment_state.task_frontiers[task_split][task]
            program = task_frontier.entries[program_i].program
            replacement_tasks.append(ReplacementTask(program, primitive_to_replace, task, program_i))
        return replacement_tasks

    def remove_redundant_primitives(
            self, experiment_state, task_split: str, task_batch_ids: list,
            direct_replace_timeout: int = 1, composition_replace_timeout: int = 3,
            verbose: bool = True, evaluationTimeout: int = 5):
        """Removes primitives that can be replaced (functionally) by other primitives.
        
        Attempts to solve search tasks where the goal is to replace a primitive
        with another primitive. For example, say there is a primitive `concat`
        which is implemented as `lambda a: lambda b: a+b` and we find 2
        solutions using this primitive during search: `(lambda (concat $0 a))` 
        for task A and `(lambda (concat $0 $0))` for task B. In this step, we
        are enumerating over programs which will be marked as “correct” if they
        solve the task when substituted for concat. For example, if we have
        another primitive `append` which has the implementation 
        `lambda a: lambda b: b+a`  (the same but reversed arguments), then
        replacing concatenate with `(lambda (lambda append $0 $1))` in both
        programs will solve the task. If all of these “replacement tasks” can
        be solved for a given primitive, we replace the primitive with its
        substitutions in the solution programs and remove it from the library.

        We follow two strategies:
            First, we try to find a direct replacement for the primitive. So,
            for each primitive, we check if another primitive can be used to
            replace it in all tasks, considering permutations of the arguments.

            If we can't find a direct replacement, we enumerate over all the
            primitives to see if a composition of other primitives can replace
            the primitive in all tasks.

        TODO: Integrate with enumerateHoles instead of the ad-hoc ReplacementTask.
        TODO: Remove primitives if another program without the primitive can solve the task.
            This avoids having to do extra search and can find places where there isn't an exact
            drop-in replacement but a different program that solves the task.

        Args:
            experiment_state (ExperimentState): The current experiment state.
            task_split (str): The split to use for the tasks ("train"/"test").
            task_batch_ids (list[str]): The list of task ids to use.
            direct_replace_timeout (int): The timeout for searching for a direct
                replacement for each primitive, in seconds.
            composition_replace_timeout (int): The timeout for searching for a
                replacement using compositions of other primitives, in seconds.
            verbose (bool): Whether to print output.
            evaluationTimeout (int): The timeout for evaluating each program.
        """

        # Remove all invented primitives, which would cause issues with the
        # current implementation. Also, inventions are (by design) compositions
        # of other primitives so including them here would be unnecessary.
        experiment_state.convert_frontiers_beta_normal()

        prev_frontiers = deepcopy(experiment_state.get_frontiers_for_ids(task_split, task_batch_ids))
        prim_to_tasks = self._prim_to_tasks(experiment_state, task_split, task_batch_ids)
        all_prims = [p for _, _, p in experiment_state.models[model_loaders.GRAMMAR].productions if not p.isInvented]
        cur_prims = [p for _, _, p in experiment_state.models[model_loaders.GRAMMAR].productions if not p.isInvented]

        # Loop through each primitive from primitive in least number of tasks to most number of tasks.
        prim_to_replacable, prim_to_unreplacable, prims_translated_direct = {}, {}, set()
        sorted_primitives = sorted(prim_to_tasks.keys(), key=lambda prim: len(prim_to_tasks[prim]))
        for i, primitive in enumerate(sorted_primitives):
            if verbose:
                print(f"Checking primitive {primitive} ({i+1}/{len(prim_to_tasks)})")
            task_prog_pairs = prim_to_tasks[primitive]

            # Make tasks where the goal is to fill in the hole left by removing the primitive.
            replacement_tasks = self.make_replacement_tasks(
                task_prog_pairs, experiment_state, task_split, primitive)

            # First, try to replace each primitive with another primitive w/ an argument permutation.
            try_composition_search = True
            for replacement_prim in [p for p in cur_prims if p != primitive]:
                replacement_frontiers, _ = multicoreEnumeration(
                    LAPSGrammar.uniform(primitives=[replacement_prim]), replacement_tasks,
                    enumerationTimeout=direct_replace_timeout,
                    solver="python", maximumFrontier=1, evaluationTimeout=evaluationTimeout,
                    no_candidates_okay=True, maxMDL=15)

                if sum([1 for rf in replacement_frontiers if rf.entries]) == len(replacement_tasks):
                    try_composition_search = False
                    prims_translated_direct.add(primitive)
                    break

            # If searching over single primitives fails, then enumerate over compositions of multiple
            # primitives to see if the primitive can be removed.
            if try_composition_search:
                grammar_without_primitive = LAPSGrammar.uniform(primitives=[p for p in cur_prims if p != primitive])
                replacement_frontiers, _ = multicoreEnumeration(
                    grammar_without_primitive, replacement_tasks, enumerationTimeout=composition_replace_timeout,
                    solver="python", maximumFrontier=1, evaluationTimeout=evaluationTimeout)

            # If the primitive can be replaced in every task, remove it from the DSL
            # and update prim->task mapping using the subprogram that replaced the primitive.
            n_replaced = sum([1 for f in replacement_frontiers if f.entries])
            if verbose:
                print(f"Fraction of tasks replaced {primitive}: {n_replaced / len(replacement_tasks):.2f}")

            if n_replaced == len(replacement_tasks):

                cur_prims.remove(primitive)
                del Primitive.GLOBALS[primitive.name]

                # Replace the primitive with the replacement sub-program.
                for replacement_task, frontier in zip(replacement_tasks, replacement_frontiers):
                    task = replacement_task.original_task
                    assert len(frontier.entries) == 1
                    prim_replacement = frontier.entries[0].program

                    entry = experiment_state.task_frontiers[task_split][task].entries[replacement_task.program_index]
                    entry.program = entry.program.substitute(primitive, prim_replacement)

                    # Make sure in beta normal form.
                    # Example:
                    # If Primitive("prepend", arrow(tint, tlist(tint), tlist(tint)), lambda x: lambda l: [x] + l)
                    # is replaced with (lambda (lambda (concat (int->list $1) $0))) in the program
                    # (lambda (int->list (get_item (prepend 10 $0) 7))), the new program should be
                    # (lambda (int->list (get_item (concat (int->list 10) $0) 7))), not
                    # (lambda (int->list (get_item ((lambda (lambda (concat (int->list $1) $0))) 10 $0) 7)))
                    entry.program = entry.program.betaNormalForm()
                    assert task.check(entry.program)

                    # Update prim->task mapping.
                    for prim in get_primitives(prim_replacement):
                        prim_to_tasks.setdefault(prim, set()).add((task, replacement_task.program_index))

                # Make sure can run inside/outside. This is only to help catch bugs.
                try:
                    _ = LAPSGrammar.uniform(primitives=cur_prims).insideOutside(
                        [experiment_state.task_frontiers[task_split][task]], pseudoCounts=30, iterations=1)
                except Exception as e:
                    import pdb; pdb.set_trace()

            # Save logs.
            for replacement_task, frontier in zip(replacement_tasks, replacement_frontiers):
                if frontier.entries:
                    prim_replacement = frontier.entries[0].program
                    program = replacement_task.program
                    program = program.substitute(primitive, prim_replacement).betaNormalForm()
                    prim_to_replacable.setdefault(primitive, []).append({
                        "task": replacement_task.original_task.name,
                        "old_prog": str(replacement_task.program),
                        "replacement": str(prim_replacement),
                        "new_prog": str(entry.program)})
                else:
                    prim_to_unreplacable.setdefault(primitive, []).append({
                        "task": replacement_task.original_task.name,
                        "old_prog": str(replacement_task.program)})

        # Ensure that the grammar is still legitimate.
        # 1. Every program solves the task.
        # 2. Every program can be parsed from string.
        # 3. Can run inside outside.
        task_frontiers = experiment_state.get_frontiers_for_ids(task_split, task_batch_ids)
        for frontier in task_frontiers:
            for entry in frontier.entries:
                assert frontier.task.check(entry.program)
                assert frontier.task.check(Program.parse(str(entry.program)))
        g = LAPSGrammar.uniform(primitives=cur_prims)
        try:
            g = g.insideOutside([t for t in task_frontiers if t.entries], pseudoCounts=30, iterations=1)
        except Exception as e:
            import pdb; pdb.set_trace()
        experiment_state.models[model_loaders.GRAMMAR] = LAPSGrammar.fromGrammar(g)

        # Save results to file.
        old_prog_to_new_prog = {}
        for task_id in task_batch_ids:
            old_prog_to_new_prog[task_id] = []
            task_frontier_prev = [f for f in prev_frontiers if f.task.name == task_id][0]
            task_frontier = [f for f in task_frontiers if f.task.name == task_id][0]
            for prev_entry, rewritten_entry in zip(task_frontier_prev.entries, task_frontier.entries):
                old_prog_to_new_prog[task_id].append({
                    "old_program": str(prev_entry.program),
                    "new_program": str(rewritten_entry.program)
                })

        prim_to_res = {}
        for prim in all_prims:
            replacements = prim_to_replacable.get(prim, None)
            non_replacements = prim_to_unreplacable.get(prim, None)

            prim_to_res[prim.name] = {
                "replacements": replacements,
                "non_replacements": non_replacements,
                "replaced_1-to-1": prim in prims_translated_direct,
            }

        results = {
            "params": {
                "enumeration_timeout": composition_replace_timeout,
            },
            "summary": {
                "n_previous_primitives": len(all_prims),
                "n_current_primitives": len(cur_prims),
                "primitives_removed": [p.name for p in all_prims if p not in cur_prims],
                "primitives_kept": [p.name for p in cur_prims],
            },
            "results_by_primitive": prim_to_res,
            "results_by_task": old_prog_to_new_prog,
        }

        results_filepath = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            task_split,
            self.results_file,
        )
        os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
        with open(results_filepath, "w") as frontier:
            json.dump(results, frontier, indent=4)
        if verbose:
            print(f"Wrote results: {results_filepath}")


class ReplacementTask(Task):
    """Task where the problem is not to solve the task, but to replace a primitive in a program that solves the task.
    
    So the type of the task is not the type of the task, but the type of the primitive to replace.
    Example:
        task = add "ij" to the end of the string
        original program: (lambda (append $0 ij))`
    """

    def __init__(self, program: Program, primitive_to_replace: Primitive, task: Task, program_index: int):
        self.original_task = task
        request = primitive_to_replace.tp
        super().__init__(
            task.name + f" ({program} - {primitive_to_replace})", request, task.examples,
            task.features, task.cache, task.desc)
        self.program = program
        self.primitive_to_replace = primitive_to_replace
        self.name += f" ({self.program} - {self.primitive_to_replace})"
        self.program_index = program_index

    def __repr__(self):
        return "ReplacementTask(name={self.name}, request={self.request}, examples={self.examples}".format(
            self=self
        )

    def check(self, e, timeout=None, verbose=False):
        """Check if the program `e` solves the task."""

        replaced_prog = self.program.substitute(self.primitive_to_replace, e)

        # Check to see if the program has the correct type.
        # Without this, there can be issues with the resuling program.
        # For example, if the program is (lambda (int->list (get-item [10] 0))),
        # and the primitive to replace is 10, and we find the program
        # (lambda (int->list const_list_10)) where const_list_10 = [10],
        # then the program will evaluate to the correct solution / solve the
        # task, but int->list accept the wrong type. We find this solution
        # in the first place because get-item returns a type variable so if we
        # try to replace it we are trying to replace something with any type.
        # TODO: Infer type for primitives based on context before search.
        can_have_type = replaced_prog.canHaveType(self.original_task.request)
        if not can_have_type:
            return False

        res = super().check(replaced_prog, timeout, verbose=verbose)
        return res
