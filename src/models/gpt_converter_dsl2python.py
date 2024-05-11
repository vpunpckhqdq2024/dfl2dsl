"""
gpt_python_to_dsl.py | Author: Sam Acquaviva.

Queries GPT to convert solutions in DSL to solutions in python.
"""

from copy import deepcopy

import src.models.model_loaders as model_loaders
from src.models.gpt_base import GPTBase, DSL2PythonPrompt
from src.models.gpt_solver_python import GPTSolverPython
from dreamcoder.frontier import Frontier, FrontierEntry

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LLM_CONVERTER_DSL_PYTHON]
DEFAULT_LINE_SEPARATOR = "\n"

@ModelRegistry.register
class GPTConverterDSL2Python(GPTSolverPython):    # type: ignore
    """Converts solutions in the DSL to solutions in python."""

    name = "gpt_converter_dsl2python"
    results_file = "gpt_converter_dsl2python_results.json"

    @staticmethod
    def load_model(experiment_state, engine=None, **kwargs):
        return GPTConverterDSL2Python(experiment_state=experiment_state, engine=engine)  # type: ignore

    def __init__(self, experiment_state=None, engine=None):
        super().__init__(self, engine=engine)
        self.line_separator = DEFAULT_LINE_SEPARATOR

    def convert_frontiers_to_python(self, experiment_state, task_split, task_batch_ids, max_tokens: int = 512):
        """Converts frontiers to python."""

        # Make prompt to convert to python.
        prompts = []
        tasks = []
        for task_id in task_batch_ids:

            task = experiment_state.get_tasks_for_ids(task_split, [task_id])[0]
            frontier = experiment_state.task_frontiers[task_split].get(task, [])
            python_frontier = experiment_state.task_frontiers_python[task_split].get(task, [])
            if (not frontier) or python_frontier:     # If no solution (or already converted), then there is nothing to convert.
                continue

            prompts.append(DSL2PythonPrompt(
                program=frontier.entries[0].program,
                experiment_state=experiment_state,
                final_task_id=task_id,
                final_task_split=task_split,))
            tasks.append(task)

        # Convert to python / check solutions.
        responses = GPTBase.query_batch(
            prompts,
            model_name=self.ENGINE,
            n_samples=1,
            max_tokens=max_tokens,
            temperature=0.0,
            stop_token=None,
        )

        python_solutions = {}
        for response, task in zip(responses, tasks):
            parse_results = self.parse_code_response(
                response, experiment_state, task_split, [task.name], evaluate_samples=True, verbose=True)
            assert len(parse_results) == 1
            parse_results = parse_results[0]
            if not parse_results["tasks_solved"]:
                continue    # Failed to convert.

            python_solutions[task] = parse_results["program"]

        # Update frontiers.
        for task, python_solution in python_solutions.items():
            assert not experiment_state.task_frontiers_python[task_split].get(task)
            experiment_state.task_frontiers_python[task_split][task] = Frontier(
                frontier=[FrontierEntry(
                    program=python_solution, logPrior=0.0, logLikelihood=0.0,
                    origin=f"gpt_converter_dsl2python {task.name}", tokens=[])], task=task)
