"""
gpt_solver_python.py | Author: Sam Acquaviva.

Queries GPT to solve tasks in Python.
"""

import json
import os
from collections import defaultdict
import re

import src.models.model_loaders as model_loaders
from src.models.gpt_base import DEFAULT_LINE_SEPARATOR, Task2PythonPrompt, Task2NLPrompt, GPTBase, Prompt
from src.models.laps_grammar import LAPSGrammar
from src.models.sample_generator import GPTSampleGenerator
from src.task_loaders import LANGUAGE, PROGRAMS, TRAIN
from src.python_utils import execute_function, extract_program
from src.experiment_iterator import SKIPPED_MODEL_FN, ExperimentState
from dreamcoder.frontier import Frontier, FrontierEntry

TASK2PYTHON = "task2python"
TASK2NL2PYTHON = "task2nl2python"
ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LLM_SOLVER_PYTHON]


@ModelRegistry.register
class GPTSolverPython(GPTSampleGenerator):
    """Solves tasks in Python."""

    name = "gpt_solver_python"
    results_file = "gpt_solver_python_results.json"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return GPTSolverPython(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None, engine=None):
        super().__init__(self, engine=engine)

    def infer_programs_for_tasks(
        self,
        experiment_state,
        task_split: str,
        task_batch_ids: list,
        strategy: str = TASK2PYTHON,
        # Sampling
        n_samples_per_query: int = None,
        n_queries_per_task: int = None,     # To allow early stopping.
        early_stop_on_solution: bool = True,
        # Prompt construction
        body_task_selection: str = "random",
        body_task_types: list = [LANGUAGE, PROGRAMS],
        final_task_types: list = [LANGUAGE],
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        n_exs: int = 3,
        max_few_shot: int = 100,
        apply_fewshot_progs_to_exs: bool = True,
        example_selection_strategy: str = "first",
        # GPT parameters
        temperature: float = 0.40,
        max_tokens_completion_beta: float = 2.0,
        max_tokens: int = None,
        # Resume from prior runs
        resume_strategy: str = None,
        # Utilities
        verbose: bool = False,
        n_tasks = None
    ):

        if (resume_strategy == "first" and experiment_state.is_first_iteration()) or (
            resume_strategy == "every"
        ):
            # If RESUME_CHECKPOINT_DIRECTORY not defined, default to self checkpoint directory
            results_filepath_ext = os.path.join(
                os.getcwd(),
                experiment_state.get_checkpoint_directory_maybe_resume(),
                task_split,
                self.results_file,
            )
            if os.path.exists(results_filepath_ext):
                with open(results_filepath_ext, "r") as f:
                    results_json = json.load(f)

                # Update experiment state from file
                self.add_python_samples_to_experiment_state(
                    experiment_state=experiment_state,
                    task_split=task_split,
                    parse_results_valid=results_json["parse_results_valid"],
                )

                # Copy external results file to checkpoint directory
                results_filepath = os.path.join(
                    os.getcwd(),
                    experiment_state.get_checkpoint_directory(),
                    task_split,
                    self.results_file,
                )
                os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
                with open(results_filepath, "w") as f:
                    json.dump(results_json, f, indent=4)

                print(f"Loaded GPT results from: {results_filepath_ext}")
                return {
                    SKIPPED_MODEL_FN: True,
                }
            else:
                print(f"GPT results not found at: {results_filepath_ext}")
                if experiment_state.is_first_iteration() and task_split == TRAIN:
                    raise ValueError("Unable to resume first iteration.")

        task_to_solutions = defaultdict(list)
        results_by_query = []
        parse_results_valid = []

        # Ignore already solved tasks that we tried to solve last round.
        task_batch_ids = list(task_batch_ids)
        if n_tasks:
            task_batch_ids = task_batch_ids[n_tasks*experiment_state.curr_iteration:n_tasks*(experiment_state.curr_iteration+1)]
        for f in experiment_state.get_non_empty_python_frontiers_for_split(task_split):
            if f.task.name in task_batch_ids:
                task_batch_ids.remove(f.task.name)

        strategy_to_proc = {
            TASK2PYTHON: self.infer_programs_python,
            TASK2NL2PYTHON: self.infer_programs_nl2python,
        }
        strategy_to_prompt = {
            TASK2PYTHON: Task2PythonPrompt,
            TASK2NL2PYTHON: None,
        }

        # Run prompts as batches.
        if n_queries_per_task > 1:
            raise NotImplementedError

        prompts_, completions_, parse_results_ = strategy_to_proc[strategy](
            experiment_state=experiment_state,
            task_split=task_split,
            task_batch_ids=task_batch_ids,
            line_separator=line_separator,
            n_exs=n_exs,
            n_samples_per_query=n_samples_per_query,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            final_task_types=final_task_types,
            max_few_shot=max_few_shot,
            apply_fewshot_progs_to_exs=apply_fewshot_progs_to_exs,
            example_selection_strategy=example_selection_strategy,
        )

        for task_i, task_id in enumerate(task_batch_ids):
            for query_i in range(n_queries_per_task):
                prompts_sequence = [p[task_i] for p in prompts_]
                completions_sequence = [c[task_i] for c in completions_]
                parse_results = [p[task_i] for p in parse_results_]

                query_res = []
                for prompt, completion, parse_result in zip(prompts_sequence, completions_sequence, parse_results):
                    query_res.append(
                        {
                            "prompt": prompt.to_dict() if isinstance(prompt, Prompt) else [p.to_dict() for p in prompt],
                            "completion": [c.to_dict_recursive() for c in completion] if isinstance(completion, list) else completion.to_dict_recursive(),
                            "parse_results": parse_result,
                        })

                results_by_query.append(
                    {
                        "task_id": task_id,
                        "query_i": query_i,
                        "results": query_res,
                    }
                )

                # If pipeline w/ multiple steps, we only care about the end step.
                parse_results = parse_results[-1]

                task_solved = False
                for result_datas in parse_results:
                    for result_data in (result_datas if isinstance(result_datas, list) else [result_datas]):
                        result_data["query_i"] = query_i

                        if result_data["valid"]:
                            parse_results_valid.append(result_data)

                        if result_data.get("tasks_solved"):
                            # Sanity check
                            assert len(result_data["tasks_solved"]) == 1
                            assert result_data["tasks_solved"][0] == task_id
                            task_to_solutions[task_id].append(result_data)
                            task_solved = True

                # Print query results
                if verbose:
                    print("-" * 12)
                    for i, prompt in enumerate(prompts_sequence):
                        if len(prompts_sequence) > 1: print(f"Step {i+1}/{len(prompts_sequence)}")
                        prompt = [prompt] if isinstance(prompt, Prompt) else prompt
                        for j, p in enumerate(prompt):
                            if len(prompt) > 1: print(f"Sample {j+1}/{len(prompt)}")
                            print(p)
                        print("-" * 12)

                    print(f"GPT ({self.ENGINE}) completions:")
                    for result_datas in parse_results:
                        for result_data in (result_datas if isinstance(result_datas, list) else [result_datas]):
                            if result_data.get("tasks_solved"):
                                status_emoji = "🏆"
                            elif result_data["valid"]:
                                status_emoji = "❎"
                            else:
                                status_emoji = "❌"
                            print(f"{status_emoji} {result_data['text']}")
                    print("")

                print(
                    f"[TASK {task_i+1}/{len(task_batch_ids)} QUERY {query_i}/{n_queries_per_task}]: {task_id}",
                    flush=True,
                )

                n_tasks_solved = len(
                    [
                        t
                        for t, results in task_to_solutions.items()
                        if len(results) > 0
                    ]
                )
                print(
                    f"Tasks solved so far: {n_tasks_solved}/{task_i+1}", flush=True
                )

                if task_solved and early_stop_on_solution:
                    break

            tasks_solved = [
                t for t, results in task_to_solutions.items() if len(results) > 0
            ]

            # Collect results
            results = {
                "params": {
                    "n_samples_per_query": n_samples_per_query,
                    "n_queries_per_task": n_queries_per_task,
                    "temperature": temperature,
                    "engine": self.ENGINE,
                    "line_separator": line_separator,
                    "body_task_types": body_task_types,
                    "final_task_types": final_task_types,
                    "function_name_classes": function_name_classes,
                },
                "summary": {
                    "n_tasks_solved": len(tasks_solved),
                    "tasks_solved": list(tasks_solved),
                },
                "task_to_solutions": task_to_solutions,
                "parse_results_valid": parse_results_valid,
                "results_by_query": results_by_query,
            }

            # Save results to file
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

        # Update experiment_state
        self.add_python_samples_to_experiment_state(
            experiment_state=experiment_state,
            task_split=task_split,
            parse_results_valid=parse_results_valid,
        )

    def infer_programs_python(
            self, experiment_state, task_split, task_batch_ids, line_separator, n_exs, n_samples_per_query, temperature,
            max_tokens, verbose, final_task_types, max_few_shot, apply_fewshot_progs_to_exs, example_selection_strategy,
            few_shot_representation=("ex", "code")):
        """Infers programs for a task by converting directly to python."""

        prompts = []
        for task_id in task_batch_ids:
            prompt = Task2PythonPrompt(
                experiment_state=experiment_state,
                final_task_id=task_id,
                final_task_split=task_split,
                line_separator=line_separator,
                n_exs=n_exs,
                final_task_types=final_task_types,
                max_few_shot=max_few_shot,
                apply_fewshot_progs_to_exs=apply_fewshot_progs_to_exs,
                example_selection_strategy=example_selection_strategy,
                few_shot_representation=few_shot_representation,
            )
            prompts.append(prompt)

        completions = GPTBase.query_batch(
            prompts,
            model_name=self.ENGINE,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_token=None,
            n_samples=n_samples_per_query)

        parse_results = []
        for i, completion in enumerate(completions):
            parse_results.append(self.parse_code_response(
                completion,
                experiment_state=experiment_state,
                task_split=task_split,
                task_ids=[task_batch_ids[i]],
                evaluate_samples=True,
                verbose=verbose,
            ))

        return [prompts], [completions], [parse_results]
    
    def infer_programs_nl2python(
            self, experiment_state, task_split, task_batch_ids, line_separator, n_exs, n_samples_per_query, temperature,
            max_tokens, verbose, final_task_types, max_few_shot, apply_fewshot_progs_to_exs, example_selection_strategy):
        """Infers programs for a task by converting to natural language then to python."""

        # Step 1: Convert to natural language.
        task2nl_prompts = []
        for task_id in task_batch_ids:
            prompt = Task2NLPrompt(
                experiment_state=experiment_state,
                final_task_id=task_id,
                final_task_split=task_split,
                line_separator=line_separator,
                n_exs=n_exs,
                final_task_types=final_task_types,
                max_few_shot=max_few_shot,
                apply_fewshot_progs_to_exs=apply_fewshot_progs_to_exs,
                example_selection_strategy=example_selection_strategy,
            )
            task2nl_prompts.append(prompt)
        task2nl_completions = GPTBase.query_batch(
            task2nl_prompts,
            model_name=self.ENGINE,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_token=None,
            n_samples=n_samples_per_query)

        task2nl_parses = [self.parse_nl_response(completion) for completion in task2nl_completions]
        assert len(task_batch_ids) == len(task2nl_parses)

        # Step 2: Add language to experiment state.
        prompt_pairs = []
        for task_id, rules in zip(task_batch_ids, task2nl_parses):
            rules_to_add = [rule for rule in rules if rule not in experiment_state.task_language[task_split][task_id]]
            experiment_state.task_language[task_split][task_id].extend(rules_to_add)
            prompt_pairs.extend([(task_id, rule) for rule in rules])

        # Step 3: Convert to python.
        assert "language" not in final_task_types
        python_prompts = []
        python_prompts_by_task = {}
        for task_id, nl_rule in prompt_pairs:
            prompt = Task2PythonPrompt(
                experiment_state=experiment_state,
                final_task_id=task_id,
                final_task_split=task_split,
                final_task_lang=nl_rule,
                line_separator=line_separator,
                n_exs=n_exs,
                final_task_types=final_task_types + ["language"],
                max_few_shot=max_few_shot,
                apply_fewshot_progs_to_exs=apply_fewshot_progs_to_exs,
                example_selection_strategy=example_selection_strategy,
                few_shot_representation=("nl", "code")
            )
            python_prompts.append(prompt)
            python_prompts_by_task.setdefault(task_id, []).append(prompt)

        python_completions = GPTBase.query_batch(
            python_prompts,
            model_name=self.ENGINE,     # Good enough to do NL -> python w/ weaker proposal engine, probably. TODO: Make this a parameter.
            max_tokens=max_tokens,
            temperature=0,
            stop_token=None,
            n_samples=1)

        python_parses = {}
        python_completions_by_task = {}
        for completion, pp in zip(python_completions, prompt_pairs):
            task_id, _ = pp
            python_parses.setdefault(task_id, []).append(self.parse_code_response(
                completion,
                experiment_state=experiment_state,
                task_split=task_split,
                task_ids=[task_id],
                evaluate_samples=True,
                verbose=verbose,
            ))
            python_completions_by_task.setdefault(task_id, []).append(completion)
        
        # Collapse parses to single list.
        python_parses = [python_parses[task_id] for task_id in task_batch_ids]
        python_prompts = [python_prompts_by_task[task_id] for task_id in task_batch_ids]
        python_completions = [python_completions_by_task[task_id] for task_id in task_batch_ids]

        return [task2nl_prompts, python_prompts], [task2nl_completions, python_completions], [task2nl_parses, python_parses]

    def parse_nl_response(self, response):
        def extract_response(prefixes, response):
            patterns = [f"{prefix}: (.*)" for prefix in prefixes]
            for pattern in patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                if matches:
                    assert len(matches) == 1
                    return matches[0].strip()

        completions = [choice["message"]["content"] for choice in response["choices"]]
        return [extract_response(["Rule"], completion) for completion in completions]

    def parse_code_response(
        self,
        completion,
        experiment_state,
        task_split: str,
        task_ids: list,
        evaluate_samples: bool = True,
        verbose: bool = False,
    ):
        """Parses the completion + checks if it solves any tasks."""

        # TODO: Check if program is "valid". Currently, we assume all programs are valid.

        def prog_solves(task, program):
            task_inps, task_outs = zip(*task.examples)
            task_inps = [t[0] for t in task_inps]
            task_inps = [f"\"{t}\"" if isinstance(t, str) else str(t) for t in task_inps]
            res = execute_function(program, task_inps, verbose=verbose)
            return all([r == o for r, o in zip(res, task_outs)])

        parse_results = []
        for choice in completion["choices"]:
            program_str_gpt = choice["text"]
            program = extract_program(program_str_gpt)

            # Does the program solve any tasks?
            task_attempted = None
            tasks_solved = []
            if evaluate_samples:
                for task in experiment_state.get_tasks_for_ids(task_split, task_ids):

                    # Format [((inp,), out)] -> ["inp"] so we can execute as string.
                    # TODO: Currently wrapping actual strings w/ "" but this is sketchy.
                    task_inps, task_outs = zip(*task.examples)
                    task_inps = [t[0] for t in task_inps]

                    task_inps = [f"\"{t}\"" if isinstance(t, str) else str(t) for t in task_inps]
                    res = execute_function(program, task_inps, verbose=verbose)
                    if all([r == o for r, o in zip(res, task_outs)]):
                        tasks_solved.append(task.name)
                        break   # Breaking after first task solved.

                    if len(task_ids) == 1:
                        task_attempted = task_ids[0]

            # Try removing comments / tests / unrelated code.
            prog_lines = program.split("\n")
            trimmed_lines = []
            for line in prog_lines:
                if line.startswith("\t") or line.startswith(" ") or line.startswith("def"):
                    trimmed_lines.append(line)
            
            trimmed_program = "\n".join(trimmed_lines)
            trimmed_solves = True
            if evaluate_samples:
                for task in experiment_state.get_tasks_for_ids(task_split, tasks_solved):
                    if not prog_solves(task, trimmed_program):
                        trimmed_solves = False
                        break

            # TODO: Log failure instead of breakpoint.
            og_prog = program
            if trimmed_solves:
                program = trimmed_program
            if not trimmed_solves:
                import pdb; pdb.set_trace()

            parse_results.append(
                {
                    "text": program_str_gpt,
                    "valid": True,
                    "program": str(program),
                    "hash": abs(hash(str(program))),
                    "task_attempted": task_attempted,
                    "tasks_solved": tasks_solved,
                }
            )

        return parse_results

    def add_python_samples_to_experiment_state(
        self,
        experiment_state: ExperimentState,
        task_split: str,
        parse_results_valid: list,
    ):
        for result_data in parse_results_valid:

            program = result_data["program"]

            # If the program solves any tasks, add it to the respective task frontier(s).
            if len(result_data["tasks_solved"]) > 0:
                for task in experiment_state.get_tasks_for_ids(
                    task_split=task_split, task_ids=result_data["tasks_solved"]
                ):

                    new_frontier = Frontier(
                        frontier=[
                            FrontierEntry(
                                program=program,
                                logPrior=0.0,
                                logLikelihood=0.0,
                                origin=self.name,
                                tokens=[]
                            )
                        ],
                        task=task,
                    )

                    experiment_state.task_frontiers_python[task_split][
                        task
                    ] = experiment_state.task_frontiers_python[task_split][task].combine(
                        new_frontier
                    )

            # Otherwise, discard the sample
            else:
                continue
