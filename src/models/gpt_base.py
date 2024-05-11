"""
gpt_base.py | Author: Gabe Grand.

Base class containing utilities for working with the GPT language model.
"""

import json
import os
from abc import ABCMeta, abstractmethod
from typing import Union, Optional

import openai
from transformers import GPT2TokenizerFast
from Levenshtein import distance as levenshtein_distance

from src.experiment_iterator import RANDOM_GENERATOR
from src.models.laps_grammar import LAPSGrammar
from src.models.model_loaders import GRAMMAR
from src.task_loaders import LANGUAGE, PROGRAMS, TEST, TRAIN
from src.query_utils import query_batch
from src.python_utils import execute_function, FailedExecutionResult
from src.embedding_utils import cosine_similarity, embed_batch
from prompts.type_to_description import type_to_descs

DEFAULT_LINE_SEPARATOR = "\n"
RANDOM, FIRST, STR_DISTANCE, EMBEDDING_DISTANCE, ORACLE_CODE, ORACLE_NL = (
    "random", "first", "str_distance", "embedding_distance", "oracle_code", "oracle_nl")

class BasePrompt(metaclass=ABCMeta):
    TASK_TYPES = [LANGUAGE, PROGRAMS]

    DEFAULT_MESSAGE_SEPARATOR = (
        DEFAULT_LINE_SEPARATOR + "======" + DEFAULT_LINE_SEPARATOR
    )

    DEFAULT_PREFIX_PROGRAM = ""
    DEFAULT_PREFIX_LANGUAGE = "-- "  # Haskell-style comment

    # https://platform.openai.com/docs/api-reference/chat
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    ROLE_USER = "user"

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def load_from_dict(self):
        pass

    @abstractmethod
    def to_chat_format(self):
        pass

    def __repr__(self):
        return self.json()

    def json(self):
        return json.dumps(self.to_dict(), indent=4)

    def serialize(self):
        return self.__str__()

    def chat_message(self, text, role=None):
        role = role or self.ROLE_USER
        return {
            "role": role,
            "content": text,
        }


class Prompt(BasePrompt):
    def __init__(
        self,
        experiment_state,
        body_task_ids: list,
        final_task_id: str,
        body_task_types: list = [LANGUAGE, PROGRAMS],
        final_task_types: list = [LANGUAGE],
        final_task_split: str = TRAIN,
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        prefix_language: str = BasePrompt.DEFAULT_PREFIX_LANGUAGE,
        prefix_program: str = BasePrompt.DEFAULT_PREFIX_PROGRAM,
        function_name_classes: list = [
            LAPSGrammar.HUMAN_READABLE,
            LAPSGrammar.DEFAULT_FUNCTION_NAMES,
        ],
        include_abstractions: bool = True,
        prepend_dsl_description: bool = False,
    ):
        assert isinstance(body_task_ids, list)
        assert len(body_task_ids) > 0

        assert isinstance(final_task_id, str)
        assert final_task_split in (TRAIN, TEST)

        # Enforce canonical ordering of task_types
        body_task_types = [t for t in self.TASK_TYPES if t in body_task_types]
        final_task_types = [t for t in self.TASK_TYPES if t in final_task_types]
        assert len(body_task_types) > 0
        assert len(final_task_types) > 0
        assert PROGRAMS in body_task_types

        self.experiment_state = experiment_state
        self.grammar = experiment_state.models[GRAMMAR]
        self.rng = experiment_state.metadata[RANDOM_GENERATOR]

        self.body_task_types = body_task_types
        self.final_task_types = final_task_types
        self.final_task_split = final_task_split

        self.line_separator = line_separator
        self.prefix_language = prefix_language
        self.prefix_program = prefix_program

        self.function_name_classes = function_name_classes

        self.body_task_data = [
            self._get_task_data(
                task_split=TRAIN,
                task_id=task_id,
                task_types=body_task_types,
                beta_reduce_program=(not include_abstractions),
            )
            for task_id in body_task_ids
        ]

        self.final_task_data = self._get_task_data(
            task_id=final_task_id,
            task_types=final_task_types,
            task_split=final_task_split,
            beta_reduce_program=(not include_abstractions),
        )

        self.prepend_dsl_description = prepend_dsl_description
        self.dsl_description = (
            self._get_dsl_description(include_abstractions=include_abstractions)
            if prepend_dsl_description
            else ""
        )

    def __len__(self):
        return len(self.body_task_data) + 1

    def __str__(self):
        return (
            self.line_separator.join([x["content"] for x in self.to_message_list()])
            + "\n"
        )

    def to_message_list(self):
        prompt_list = []
        if self.prepend_dsl_description:
            prompt_list += [
                self.chat_message(self.dsl_description, role=self.ROLE_SYSTEM)
            ]
        # Write the body tasks
        prompt_list += [self.chat_message("Here are some example programs:")]
        for task_data in self.body_task_data:
            if LANGUAGE in self.body_task_types:
                prompt_list += [
                    self.chat_message(self.prefix_language + task_data["task_language"])
                ]
            if PROGRAMS in self.body_task_types:
                prompt_list += [
                    self.chat_message(
                        self.prefix_program + task_data["task_program"],
                        role=self.ROLE_ASSISTANT,
                    )
                ]
        # Write the final task
        if LANGUAGE in self.final_task_types:
            prompt_list += [
                self.chat_message(
                    self.prefix_language + self.final_task_data["task_language"],
                )
            ]
        if PROGRAMS in self.final_task_types:
            prompt_list += [
                self.chat_message(
                    self.prefix_program + self.final_task_data["task_program"],
                    role=self.ROLE_ASSISTANT,
                )
            ]
        return prompt_list

    def to_chat_format(self):
        messages = self.to_message_list()
        return messages

    def to_dict(self):
        return {
            "dsl_description": self.dsl_description,
            "body_task_data": self.body_task_data,
            "final_task_data": self.final_task_data,
        }

    def load_from_dict(self, d):
        self.dsl_description = d["dsl_description"]
        self.body_task_data = d["body_task_data"]
        self.final_task_data = d["final_task_data"]

    def get_last_program(self):
        if PROGRAMS in self.final_task_types:
            return self.final_task_data["task_program"]
        else:
            return self.body_task_data[-1]["task_program"]

    def remove_last_body_task(self):
        if len(self.body_task_data) > 1:
            self.body_task_data = self.body_task_data[:-1]
        else:
            raise ValueError("Cannot remove single remaining body task from prompt.")

    def _get_task_data(
        self,
        task_id: str,
        task_types: list,
        task_split: str = TRAIN,
        use_mdl_program: bool = True,
        beta_reduce_program: bool = False,
    ):
        frontier = self.experiment_state.get_frontiers_for_ids(task_split, [task_id])[0]

        # Optionally, get the program
        if PROGRAMS in task_types:
            programs = [e.program for e in frontier.entries]
            if use_mdl_program:
                task_program = self.rng.choice(self.grammar.get_mdl_programs(programs))
            else:
                task_program = self.rng.choice(programs)
            if beta_reduce_program:
                task_program = task_program.betaNormalForm()
            task_program = self.grammar.show_program(
                task_program, name_classes=self.function_name_classes
            )
        else:
            task_program = None

        # Optionally, get the language
        if LANGUAGE in task_types:
            task_language = self.rng.choice(
                self.experiment_state.get_language_for_ids(task_split, [task_id])[0]
            )
            # Remove any line separators from the language
            task_language = task_language.replace(self.line_separator, " ")
        else:
            task_language = None

        return {
            "task_id": task_id,
            "task_program": task_program,
            "task_language": task_language,
        }

    def _get_dsl_description(self, include_abstractions: bool = True):
        dsl_fns = []
        for primitive in self.grammar.primitives:
            if primitive.isInvented and (not include_abstractions):
                # Optionally, skip abstractions
                continue
            fn_name = self.grammar.get_name(
                production_key=str(primitive), name_classes=self.function_name_classes
            )
            fn_type = primitive.infer()
            if primitive.isInvented:
                fn_body = str(
                    self.grammar.show_program(
                        str(primitive)[
                            1:
                        ],  # Remove leading `#` so that any inlined abstractions are replaced with their fn_name
                        name_classes=[
                            LAPSGrammar.HUMAN_READABLE,
                            LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                        ],
                    )
                )
            else:
                fn_body = str(primitive)
            fn_description = self.grammar.get_function_description(str(primitive))
            dsl_fns.append((primitive, fn_name, fn_type, fn_body, fn_description))

        dsl_description = (
            "You are an expert programmer working in a language based on lambda calculus.\n"
            + "Your goal is to write programs that accomplish the tasks specified by the user.\n"
        )
        if "dsl_description_prefix" in self.experiment_state.metadata:
            dsl_description += (
                self.experiment_state.metadata["dsl_description_prefix"] + "\n"
            )

        dsl_description += "\nWrite programs using the available functions:\n\n"

        for primitive, fn_name, fn_type, fn_body, fn_description in dsl_fns:
            docstring = f"{fn_name} :: {fn_type}"
            if primitive.isInvented:
                docstring += f"\n{fn_body}"
            if fn_description is not None:
                docstring += f"\ndescription: {fn_description}"
            dsl_description += docstring + "\n\n"

        return dsl_description


class Task2PythonPrompt(Prompt):
    """Prompt to solve tasks in python."""

    def __init__(
        self,
        experiment_state,
        final_task_id: str,
        final_task_split: str = TRAIN,
        final_task_lang: str = None,
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        n_exs: int = 3,
        final_task_types: list = [LANGUAGE, PROGRAMS],
        apply_fewshot_progs_to_exs: bool = True,
        max_few_shot: int = 100,
        example_selection_strategy=FIRST,
        few_shot_representation=("ex", "code")
    ):
        assert isinstance(final_task_id, str)
        assert final_task_split in (TRAIN, TEST)

        self.experiment_state = experiment_state
        self.rng = experiment_state.metadata[RANDOM_GENERATOR]
        self.line_separator = line_separator

        self.final_task_split = final_task_split
        self.final_task_data = self._get_task_data(
            task_id=final_task_id,
            task_split=final_task_split,
        )
        self.final_task_lang = final_task_lang

        self.n_exs = n_exs
        self.final_task_types = final_task_types

        self.few_shot_ios_per_example = n_exs
        self.max_few_shot = max_few_shot
        self.apply_to_new_task = apply_fewshot_progs_to_exs
        self.example_selection_strategy = example_selection_strategy
        self.few_shot_representation = few_shot_representation

        if (not apply_fewshot_progs_to_exs) and (example_selection_strategy not in (RANDOM, FIRST)):
            raise ValueError(f"Strategy {example_selection_strategy} must be applied to new task.")
        if example_selection_strategy not in (RANDOM, FIRST, STR_DISTANCE, EMBEDDING_DISTANCE, ORACLE_CODE, ORACLE_NL):
            raise ValueError(f"Invalid strategy: {example_selection_strategy}")


    def __len__(self):
        return 1

    def __str__(self):
        return (
            self.line_separator.join([x["content"] for x in self.to_message_list()])
            + "\n")

    def to_message_list(self):
        desc = self.final_task_lang if self.final_task_lang else self.final_task_data["task_language"]
        return [self.chat_message(
            self._make_convert_to_python_prompt(
                domain_description=self.experiment_state.metadata["domain_description"],
                examples=self._format_ios(self.final_task_data["task_examples"][:self.n_exs]),
                description=desc if LANGUAGE in self.final_task_types else None,
                code=None,   # TODO
                few_shot_examples=self._get_examples()
        ))]

    def _get_examples(self):
        """Chooses + formats few-shot examples."""

        solved_tasks = self.experiment_state.get_non_empty_python_frontiers_for_split(self.final_task_split)
        if not solved_tasks or self.max_few_shot == 0:
            return None

        strategy = self.example_selection_strategy
        n_fewshot = min(self.max_few_shot, len(solved_tasks))

        # Randomly choose examples.
        if strategy == RANDOM:
            solved_tasks = self.rng.choice(solved_tasks, size=n_fewshot, replace=False)
        
        # Choose the first n examples.
        elif strategy == FIRST:
            solved_tasks = solved_tasks[:n_fewshot]

        # All other strategies require applying the programs to the examples.
        n_per = self.few_shot_ios_per_example
        if self.apply_to_new_task:
            task_exs = self.final_task_data["task_examples"]
            example_ios = []
            programs = []
            language = []
            for ex in solved_tasks:

                # Execute the program on the examples.
                ex_ins = [ex[0][0] for ex in task_exs]
                ex_ins_formatted = [("'" + e + "'" if isinstance(e, str) else str(e)) for e in ex_ins]
                ex_prog = ex.entries[0].program
                exec_res = execute_function(ex_prog, ex_ins_formatted)

                # Find non-degenerate examples.
                valid_exs = []
                for exec_out, ex_in in zip(exec_res, ex_ins):
                    if exec_out != ex_in and not isinstance(exec_out, FailedExecutionResult):
                        valid_exs.append(((ex_in,), exec_out))
                
                # Add examples.
                if len(valid_exs) >= n_per:
                    example_ios.append(valid_exs[:n_per])
                    programs.append(ex_prog)
                    task_lang = self.experiment_state.get_language_for_ids(self.final_task_split, [ex.task.name])[0]
                    assert task_lang or ("nl" not in self.few_shot_representation)
                    language.append(task_lang[0] if task_lang else None)
                else:
                    pass
        else:
            example_ios = [ex.task.examples[:n_per] for ex in solved_tasks]
            programs = [ex.entries[0].program for ex in solved_tasks]
            ls = self.experiment_state.get_language_for_ids(self.final_task_split, [ex.task.name for ex in solved_tasks])
            language = [l[0] if l else None for l in ls]

        final_task_ios = self.final_task_data["task_examples"]
        # Apply sol'n progs to all tasks, choose examples with outputs closest to final task, as measured
        # by string distance.
        if strategy == STR_DISTANCE:
            distances = []
            for ios in example_ios:
                dist = sum([levenshtein_distance(str(o), str(final_task_ios[i][1])) for i, (_, o) in enumerate(ios)])
                distances.append(dist)
            
            # Choose the n closest examples.
            sorted_inds = sorted(range(len(distances)), key=lambda i: distances[i])
            example_ios = [example_ios[i] for i in sorted_inds[:n_fewshot]]
            programs = [programs[i] for i in sorted_inds[:n_fewshot]]
            language = [language[i] for i in sorted_inds[:n_fewshot]]
        
        # Apply sol'n progs to all tasks, choose examples with outputs closest to final task, as measured
        # by embedding distance.
        elif strategy == EMBEDDING_DISTANCE:
            io_strs = [self._format_ios(ios) for ios in example_ios]
            final_io_str = self._format_ios(final_task_ios)
            embeddings = embed_batch(io_strs + [final_io_str], "text-embedding-3-small")
            ex_embeddings, final_task_embedding = embeddings[:-1], embeddings[-1]
            distances = [-cosine_similarity(e, final_task_embedding) for e in ex_embeddings]

            # Choose the n closest examples.
            sorted_inds = sorted(range(len(distances)), key=lambda i: distances[i])
            example_ios = [example_ios[i] for i in sorted_inds[:n_fewshot]]
            programs = [programs[i] for i in sorted_inds[:n_fewshot]]
            language = [language[i] for i in sorted_inds[:n_fewshot]]

        # Choose tasks where the code is closest in embedding space to the code for this task.
        elif strategy == ORACLE_CODE:
            raise NotImplementedError()

        # Choose tasks where the natural language description is closest in embedding space to the
        # description for this task.
        elif strategy == ORACLE_NL:
            raise NotImplementedError()

        if not example_ios:
            import pdb; pdb.set_trace()

        # Format examples.
        examples_str = ""
        for ios, prog, lang in zip(example_ios, programs, language):
            for rep_type in self.few_shot_representation:
                if "ex" == rep_type:
                    examples_str += f"Examples:\n{self._format_ios(ios)}\n"
                if "code" == rep_type:
                    assert "def fn" in prog
                    examples_str += f"Code:\n```{prog}\n```\n"
                if "nl" == rep_type:
                    examples_str += f"Rule: {lang}\n"
            examples_str += "\n"

        return examples_str.strip()

    def _make_convert_prompt(
        self, job_str: str, domain_description: Optional[str], examples: str, description: Optional[str],
        code: Optional[str], final_instruction: Optional[str] = None) -> str:
        """Makes the prompts for converting a set of examples/language/code."""

        assert examples
        
        prompt = ""
        if domain_description:
            prompt += self._get_domain_description() + "\n\n"
        
        provided_info = [x_str for x, x_str in ((examples, "examples"), (description, "a description"), (code, "code")) if x]
        if len(provided_info) > 1:
            prompt += f"You will be given {', '.join(provided_info[:-1])} and {provided_info[-1]} of a single transformation rule.\n\n"
        elif provided_info:
            prompt += f"You will be given {provided_info[0]} of a single transformation rule.\n\n"

        prompt += job_str + "\n\n"

        if examples:
            prompt += f"Here are examples of the transformation:\n{examples}\n\n"
        if description:
            prompt += f"Here is a description of the task: \"{description}\"\n\n"
        if code:
            prompt += f"\nHere is the code that solves the task.\n{code}\n\n"

        if final_instruction:
            prompt += final_instruction

        return prompt.strip()

    def _get_domain_description(self):
        tin, tout = self._describe_type()
        return (f"This task is in the domain of {self.experiment_state.metadata['domain_description']}. "
                f"The solution should take in {tin} and return {tout}.")

    def _make_convert_to_python_prompt(
            self, domain_description: Optional[str], examples: str, description: Optional[str], code: Optional[str],
            few_shot_examples=None) -> str:
        """Makes the prompts for converting a set of examples to a Python function."""

        job_prompt = "Write a Python function `fn` that implements the underlying rule mapping each input to its output."
        final_instruction = ""
        if few_shot_examples:
            final_instruction += "Here are examples of other transformations and the code for the underlying rule:\n\n"
            final_instruction += few_shot_examples + "\n"
            final_instruction += "Now, determine the underlying rule for the earlier examples and write a Python function `fn` that implements this rule.\n\n"
        final_instruction += "Do not import  any libraries."
        return self._make_convert_prompt(job_prompt, domain_description, examples, description, code, final_instruction)

    def _format_ios(self, examples):

        ios = []
        for i, o in examples:

            assert isinstance(i, (tuple, list)) and len(i) == 1
            ios.append((i[0], o))

        return "\n".join([f"{i} -> {o}" for i, o in ios])

    def _describe_type(self):
        """Generates a readable description for a primitive type."""

        task_type = self.final_task_data["task_type"]
        assert task_type["constructor"] == "->" and len(task_type["arguments"]) == 2
        tin, tout = task_type["arguments"]

        # Special case for strings = list(char).
        if tin["constructor"] == "list" and tin["arguments"][0]["constructor"] == "char":
            tin = {"constructor": "tstr", "arguments": []}
        if tout["constructor"] == "list" and tout["arguments"][0]["constructor"] == "char":
            tout = {"constructor": "tstr", "arguments": []}

        # Handle lists.
        if tin["constructor"] == "list":
            tin = "a list of " + type_to_descs[tin["arguments"][0]["constructor"]][1]
        else:
            tin = type_to_descs[tin["constructor"]][0]

        if tout["constructor"] == "list":
            tout = "a list of " + type_to_descs[tout["arguments"][0]["constructor"]][1]
        else:
            tout = type_to_descs[tout["constructor"]][0]

        return tin, tout

    def to_dict(self):
        return {
            "final_task_data": self.final_task_data,
        }

    def load_from_dict(self, d):
        self.final_task_data = d["final_task_data"]

    def _get_task_data(
        self,
        task_id: str,
        task_split: str = TRAIN,
    ):

        # Add examples
        task_info = self.experiment_state.get_tasks_for_ids(task_split, [task_id])[0]
        task_examples = task_info.examples

        # Optionally, get the language
        task_language = self.experiment_state.get_language_for_ids(task_split, [task_id])[0]
        task_language = self.rng.choice(task_language) if task_language else None
        # Remove any line separators from the language
        task_language = task_language.replace(self.line_separator, " ") if task_language else None

        return {
            "task_id": task_id,
            "task_language": task_language,
            "task_examples": task_examples,
            "task_type": task_info.request.json(),
        }

    def _get_system_msg(self):
        return None


class DSL2PythonPrompt(Task2PythonPrompt):
    """Prompt to convert DSL code to Python."""

    def __init__(
        self,
        program,
        experiment_state,
        final_task_id: str,
        final_task_split: str = TRAIN,
        n_exs: int = 3,
        final_task_types: list = [LANGUAGE, PROGRAMS],
    ):
        self.program = program
        assert isinstance(final_task_id, str)
        assert final_task_split in (TRAIN, TEST)

        self.experiment_state = experiment_state
        self.rng = experiment_state.metadata[RANDOM_GENERATOR]

        self.line_separator = DEFAULT_LINE_SEPARATOR

        self.final_task_split = final_task_split
        self.final_task_data = self._get_task_data(
            task_id=final_task_id,
            task_split=final_task_split,
        )

        self.n_exs = n_exs
        self.final_task_types = final_task_types


    def to_message_list(self):
        return [self.chat_message(
            self._make_convert_to_python_prompt(
                domain_description=self.experiment_state.metadata["domain_description"],
                examples=self._format_ios(self.final_task_data["task_examples"][:self.n_exs]),
                description=self.final_task_data["task_language"]if LANGUAGE in self.final_task_types else None,
                code=self.program,
                few_shot_examples=None
        ))]


class Task2NLPrompt(Task2PythonPrompt):
    """Prompt to solve tasks by proposing NL hypotheses then converting to python."""

    def __init__(
        self,
        experiment_state,
        final_task_id: str,
        final_task_split: str = TRAIN,
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        n_exs: int = 3,
        final_task_types: list = [LANGUAGE, PROGRAMS],
        max_few_shot: int = 100,
        example_selection_strategy=FIRST,
        apply_fewshot_progs_to_exs=False,
    ):
        assert LANGUAGE not in final_task_types
        super().__init__(
            experiment_state,
            final_task_id,
            final_task_split,
            line_separator=line_separator,
            n_exs=n_exs,
            final_task_types=final_task_types,
            max_few_shot=max_few_shot,
            example_selection_strategy=example_selection_strategy,
            apply_fewshot_progs_to_exs=apply_fewshot_progs_to_exs,
            few_shot_representation=("ex", "nl")
        )

    def to_message_list(self):
        return [self.chat_message(
            self._make_convert_to_nl_prompt(
                domain_description=self.experiment_state.metadata["domain_description"],
                examples=self._format_ios(self.final_task_data["task_examples"][:self.n_exs]),
                description=self.final_task_data["task_language"]if LANGUAGE in self.final_task_types else None,
                code=None,   # TODO
                few_shot_examples=self._get_examples()
        ))]

    def _make_convert_to_nl_prompt(
            self, domain_description: Optional[str], examples: str, description: Optional[str], code: Optional[str], few_shot_examples=None) -> str:
        """Makes the prompts for converting a set of examples to a natural language description."""

        job_prompt = "Figure out the underlying rule mapping each input to its output."
        final_instruction = ""
        if few_shot_examples:
            final_instruction += "Here are examples of other transformations and the underlying rule:\n\n"
            final_instruction += few_shot_examples + "\n\n"
            final_instruction += "Now, determine the underlying rule for the first examples.\n\n"

        final_instruction += "Format your rule as follows:\n\nRule: <Your rule>"
        return self._make_convert_prompt(job_prompt, domain_description, examples, description, code, final_instruction)



class GPTBase(object):
    # https://platform.openai.com/docs/models
    ENGINE_CODEX = "code-davinci-002"
    ENGINE_GPT_3_5_TURBO = "gpt-3.5-turbo"
    ENGINE_GPT_3_5_TURBO_0301 = "gpt-3.5-turbo-0301"
    ENGINE_GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    ENGINE_GPT_4 = "gpt-4"
    ENGINE_GPT_4_0613 = "gpt-4-0613"
    ENGINE_GPT_4_0314 = "gpt-4-0314"
    ENGINE_GPT_4_TURBO = "gpt-4-turbo"
    ENGINE_GPT_4_TURBO_0125 = "gpt-4-0125-preview"
    ENGINE_GPT_4_TURBO_0409 = "gpt-4-turbo-2024-04-09"
    ENGINE_DEFAULT = ENGINE_CODEX

    # Max tokens for BOTH the prompt and the completion.
    MAX_TOKENS_PER_ENGINE = {
        ENGINE_CODEX: 4096,  # 8001
        ENGINE_GPT_3_5_TURBO: 16385,
        ENGINE_GPT_3_5_TURBO_0301: 4096,
        ENGINE_GPT_3_5_TURBO_0125: 16385,
        ENGINE_GPT_4: 8192,
        ENGINE_GPT_4_0613: 8192,
        ENGINE_GPT_4_0314: 8192,
        ENGINE_GPT_4_TURBO: 128000,
        ENGINE_GPT_4_TURBO_0125: 128000,
        ENGINE_GPT_4_TURBO_0409: 128000,
    }

    # Models that use chat completion format
    CHAT_ENGINES = [
        ENGINE_GPT_3_5_TURBO,
        ENGINE_GPT_3_5_TURBO_0301,
        ENGINE_GPT_3_5_TURBO_0125,
        ENGINE_GPT_4,
        ENGINE_GPT_4_0314,
        ENGINE_GPT_4_0613,
        ENGINE_GPT_4,
        ENGINE_GPT_4_TURBO,
        ENGINE_GPT_4_TURBO_0125,
        ENGINE_GPT_4_TURBO_0409,
    ]

    def __init__(self, experiment_state=None, engine=None):
        super().__init__()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set this in the shell via `export OPENAI_API_KEY=...`"
            )
        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.ENGINE = engine or self.ENGINE_DEFAULT
        self.ENGINE_MAX_TOKENS = self.MAX_TOKENS_PER_ENGINE[self.ENGINE]

        # Used for computing approximate token counts for queries
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.model_max_length = self.ENGINE_MAX_TOKENS
        os.environ["TOKENIZERS_PARALLELISM"] = str(False)

    @classmethod
    def query_batch(cls, prompts, model_name, max_tokens, temperature, stop_token, n_samples, only_sample_once=True):

        messages_batch = []
        for prompt in prompts:
            messages = prompt.to_chat_format()
            messages_batch.append(messages)
        batch_responses = query_batch(
            messages_batch=messages_batch,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_token=stop_token,
            n_samples=n_samples,
            only_sample_once=only_sample_once,
        )

        # Convert ChatCompletion -> Completion format
        for resp in batch_responses:
            for choice in resp["choices"]:
                choice["text"] = choice["message"]["content"]
        
        return batch_responses

    def query_completion(
        self,
        prompt: Union[Prompt, str],
        n_samples: int,
        temperature: float = None,
        max_tokens: int = 256,  # Max tokens for completion only.
        stop: str = DEFAULT_LINE_SEPARATOR,
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        top_p=None,
        logprobs=None,
    ):
        if top_p or logprobs or line_separator != DEFAULT_LINE_SEPARATOR:
            raise NotImplementedError()
        if self.is_chat_format():

            # Convert prompt text to ChatCompletion format
            if isinstance(prompt, BasePrompt):
                messages = prompt.to_chat_format()
            else:
                messages = [{"role": "user", "content": str(prompt)}]

            completion = query_batch(
                messages_batch=[messages],
                model_name=self.ENGINE,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_token=stop,
                n_samples=n_samples,
            )[0]

            # Convert ChatCompletion -> Completion format
            for choice in completion["choices"]:
                choice["text"] = choice["message"]["content"]
        else:
            raise NotImplementedError()

        return completion

    def is_chat_format(self):
        return self.ENGINE in self.CHAT_ENGINES

    def count_tokens_gpt2(self, text):
        # TODO(gg): Consider preprocessing to collapse whitespace, which could
        # bring the behavior more in line with the Codex tokenizer.
        return len(self.tokenizer(text, truncation=False)["input_ids"])
