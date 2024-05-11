"""
task_loaders.py | Author : Gabe Grand
Utilities for autogenerating configs for experiments based on templates.

"""

import json
import os
import subprocess
from enum import Enum

# import data.drawings.make_tasks as drawing_tasks # zyzzyva@ Temporarily disable this domain., which is causing
from src.experiment_iterator import (
    AWS_S3_SYNC_BASE_PATH,
    CURR_ITERATION,
    EXPERIMENT_BLOCK_TYPE,
    EXPERIMENT_BLOCK_TYPE_CHECKPOINT,
)
from src.models.laps_dreamcoder_recognition import LAPSDreamCoderRecognition
from src.models.laps_grammar import LAPSGrammar
from src.models.model_loaders import (
    AMORTIZED_SYNTHESIS,
    GRAMMAR,
    INITIALIZE_GROUND_TRUTH,
    LIBRARY_LEARNER,
    LIBRARY_NAMER,
    LLM_SOLVER,
    LLM_SOLVER_PYTHON,
    PROGRAM_REWRITER,
    SAMPLE_GENERATOR,
)
from src.models.stitch_proposer import StitchProposerLibraryLearner
from src.task_loaders import ALL, RandomShuffleOrderedTaskBatcher

# @zyzzyva (April 19): Temporarily disable the drawings domain, which is causing Primitive import conflicts.


DEFAULT_EXPERIMENT_DIR = "experiments_iterative"
DEFAULT_TEMPLATE_DIR = os.path.join(DEFAULT_EXPERIMENT_DIR, "templates")


DEFAULT_STITCH_PARAMS = {
    "max_arity": 3,
    "iterations": 10,
    "candidates_per_iteration": 1,
}

DEFAULT_GPT_PARAMS = {
    "debug": False,
    "use_cached": False,
    "n_samples": 50,
    "n_samples_per_query": 5,
    "temperature": 0.40,
    "max_tokens_completion_beta": 2.0,
    "function_name_classes": ["human_readable", "default"],
    "final_task_origin": "default",
    "body_task_types": ["programs"],
    "final_task_types": ["programs"],
    "prepend_dsl_description": False,
}

DEFAULT_GPT_SOLVER_PARAMS = {
    "temperature": 0.90,
    "max_tokens_completion_beta": 4.0,
    "function_name_classes": ["human_readable", "default"],
    "body_task_selection": "random",
}

DEFAULT_GPT_SOLVER_PYTHON_PARAMS = {
    "temperature": 0.90,
    "max_tokens_completion_beta": 4.0,
    "function_name_classes": ["human_readable", "default"],
}


class ExperimentType(str, Enum):
    # Partial synthesis experiments
    ORACLE = "oracle"
    ORACLE_TRAIN_TEST = "oracle_train_test"
    STITCH = "stitch"
    STITCH_CODEX = "stitch_codex"
    STITCH_CODEX_LANGUAGE = "stitch_codex_language"
    STITCH_CODEX_LANGUAGE_ORIGIN_RANDOM_TEST = (
        "stitch_codex_language_origin_random_test"
    )
    STITCH_CODEX_DSL_DESCRIPTION = "stitch_codex_dsl_description"
    # Full synthesis experiments
    BASE_DSL = "base_dsl"
    DREAMCODER = "baseline_dreamcoder"
    GPT_SOLVER = "gpt_solver"
    GPT_SOLVER_PYTHON = "gpt_solver_python"
    GPT_SOLVER_SEARCH = "gpt_solver_search"
    GPT_SOLVER_STITCH = "gpt_solver_stitch"
    GPT_SOLVER_STITCH_NAMER = "gpt_solver_stitch_namer"
    GPT_SOLVER_STITCH_NAMER_HYBRID_DSL = "gpt_solver_stitch_namer_hybrid_dsl"
    GPT_SOLVER_STITCH_NAMER_SEARCH = "gpt_solver_stitch_namer_search"
    GPT_SOLVER_LEARNER = "gpt_solver_learner"
    RANDOM = "random"
    COS_SIMILAR = "cos_similar"


def get_domain_metadata(domain: str):
    METADATA = {
        "logo": {
            "tasks_loader": "compositional_graphics_200",
            "task_language_loader": "compositional_graphics_200_synthetic",
            "ocaml_special_handler": "LOGO",
            "dsl_description_prefix": "This is a domain-specific language for Logo turtle graphics.",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 200],
            "n_tasks_train": 200,
            "n_tasks_test": 111,

            "domain_description": " Logo turtle graphic",
            "existing_types": []
        },
        "logo_human": {
            "tasks_loader": "compositional_graphics_200",
            "task_language_loader": "compositional_graphics_200_human",
            "ocaml_special_handler": "LOGO",
            "dsl_description_prefix": "This is a domain-specific language for Logo turtle graphics.",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 200],
            "n_tasks_train": 200,
            "n_tasks_test": 111,

            "domain_description": " Logo turtle graphic",
            "existing_types": []
        },
        "clevr": {
            "tasks_loader": "clevr",
            "task_language_loader": "clevr_synthetic",
            "ocaml_special_handler": "clevr",
            "dsl_description_prefix": "This is a domain-specific language for CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning.",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 191],
            "n_tasks_train": 191,
            "n_tasks_test": 103,

            "domain_description": "CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning",
            "existing_types": ["tint", "tbool", "arrow", "tlist", "tclevrcolor", "tclevrshape", "tclevrmaterial", "tclevrsize", "tclevrrelation", "tclevrobject"]
        },
        "clevr_human": {
            "tasks_loader": "clevr",
            "task_language_loader": "clevr_human",
            "ocaml_special_handler": "clevr",
            "dsl_description_prefix": "This is a domain-specific language for CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning.",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 191],
            "n_tasks_train": 191,
            "n_tasks_test": 103,

            "domain_description": "CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning",
            "existing_types": ["tint", "tbool", "arrow", "tlist", "tclevrcolor", "tclevrshape", "tclevrmaterial", "tclevrsize", "tclevrrelation", "tclevrobject"]
        },
        "re2": {
            "tasks_loader": "re2",
            "task_language_loader": "re2_synthetic",
            "ocaml_special_handler": "re2",
            "dsl_description_prefix": "This is a domain-specific language for regular expressions that specify string transformations.",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 200, 300, 400, 491],
            "n_tasks_train": 491,
            "n_tasks_test": 500,

            "domain_description": "regular expressions",
            "existing_types": ["tint", "tbool", "arrow", "tstr"]
        },
        "re2_human": {
            "tasks_loader": "re2",
            "task_language_loader": "re2_human",
            "ocaml_special_handler": "re2",
            "dsl_description_prefix": "This is a domain-specific language for regular expressions that specify string transformations.",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 200, 300, 400, 491],
            "n_tasks_train": 491,
            "n_tasks_test": 500,

            "domain_description": "regular expressions",
            "existing_types": ["tint", "tbool", "arrow", "tstr"]
        },

        "list_functions": {
            "tasks_loader": "list_functions",
            "task_language_loader": "list_functions_synthetic",
            "ocaml_special_handler": "list_functions",
            "dsl_description_prefix": "This is a domain-specific language for list functions.",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100],
            "n_tasks_train": 100,
            "n_tasks_test": 117,

            "domain_description": "list functions",
            "existing_types": ["tint", "tbool", "arrow", "tlist"]
        },
    }

    return METADATA[domain]


def build_config(
    experiment_name: str,
    experiment_type: str,
    domain: str,
    custom_experiment_type: str = None,
    output_directory: str = DEFAULT_EXPERIMENT_DIR,
    random_seed: int = 0,
    iterations: int = 1,
    init_iteration: int = 0,
    task_batcher: str = RandomShuffleOrderedTaskBatcher.name,
    global_batch_size: int = ALL,
    enumeration_timeout: int = None,
    recognition_train_steps: int = None,
    encoder: str = None,
    stitch_params: dict = DEFAULT_STITCH_PARAMS,
    gpt_params: dict = DEFAULT_GPT_PARAMS,
    compute_likelihoods: bool = True,
    compute_description_lengths: bool = True,
    increment_task_batcher: bool = True,
    init_frontiers_from_checkpoint: bool = False,
    init_frontiers_every_iteration: bool = False,
    init_grammar_from_checkpoint: bool = False,
    resume_checkpoint_directory: bool = False,
    s3_sync: bool = True,
    n_train: int = None,
    n_test: int = None,
    type_override: str = None,
    solver: str = "python",
    no_lang: bool = False,
    n_samples_python: int = 1,
):
    config = {}
    config.update(
        build_config_body(
            experiment_type=experiment_type,
            domain=domain,
            iterations=iterations,
            task_batcher=task_batcher,
            global_batch_size=global_batch_size,
            enumeration_timeout=enumeration_timeout,
            recognition_train_steps=recognition_train_steps,
            encoder=encoder,
            stitch_params=stitch_params,
            gpt_params=gpt_params,
            compute_likelihoods=compute_likelihoods,
            compute_description_lengths=compute_description_lengths,
            increment_task_batcher=increment_task_batcher,
            s3_sync=s3_sync,
            solver=solver,
            n_samples_python=n_samples_python,
        )
    )
    config.update(
        build_config_metadata(
            experiment_name=experiment_name,
            domain=domain,
            experiment_type=experiment_type,
            custom_experiment_type=custom_experiment_type,
            global_batch_size=global_batch_size,
            enumeration_timeout=enumeration_timeout,
            recognition_train_steps=recognition_train_steps,
            encoder=encoder,
            output_directory=output_directory,
            init_iteration=init_iteration,
            init_frontiers_from_checkpoint=init_frontiers_from_checkpoint,
            init_frontiers_every_iteration=init_frontiers_every_iteration,
            init_grammar_from_checkpoint=init_grammar_from_checkpoint,
            resume_checkpoint_directory=resume_checkpoint_directory,
            random_seed=random_seed,
            n_train=n_train,
            n_test=n_test,
            type_override=type_override,
            no_lang=no_lang,
        )
    )
    return config


def build_config_metadata(
    experiment_name: str,
    domain: str,
    experiment_type: str,
    custom_experiment_type: str = None,
    global_batch_size: int = ALL,
    enumeration_timeout: int = None,
    recognition_train_steps: int = None,
    encoder: str = None,
    output_directory: str = DEFAULT_EXPERIMENT_DIR,
    init_iteration: int = 0,
    init_frontiers_from_checkpoint: bool = False,
    init_frontiers_every_iteration: bool = False,
    init_grammar_from_checkpoint: bool = False,
    resume_checkpoint_directory: bool = False,
    random_seed: int = 0,
    n_train: int = None,
    n_test: int = None,
    type_override: str = None,
    no_lang: bool = False,
):
    domain_meta = get_domain_metadata(domain)

    export_directory = os.path.join(
        output_directory,
        "outputs",
        experiment_name,
        "domains",
        domain,
        custom_experiment_type or experiment_type,
        f"seed_{random_seed}",
    )
    log_directory = os.path.join(
        output_directory,
        "logs",
        experiment_name,
        "domains",
        domain,
        custom_experiment_type or experiment_type,
        f"seed_{random_seed}",
    )

    if experiment_type in ("dfl", "llm") and domain == "re2":
        assert type_override == "tstr", f"Type override must be 'tstr' for DFL on RE2, not {type_override}."
    elif type_override is not None:
        raise ValueError(
            f"Type override is not supported for experiment_name: {experiment_type} and domain: {domain}."
        )

    return {
        "metadata": {
            "experiment_name": experiment_name,
            "experiment_id": f"{custom_experiment_type or experiment_type}_{global_batch_size}",
            "human_readable": "Autogenerated iterative experiment.",
            "export_directory": export_directory,
            "log_directory": log_directory,
            "tasks_loader": domain_meta["tasks_loader"],
            "task_language_loader": domain_meta["task_language_loader"],
            "dsl_description_prefix": domain_meta["dsl_description_prefix"],
            "export_with_timestamp": False,
            "resume_checkpoint_directory": resume_checkpoint_directory,
            "init_frontiers_from_checkpoint": init_frontiers_from_checkpoint,
            "init_frontiers_every_iteration": init_frontiers_every_iteration,
            "init_grammar_from_checkpoint": init_grammar_from_checkpoint,
            "ocaml_special_handler": domain_meta["ocaml_special_handler"],
            "global_batch_size": global_batch_size,
            "enumeration_timeout": enumeration_timeout,
            "recognition_train_steps": recognition_train_steps,
            "encoder": encoder,
            "random_seed": random_seed,
            "curr_iteration": init_iteration,
            "n_tasks_train": n_train,
            "n_tasks_test": n_test,

            "domain_description": domain_meta["domain_description"],
            "existing_types": domain_meta["existing_types"],

            "type_override": type_override,

            "no_lang": no_lang,
        }
    }


def build_config_body(
    experiment_type: str,
    domain: str,
    iterations: int = 1,
    task_batcher: str = RandomShuffleOrderedTaskBatcher.name,
    global_batch_size: int = ALL,
    enumeration_timeout: int = None,
    recognition_train_steps: int = None,
    encoder: str = None,
    stitch_params: dict = DEFAULT_STITCH_PARAMS,
    gpt_params: dict = DEFAULT_GPT_PARAMS,
    compute_likelihoods: bool = True,
    compute_description_lengths: bool = True,
    increment_task_batcher: bool = True,
    s3_sync: bool = True,
    solver: str = "python",
    n_samples_python: int = 1,
):
    template_path = os.path.join(
        DEFAULT_TEMPLATE_DIR, f"template_{experiment_type}.json"
    )
    with open(template_path, "r") as f:
        config = json.load(f)

    domain_meta = get_domain_metadata(domain)

    model_initializers = config["model_initializers"]
    model_initializers[0]["model_loader"] = domain_meta["ocaml_special_handler"]
    config["model_initializers"] = model_initializers

    # Update recognition model params to match domain if there is a recognition model and
    # it is set on the command-line.
    recognition_encoder_initializer = next(
        (
            initializer
            for initializer in model_initializers
            if initializer["model_type"] == "examples_encoder"
        ),
        None,
    )

    if encoder:
        if not recognition_encoder_initializer:
            raise ValueError(
                "Encoder is provided by command-line arguments but there is no encoder being initialized in the template."
            )
        recognition_encoder_initializer["model_loader"] = encoder
    elif (
        recognition_encoder_initializer
        and recognition_encoder_initializer["model_loader"] is None
    ):
        raise ValueError(
            "Encoder is not provided by command-line arguments but there is an encoder being initialized in the template."
        )

    config["experiment_iterator"]["max_iterations"] = iterations
    config["experiment_iterator"]["task_batcher"]["model_type"] = task_batcher
    config["experiment_iterator"]["task_batcher"]["params"][
        "global_batch_size"
    ] = global_batch_size

    config["experiment_iterator"]["task_batcher"]["params"][
        "increment_at_global_iteration"
    ] = increment_task_batcher

    # params updates use the following precedence order (highest to lowest):
    # 1. params from CLI (e.g., stitch_params)
    # 2. params from template (e.g., block["params"])
    # 3. params from config_builder globals (e.g., DEFAULT_STITCH_PARAMS)

    loop_blocks = []
    for block in config["experiment_iterator"]["loop_blocks"]:
        if (
            block.get("model_type") == LAPSGrammar.GRAMMAR
            and block.get("model_fn") == LAPSGrammar.infer_programs_for_tasks.__name__
        ) or (
            block.get("model_type") == AMORTIZED_SYNTHESIS
            and block.get("model_fn")
            == LAPSDreamCoderRecognition.infer_programs_for_tasks.__name__
        ):
            block["params"]["solver"] = solver
            if enumeration_timeout is not None:
                block["params"]["enumeration_timeout"] = enumeration_timeout
        if block.get("model_type") == SAMPLE_GENERATOR:
            _gpt_params = DEFAULT_GPT_PARAMS
            _gpt_params.update(block["params"])
            _gpt_params.update(gpt_params)
            block["params"] = _gpt_params
        if block.get("model_type") == LLM_SOLVER:
            _gpt_params = DEFAULT_GPT_SOLVER_PARAMS
            _gpt_params.update(block["params"])
            _gpt_params.update(gpt_params)
            block["params"] = _gpt_params
        if block.get("model_type") == LLM_SOLVER_PYTHON:
            _gpt_params = DEFAULT_GPT_SOLVER_PYTHON_PARAMS
            _gpt_params.update(block["params"])
            _gpt_params.update(gpt_params)
            if n_samples_python is not None:
                _gpt_params["n_samples_per_query"] = n_samples_python
            block["params"] = _gpt_params
        # TODO(SA): Setup program translation.
        if block.get("model_type") == LIBRARY_NAMER:
            _gpt_params = block["params"]
            _gpt_params.update(gpt_params)
        if block.get("model_type") == LIBRARY_LEARNER:
            _stitch_params = DEFAULT_STITCH_PARAMS
            _stitch_params.update(block["params"])
            if (
                block.get("model_fn")
                == StitchProposerLibraryLearner.get_compressed_grammar_mdl_prior_rank.__name__
            ):
                _stitch_params.update(stitch_params)
                block["params"] = _stitch_params
        if (
            block.get("model_type")
            in [
                LAPSGrammar.GRAMMAR,
                SAMPLE_GENERATOR,
                PROGRAM_REWRITER,
            ]
            or block.get("state_fn") == INITIALIZE_GROUND_TRUTH
        ):
            block["params"].update(
                {
                    "compute_likelihoods": compute_likelihoods,
                }
            )
        if (
            block.get("model_type") == AMORTIZED_SYNTHESIS
            and block.get("model_fn")
            == LAPSDreamCoderRecognition.optimize_model_for_frontiers.__name__
            and recognition_train_steps is not None
        ):
            block["params"].update(
                {
                    "recognition_train_steps": recognition_train_steps,
                }
            )
        if (
            block.get(EXPERIMENT_BLOCK_TYPE) == EXPERIMENT_BLOCK_TYPE_CHECKPOINT
        ) and block.get(AWS_S3_SYNC_BASE_PATH):
            if s3_sync:
                # Verify that AWS CLI is configured on the machine
                subprocess.run(
                    "aws sts get-caller-identity",
                    shell=True,
                    capture_output=True,
                    check=True,
                )
                # Verify that the bucket exists
                subprocess.run(
                    f"aws s3 ls {block[AWS_S3_SYNC_BASE_PATH]}",
                    shell=True,
                    capture_output=True,
                    check=True,
                )
            else:
                # Disable S3 upload
                block[AWS_S3_SYNC_BASE_PATH] = None

        loop_blocks.append(block)
    config["experiment_iterator"]["loop_blocks"] = loop_blocks

    return config
