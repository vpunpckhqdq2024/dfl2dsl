"""
list_functions: make_tasks.py | Author : Sam Acquaviva

Loading tasks and language for the list functions domain
"""
import os

from src.task_loaders import *
from data.re2.grammar import *
from dreamcoder.domains.list.main import retrieveJSONTasks
from dreamcoder.program import Program
from dreamcoder.utilities import ParseFailure
from dreamcoder.type import arrow, tstr

DOMAIN_NAME = "list_functions"
ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/list/")
DEFAULT_DATASET = "list_functions250.json"
DEFAULT_TASKS_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS)
# DEFAULT_LANGUAGE_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, LANGUAGE)

N_TRAIN = 100

@TaskLoaderRegistry.register
class ListFunctionsLoader(TaskDataLoader):
    name = DOMAIN_NAME

    def load_tasks(self, type_override=None):
        if type_override:
            raise NotImplementedError
        dataset_path = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS, DEFAULT_DATASET)

        tasks = retrieveJSONTasks(dataset_path)

        tasks = {TRAIN: tasks[:N_TRAIN], TEST: tasks[N_TRAIN:]}

        for split in tasks.keys():
            for t in tasks[split]:
                t.supervisedSolution = None
                t.groundTruthProgram = None
        return tasks


@TaskLanguageLoaderRegistry.register
class ListFunctionsHumanLanguageLoader(TaskDataLoader):
    name = "list_functions_human"

    def load_task_language(self):
        raise NotImplementedError


@TaskLanguageLoaderRegistry.register
class ListFunctionsSyntheticLanguageLoader(TaskDataLoader):
    name = "list_functions_synthetic"

    def load_task_language(self):

        dataset_path = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS, DEFAULT_DATASET)
        tasks = retrieveJSONTasks(dataset_path)

        tasks = {TRAIN: tasks[:N_TRAIN], TEST: tasks[N_TRAIN:]}

        language = {}
        vocab = {}
        for split in tasks.keys():
            language[split] = {}
            vocab[split] = []
            for t in tasks[split]:
                desc = t.desc or t.name
                language[split][t.name] = [desc]
                vocab[split].extend(desc.split())

        return language, vocab
