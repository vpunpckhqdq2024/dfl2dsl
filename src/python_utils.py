# Reference: https://github.com/langchain-ai/langchain/blob/4a07fba9f0f7d949c9c5cb057a4d09c7db1dfb42/libs/langchain/langchain/utilities/python.py

import collections
import copy
import logging
import math
import multiprocessing
from queue import Queue
import dill
import re
import types
from collections import Counter
import functools
import signal
import pickle
import os
import traceback

logger = logging.getLogger(__name__)

TIMEOUT = None  # Bug when using timeout / multiprocessing, so we disable it for now.

class TimeoutError(Exception):
    """Timeout error."""
    pass


class FailedExecutionResult:
    """Failed execution result."""
    def __init__(self, exception, traceback):
        self.exception = exception
        self.traceback = traceback

    def __repr__(self) -> str:
        return f"FailedExecutionResult({self.exception})"
    def __str__(self) -> str:
        return f"FailedExecutionResult({self.exception})"


class DilledExecutionResult:
    """Dilled execution result, to allow putting un-pickleable objects on a queue."""
    def __init__(self, result, should_dill=True):
        self.dilled_result = dill.dumps(result) if should_dill else result
    
    @property
    def result(self):
        return dill.loads(self.dilled_result)



def handler(signum, frame):
    raise TimeoutError("Function ran out of time.")

def is_pickleable(obj):
    """Check if an object can be pickled without error."""
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, AttributeError):
        return False

class PythonExecutor:
    """Simulates a standalone Python Executor."""

    def __init__(self, globals=None, locals=None):
        self.globals = globals if globals is not None else {}
        self.locals = locals if locals is not None else {}

        # Set up the signal handler.
        signal.signal(signal.SIGALRM, handler)

    @classmethod
    def worker(
        cls,
        command,
        globals,
        queue,
    ):
        try:

            # Update globals with module references
            globals["math"] = math
            globals["Counter"] = Counter
            globals["collections"] = collections
            globals["copy"] = copy
            globals["functools"] = functools

            local_namespace = {}
            exec(command, globals, local_namespace)
            result = local_namespace.get("result")
            if isinstance(result, types.GeneratorType):
                result = list(result)
            if isinstance(result, list):
                result = [
                    list(x) if isinstance(x, types.GeneratorType) else x for x in result
                ]
            if is_pickleable(result):
                queue.put(result)
            else:
                queue.put(DilledExecutionResult(result, should_dill=True))
        except BaseException as e:
            queue.put(FailedExecutionResult(e, traceback.format_exc()))

    def run(self, command, timeout = None):
        """Run command with own globals/locals and returns the result.
        Timeout after the specified number of seconds."""

        # Only use multiprocessing if we are enforcing a timeout
        if timeout is not None:
            queue = multiprocessing.Queue()

            # create a Process
            p = multiprocessing.Process(
                target=self.worker, args=(command, self.globals, queue)
            )

            # start it
            p.start()

            # wait for the process to finish or kill it after timeout seconds
            p.join(timeout)

            if p.is_alive():
                p.terminate()
                return None
        else:
            queue = Queue()
            # Serial execution timeout logic.
            # If reaches timeout, will return the TimeoutError.

            max_serial_time = 1
            signal.alarm(max_serial_time)
            self.worker(command, self.globals, queue)
            signal.alarm(0)

        # Get the result from the worker function
        res = queue.get()
        if isinstance(res, str) and "TimeoutError" in res:
            return None
        if isinstance(res, DilledExecutionResult):
            res = res.result

        return res


def extract_program(response):
    """Extracts the python code from LLM response."""
    # TODO: Remove test cases / print statements from string.
    pattern = r"```python\s*([\s\S]+?)\s*```"
    matches = re.findall(pattern, response)
    matches = [match for match in matches if "def" in match]
    if matches:
        return "\n".join(matches)
    return response


def extract_function_names(function_string):
    matches = re.findall(r"def (\w+)\(", function_string)
    return matches


def make_eval_cache_key(function_string, inp):

    # List domain, convert to tuple.
    if isinstance(inp, list) and (len(inp) == 0 or isinstance(inp[0], int)):
        inp = tuple(inp)

    return (function_string, inp)

EVAL_CACHE = {}

def execute_function(function_string, inputs, timeout=TIMEOUT, verbose=False, cache_path=None, globals=None, locals=None, ignore_cache=False):
    """Execute a function on a list of inputs. Return a list of outputs."""

    # TODO: Handle arguments better. Currently, just pass inputs[i] directly,
    # but for multiple arguments, this means inputs[i] must be formatted with
    # "*" and "**" for unpacking. Additionally, string inputs must be formatted
    # with quotes. This is not ideal. We should handle this better.

    # Load the eval cache once to save loading time.
    global EVAL_CACHE
    if cache_path and os.path.exists(cache_path) and not EVAL_CACHE:
        try:
            with open(cache_path, "rb") as f:
                EVAL_CACHE = pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            logging.warning("Eval pickle corrupted.")
            EVAL_CACHE = {}
    cache = EVAL_CACHE if not ignore_cache else {}

    outputs = []
    function_names = extract_function_names(function_string)
    try:
        fn_name = "fn" if "fn" in function_names else function_names[-1]
    except: # pylint: disable=bare-except
        return [None] * len(inputs)

    # Load from cache if already evaluated. Otherwise, evaluate and save to
    # cache.
    new_cache = {}
    executor = PythonExecutor(globals=globals)
    for inp in inputs:

        key = make_eval_cache_key(function_string, inp)
        if key in cache:
            outputs.append(cache[key])
            continue

        try:
            output = executor.run(f"{function_string}\nresult = {fn_name}({inp})", timeout=timeout)
        except BaseException as e:
            output = FailedExecutionResult(e, traceback.format_exc())
        if verbose:
            if output is None:
                print(f"Timeout on {inp} using:")
                print(function_string)
        outputs.append(output)

        new_cache[key] = output

    # Save the new cache to disk.
    cache.update(new_cache)
    if cache_path and new_cache:
        # Once overwrote the cache with empty by ^C, will not make that
        # mistake again.
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
        except KeyboardInterrupt:
            print("First dumping to file, then quitting.")
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            print("Dumped to file.")
            exit(0)

    return outputs

if __name__ == "__main__":
    fn = """def fn(lst):
    if len(lst) >= 3:
        return [lst[2] + math.sqrt(2)]
    else:
        return "Error: The input list must contain at least 3 elements.\""""
    
    inputs = [[2, 4, 3, 2]]
    outputs = execute_function(fn, inputs, globals={"math": math})
    print(outputs)

    from dreamcoder.program import Primitive, Program
    from dreamcoder.type import tint, arrow, baseType, tbool, t0, tstr
    import dill

    def to_f_string(primitives, prog_str):
        f_lines = ["def fn(inp):"]
        prim_str = "_ = " + str(primitives) + "\n"
        f_lines.append(prim_str)
        f_lines.append("p = Program.parse('" + prog_str + "')")

        f_lines.append("init_ = p.evaluate([])")
        f_lines.append("for inp in (inp if isinstance(inp, list) else [inp]):")
        f_lines.append("\tinit_ = init_(inp)")
        f_lines.append("return init_")

        f_lines = [f_lines[0] ]+ ["\t" + line for line in f_lines[1:]]
        return "\n".join(f_lines)

    int_exs = """[
        Primitive("if", arrow(tbool, t0, t0, t0), lambda b: lambda x: lambda y: x if b else y),
        Primitive("+", arrow(tint, tint, tint), lambda x: lambda y: x + y),
        Primitive("0", tint, 0),
        Primitive("is_even", arrow(tint, tbool), lambda x: x % 2 == 0),
        Primitive("and", arrow(tbool, tbool, tbool), lambda x: lambda y: x and y),
        Primitive("*", arrow(tint, tint, tint), lambda x: lambda y: x * y)
    ]"""

    p_raw = "(lambda (lambda (if (and (is_even $0) (is_even $1)) (+ $0 $1) (* $0 $1))))"
    globs = {
        "Program": Program, "Primitive": Primitive, "arrow": arrow, "tint": tint,
        "tbool": tbool, "t0": t0, "baseType": baseType}
    str_repr = to_f_string(int_exs, p_raw)

    inputs = [[5, 4], [8, 4]]
    globs["dill"] = dill

    outputs = execute_function(str_repr, inputs, globals=globs)
    print(outputs)
