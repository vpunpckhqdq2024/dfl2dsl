"""File to call pypy on. This file is called by the compiled driver to run the function in a separate process."""

import dill
import sys
import time
import traceback
import pickle as pickle
import os
import mock


if __name__ == "__main__":

    sys.setrecursionlimit(10000)
    repo_root = os.path.join(os.path.dirname(__file__), os.path.pardir)
    sys.path.append(repo_root)

    from dreamcoder.utilities import eprint

    # Make dummy modules for modules that cannot be installed in pypy.
    # Enumeration, the only place where pypy should be called, does not use these modules.
    # TODO: requirements.txt for pypy.
    sys.modules["torch"] = mock.MagicMock()
    sys.modules["torch.nn"] = mock.MagicMock()
    sys.modules["torch.nn.functional"] = mock.MagicMock()
    sys.modules["torch.autograd"] = mock.MagicMock()
    sys.modules["torch.nn.utils.rnn"] = mock.MagicMock()
    sys.modules["seaborn"] = mock.MagicMock()

    start = time.time()
    request = dill.load(sys.stdin.buffer)
    dt = time.time() - start
    if dt > 1:
        eprint(
            "(compiled driver warning: SLOW) Compiled driver unpacked the message in time",
            dt)

    response = (False, None)
    try:
        start = time.time()
        f = request["function"]
        result = f(*request["arguments"],
                   **request["keywordArguments"])
        response = (True, result)
    except Exception as e:  # pylint: disable=broad-except
        eprint(f"Exception thrown in pypy process for %s: {f.__name__}")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
    finally:
        start = time.time()
        pickle.dump(response, sys.stdout.buffer)
        dt = time.time() - start
        if dt > 1:
            eprint(
                "(compiled driver warning: SLOW) Compiled driver packed the message in time",
                dt)
