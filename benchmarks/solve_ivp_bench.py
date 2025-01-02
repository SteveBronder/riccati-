# %%
import numpy as np
import pyriccaticpp as ric
import scipy.special as sp
from scipy.integrate import solve_ivp
import os
from pathlib import Path
import time
import polars as pl
from enum import Enum
import timeit
import itertools
import inspect
from typing import Any, Callable, Dict, List, Tuple


# %%
class Algo(Enum):
    """Enumeration of available algorithms for solving differential equations."""

    PYRICCATICPP = 1
    DOP853 = 2
    RK45 = 3
    RK23 = 4
    Radau = 5
    BDF = 6
    LSODA = 7

    def __str__(self) -> str:
        """Returns the name of the algorithm."""
        return str(self.name)


str(Algo.PYRICCATICPP)


class Problem(Enum):
    """Enumeration of available differential equation problems."""

    BREMER237 = 1
    BURST = 2
    AIRY = 3
    SCHRODINGER = 4
    FLAME_PROP = 5

    def __str__(self) -> str:
        """Returns the name of the problem."""
        return str(self.name)


def get(d: Dict, *args: Any) -> Any:
    """
    Recursively retrieves a value from a nested dictionary using a sequence of keys.

    Args:
      d (Dict): The dictionary to retrieve values from.
      *args: A sequence of keys to traverse the nested dictionaries.

    Returns:
      Any: The value retrieved from the nested dictionaries.

    Example:
      d = {'a': {'b': {'c': 1}}}
      get(d, 'a', 'b', 'c')  # Returns 1
    """
    if len(args) == 1:
        return d[args[0]]
    else:
        return get(d[args[0]], *args[1:])


# %%
class BaseProblem:
    """
    Base class for defining a problem to solve.

    Attributes:
      range (List[int]): The interval over which to solve the problem.
      y1 (float): The expected value at the end of the interval.
      relative_error (float): The acceptable relative error in the solution.
    """

    def __init__(self, start: int, end: int, y1: float, relative_error: float):
        """
        Initializes the BaseProblem with the given parameters.

        Args:
          start (int): The start of the interval.
          end (int): The end of the interval.
          y1 (float): The expected value at the end of the interval.
          relative_error (float): The acceptable relative error.
        """
        self.range = [start, end]
        self.y1 = y1
        self.relative_error = relative_error


## Bremer

# Self-reported accuracy of Bremer's phase function method on
# Eq. (237) in Bremer 2018.
# "On the numerical solution of second order ordinary
#  differential equations in the high-frequency regime".
dir_path = os.getcwd()
if dir_path.endswith("benchmarks"):
    reftable = "./data/eq237.csv"
else:
    reftable = "./benchmarks/data/eq237.csv"
try:
    bremer_ref = pl.read_csv(reftable, separator=",")
except FileNotFoundError:
    print("Current Directory is ", dir_path)
    print(
        "./data/eq237.csv was not found. This script should be run from the top level of the repository."
    )

bremer_ref = bremer_ref.with_columns(pl.lit(-1).alias("start"))
bremer_ref = bremer_ref.with_columns(pl.lit(1).alias("end"))


# %%
def get_args(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters keyword arguments to only include those accepted by the given function.

    Args:
      func (Callable): The function to inspect.
      kwargs (Dict[str, Any]): The dictionary of keyword arguments.

    Returns:
      Dict[str, Any]: A dictionary of keyword arguments accepted by the function.
    """
    return {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(func).args}


class Bremer(BaseProblem):
    """
    Represents the differential equation problem from Bremer (2018), Eq. (237).

    This class defines the problem parameters, functions, and initial conditions
    needed to solve the equation using different numerical methods.

    Attributes:
      lamb (float): Parameter λ in the differential equation.
    """

    def __init__(
        self, start: int, end: int, y1: float, relative_error: float, lamb: float
    ):
        """
        Initializes the Bremer problem with the given parameters.

        Args:
          start (int): The start of the interval.
          end (int): The end of the interval.
          y1 (float): The expected value at the end of the interval.
          relative_error (float): The acceptable relative error.
          lamb (float): Parameter λ in the differential equation.
        """
        super().__init__(start, end, y1, relative_error)
        self.lamb = lamb

    def w_gen(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Generates the function w(x) for the differential equation.

        Returns:
          Callable[[np.ndarray], np.ndarray]: The function w(x).
        """
        return lambda x: self.lamb * np.sqrt(1 - x**2 * np.cos(3 * x))

    def g_gen(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Generates the function g(x) for the differential equation.

        Returns:
          Callable[[np.ndarray], np.ndarray]: The function g(x), which is zero in this case.
        """
        return lambda x: np.zeros_like(x)

    def f_gen(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """
        Generates the function f(t, y) for use with numerical solvers.

        Returns:
          Callable[[float, np.ndarray], np.ndarray]: The function f(t, y).
        """

        def f(t: float, y: np.ndarray) -> np.ndarray:
            yp = np.zeros_like(y)
            yp[0] = y[1]
            yp[1] = -(self.lamb**2) * (1 - t**2 * np.cos(3 * t)) * y[0]
            return yp

        return f

    def yi_init(self) -> complex:
        """
        Provides the initial condition y(t0).

        Returns:
          complex: The initial value y(t0).
        """
        return complex(0.0)

    def dyi_init(self) -> complex:
        """
        Provides the initial condition y'(t0).

        Returns:
          complex: The initial derivative y'(t0).
        """
        return complex(self.lamb)

    def print_params(self) -> str:
        """
        Returns a string representation of the problem parameters.

        Returns:
          str: The problem parameters as a string.
        """
        return f"start={self.range[0]},end={self.range[1]},lamb={self.lamb}"

    def __str__(self) -> str:
        """
        Returns a string representation of the problem.

        Returns:
          str: The problem as a string.
        """
        return "Bremer237: " + self.print_params()


problem_dictionary = {Problem.BREMER237: {"class": Bremer, "data": bremer_ref}}

## Airy


class Airy(BaseProblem):
    """
    Represents the Airy differential equation problem.

    This class defines the problem parameters, functions, and initial conditions
    needed to solve the Airy equation using different numerical methods.
    """

    def __init__(self, start: int, end: int, y1: float, relative_error: float):
        """
        Initializes the Airy problem with the given parameters.

        Args:
          start (int): The start of the interval. QUESTION: why is this an integer? Should be float.
          end (int): The end of the interval. QUESTION: same as above.
          y1 (float): The expected value at the end of the interval.
          relative_error (float): The acceptable relative error.
        """
        super().__init__(start, end, y1, relative_error)

    def w_gen(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Generates the function w(x) for the Airy equation.

        Returns:
          Callable[[np.ndarray], np.ndarray]: The function w(x).
        """
        return lambda x: np.sqrt(x)

    def g_gen(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Generates the function g(x) for the Airy equation.

        Returns:
          Callable[[np.ndarray], np.ndarray]: The function g(x), which is zero in this case.
        """
        return lambda x: np.zeros_like(x)

    def f_gen(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """
        Generates the function f(t, y) for use with numerical solvers.

        Returns:
          Callable[[float, np.ndarray], np.ndarray]: The function f(t, y).
        """
        return lambda t, y: [t * y[1], y[0]]

    def yi_init(self) -> complex:
        """
        Provides the initial condition y(t0).

        Returns:
          complex: The initial value y(t0).
        """
        return complex(sp.airy(-self.range[0])[0] + 1j * sp.airy(-self.range[0])[2])

    def dyi_init(self) -> complex:
        """
        Provides the initial condition y'(t0).

        Returns:
          complex: The initial derivative y'(t0).
        """
        # NOTE: Hard coded init position as 1 here :(
        return complex(-sp.airy(-self.range[0])[1] - 1j * sp.airy(-self.range[0])[3])

    def print_params(self) -> str:
        """
        Returns a string representation of the problem parameters.

        Returns:
          str: The problem parameters as a string.
        """
        return f"start={self.range[0]},end={self.range[1]}"

    def __str__(self) -> str:
        """
        Returns a string representation of the problem.

        Returns:
          str: The problem as a string.
        """
        return "Airy: " + self.print_params()


airy_data = pl.DataFrame(
    {"start": [-1.0], "end": [1.0], "y1": [0.0], "relative_error": [1e-12]}
)
problem_dictionary[Problem.AIRY] = {"class": Airy, "data": airy_data}

## Stiff problem

class Stiff(BaseProblem):
    """
    Represents the differential equation problem y'' + (t + 21)*y' + 21*t*y = 0,
    on the interval [0, 200], with initial conditions y(0) = 0, y'(0) = 1.

    This class defines the problem parameters, functions, and initial conditions
    needed to solve the equation using different numerical methods.

    Attributes:
    """

    def __init__(
        self, start: int, end: int, y1: float, relative_error: float):
        """
        Initializes the Flame propagation problem with the given parameters.

        Args:
          start (int): The start of the interval.
          end (int): The end of the interval.
          y1 (float): The expected value at the end of the interval.
          relative_error (float): The acceptable relative error.
          lamb (float): Parameter λ in the differential equation.
        """
        super().__init__(start, end, y1, relative_error)

    def w_gen(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Generates the function w(x) for the differential equation.

        Returns:
          Callable[[np.ndarray], np.ndarray]: The function w(x).
        """
        return lambda x: np.sqrt(21.0 * x) 

    def g_gen(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Generates the function g(x) for the differential equation.

        Returns:
          Callable[[np.ndarray], np.ndarray]: The function g(x), which is zero in this case.
        """
        return lambda x: 0.5 * (21.0 + x)

    def f_gen(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """
        Generates the function f(t, y) for use with numerical solvers.

        Returns:
          Callable[[float, np.ndarray], np.ndarray]: The function f(t, y).
        """

        def f(t: float, y: np.ndarray) -> np.ndarray:
            yp = np.zeros_like(y)
            yp[0] = y[1]
            yp[1] = - (t + 21.0) * y[1] - 21.0 * t * y[0]
            return yp

        return f

    def yi_init(self) -> complex:
        """
        Provides the initial condition y(t0).

        Returns:
          complex: The initial value y(t0).
        """
        return complex(0.0)

    def dyi_init(self) -> complex:
        """
        Provides the initial condition y'(t0).

        Returns:
          complex: The initial derivative y'(t0).
        """
        return complex(1.0)

    def print_params(self) -> str:
        """
        Returns a string representation of the problem parameters.

        Returns:
          str: The problem parameters as a string.
        """
        return f"start={self.range[0]},end={self.range[1]},lamb={self.lamb}"

    def __str__(self) -> str:
        """
        Returns a string representation of the problem.

        Returns:
          str: The problem as a string.
        """
        return "Stiff: " + self.print_params()

# Not sure what to write here
#stiff_data = pl.DataFrame(
#    {"start": [0.0], "end": [200.0], "y1": [0.0], "relative_error": [1e-12]}
#)

problem_dictionary = {Problem.STIFF: {"class": Stiff} #, "data": bremer_ref}} # Do we need this?




## Burst


def gen_problem(problem: Any, data: pl.DataFrame):
    """
    Generator function to create problem instances from data.

    Args:
      problem: The problem class to instantiate.
      data (pl.DataFrame): A dataframe containing problem parameters.

    Yields:
      Any: Instances of the problem class initialized with parameters from the data.
    """
    for x in data.iter_rows(named=True):
        yield problem(**x)


##
# %%
def construct_riccati_args(
    problem: BaseProblem, eps: float, epsh: float, n: int
) -> Dict[str, Any]:
    """
    Constructs the arguments required for solving the problem using pyriccaticpp.

    Args:
      problem (BaseProblem): The problem instance.
      eps (float): The epsilon parameter for the solver.
      epsh (float): The epsilon_h parameter for the solver.
      n (int): The parameter n for the solver.

    Returns:
      Dict[str, Any]: A dictionary containing the arguments for the solver.
    """
    return {
        "print_args": {"n": n, "p": n, "epsh": epsh},
        "init_args": (problem.w_gen(), problem.g_gen(), 8, max(32, n), n, n),
        "solver_args": {
            "xi": problem.range[0],
            "xf": problem.range[1],
            "yi": problem.yi_init(),
            "dyi": problem.dyi_init(),
            "eps": eps,
            "epsilon_h": epsh,
            "hard_stop": True,
        },
    }


def construct_solve_ivp_args(
    problem: BaseProblem, method: str, rtol: float, atol: float
) -> Dict[str, Any]:
    """
    Constructs the arguments required for solving the problem using scipy's solve_ivp.

    Args:
      problem (BaseProblem): The problem instance.
      method (str): The method to use for solving.
      rtol (float): The relative tolerance.
      atol (float): The absolute tolerance.

    Returns:
      Dict[str, Any]: A dictionary containing the arguments for solve_ivp.
    """
    return {
        "fun": problem.f_gen(),
        "t_span": problem.range,
        "y0": [problem.yi_init(), problem.dyi_init()],
        "method": str(method),
        "rtol": rtol,
        "atol": atol,
    }


def construct_algo_args(
    algo: Algo, problem: BaseProblem, args: Tuple[Any, ...]
) -> Dict[str, Any]:
    """
    Constructs the algorithm-specific arguments based on the chosen algorithm.

    Args:
      algo (Algo): The algorithm to use.
      problem (BaseProblem): The problem instance.
      args (Tuple[Any, ...]): Additional arguments required for the algorithm.

    Returns:
      Dict[str, Any]: A dictionary containing the arguments for the algorithm.
    """
    match algo:
        case Algo.PYRICCATICPP:
            # Here it is eps, epsh, n as first 3 of args
            return construct_riccati_args(problem, *args)
        case _:
            return construct_solve_ivp_args(problem, str(algo), *args)


## Start Benchmark
epss, epshs, ns = [1e-12, 1e-6], [1e-13, 1e-9], [35, 20]
atol = [1e-14, 1e-6]
# rtol = [1e-3, 1e-6]
algorithm_dict = {  # Algo.LSODA: {"args":[epss, atol], "iters": 1},
    Algo.BDF: {"args": [epss, atol], "iters": 1},
    Algo.Radau: {"args": [epss, atol], "iters": 1},
    Algo.RK45: {"args": [epss, atol], "iters": 1},
    Algo.DOP853: {"args": [epss, atol], "iters": 1},
    Algo.PYRICCATICPP: {"args": [epss, epshs, ns], "iters": 1000},
}


def benchmark(
    problem_info: BaseProblem, algo_args: Dict[str, Any], N: int = 1000
) -> pl.DataFrame:
    """
    Benchmarks a numerical method on a given problem.

    Args:
      problem_info (BaseProblem): The problem instance.
      algo_args (Dict[str, Any]): The arguments for the algorithm.
      N (int, optional): The number of times to run the benchmark. Defaults to 1000.

    Returns:
      pl.DataFrame: A dataframe containing the timing and error results.
    """
    match algo_args["method"]:
        case Algo.PYRICCATICPP:
            info = ric.Init(*algo_args["method_args"]["init_args"])
            solver_args = algo_args["method_args"]["solver_args"]
            algo_args["method_args"]["init_step_args"] = (
                info,
                solver_args["xi"],
                solver_args["xf"],
                solver_args["epsilon_h"],
            )
            init_step = ric.choose_nonosc_stepsize(
                *algo_args["method_args"]["init_step_args"]
            )
            solver_args["init_stepsize"] = init_step
            runtime = (
                timeit.timeit(lambda: ric.evolve(info=info, **solver_args), number=N)
                / N
            )
            _, ys, _, _, _, _, _, _, _ = ric.evolve(info=info, **solver_args)
            ys = np.array(ys)
            # Compute statistics
            yerr = np.abs((problem_info.y1 - ys[-1]) / problem_info.y1)
            print_args = algo_args["method_args"]["print_args"]
            timing_df = pl.DataFrame(
                {
                    "eq_name": algo_args["function_name"],
                    "method": str(algo_args["method"]),
                    "eps": algo_args["eps"],
                    "relerr": yerr,
                    "walltime": runtime,
                    "errlessref": bool(yerr < problem_info.relative_error),
                    "problem_params": problem_info.print_params(),
                    "params": f"n={print_args['n']};p={print_args['n']};epsh={print_args['epsh']}",
                }
            )
            return timing_df
        # All Python IVPs use the same scheme
        case _:
            runtime = (
                timeit.timeit(lambda: solve_ivp(**algo_args["method_args"]), number=N)
                / N
            )
            sol = solve_ivp(**algo_args["method_args"])
            yerr = np.abs((sol.y[0, -1] - problem_info.y1) / problem_info.y1)
            timing_df = pl.DataFrame(
                {
                    "eq_name": algo_args["function_name"],
                    "method": str(algo_args["method"]),
                    "eps": algo_args["eps"],
                    "relerr": yerr,
                    "walltime": runtime,
                    "errlessref": bool(yerr < problem_info.relative_error),
                    "problem_params": problem_info.print_params(),
                    "params": f"rtol={algo_args['method_args']['rtol']};atol={algo_args['method_args']['atol']}",
                }
            )
            return timing_df


timing_dfs = []
for problem_key, problem_item in problem_dictionary.items():
    for algo, algo_params in algorithm_dict.items():
        for algo_iter in list(itertools.product(*algo_params["args"])):
            for problem_info in gen_problem(
                problem_item["class"], problem_item["data"]
            ):
                algo_args = {
                    "method": algo,
                    "function_name": str(problem_key),
                    # Eps must always be first arg of tuple
                    "eps": algo_iter[0],
                    "method_args": construct_algo_args(algo, problem_info, algo_iter),
                }
                print("Running ", str(algo), "on", str(problem_key))
                print("\tProblem Info: ", problem_info)
                match algo_args["method"]:
                    case Algo.PYRICCATICPP:
                        print_args = algo_args["method_args"]["print_args"]
                        print(
                            "\tArgs: ",
                            f"n={print_args['n']};p={print_args['n']};epsh={print_args['epsh']}",
                        )
                    case _:
                        print(
                            "\tArgs: ",
                            f"rtol={algo_args['method_args']['rtol']};atol={algo_args['method_args']['atol']}",
                        )
                timing_dfs.append(
                    benchmark(problem_info, algo_args, N=algo_params["iters"])
                )

algo_times = pl.concat(timing_dfs, rechunk=True, how="vertical_relaxed")
algo_times
dir_path = os.getcwd()
if dir_path.endswith("benchmarks"):
    output_path = "./output/solve_ivp_times.csv"
else:
    output_path = "./benchmarks/output/solve_ivp_times.csv"
algo_times.write_csv(output_path)

# %%
