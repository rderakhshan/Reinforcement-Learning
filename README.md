# RL LAB: Reinforcement Learning Algorithms

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00.svg)](https://www.tensorflow.org/)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-purple.svg)](https://github.com/astral-sh/uv)

## Project Overview

RL LAB is an educational Reinforcement Learning laboratory designed to explore and implement fundamental concepts and algorithms spanning tabular and deep reinforcement learning.

Rather than focusing solely on deep RL in complex environments, this codebase provides a solid foundation by implementing classical algorithms in custom Grid World environments (`Utils/grid_env.py`) before scaling up to Deep Q-Networks (DQN) and more advanced architectures.

This project is fully managed using `uv` for lightning-fast Python dependency management and runs cleanly on Python 3.11+.

---

## Implemented Algorithms

The codebase includes implementations spanning across multiple families of reinforcement learning algorithms located in the `src/back/` directory:

- **Value Iteration & Policy Iteration**: Dynamic programming methods for solving known MDPs.
- **Monte Carlo Methods**: Model-free episode-based learning (`MC_Basic`).
- **Temporal-Difference Learning**: Step-by-step model-free learning (e.g., Q-Learning, SARSA).
- **Stochastic Approximation**: Foundations for function approximation.
- **Value Function Approximation**: Scaling classical tabular methods to continuous or large state spaces.
- **DQN & DDQN**: Deep Q-Networks and Double Deep Q-Networks (PyTorch and TensorFlow implementations).
- **Policy Gradient & Actor-Critic**: Advanced architectures for continuous control and policy optimization.

---

## Installation

This project utilizes `uv` as its primary package manager.

```bash
# Clone the repository
git clone https://github.com/rderakhshan/Reinforcement-Learning-Q-Learning-Family-.git
cd "RL LAB"

# Install uv if you haven't already (https://docs.astral.sh/uv/)
# Under the project root, let uv automatically create the virtual environment and sync dependencies:
uv sync

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### Core Dependencies

- `Python 3.11+`
- `PyTorch` / `TensorFlow` (For Deep RL algorithms)
- `Gymnasium` (Environment support)
- `NumPy`, `Pandas`
- `Matplotlib`, `Plotly`, `OpenCV-Python` (Visualization & observation manipulation)
- `tqdm` (Progress bars for long training loops)

---

## Project Structure

```text
RL LAB/
├── src/
│   └── back/
│       ├── Actor Critic/
│       ├── DDQN/
│       ├── DQN/
│       │   ├── PT/                       # PyTorch Implementation Domain
│       │   └── TF/                       # TensorFlow Implementation Domain
│       ├── Monte Carlo Methods/          # Episodic Tabular RL
│       ├── Policy Gradient/
│       ├── Stochastic_approximation/
│       ├── Temporal-Difference learning/ # Q-Learning, SARSA
│       ├── Utils/                        # Core Environment configs (grid_env)
│       ├── Value Function Approximaton/
│       └── Value iteration and Policy iteration/
├── pyproject.toml                        # Project definitions & dependencies
├── uv.lock                               # Locked dependency resolution
└── README.md                             # You are here
```

---

## Quick Start Examples

You can run individual algorithmic scripts using the `uv run` command. For instance, to verify the Monte Carlo Basic properties and policy visualization on the grid environment:

```bash
uv run "src\back\Monte Carlo Methods\MC_Basic.py"
```

To run the classic Value Iteration dynamic programming approach:

```bash
uv run "src\back\Value iteration and Policy iteration\value iteration.py"
```

_Note: The custom `grid_env.py` (under Utils) provides an interactive Matplotlib rendering UI. Once a script's computation completes, a blocking window might open up visually mapping the learned policies and estimated State-Values onto the grid._

---

## Key Features

- **Cross-Framework Support**: Cleanly separates algorithmic logic between standard Tabular math, PyTorch, and TensorFlow paradigms.
- **Educational Annotations**: The underlying files are deeply documented with English & inline mathematical explanations for study.
- **Grid World Environment**: Custom UI mapping arrows for deterministic policy tracing.
- **Seamless Package Management**: Completely standardized under `pyproject.toml` specs for one-command replication via `uv`.

---

## Troubleshooting

1. **"ModuleNotFoundError: No module named 'grid_env'"**  
   _Solution_: The scripts are designed to dynamically map their relative paths utilizing `pathlib` and `sys.path.insert`. Execute the scripts utilizing `uv run "src\back\..."` from the root directory of the project, avoiding running them deep within relative subdirectories unless your IDE handles root-level execution paths for you.

2. **TensorFlow oneDNN warnings**
   _Solution_: This is automatically handled within most entry point scripts via `os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"`.

---

## License & Acknowledgements

This project was built iteratively to serve as an in-depth laboratory for internal experimentation with reinforcement learning concepts spanning the classic Sutton & Barto literature up to modern continuous control spaces.

---

## Contributing

We openly welcome community involvement and contributions! Please refer to our `CONTRIBUTING.md` (when available) for guidelines handling:

- Opening constructive bug reports via Issues.
- Local repository setup and integration.

---

## License

This collective iteration is openly published under the guidelines and agreements stipulated by the **[MIT License](LICENSE)** structure. See `LICENSE` inside the repository for extended documentation boundaries and provisions attached.

---

## Citation

If utilizing this codebase mapping alongside academic, diagnostic, or educational purposes simulating historic DeepMind parameters, please reference the original architects:

```bibtex
@article{mnih2015human,
  title={Human-level control through deep reinforcement learning},
  author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A and Veness, Joel and Bellemare, Marc G and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K and Ostrovski, Georg and others},
  journal={Nature},
  volume={518},
  number={7540},
  pages={529--533},
  year={2015},
  publisher={Nature Publishing Group}
}
```

## Acknowledgments

Deep reinforcement learning environments depend on monumental public frameworks:

- The authors and researchers of the **DeepMind** 2015 publication group defining Deep Q-learning paradigms.
- **Gymnasium (Farama Foundation)** & Open-Source legacy **OpenAI Gym** maintainers structuring emulation environments.
- Active contributors to the mathematical architectures mapping the **PyTorch** and **TensorFlow** deployment distributions.
