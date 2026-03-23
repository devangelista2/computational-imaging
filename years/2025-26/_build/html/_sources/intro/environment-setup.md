# Environment Setup

This page explains how to set up a working Python environment for the course from scratch, even on a machine where Python is not yet installed. The tool we will use is [uv](https://docs.astral.sh/uv/), a modern Python package and project manager.

## Why uv

Traditional Python tooling has a well-known fragmentation problem. You need Python installed before you can install a package manager, you need a package manager to install a virtual-environment tool, and the interaction between system Python, pip, and virtual environments is a recurring source of confusion.

`uv` collapses this stack into a single binary. It can:

- **download and manage Python** versions for you, without touching the system Python;
- **create virtual environments** in one command;
- **resolve and install packages** an order of magnitude faster than pip, using a Rust-based resolver;
- produce a **lockfile** (`uv.lock`) that pins every dependency to an exact version, making your environment reproducible on any machine.

For a course like this one, where code is executed interactively and reproducibility matters, `uv` removes a whole class of "it works on my machine" problems.

```{info}
A **virtual environment** in Python is an isolated environment on your computer, where you can run and test your Python projects. It allows you to manage project-specific dependencies without interfering with other projects or the original Python installation. It is recommended to always work within virtual environments, to avoid breaking the main Python environment due to conflicts caused by different dependencies.
```

## Installing uv

`uv` is distributed as a standalone binary. You do not need Python, pip, or any other tool to install it.

**On macOS and Linux**, open a terminal and run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows**, open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you already have `pip` installed, you can easily install `uv` as

```bash
pip install uv
```

After the installer finishes, restart your terminal (or open a new one) so that the `uv` command is available on your path. Verify the installation with:

```bash
uv --version
```

If you need help regarding the installation of `uv`, please check the official website at [](https://docs.astral.sh/uv/getting-started/installation/).

## Creating the Course Folder

During this course you will write code to run experiments. I suggest to keep all the files produced for this course in a single folder called `computational-imaging`. Create it wherever you prefer and enter it:

```bash
mkdir computational-imaging
cd computational-imaging
```

Every command in the rest of this guide should be run from inside this folder.

## Initializing the uv Project

Initialize a new `uv` project in the current directory:

```bash
uv init .
```

This creates a `pyproject.toml` file that describes the project and its dependencies, and a `.python-version` file that pins the Python version. `uv` will automatically download that version of Python if it is not already present on your system.

If you want to target a specific Python version explicitly (Python 3.12 is recommended for this course), run:

```bash
uv python pin 3.12
```

## Installing the Required Packages

The course notebooks use the following external libraries:

| Package | Purpose |
|---|---|
| `torch` | Neural networks and automatic differentiation (PyTorch) |
| `torchvision` | Image datasets, transforms, and pretrained models |
| `numpy` | Numerical arrays and linear algebra |
| `matplotlib` | Plotting and visualisation |
| `Pillow` | Image file I/O |
| `scipy` | Scientific computing and signal processing |
| `scikit-image` | Image processing algorithms |
| `scikit-learn` | Classical machine learning utilities |
| `numba` | JIT compilation for numerical code |
| `pandas` | Tabular data handling |

Install them all in one command:

```bash
uv add torch torchvision numpy matplotlib Pillow scipy scikit-image scikit-learn numba pandas jupyterlab
```

`uv` will resolve a compatible set of versions, download the packages, and write the exact versions to `uv.lock`. This step only needs to be done once per machine.

```{note}
If your machine has an NVIDIA GPU and you want to enable GPU acceleration, replace `torch torchvision` in the command above with the appropriate CUDA-enabled wheels. For example, for CUDA 12.4:

    uv add torch torchvision --index-url https://download.pytorch.org/whl/cu124

Check the [PyTorch installation page](https://pytorch.org/get-started/locally/) for the correct URL for your CUDA version.
```

## The IPPy Package

Several algorithms in the course use **IPPy**, a small imaging library developed specifically for this course, which you should've already encountered during the previous Module. `IPPy` is available for download via Github, simply running the command:

```bash
git clone https://github.com/devangelista2/IPPy.git
```

By running this command inside your `computational-imaging` directory, it should place an `IPPy` folder inside the project itself. However, to be able to run functions, you should extract the inner `IPPy` folder (which contains the actual functionalities), and place it in the root `computational-imaging` folder.

To avoid possible conficts with different versions of `IPPy`, a static version of the package is uploaded on Virtuale and available for download.

Your directory should look like this:

```
computational-imaging/
├── IPPy/
│   ├── __init__.py
│   ├── operators.py
│   ├── solvers.py
│   ├── nn/
│   └── utilities/
├── pyproject.toml
├── uv.lock
└── ... (your notebooks and scripts)
```

Because `IPPy` sits in the same directory as your notebooks, Python can import it directly without any installation step.

## Syncing on a New Machine

If you move your `computational-imaging` folder to another machine (or clone it from a version control system), you do not need to repeat the `uv add` step. Run:

```bash
uv sync
```

`uv` will read `uv.lock` and recreate the exact same environment, including the same Python version, in seconds.
