# Install

This page holds the install details that the [README](https://github.com/yzhao062/auditable/blob/main/README.md#install) leaves out: the extras with their dependencies, the graph methods on Colab and on torch patch releases, running the examples from a clone, and the platform checks.

## Extras

| Command | Adds | Needed for |
|---|---|---|
| `pip install auditable` | the registry and the configuration rules, with no third-party dependencies | `config` rules, the step-by-step and binary-search LLM judges, listing the registry |
| `pip install "auditable[tabular-ad]"` | NumPy, scikit-learn, PyOD, NetworkX | run, step, and online methods on trace features |
| `pip install "auditable[bench]"` | the `tabular-ad` packages, plus agentpile with its Hugging Face loaders | the leaderboards (`python -m auditable.bench`) |
| `pip install "auditable[agentpile]"` | agentpile with its Hugging Face loaders, and NumPy | `from_agentpile`, `configs_from_agentpile`, `load_traces` |
| `pip install "auditable[graph-ad]"` | the `tabular-ad` packages, plus torch, PyTorch Geometric, and PyGOD | the graph methods (PyGOD detectors, GUARDIAN-Lite, G-Safeguard-Lite) |

The structural graph analyses (`analyze_run`, `analyze_plan`) need `auditable[graph]`. Further extras cover the capture integrations (`langgraph`, `langchain`, `mcp`), the audit report PDF (`report`), and the real-LLM example (`llm`). The LLM judges take any `prompt -> reply` function as their backend, and `OpenAICompatibleBackend` reaches an OpenAI-compatible endpoint with the standard library alone.

The examples, tests, and benchmark harnesses are in the repository and not in the wheel. To run them, install from a clone:

```bash
pip install -e ".[dev,tabular-ad,bench]"
```

## Graph Methods

The graph methods, and so the graph rows of the leaderboards, need Python 3.10 or newer, because torch 2.12.1 and `pyg-lib` publish no wheels for 3.9. They also need a sampling backend built for the installed torch, which pip cannot resolve from the `graph-ad` extra. Install torch and the matching `pyg-lib` wheel first; the steps are the same on Linux, macOS on Apple silicon, and Windows:

```bash
pip install "torch==2.12.1" --index-url https://download.pytorch.org/whl/cpu
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.12.1+cpu.html
pip install "auditable[graph-ad]"
python -c "import torch_geometric.typing as t; assert t.WITH_PYG_LIB"
```

Without the backend the graph rows of a board are listed as skipped, with the extra named, and the rest of the board is still scored.

The PyGOD methods run on the CPU (`gpu=-1`) even when a GPU is present.

### Colab and Other Preinstalled torch

Where torch is already installed, as on Colab, keep it. Skip the first command, and in the `pyg-lib` URL replace `2.12.1+cpu` with the installed version, which `python -c "import torch; print(torch.__version__)"` prints. The pinned CPU build would replace Colab's torch and break its preinstalled torchvision.

### Patch Releases Without pyg-lib Wheels

PyG does not publish wheels for every patch release. When pip finds no `pyg_lib` at the URL for the installed version, use the `.0` release with the same CUDA tag. Colab's torch `2.14.1+cu130` takes `https://data.pyg.org/whl/torch-2.14.0+cu130.html`.

### Platform Checks

The graph-ad install was checked for 0.4.0 on a Colab T4 runtime with its torch 2.14.1+cu130, and for 0.3.0 on Colab CPU and T4 runtimes with Colab's torch and on Linux aarch64 (DGX Spark) with the pinned one. A weekly workflow (`bench-platforms.yml`) repeats the pinned steps from a clean install on Linux, macOS, and Windows and checks three boards against their golden files.
