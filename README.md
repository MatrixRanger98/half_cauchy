# Half-Cauchy and EHMP for Integrating Studies


This repo provides code for the paper:

[**A Heavily Right Strategy for Integrating Dependent Studies in Any Dimension**](https://arxiv.org/abs/2501.01065) 

by *Tianle Liu*, *Xiao-Li Meng*, and *Natesh S. Pillai*, 2025.


To install the dependencies, run

```Bash
pip install -r requirements.txt
```

If using the package manager `uv`, run

```Bash
uv pip install -r requirements.txt
```

The folder `src` contains relevant functions and classes of our methods and `scripts` contains experiments. For example, to replicate the simulation for 1d normal studies, execute the following command under the directory of this repo:

```Bash
export PYTHONPATH=$(pwd)
python ./scripts/simulation_1_1d_coverage.py
```