# HTBB
Hierarchical Tucker for Black Box gradient-free discrete approximation and optimization **HTBB**

## Installation

1. Install [anaconda](https://www.anaconda.com) package manager with [python](https://www.python.org) (version 3.8);

2. Create a virtual environment:
    ```bash
    conda create --name htbb python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate htbb
    ```

4. Install dependencies:
    ```bash
    pip install teneva_opti==0.5.3
    ```
    > When update `teneva_opti` version, please, do before: `pip uninstall teneva_opti -y`.

5. Install dependencies for benchmarks (it is optional now, since we run only analytic functions!):
    ```bash
    wget https://raw.githubusercontent.com/AndreiChertkov/teneva_bm/main/install_all.py && python install_all.py --env htde
    ```
    > You can then remove the downloaded file as `rm install_all.py`. In the case of problems with `scikit-learn`, uninstall it as `pip uninstall scikit-learn` and then install it from the anaconda: `conda install -c anaconda scikit-learn`.

6. Optionally delete virtual environment at the end of the work:
    ```bash
    conda activate && conda remove --name htbb --all -y
    ```


## Computations

1. Run the approximation problems as:
    ```bash
    python run_func_appr.py
    ```
    > The results (for $d = 256$) will be in the `result_func_appr` folder. You can use the flag `--show` to only present the saved computation results. For the case of higher dimensions (`d = 512` and `d = 1024`) we saved the results in the `result_func_appr_d[d]` folder. To show the results, please, run the script like `python run_func_appr.py --show --fold result_func_appr_d512 --without_bs`.


2. Run the optimization problems as:
    ```bash
    python run_func_opti.py
    ```
    > The results will be in the `result_func_opti` folder. You can use the flags `--with_no_calc` to only present the saved computation results.


## Authors

- [Gleb Ryzhakov](https://github.com/G-Ryzhakov) (Basic ideas; raw code for proof of concept)
- [Andrei Chertkov](https://github.com/AndreiChertkov) (Code speed rewriting & speed up; most of experiments; checking)
- [Artem Basharin](https://github.com/a-wernon) (PEPS part; testing)
- [Ivan Oseledets](https://github.com/oseledets) (Supervision)


---


> ✭__🚂  The stars that you give to **HTBB**, motivate us to develop faster and add new interesting features to the code 😃
