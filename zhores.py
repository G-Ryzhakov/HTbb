"""Script to run computations on zhores cluster.

Run this script as `python zhores.py`.
To check status, use "squeue"; to delete the running task do "scancel NUMBER".

Please, install packages within the environment before the run of this script:
$ module purge && module load python/anaconda3
$ conda activate && conda remove --name htde --all -y
$ conda create --name htde -y && source activate htde
$ conda install -n htde python=3.8 -y && pip install --upgrade pip
$ pip install teneva_opti==0.5.1

"""
import os
import subprocess
import sys


DIMENSION = 1024
OPTIONS = {
    'args': {
        'd': DIMENSION,
        'without_bs': True,
        'fold': f'result_func_appr_d{DIMENSION}'
    },
    'opts': {
        'env': 'htde',
        'file': 'run_func_appr',
        'days': 4,
        'hours': 0,
        'memory': 40,
        'out': f'result_func_appr_d{DIMENSION}',
        'gpu': False
    }
}
TASKS = {}
for s in [6, 7, 8, 9, 10]:
    TASKS[f'{s}-htde'] = {
        'args': {
            'seed_only': s,
            'postfix': f'_seed{s}',
        }
    }


def zhores(kind='main'):
    if kind == 'main':
        options, tasks = OPTIONS, TASKS
    else:
        raise NotImplementedError

    for task_name, task in tasks.items():
        opts = {**options.get('opts', {}), **task.get('opts', {})}

        text = '#!/bin/bash -l\n'

        text += f'\n#SBATCH --job-name={task_name}'

        fold = opts.get('out', '.')
        file = opts.get('out_name', f'zhores_out_{task_name}.txt')
        text += f'\n#SBATCH --output={fold}/{file}'
        os.makedirs(fold, exist_ok=True)

        d = str(opts['days'])
        h = str(opts['hours'])
        h = '0' + h if len(h) == 1 else h
        text += f'\n#SBATCH --time={d}-{h}:00:00'

        if opts['gpu']:
            text += '\n#SBATCH --partition gpu'
        else:
            text += '\n#SBATCH --partition mem'

        text += '\n#SBATCH --nodes=1'

        if opts['gpu']:
            text += '\n#SBATCH --gpus=1'

        mem = str(opts['memory'])
        text += f'\n#SBATCH --mem={mem}GB'

        text += '\n\nmodule purge'

        text += '\nmodule load python/anaconda3'

        if opts['gpu']:
            text += '\nmodule load gpu/cuda-11.3'

        env = opts['env']
        text += '\neval "$(conda shell.bash hook)"'
        text += f'\nsource activate {env}\n'

        args_list = task.get('args', {})
        if isinstance(args_list, dict):
            args_list = [args_list]

        for args in args_list:
            text += f'\nsrun python3 {opts["file"]}.py'
            args = {**options.get('args', {}), **args}
            for name, value in args.items():
                if isinstance(value, bool):
                    text += f' --{name}'
                elif value is not None:
                    text += f' --{name} {value}'

        text += '\n\nexit 0'

        with open(f'___zhores_run_{task_name}.sh', 'w') as f:
            f.write(text)

        prc = subprocess.getoutput(f'sbatch ___zhores_run_{task_name}.sh')
        os.remove(f'___zhores_run_{task_name}.sh')

        if 'command not found' in prc:
            print('!!! Error: can not run "sbatch"')
        else:
            print(prc)


if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'main'
    zhores(kind)
