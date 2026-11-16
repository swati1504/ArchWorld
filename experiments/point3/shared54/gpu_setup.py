"""Notebook setup variants, including CUDA 12.8 wheels for Blackwell GPUs."""
import textwrap


def gpu_stack(capabilities):
    if not capabilities:raise ValueError('Select a GPU runtime before installing model environments')
    blackwell=max(float(c) for c in capabilities)>=10
    return {'blackwell':blackwell,'eval_torch':'2.7.0' if blackwell else '2.3.1',
            'eval_vision':'0.22.0' if blackwell else '0.18.1',
            'eval_cuda':'12.8' if blackwell else '12.1',
            'sam_cuda':'12.8' if blackwell else '12.6'}


def setup_for(base,mode):
    if mode not in ('sam','evaluation'):raise ValueError('Unknown setup environment')
    def change(old,new):
        nonlocal base
        if base.count(old)!=1:raise ValueError('Setup template changed: '+old)
        base=base.replace(old,new)
    if mode=='sam':
        change('eval_ready = ready(EVAL_PY, eval_probe, vggt_dir, vggt_revision)','eval_ready = True')
        change("checked([EVAL_PY, '-c', eval_probe], 'Validating VGGT imports and GPU')",'pass # No evaluation environment in mask preparation.')
        change("[('evaluation', EVAL_PY), ('sam', SAM_PY)]","[('sam', SAM_PY)]")
    else:
        change('sam_ready = ready(SAM_PY, sam_probe, sam_dir, sam_revision)','sam_ready = True')
        change("checked([SAM_PY, '-c', sam_probe], 'Validating SAM imports and GPU')",'pass # Experiment notebooks only read masks.')
        change("[('evaluation', EVAL_PY), ('sam', SAM_PY)]","[('evaluation', EVAL_PY)]")
    change('"assert torch.__version__.startswith(\'2.3.1\'),torch.__version__;"',
           'f"assert torch.__version__.split(\'+\')[0]=={GPU_STACK[\'eval_torch\']!r},torch.__version__;"')
    change('"assert torch.cuda.is_available(),\'Select a GPU runtime\';"',
           'f"assert torch.version.cuda=={GPU_STACK[\'eval_cuda\']!r},torch.version.cuda;"\n'
           '              "assert torch.cuda.is_available(),\'Select a GPU runtime\';torch.ones(2,device=\'cuda\').sum().item();"')
    change('"assert torch.cuda.is_available(),\'SAM requires a GPU\'")',
           'f"assert torch.version.cuda=={GPU_STACK[\'sam_cuda\']!r},torch.version.cuda;"\n'
           '             "assert torch.cuda.is_available(),\'SAM requires a GPU\';torch.ones(2,device=\'cuda\').sum().item()")')
    change("['torch==2.3.1', 'torchvision==0.18.1', '--index-url', 'https://download.pytorch.org/whl/cu121']",
           "['torch=='+GPU_STACK['eval_torch'], 'torchvision=='+GPU_STACK['eval_vision'], '--index-url', 'https://download.pytorch.org/whl/cu'+GPU_STACK['eval_cuda'].replace('.','')]")
    change("'https://download.pytorch.org/whl/cu126'", "'https://download.pytorch.org/whl/cu'+GPU_STACK['sam_cuda'].replace('.','')")
    prefix=textwrap.dedent('''
        from gpu_setup import gpu_stack
        capabilities = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], text=True, timeout=15).strip().splitlines()
        GPU_STACK = gpu_stack(capabilities)
        if GPU_STACK['blackwell']:
            EVAL_PY = WORK/'envs/vggt_cu128/bin/python'
            SAM_PY = WORK/'envs/sam3_cu128/bin/python'
        print('GPU library profile:', GPU_STACK)
        print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True, timeout=15))
        from protocol import save
        save(OUTPUT/'gpu_setup.json', GPU_STACK)
    ''')
    return prefix+'\n'+base
