import importlib.util
from pathlib import Path
import sys
import types
from common import sha

def math_setup():
    site=Path('/scratch/project_462001687/yiminliu/ouro_motivation_accuracy/venv/lib/python3.12/site-packages')
    utility=site/'lm_eval/tasks/hendrycks_math/utils.py'
    assert sha(utility)=='8332e42f23c62043b23d74ee6d3934f997b209cd60a1ad45b7882d6b00929d5c'
    for name in ('lm_eval','lm_eval.tasks','lm_eval.tasks.hendrycks_math'):
        assert name not in sys.modules
        module=types.ModuleType(name);module.__path__=[str(site.joinpath(*name.split('.')))];sys.modules[name]=module
    spec=importlib.util.spec_from_file_location('lm_eval.tasks.hendrycks_math.utils',utility)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


def repeat(ids):
    seen={};flags=[]
    for i in range(max(0,len(ids)-7)):
        gram=tuple(ids[i:i+8]);flags.append(gram in seen and seen[gram]<=i-8);seen.setdefault(gram,i)
    return sum(flags[-256:])/len(flags[-256:]) if flags else 0.

