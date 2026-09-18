"""Reading a project's own optimiser parameter ranges."""
from pathlib import Path
from .core import read

def active_bounds(project):
    p=Path(project).with_suffix('')/'Model/3D/Model.jopt';out={}
    def visit(v):
        if isinstance(v,dict):
            if v.get('Active') in ('True',True) and 'Parameter Name' in v:
                try:out[v['Parameter Name']]=[float(v['Min Value']),float(v['Max Value'])]
                except (ValueError,KeyError):pass
            for item in v.values():visit(item)
        elif isinstance(v,list):
            for item in v:visit(item)
    if p.exists():visit(read(p))
    return out
