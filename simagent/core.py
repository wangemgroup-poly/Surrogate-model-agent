from __future__ import annotations
import json,hashlib,os,datetime,contextlib,re
from pathlib import Path
import numpy as np

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def signature(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
IDENTITY_EXCLUDED=('simulation_budget','cpus','timeout_minutes','source_sha256','geometry_sha256','fixed_parameters_sha256')
def config_signature(c):
    """Identity of targets, parameters and method; budget, CPUs, timeout and source hashes may change without invalidating models or frozen batches."""
    return signature({k:v for k,v in c.items() if k not in IDENTITY_EXCLUDED})
def event(task,message,**details):
    record=dict(time=now(),message=message,**details);print(message,flush=True)
    with (Path(task)/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
@contextlib.contextmanager
def lock(task):
    p=Path(task);p.mkdir(parents=True,exist_ok=True)
    f=(p/'.agent.lock').open('a+b');f.seek(0);f.write(b'1');f.flush();f.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:f.close();raise RuntimeError('此任务已有Agent操作在运行。')
    try:yield
    finally:f.close()

def validate_config(c):
    assert c.get('schema_version')==1,'配置schema_version必须为1'
    assert c.get('parameters') and c.get('metrics'),'必须定义参数和指标'
    for name,b in c['parameters'].items():
        assert re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',name),f'不支持参数名{name}'
        assert len(b)==2 and np.isfinite(b).all() and b[0]<b[1],f'参数范围无效{name}'
    assert len({m['name'] for m in c['metrics']})==len(c['metrics'])
    for m in c['metrics']:
        assert m['kind'] in ['curve','sll_phi_cut'],'指标kind仅支持curve/sll_phi_cut'
        if m['kind']=='curve':
            assert m['reduce'] in ['max','min','ripple','at']
            assert m.get('transform','real') in ['real','db20','abs']
            assert m.get('tree'),'curve需要CST结果树路径'
            if 'band' in m:assert len(m['band'])==2 and m['band'][0]<m['band'][1]
        if 'limit' in m:
            assert m['op'] in ['<=','<','>=','>'];assert np.isfinite(m['limit'])
        assert m.get('scale',1)>0
    assert any('limit' in m for m in c['metrics']),'至少一个性能约束'
    assert c.get('solver','frequency')=='frequency','首版CST执行器支持频域求解器'
    assert 0<c.get('trust_radius',.05)<=1
    assert c.get('proposal_method','random_pool') in ['random_pool','optimize','sb_sadea'],'proposal_method仅支持random_pool/optimize/sb_sadea'
    assert all(m.get('penalty_weight',1)>0 for m in c['metrics']),'penalty_weight必须为正'
    assert c.get('sb_sadea',{}).get('surrogate','gp') in ['gp','bnn'],'sb_sadea.surrogate仅支持gp/bnn'
    assert c.get('search_domain','trust_region') in ['trust_region','full'],'search_domain仅支持trust_region/full'
    assert c.get('conservative_sigma',1)>=0,'conservative_sigma不能为负'
    mat=c.get('materialize',{})
    assert isinstance(mat,dict) and mat.get('on','improvement') in ['improvement','pass','never'],'materialize.on仅支持improvement/pass/never'
    assert isinstance(mat.get('max',3),int) and mat.get('max',3)>=0,'materialize.max必须为非负整数'
    assert c.get('extrapolation_ratio',2)>0 and c.get('disagreement_limit',2)>0,'extrapolation_ratio/disagreement_limit必须为正'
    for key in ('geometry_sha256','fixed_parameters_sha256'):
        if c.get(key) is not None:assert isinstance(c[key],str) and len(c[key])==64,key+'无效'
    assert isinstance(c.get('simulation_budget',0),int) and c.get('simulation_budget',0)>=0,'总仿真预算必须为非负整数'
    assert 1<=c.get('cpus',16)<=1024 and c.get('timeout_minutes',45)>0,'CPU数或等待时限无效'
    return c
def arrays(rows,c):
    names=list(c['parameters']);metrics=[m['name'] for m in c['metrics']]
    x=np.array([[r['parameters'][n] for n in names] for r in rows],float)
    y=np.array([[r['metrics'][n] for n in metrics] for r in rows],float)
    return x,y
def normalize(x,c):
    b=np.array(list(c['parameters'].values()));return (np.asarray(x)-b[:,0])/(b[:,1]-b[:,0])
def denormalize(z,c):
    b=np.array(list(c['parameters'].values()));return b[:,0]+np.asarray(z)*(b[:,1]-b[:,0])
def margins(y,c):
    y=np.atleast_2d(y);out=[]
    for i,m in enumerate(c['metrics']):
        if 'limit' not in m:continue
        d=m['limit']-y[:,i] if m['op'] in ['<=','<'] else y[:,i]-m['limit']
        out.append(d/m.get('scale',1))
    return np.column_stack(out)
def passes(y,c):
    y=np.atleast_2d(y);p=np.ones(len(y),bool)
    for i,m in enumerate(c['metrics']):
        if 'limit' not in m:continue
        v=y[:,i];t=m['limit'];p &= {'<=':v<=t,'<':v<t,'>=':v>=t,'>':v>t}[m['op']]
    return p
def loss(y,c):return -margins(y,c).min(1)
def fitness(y,c):
    """Penalty fitness of Liu et al. (IEEE TAP 2022, eqs. 16-17): weighted sum of constraint violations, 0 when all hold."""
    y=np.atleast_2d(y);f=np.zeros(len(y))
    for i,m in enumerate(c['metrics']):
        if 'limit' not in m:continue
        v=y[:,i]-m['limit'] if m['op'] in ['<=','<'] else m['limit']-y[:,i]
        f+=m.get('penalty_weight',1)*np.maximum(v,0)
    return f
def ingest(task,incoming,provenance):
    task=Path(task);c=read(task/'config.json');db=read(task/'data.json');names=list(c['parameters']);rejected=[];added=0;repeats=0
    keys={tuple(round(r['parameters'][n],11) for n in names):r for r in db['rows']}
    for r in incoming:
        try:
            x,y=arrays([r],c)
            assert np.isfinite(x).all() and np.isfinite(y).all(),'非有限数值'
            assert ((normalize(x,c)>=-1e-9)&(normalize(x,c)<=1+1e-9)).all(),'参数超出任务范围'
            q=dict(id=r.get('id',r.get('sample_id',str(len(db['rows'])+1))),parameters=dict(zip(names,x[0].tolist())),metrics=dict(zip([m['name'] for m in c['metrics']],y[0].tolist())),provenance=r.get('provenance',provenance),imported=now())
            key=tuple(round(v,11) for v in x[0])
            if key in keys:
                # Identical reimport is idempotent; numerical repeats retained separately.
                if q['metrics']!=keys[key]['metrics'] and not any(t['parameters']==q['parameters'] and t['metrics']==q['metrics'] for t in db['repeats']):db['repeats'].append(q);repeats+=1
            else:db['rows'].append(q);keys[key]=q;added+=1
        except (KeyError,ValueError,AssertionError) as exc:rejected.append(dict(id=r.get('id',r.get('sample_id')),reason=str(exc)))
    write(task/'data.json',db);write(task/'last_import.json',dict(added=added,repeats=repeats,rejected=rejected))
    event(task,f'数据导入：新增{added}组，新增重复记录{repeats}条，拒绝{len(rejected)}条；有效参数组共{len(db["rows"])}组。')
    return added
