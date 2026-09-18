"""Read-only result importer and isolated CST frequency-domain job adapter."""
import os,sys,sqlite3,struct,shutil,subprocess,time,uuid,hashlib,json
from pathlib import Path
import numpy as np
from .core import read,write,digest,now

GEOMETRY_FILES=('Model/3D/Model.mod','Model/3D/ModelHistory.json','Model/simulationproperties.docstore')

def ascii_path(p):
    """CST's Python result library cannot open non-ASCII paths; reach the file through a stable junction to its directory."""
    p=Path(p)
    if str(p).isascii() or os.name!='nt':return p
    parent=p.parent.resolve();alias=Path(r'C:\Users\Public\Documents')/('simagent_alias_'+hashlib.sha256(str(parent).lower().encode()).hexdigest()[:10])
    if not alias.exists():subprocess.run(['cmd','/c','mklink','/J',str(alias),str(parent)],check=True,capture_output=True)
    assert (alias/p.name).exists(),'目录联接未生效：'+str(alias)
    return alias/p.name

def long_path(p):
    """Windows extended-length form: CST result trees nest deeply and a backup copy easily passes MAX_PATH (260)."""
    s=str(Path(p).resolve())
    return '\\\\?\\'+s if os.name=='nt' and not s.startswith('\\\\?\\') else s

def has_geometry(project):return (Path(project).with_suffix('')/GEOMETRY_FILES[0]).exists()
def geometry_signature(project,files=GEOMETRY_FILES):
    """Identity of the modelled structure and solver properties; saving further parametric runs into the project does not change it.
    Pass GEOMETRY_FILES[:2] to compare structure only (agent job copies carry altered solver properties)."""
    base=Path(project).with_suffix('');parts=[]
    for rel in files:
        f=base/rel;parts.append(rel+':'+(digest(f) if f.exists() else 'missing'))
    assert 'missing' not in parts[0],'找不到Model/3D/Model.mod，无法识别结构：'+str(project)
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()
def literal_parameters(project):
    """Numeric parameter values stored in the project (expressions excluded). Fixed dimensions live here, not in Model.mod."""
    f=Path(project).with_suffix('')/'Model/Parameters.json';out={}
    if not f.exists():return out
    for v in read(f).get('parameters',[]):
        try:out[v['name']]=float(v['expr'])
        except (ValueError,KeyError,TypeError):pass
    return out
def fixed_parameters_signature(project,free):
    """Values of every numeric parameter the task does not optimise; a changed fixed dimension is a different structure."""
    free=set(free);lit=literal_parameters(project)
    return hashlib.sha256(json.dumps({k:round(v,12) for k,v in sorted(lit.items()) if k not in free},sort_keys=True).encode()).hexdigest()
def structure_differences(a,b,free=()):
    """Reasons two projects cannot share full-wave data. Identical geometry history is not enough: two versions can build
    the same history with different fixed dimensions (e.g. 12-slot V3 b=7.5 vs V5_3 b=7.8)."""
    out=[]
    if geometry_signature(a)!=geometry_signature(b):out.append('几何或求解属性文件不同（Model.mod、ModelHistory.json、simulationproperties.docstore）')
    la,lb=literal_parameters(a),literal_parameters(b);names=(set(la)|set(lb))-set(free)
    only=sorted(n for n in names if (n in la)!=(n in lb))
    if only:out.append('固定参数的定义不同：'+'、'.join(only))
    diff=[n for n in sorted(names) if n in la and n in lb and abs(la[n]-lb[n])>1e-9]
    if diff:out.append('固定参数取值不同：'+'、'.join(f'{n}({la[n]:g}≠{lb[n]:g})' for n in diff))
    return out
def same_structure(a,b,free=()):return not structure_differences(a,b,free)
def check_source(c,source=None,file_sha256=None):
    """Tasks with a registered geometry accept a source that only gained saved runs; older tasks fall back to the .cst hash.
    When fixed parameters are registered, an edited fixed dimension in the source is rejected as well."""
    source=Path(source or c['project'])
    if c.get('geometry_sha256'):assert geometry_signature(source)==c['geometry_sha256'],'源工程的几何或求解属性已改变；请为新版本新建任务'
    else:assert digest(source)==(file_sha256 or c['source_sha256']),'源CST文件自任务创建后改变；请为新版本新建任务'
    if c.get('fixed_parameters_sha256'):assert fixed_parameters_signature(source,c['parameters'])==c['fixed_parameters_sha256'],'源工程中未优化的固定参数取值已改变（结构不同）；请为新版本新建任务'

def module(c,project=None):
    base=Path(c['cst_install'])/'AMD64'
    if not base.exists():raise RuntimeError('找不到CST安装目录，请设置cst_install。')
    if os.name!='nt':raise RuntimeError('CST执行器需要Windows及对应CST Python库。')
    global _dll
    _dll=os.add_dll_directory(str(base));sys.path[:0]=[str(base),str(base/'python_cst_libraries')]
    import cst.results
    return cst.results.ProjectFile(str(ascii_path(project or c['project'])),allow_interactive=True).get_3d()

def inspect(c):
    p=Path(c['project']);base=p.with_suffix('');mod=module(c)
    definitions=read(base/'Model/Parameters.json') if (base/'Model/Parameters.json').exists() else {}
    return dict(project=str(p),sha256=digest(p),run_ids=mod.get_all_run_ids(),parameter_definitions=definitions,tree=mod.get_tree_items() if hasattr(mod,'get_tree_items') else [],solver_support='本执行器仅运行参数化CST频域模型；其他求解器需新适配器。')

def extract(c,project,run,mod=None):
    mod=mod or module(c,project)
    if run==0:
        ids=[i for i in mod.get_all_run_ids() if i>0]
        assert len(ids)==1,'独立任务应恰有一个已保存Run，拒绝含糊的Run 0别名'
        run=ids[0]
    pars=mod.get_parameter_combination(run);values={};counts={}
    for spec in c['metrics']:
        if spec['kind']=='curve':
            item=mod.get_result_item(spec['tree'],run);freq=np.asarray(item.get_xdata(),float);v=np.asarray(item.get_ydata());transform=spec.get('transform','real')
            if transform=='db20':v=20*np.log10(np.maximum(abs(v),1e-300))
            elif transform=='abs':v=abs(v)
            else:
                assert not np.iscomplexobj(v) or np.max(abs(v.imag))<1e-10,'real指标不能静默忽略虚部'
                v=v.real
            if 'band' in spec:
                lo,hi=spec['band'];mask=(freq>=lo-1e-8)&(freq<=hi+1e-8);freq=freq[mask];v=v[mask]
                assert len(v)>1 and np.isclose(freq[0],lo,atol=1e-7) and np.isclose(freq[-1],hi,atol=1e-7),'缺少频带端点；请增加监测频点'
            assert np.isfinite(v).all() and len(v)>0
            reducer=spec['reduce']
            if reducer=='at':
                ix=int(np.argmin(abs(freq-spec['frequency'])));assert abs(freq[ix]-spec['frequency'])<1e-7,'没有指定频点';value=v[ix]
            else:value={'max':np.max,'min':np.min,'ripple':np.ptp}[reducer](v)
            counts[spec['name']]=len(v)
        else:
            # This storage layout is explicitly checked; unsupported variants fail closed.
            dbpath=ascii_path(project).with_suffix('')/'Result/Storage.sdb'
            with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
                name=spec.get('storage_name',f"Farfield_Cut_farfield (f={spec['frequency']:g})_Phi={spec.get('phi',0):g}_[{spec.get('port',1)}]_0.sig")
                sid=db.execute('select sig_id from SigHeader where name=? and choice=?',(name,run)).fetchone()
                if sid is None and run==0:
                    ids=[x for x in mod.get_all_run_ids() if x>0];assert len(ids)==1,'当前别名不明确';sid=db.execute('select sig_id from SigHeader where name=? and choice=?',(name,ids[0])).fetchone()
                assert sid is not None,'未保存所需方向图切面'
                b=db.execute('select bData from SigData10 where sig_id=?',(sid[0],)).fetchone()[0]
            n,d,k,w1,w2=struct.unpack_from('<5Q',b);assert (n,d,k,w1,w2)==(361,1,8,8,8),'不支持的CST方向图存储格式'
            a=np.frombuffer(b,dtype='<f8',offset=40+n*8).reshape(n,8)[:360];power=(a[:,2:]**2).sum(1)
            peaks=np.flatnonzero((power>np.roll(power,1))&(power>=np.roll(power,-1)));assert len(peaks)>=2 and np.isfinite(power).all()
            pp=np.sort(power[peaks]);value=10*np.log10(pp[-2]/pp[-1]);counts[spec['name']]=360
        values[spec['name']]=float(value)
    return dict(id=f'CST_Run_{run}',parameters={n:float(pars[n]) for n in c['parameters']},metrics=values,provenance=dict(project=str(project),run_id=run,frequency_or_angle_sample_counts=counts))

def import_existing(c,project=None):
    p=Path(project or c['project']);mod=module(c,p);db=p.with_suffix('')/'Result/Storage.sdb';stamp=(db.stat().st_size,db.stat().st_mtime_ns)
    rows=[];errors=[];allpars=[]
    for run in mod.get_all_run_ids():
        if run==0:continue
        try:
            r=extract(c,p,run,mod);allpars.append(mod.get_parameter_combination(run));rows.append(r)
        except Exception as e:errors.append(dict(run_id=run,error=str(e)))
    # Varying literal geometry dimensions omitted from inputs would create ambiguous labels.
    defs=read(p.with_suffix('')/'Model/Parameters.json');literal=[]
    for v in defs.get('parameters',[]):
        try:float(v['expr']);literal.append(v['name'])
        except (ValueError,KeyError):pass
    missing=[n for n in literal if n not in c['parameters'] and len({round(float(v[n]),10) for v in allpars if n in v})>1]
    if missing:raise ValueError('已有数据中还存在变化的独立参数，必须纳入输入或先筛选固定值：'+','.join(missing))
    assert stamp==(db.stat().st_size,db.stat().st_mtime_ns),'导入期间结果发生变化，请等待仿真结束后重试'
    return rows,errors

def prepare(c,folder,parameters):
    d=Path(folder);source=Path(c['project']);assert not (source.with_suffix('')/'Model.lok').exists(),'源工程正在打开；请先保存并关闭该工程，避免复制活动状态'
    check_source(c,source)
    if (d/'job.json').exists():return read(d/'job.json')
    assert not (d/'project.cst').exists(),'存在不完整准备文件，请检查后新建任务，不覆盖'
    d.mkdir(parents=True,exist_ok=True);shutil.copy2(source,d/'project.cst')
    for name in ['Model','CADData']:
        if (source.with_suffix('')/name).exists():shutil.copytree(source.with_suffix('')/name,d/'project'/name)
    temp=Path(os.environ.get('TEMP',r'C:\Users\Public'))
    if not str(temp).isascii():temp=Path(r'C:\Users\Public\Documents')
    alias=temp/('simagent_'+uuid.uuid4().hex)
    subprocess.run(['cmd','/c','mklink','/J',str(alias),str(d.resolve())],check=True,capture_output=True)
    lines=['Option Explicit','Sub Main()',' On Error GoTo Failed',' Dim ok As Boolean, ret As Long',f' OpenFile "{alias / "project.cst"}"']
    for n,v in parameters.items():lines.append(f' StoreDoubleParameter "{n}", {float(v):.17g}')
    lines+=[' ok=RebuildOnParametricChange(True,False)',' If Not ok Then Err.Raise 25001, "simagent", "Full rebuild failed"',' Save',f' FDSolver.MaxCPUs "{int(c.get("cpus",16))}"',' FDSolver.AcceleratedRestart False',' ret=FDSolver.Start',f' Open "{alias / "status.txt"}" For Append As #1',' Print #1,"SOLVER_RETURN=" & CStr(ret)',' Close #1',' If ret<>1 Then Err.Raise 25002, "simagent", "Solver failed"',' Save',f' Open "{alias / "status.txt"}" For Append As #1',' Print #1,"DONE"',' Close #1',' Quit',' Exit Sub','Failed:',' Dim why As String',' why=Err.Description',' On Error Resume Next',' Close #1',f' Open "{alias / "status.txt"}" For Append As #1',' Print #1,"ERROR: " & why',' Close #1',' Save',' Quit','End Sub']
    (d/'solve.bas').write_text('\n'.join(lines),encoding='ascii')
    job=dict(state='prepared',project=str(d/'project.cst'),macro=str(alias/'solve.bas'),source_sha256=digest(source),geometry_sha256=c.get('geometry_sha256'),parameters=parameters,created=now());write(d/'job.json',job);return job

def materialize(c,parameters,backup,notify=print):
    """Solve one design inside the source project itself and save it there, so its curves can be viewed in CST.

    The project is backed up in full first, and its own solver settings are kept: unlike the isolated batch jobs
    this does not touch MaxCPUs or AcceleratedRestart. Adding a saved run changes the .cst file hash but not the
    geometry signature, so tasks registered by geometry stay valid; both are checked before and after.
    """
    source=Path(c['project']);base=source.with_suffix('')
    assert not (base/'Model.lok').exists(),'源工程正在打开；请先保存并关闭后再写回'
    check_source(c)
    sha_before=digest(source);geo=geometry_signature(source);fixed=fixed_parameters_signature(source,parameters)
    backup=Path(backup);assert not backup.exists(),'备份目录已存在，请先检查：'+str(backup)
    backup.mkdir(parents=True);shutil.copy2(long_path(source),long_path(backup/source.name));shutil.copytree(long_path(base),long_path(backup/base.name))
    assert digest(backup/source.name)==sha_before,'备份文件哈希与源工程不一致'
    notify('已完整备份源工程：'+str(backup))
    mod=module(c,source);before_ids=set(mod.get_all_run_ids());del mod
    temp=Path(os.environ.get('TEMP',r'C:\Users\Public'))
    if not str(temp).isascii():temp=Path(r'C:\Users\Public\Documents')
    work=temp/('simagent_mat_'+uuid.uuid4().hex);work.mkdir(parents=True);status=work/'status.txt';alias=ascii_path(source)
    lines=['Option Explicit','Sub Main()',' On Error GoTo Failed',' Dim ok As Boolean, ret As Long',f' OpenFile "{alias}"']
    for n,v in parameters.items():lines.append(f' StoreDoubleParameter "{n}", {float(v):.17g}')
    lines+=[' ok=RebuildOnParametricChange(True,False)',' If Not ok Then Err.Raise 25001, "simagent", "Full rebuild failed"',' Save',' ret=FDSolver.Start',
            f' Open "{status}" For Append As #1',' Print #1,"SOLVER_RETURN=" & CStr(ret)',' Close #1',
            ' If ret<>1 Then Err.Raise 25002, "simagent", "Solver failed"',' Save',
            f' Open "{status}" For Append As #1',' Print #1,"DONE"',' Close #1',' Quit',' Exit Sub','Failed:',
            ' Dim why As String',' why=Err.Description',' On Error Resume Next',' Close #1',
            f' Open "{status}" For Append As #1',' Print #1,"ERROR: " & why',' Close #1',' Save',' Quit','End Sub']
    macro=work/'materialize.bas';macro.write_text('\n'.join(lines),encoding='ascii')
    exe=Path(c['cst_install'])/'CST DESIGN ENVIRONMENT.exe';si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
    proc=subprocess.Popen([str(exe),'-m',str(macro)],startupinfo=si,creationflags=subprocess.CREATE_NO_WINDOW);start=time.monotonic()
    while proc.poll() is None:
        time.sleep(5)
        if int(time.monotonic()-start)%60<5:notify('写回源工程求解中…')
        if time.monotonic()-start>c.get('timeout_minutes',45)*60:raise TimeoutError('写回求解超过等待时限，保留现场；未终止求解器。')
    text=status.read_text() if status.exists() else ''
    assert proc.returncode==0 and 'DONE' in text.splitlines() and 'SOLVER_RETURN=1' in text,'写回求解未成功完成：'+text.strip()[-200:]
    assert not (base/'Model.lok').exists(),'写回后源工程仍有Model.lok'
    mod=module(c,source);new=sorted(set(mod.get_all_run_ids())-before_ids)
    assert len(new)==1,f'源工程新增Run数异常：{new}'
    r=extract(c,source,new[0],mod)
    for n,v in parameters.items():assert np.isclose(r['parameters'][n],v,atol=1e-12,rtol=1e-10),'参数回读不一致'
    assert geometry_signature(source)==geo,'写回后几何或求解属性发生变化，请审计'
    assert fixed_parameters_signature(source,parameters)==fixed,'写回后固定参数取值发生变化，请审计'
    shutil.rmtree(work,ignore_errors=True)
    return dict(run_id=new[0],parameters=r['parameters'],metrics=r['metrics'],backup=str(backup),sha256_before=sha_before,sha256_after=digest(source),geometry_sha256=geo,seconds=round(time.monotonic()-start,1))

def run(c,folder,notify=print):
    d=Path(folder);job=read(d/'job.json');status=d/'status.txt'
    if job['state']=='complete':return extract(c,d/'project.cst',0)
    if job['state']=='running':
        # Never start a second process over a possibly running or interrupted job.
        if (d/'project/Model.lok').exists() or not status.exists() or 'DONE' not in status.read_text().splitlines():raise RuntimeError('上次求解仍在运行或中断；保留现场，请等待后重试恢复，不重复启动。')
    elif job['state']=='prepared':
        exe=Path(c['cst_install'])/'CST DESIGN ENVIRONMENT.exe';si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
        proc=subprocess.Popen([str(exe),'-m',job['macro']],startupinfo=si,creationflags=subprocess.CREATE_NO_WINDOW)
        job.update(state='running',pid=proc.pid,started=now());write(d/'job.json',job);start=time.monotonic()
        while proc.poll() is None:
            time.sleep(5)
            if int(time.monotonic()-start)%30<5:notify('CST求解中：'+d.name)
            if time.monotonic()-start>c.get('timeout_minutes',45)*60:raise TimeoutError('求解超过预算时间，保留运行现场；未终止求解器。')
        job['exit_code']=proc.returncode;write(d/'job.json',job)
        if proc.returncode:raise RuntimeError(f'CST退出码{proc.returncode}')
    assert status.exists() and 'DONE' in status.read_text().splitlines() and 'SOLVER_RETURN=1' in status.read_text(),'求解未成功完成'
    log=(d/'project/Result/Model.log').read_text(errors='replace')
    assert 'All broadband sweep convergence criteria have been satisfied' in log,'未确认宽带收敛'
    result=extract(c,d/'project.cst',0)
    for n,v in job['parameters'].items():assert np.isclose(result['parameters'][n],v,atol=1e-12,rtol=1e-10),'参数回读不一致'
    check_source(c,file_sha256=job['source_sha256'])
    job.update(state='complete',finished=now(),result=result);write(d/'job.json',job);return result
