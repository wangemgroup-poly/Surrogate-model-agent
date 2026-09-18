import json,time,uuid,html,re
from pathlib import Path
import numpy as np,joblib
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from scipy.optimize import differential_evolution
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from .core import read,write,digest,signature,config_signature,event,validate_config,arrays,normalize,denormalize,loss,passes,now,ingest,fitness,margins
from .models import Predictor,LABELS,eligible,batched_bnn_predict
from . import cst
from .configure import active_bounds

def init(task,config):
    t=Path(task);t.mkdir(parents=True,exist_ok=True)
    if (t/'config.json').exists():raise ValueError('任务已存在，不覆盖。')
    c=validate_config(config);p=Path(c['project']).resolve();assert p.suffix.lower()=='.cst' and p.is_file(),'请选择现有CST文件'
    c['project']=str(p);c['source_sha256']=digest(p)
    if cst.has_geometry(p):c['geometry_sha256']=cst.geometry_signature(p);c['fixed_parameters_sha256']=cst.fixed_parameters_signature(p,c['parameters'])
    write(t/'config.json',c);write(t/'data.json',dict(rows=[],repeats=[]));write(t/'state.json',dict(created=now(),phase='initialized',model=None,batch=None,simulations_started=0,radius=c.get('trust_radius',.05),diversity_history=[]))
    event(t,'任务已创建。可导入已有数据，或在明确计算预算内生成初始全波样本。')
    return t

def import_json(task,path):
    raw=read(path);rows=raw.get('rows',[]) if isinstance(raw,dict) else raw
    assert isinstance(rows,list) and rows,'输入应为非空rows列表'
    # Generic metric data only. Legacy curve datasets are explicitly converted by the example builder.
    return ingest(task,rows,dict(file=str(Path(path).resolve()),sha256=digest(path)))

def version_tag(p):
    m=re.search(r'V\d+(?:_\d+)?$',Path(p).stem);return m.group(0) if m else Path(p).stem

def union_bounds(projects):
    """Per-project optimizer ranges (Model.jopt) and their union; the union is the range a merged task needs."""
    per={str(Path(p).resolve()):active_bounds(p) for p in projects};union={}
    for b in per.values():
        for k,v in b.items():union[k]=[min(v[0],union[k][0]),max(v[1],union[k][1])] if k in union else list(v)
    return dict(projects=per,union=union)

def import_cst(task,projects=None):
    """Import saved runs from the source project and, optionally, sibling versions of the same structure (byte-identical geometry files)."""
    t=Path(task);c=read(t/'config.json');source=Path(c['project']).resolve();projects=[Path(p).resolve() for p in projects] if projects else [source];errors={};total=0;seen=0;observed={};summary={}
    for p in projects:
        assert p.is_file() and p.suffix.lower()=='.cst',f'不是CST文件：{p}'
        if p!=source:
            why=cst.structure_differences(source,p,c['parameters'])
            assert not why,f'{p.name} 与任务源工程不是同一结构，不能合并数据：'+'；'.join(why)
        rows,errs=cst.import_existing(c,p);errors[str(p)]=errs;sha=digest(p);seen+=len(rows)
        for r in rows:
            if p!=source:r['id']=f'{version_tag(p)}_{r["id"]}'
            r['provenance']['project_sha256']=sha
            for k,v in r['parameters'].items():observed[k]=[min(v,observed[k][0]),max(v,observed[k][1])] if k in observed else [v,v]
        event(t,f'{p.name}：CST导入读取成功{len(rows)}组，缺失/不支持结果{len(errs)}组。')
        if rows:
            total+=ingest(t,rows,dict(project=str(p),project_sha256=sha));last=read(t/'last_import.json')
            summary[str(p)]=dict(read=len(rows),added=last['added'],repeats=last['repeats'],rejected=len(last['rejected']),without_results=len(errs))
    write(t/'cst_import_errors.json',errors);write(t/'import_summary.json',summary)
    if not seen:raise RuntimeError('没有可读的完整全波结果，请检查结果树路径或先生成数据。')
    b=union_bounds(projects)['union']
    for k,v in observed.items():b[k]=[min(v[0],b[k][0]),max(v[1],b[k][1])] if k in b else list(v)  # saved runs often lie outside the project's current optimizer range
    wider=[k for k,v in b.items() if k in c['parameters'] and (v[0]<c['parameters'][k][0]-1e-9 or v[1]>c['parameters'][k][1]+1e-9)]
    if wider:
        write(t/'bounds_advice.json',dict(projects_union=b,task_parameters=c['parameters'],suggested={k:([min(b[k][0],v[0]),max(b[k][1],v[1])] if k in b else v) for k,v in c['parameters'].items()}))
        event(t,'提示：所导入工程的优化器范围或已有样本在'+'、'.join(wider)+'上超出本任务范围，越界样本已被拒绝；要全部纳入需用并集范围新建任务，见bounds_advice.json。')
    return total

def evaluate(y,mu,std,c):
    error=abs(y-mu);actual=passes(y,c);pred=passes(mu,c)
    return dict(n=len(y),MAE=error.mean(0).tolist(),P90=np.quantile(error,.9,axis=0).tolist(),max_error=error.max(0).tolist(),false_pass=int((~actual&pred).sum()),true_pass=int((actual&pred).sum()),actual_pass=int(actual.sum()),predicted_pass=int(pred.sum()),raw_2sigma_coverage=(error<=2*std).mean(0).tolist(),uncertainty='原始模型离散度/条件后验，未校准，不能解释为合格概率')

def region_labels(task,z,c):
    """Freeze region anchors so one new point cannot permute all evaluation folds."""
    t=Path(task);path=t/'split_policy.json';bounds_signature=signature(c['parameters'])
    if path.exists():
        policy=read(path)
        if policy['bounds_signature']!=bounds_signature:raise ValueError('参数范围变化会改变验证区域，请新建任务，不能静默重划验证集。')
    else:
        reference=t/'models/model_001/data_snapshot.json'
        base=normalize(arrays(read(reference)['rows'],c)[0],c) if reference.exists() else z
        anchors=KMeans(n_clusters=6,random_state=17,n_init=10).fit(base).cluster_centers_
        policy=dict(bounds_signature=bounds_signature,anchors=anchors.tolist(),seed=17,reference_samples=len(base),note='固定初始参数区域：0为测试、1为验证、其余训练；新增样本按最近区域归属。');write(path,policy)
    return np.argmin(cdist(z,np.array(policy['anchors'])),axis=1)

def train(task,force=False):
    t=Path(task);c=validate_config(read(t/'config.json'));db=read(t/'data.json');rows=db['rows'];state=read(t/'state.json')
    if len(rows)<max(30,3*len(c['parameters'])):raise ValueError('样本不足以分组验证：至少需要max(30,3×参数数)组；请先导入或在预算内补点。')
    dh=digest(t/'data.json');ch=config_signature(c)
    if state['model'] and not force:
        meta=read(t/'models'/state['model']/'selection.json')
        if meta['data_sha256']==dh and meta['config_signature']==ch:event(t,'数据和目标未变化，沿用已验证选择的模型。');return meta
    x,y=arrays(rows,c);z=normalize(x,c);metrics=[m['name'] for m in c['metrics']];scale=np.array([m.get('scale',1) for m in c['metrics']]);pool,excluded=eligible(len(x),x.shape[1],y.shape[1],c.get('include_global_gp',False))
    event(t,f'模型选择：{len(x)}组有效样本，{x.shape[1]}个输入参数，{y.shape[1]}个输出；将比较'+ '、'.join(LABELS[a] for a in pool)+'。')
    with threadpoolctl_context():
        labels=region_labels(t,z,c)
        te=np.flatnonzero(labels==0);va=np.flatnonzero(labels==1);tr=np.flatnonzero(labels>=2)
        # Purging prevents near-identical training/test designs from making validation look easy.
        tr=tr[cdist(z[tr],z[np.r_[va,te]]).min(1)>=c.get('purge_distance',.01)]
        va=va[cdist(z[va],z[te]).min(1)>=c.get('purge_distance',.01)]
        if min(len(tr),len(va),len(te))<5:raise ValueError('分组后有效独立样本过少。请扩大覆盖范围或增加数据，不退回随机泄漏切分。')
        # Selection subset cap is explicit; final deployment refits the full database.
        pilot_cap=c.get('selection_training_cap',1800);rng=np.random.default_rng(17)
        if len(tr)>pilot_cap:tr=np.sort(rng.choice(tr,pilot_cap,replace=False))
        center=z[tr[np.argmin(loss(y[tr],c))]];records=[]
        for algorithm in pool:
            started=time.monotonic()
            try:
                model=Predictor(algorithm,center).fit(z[tr],y[tr]);mu,std=model.predict(z[va]);ev=evaluate(y[va],mu,std,c)
                score=float(np.mean(np.array(ev['MAE'])/scale)+2*ev['false_pass']/len(va));elapsed=time.monotonic()-started
                records.append(dict(algorithm=algorithm,label=LABELS[algorithm],score=score,seconds=elapsed,validation=ev,warnings=model.warnings,fit_samples=model.n_fit))
                event(t,f'{LABELS[algorithm]}：验证评分{score:.3f}，误判合格{ev["false_pass"]}组，用时{elapsed:.1f}秒。')
            except Exception as exc:records.append(dict(algorithm=algorithm,error=str(exc)));event(t,LABELS[algorithm]+'评估失败，已记录原因：'+str(exc))
        valid=[r for r in records if 'score' in r]
        if not valid:raise RuntimeError('所有候选模型失败，见日志。')
        best_score=min(r['score'] for r in valid);competitive=[r for r in valid if r['score']<=best_score*1.05+1e-8];winner=min(competitive,key=lambda r:r['seconds'])
        # Test evaluated only after algorithm choice, not used to choose winner.
        dev=np.r_[tr,va];testmodel=Predictor(winner['algorithm'],z[dev[np.argmin(loss(y[dev],c))]]).fit(z[dev],y[dev]);mu,std=testmodel.predict(z[te]);test=evaluate(y[te],mu,std,c)
        ci=int(np.argmin(loss(y,c)));deploy=Predictor(winner['algorithm'],z[ci]).fit(z,y)
    version=f'model_{len(list((t/"models").glob("model_*")))+1:03d}';folder=t/'models'/version;folder.mkdir(parents=True,exist_ok=False)
    joblib.dump(deploy,folder/'model.joblib');write(folder/'data_snapshot.json',db)
    meta=dict(version=version,algorithm=winner['algorithm'],label=winner['label'],data_sha256=dh,config_signature=ch,model_sha256=digest(folder/'model.joblib'),available_samples=len(x),fit_samples=deploy.n_fit,inputs=x.shape[1],outputs=y.shape[1],metrics=metrics,candidate_results=records,excluded=excluded,selection_rule='验证误差按指标scale归一化并惩罚误判合格；评分差5%内优先较快者。测试集不参与选择。',test=test,splits=dict(train=tr.tolist(),validation=va.tolist(),test=te.tolist()),center=rows[ci]['parameters'],center_id=rows[ci]['id'],created=now(),notes=['离线参数区域留出，不等于新的前瞻CST验证。','近重复参数已跨集合隔离；重复记录不用于独立样本计数。','若选择贝叶斯末层网络，仅末层权重为贝叶斯推断；特征网络权重不确定性未建模，不是Liu论文BNN完整复现。','逐指标模型不能保证未监测频点的连续响应。'])
    write(folder/'selection.json',meta);state.update(model=version,phase='trained');write(t/'state.json',state)
    event(t,f'已选择{meta["label"]}：数据池{len(x)}组，实际拟合{deploy.n_fit}组；依据验证误差、误判合格及耗时。完整比较和局限已保存。',model=version)
    return meta

def threadpoolctl_context():return threadpool_limits(limits=4)

def pessimistic(mu,sd,c,k):
    """Constraint outputs shifted by k*sigma toward failure; unconstrained outputs unchanged."""
    s=np.array([(1 if m['op'] in ['<=','<'] else -1) if 'limit' in m else 0 for m in c['metrics']]);return np.asarray(mu)+s*k*np.asarray(sd)

def optimize_point(model,c,z,taken,center,radius,explore,seed):
    """Differential evolution directly on the surrogate; None if no design satisfies the distance rules."""
    full=c.get('search_domain','trust_region')=='full';gap=c.get('min_candidate_distance',.03);sep=c.get('candidate_separation',.05 if full else radius*.1)
    sign=np.array([(-2 if m['op'] in ['<=','<'] else 2) if 'limit' in m else 0 for m in c['metrics']])
    kc=c.get('conservative_sigma',1.)
    def objective(Q):
        Q=Q.T;mu,sd=model.predict(Q)
        mu=mu+sign*sd if explore else mu-sign/2*kc*sd  # explore: optimistic LCB; improvement: constraints judged at mu+k*sigma on the failing side
        penalty=np.maximum(gap-cdist(Q,z).min(1),0)
        if len(taken):penalty=penalty+np.maximum(sep-cdist(Q,taken).min(1),0)
        if not full:penalty=penalty+np.maximum(np.linalg.norm(Q-center,axis=1)-radius,0)
        return loss(mu,c)+100*penalty
    with threadpoolctl_context():
        r=differential_evolution(objective,[(0,1)]*z.shape[1],seed=seed,maxiter=c.get('optimizer_iterations',150),popsize=15,tol=1e-8,polish=False,vectorized=True,updating='deferred')
    q=r.x
    ok=cdist(q[None],z).min()>=gap-1e-9 and (not len(taken) or cdist(q[None],taken).min()>=sep-1e-9) and (full or np.linalg.norm(q-center)<=radius+1e-9)
    return q if ok else None

def de_children(P,rng,F=.8,CR=.8):
    """DE/current-to-best/1 with binomial crossover (Liu et al., IEEE TAP 2022, eqs. 8-9); P is sorted best first."""
    lam,d=P.shape;out=np.empty_like(P)
    for i in range(lam):
        r1,r2=rng.choice(np.delete(np.arange(lam),i),2,replace=False)
        v=P[i]+F*(P[0]-P[i])+F*(P[r1]-P[r2])
        mask=rng.random(d)<=CR;mask[rng.integers(d)]=True
        out[i]=np.clip(np.where(mask,v,P[i]),0,1)
    return out

def best_index(v,c):
    """Lowest penalty fitness; ties (e.g. several predicted feasible) broken by worst normalized margin."""
    return int(np.lexsort((loss(v,c),fitness(v,c)))[0])

def sb_sadea_select(z,y,c,history,rng,predictor):
    """SB-SADEA Steps 3-6: lambda best designs, DE children, tau-nearest BNN per child, self-adaptive LCB."""
    d=z.shape[1];s=c.get('sb_sadea',{});lam=s.get('lambda',4*d);tau=s.get('tau',4*d);omega=s.get('omega',2 if s.get('surrogate','gp')=='gp' else 14)
    assert len(z)>=max(lam,tau),'数据库样本少于λ或τ，不能执行SB-SADEA'
    P=z[np.argsort(fitness(y,c),kind='stable')[:lam]]
    U=de_children(P,rng,s.get('F',.8),s.get('CR',.8));U=U[cdist(U,z).min(1)>1e-9]
    assert len(U),'差分进化子代全部与数据库设计重复'
    idx=np.argsort(cdist(U,z),axis=1)[:,:tau]
    mu,sd,info=predictor(z[idx],y[idx],U)
    b=best_index(mu,c);S=float(cdist(U[b:b+1],P).min())
    window=history[-10:];use_lcb=len(window)>=10 and S<np.mean(window)-.5*np.std(window,ddof=1)
    choice=b
    if use_lcb:
        sign=np.array([(-1 if m['op'] in ['<=','<'] else 1) if 'limit' in m else 0 for m in c['metrics']])
        choice=best_index(mu+sign*omega*sd,c)
    return dict(u=U[choice],mu=mu[choice],std=sd[choice],role='sb_sadea_lcb' if use_lcb else 'sb_sadea_mean',S=S,children=len(U),population=lam,tau=tau,omega=omega,predicted_fitness=float(fitness(mu[choice],c)[0]),info=info)

def propose_sb_sadea(t,c,state,meta,version,z,y,batchid,rng,count):
    if count!=1:raise ValueError('SB-SADEA每次迭代只对1个设计做全波仿真，请使用 --count 1。')
    s=c.get('sb_sadea',{})
    if s.get('surrogate','gp')=='bnn':
        def predictor(tx,ty,q):
            mu,sd,info=batched_bnn_predict(tx,ty,q,prior_std=s.get('prior_std',.1),lr=s.get('learning_rate',.05),decay=s.get('decay',.999),max_steps=s.get('max_steps',3000),patience=s.get('patience',200),samples=s.get('prediction_samples',200),seed=int(rng.integers(1000000)),device=s.get('device','auto'),kl_weight=s.get('kl_weight',1.),min_steps=s.get('min_steps',500))
            return mu,sd,dict(surrogate='bnn',**info)
    else:
        def predictor(tx,ty,q):
            # GP-ALCB variant in Liu et al. (Table III): one ARD-Matérn GP per child on its tau nearest samples.
            mus,sds=[],[]
            with threadpoolctl_context():
                for k in range(len(q)):
                    m,sd=Predictor('global_gp').fit(tx[k],ty[k]).predict(q[k:k+1]);mus.append(m[0]);sds.append(sd[0])
            return np.array(mus),np.array(sds),dict(surrogate='gp')
    history=state.get('sb_sadea_S',[]);started=time.monotonic()
    r=sb_sadea_select(z,y,c,history,rng,predictor)
    state['sb_sadea_S']=(history+[r['S']])[-100:]
    names=[m['name'] for m in c['metrics']];lcb=r['role']=='sb_sadea_lcb'
    item=dict(id=f'{batchid}_001',parameters=dict(zip(c['parameters'],denormalize(r['u'],c).tolist())),role=r['role'],source='sb_sadea',state='pending',
              prediction=dict(zip(names,r['mu'].tolist())),std=r['std'].tolist(),predicted_pass=bool(passes(r['mu'][None],c)[0]),predicted_fitness=r['predicted_fitness'],
              sb_sadea=dict(S=r['S'],lcb_used=lcb,children=r['children'],population=r['population'],tau=r['tau'],omega=r['omega'],history_length=len(history),seconds=time.monotonic()-started,**r['info']))
    flags=annotate(t,c,z,y,[item])
    batch=dict(id=batchid,created=now(),model=version,model_sha256=meta['model_sha256'],data_sha256=digest(t/'data.json'),config_signature=config_signature(c),config_snapshot=c,center=None,radius=state['radius'],
               proposal_method='sb_sadea',search_domain='full',candidates=[item],trust_update={'policy':'SB-SADEA不使用信任半径'},
               note='SB-SADEA（Liu et al., IEEE TAP 2022）：λ个最优设计→DE/current-to-best/1→每个子代τ近邻局部代理（sb_sadea.surrogate：gp为论文GP-ALCB变体，bnn为论文BNN）→自适应LCB→1次全波。不确定度未校准；预测合格不等于全波合格。')
    write(t/'batches'/batchid/'batch.json',batch);state.update(batch=batchid,phase='proposed');write(t/'state.json',state)
    event(t,f'SB-SADEA：{r["children"]}个子代各用{r["tau"]}个近邻训练{"BNN" if r["info"].get("surrogate")=="bnn" else "GP"}，'+(f'多样性不足，采用LCB(ω={r["omega"]})' if lcb else '采用预测均值')+f'，冻结1个候选，预测适应度{r["predicted_fitness"]:.3f}。'+warning_text(flags),batch=batchid)
    return batch

def data_spacing(z):
    """Median nearest-neighbour distance of the data; the yardstick for calling a candidate an extrapolation."""
    if len(z)<2:return float('nan')
    d=cdist(z,z);np.fill_diagonal(d,np.inf);return float(np.median(d.min(1)))

def annotate(t,c,z,y,items):
    """Flag candidates far from the data and predictions the fast non-GP algorithms do not support; warnings only, nothing is dropped."""
    names=[m['name'] for m in c['metrics']];scale=np.array([m.get('scale',1) for m in c['metrics']]);limited=np.array(['limit' in m for m in c['metrics']])
    ratio=c.get('extrapolation_ratio',2.);dlimit=c.get('disagreement_limit',2.);spacing=data_spacing(z);ex=un=0
    q=normalize(np.array([[r['parameters'][k] for k in c['parameters']] for r in items]),c);dist=cdist(q,z).min(1) if len(z) else np.full(len(q),np.nan)
    others={}
    if len(z)>=10 and any(r.get('prediction') for r in items):
        with threadpoolctl_context():
            for a in ('extra_trees','krr','bayes_last_layer'):
                try:others[a]=Predictor(a).fit(z,y).predict(q)[0]
                except Exception as exc:event(t,LABELS[a]+'交叉核查失败：'+str(exc))
    for i,r in enumerate(items):
        w=[];r['distance_to_data']=float(dist[i]) if np.isfinite(dist[i]) else None;r['data_spacing']=spacing if np.isfinite(spacing) else None
        if np.isfinite(dist[i]) and np.isfinite(spacing) and spacing>0 and dist[i]>ratio*spacing:w.append(f'外推：距最近样本{dist[i]/spacing:.1f}倍于数据间距');ex+=1
        if r.get('prediction') and others:
            mu=np.array([r['prediction'][n] for n in names]);P=np.array([others[a][i] for a in others]);worst=float((abs(P-mu)/scale)[:,limited].max())
            r['cross_check']=dict(algorithms=list(others),predictions={a:dict(zip(names,others[a][i].tolist())) for a in others},max_normalized_disagreement=worst,agree=worst<=dlimit,predicted_pass_all_models=bool(r.get('predicted_pass')) and all(bool(passes(others[a][i:i+1],c)[0]) for a in others))
            if worst>dlimit:w.append(f'其他模型不支持该预测（最大分歧{worst:.1f}×scale），按探索候选看待');un+=1
        r['warnings']=w
    return dict(extrapolation=ex,unsupported=un)

def warning_text(flags):return f' 预警：{flags["extrapolation"]}组外推，{flags["unsupported"]}组预测未被其他模型支持。' if flags['extrapolation'] or flags['unsupported'] else ''

def propose(task,count=3,seed_count=None):
    t=Path(task);c=read(t/'config.json');state=read(t/'state.json');rows=read(t/'data.json')['rows']
    if state.get('batch'):
        old=read(t/'batches'/state['batch']/'batch.json')
        if any(r['state']!='complete' for r in old['candidates']):raise RuntimeError('当前批次未完成，请先validate/resume，或使用discard明确放弃未运行候选。')
    if seed_count is None and not state['model']:raise RuntimeError('请先训练模型。')
    if count<1 or count>10000:raise ValueError('候选数应为1至10000。')
    if seed_count is not None and not 1<=seed_count<=10000:raise ValueError('初始样本数应为1至10000。')
    version=state['model'];meta=read(t/'models'/version/'selection.json') if version and seed_count is None else None
    if meta and (meta['data_sha256']!=digest(t/'data.json') or meta['config_signature']!=config_signature(c)):raise RuntimeError('数据/目标已更新，请先train，不能用旧模型继续优化。')
    x,y=arrays(rows,c) if rows else (np.empty((0,len(c['parameters']))),np.empty((0,len(c['metrics']))));z=normalize(x,c)
    batchid='batch_'+uuid.uuid4().hex[:10];rng=np.random.default_rng(21+len(list((t/'batches').glob('*'))));items=[];conservative={}
    if seed_count is None and c.get('proposal_method')=='sb_sadea':
        return propose_sb_sadea(t,c,state,meta,version,z,y,batchid,rng,count)
    if seed_count is not None:
        zz=qmc.LatinHypercube(d=len(c['parameters']),seed=int(rng.integers(100000))).random(seed_count);mu=std=None;selected=list(range(seed_count));roles=['initial_sampling']*seed_count;center=None
    else:
        center=normalize(np.array([meta['center'][k] for k in c['parameters']]),c);radius=state['radius'];u=rng.normal(size=(c.get('search_pool',5000),len(center)));u/=np.linalg.norm(u,axis=1)[:,None];zz=np.clip(center+u*radius*rng.random((len(u),1))**(1/len(center)),0,1)
        dist=cdist(zz,z).min(1);zz=zz[dist>min(.005,radius*.1)]
        model=joblib.load(t/'models'/version/'model.joblib')
        with threadpoolctl_context():mu,std=model.predict(zz)
        nearest=np.argmin(cdist(zz,z),axis=1);disagreement=np.mean(abs(mu-y[nearest]),axis=1)
        # Optimistic LCB-like score only chooses simulations; never certifies feasibility.
        optimistic=mu.copy()
        for j,m in enumerate(c['metrics']):
            if 'limit' in m:optimistic[:,j]+=(-2 if m['op'] in ['<=','<'] else 2)*std[:,j]
        order_mean=np.argsort(loss(mu,c));order_explore=np.argsort(loss(optimistic,c));order_dis=np.argsort(-disagreement)
        selected=[];roles=[];history=state.get('diversity_history',[])
        nearest_best=float(cdist(zz[order_mean[:1]],z[np.argsort(loss(y,c))[:min(20,len(z))]]).min())
        adaptive=len(history)>=10 and nearest_best<float(np.mean(history[-10:])-.5*np.std(history[-10:]))
        optimized=c.get('proposal_method','random_pool')=='optimize';full=c.get('search_domain','trust_region')=='full';sources=[]
        if optimized and full and meta['algorithm']=='local_gp':raise ValueError('当前模型为局部GP（仅拟合中心附近120点），不能用于全参数域优化；请开启全样本GP或改用trust_region。')
        for j in range(count):
            role,order=([('adaptive_exploration' if adaptive else 'predicted_improvement',order_explore if adaptive else order_mean),('uncertainty_exploration',order_explore),('model_disagreement',order_dis)])[j%3]
            if optimized and role!='model_disagreement':
                q=optimize_point(model,c,z,zz[selected] if selected else np.empty((0,z.shape[1])),center,radius,explore=role!='predicted_improvement',seed=int(rng.integers(1000000)))
                if q is not None:
                    with threadpoolctl_context():qm,qs=model.predict(q[None])
                    zz=np.vstack([zz,q]);mu=np.vstack([mu,qm]);std=np.vstack([std,qs]);selected.append(len(zz)-1);roles.append(role);sources.append('surrogate_optimization');conservative[len(zz)-1]=pessimistic(qm,qs,c,c.get('conservative_sigma',1.))[0];continue
                event(t,f'第{j+1}个候选的代理优化未找到满足距离规则的设计，回退随机候选池。')
            index=next((int(i) for i in order if i not in selected and (not selected or cdist(zz[i:i+1],zz[selected]).min()>=radius*.1)),None)
            if index is None:break
            selected.append(index);roles.append(role);sources.append('random_pool')
        state['diversity_history']=(history+[nearest_best])[-100:]
    names=[m['name'] for m in c['metrics']]
    for j,(i,role) in enumerate(zip(selected,roles),1):
        params=dict(zip(c['parameters'],denormalize(zz[i],c).tolist()));extra={}
        if i in conservative:extra=dict(conservative_prediction=dict(zip(names,conservative[i].tolist())),predicted_pass_conservative=bool(passes(conservative[i][None],c)[0]))
        items.append(dict(id=f'{batchid}_{j:03d}',parameters=params,role=role,source=sources[j-1] if seed_count is None else 'initial_sampling',state='pending',prediction=dict(zip(names,mu[i].tolist())) if mu is not None else None,std=std[i].tolist() if std is not None else None,predicted_pass=bool(passes(mu[i:i+1],c)[0]) if mu is not None else None,**extra))
    if not items:raise RuntimeError('当前范围无法生成不同候选，请检查参数范围和搜索半径。')
    flags=annotate(t,c,z,y,items)
    batch=dict(id=batchid,created=now(),model=version if meta else None,model_sha256=meta['model_sha256'] if meta else None,data_sha256=digest(t/'data.json'),config_signature=config_signature(c),config_snapshot=c,center=meta['center'] if meta else None,radius=state['radius'],proposal_method=c.get('proposal_method','random_pool'),search_domain=c.get('search_domain','trust_region'),candidates=items,note='预测合格不等于全波合格；不确定度未校准。贝叶斯末层/GP的LCB探索系数采用2，未照搬Liu BNN的系数14。')
    write(t/'batches'/batchid/'batch.json',batch);state.update(batch=batchid,phase='proposed');write(t/'state.json',state)
    event(t,f'已冻结{len(items)}组候选，其中预测合格{sum(r["predicted_pass"] is True for r in items)}组。尚未进行CST验收。'+warning_text(flags),batch=batchid)
    return batch

def validate(task,budget):
    t=Path(task);c=read(t/'config.json');state=read(t/'state.json');assert state['batch'],'没有候选批次';folder=t/'batches'/state['batch'];batch=read(folder/'batch.json')
    assert config_signature(c)==batch['config_signature'],'配置变更，不能验证旧批次；请恢复配置或放弃后重建'
    if batch['model']:assert digest(t/'models'/batch['model']/'model.joblib')==batch['model_sha256'],'模型快照已改变'
    assert budget>=0,'预算不能为负';started=0
    for candidate in batch['candidates']:
        if candidate['state']=='complete':continue
        if (t/'PAUSE').exists():event(t,'收到暂停请求；当前已完成结果保留。');break
        d=folder/'cst'/candidate['id'];job=read(d/'job.json') if (d/'job.json').exists() else None
        is_new=not job or job['state']=='prepared'
        if is_new and (started>=budget or state['simulations_started']>=c.get('simulation_budget',0)):event(t,'已达到本次或任务总仿真预算，保留候选等待续跑。');break
        try:
            if not job:cst.prepare(c,d,candidate['parameters'])
            if is_new:
                state['simulations_started']+=1;started+=1;write(t/'state.json',state)
            candidate['state']='running';write(folder/'batch.json',batch)
            event(t,'开始/恢复全波验证：'+candidate['id'])
            result=cst.run(c,d,lambda msg:event(t,msg));result['id']=candidate['id']
            ingest(t,[result],result['provenance']);candidate.update(state='complete',actual=result['metrics'],project=str(d/'project.cst'),run_id=result['provenance']['run_id'])
            a=np.array([[result['metrics'][m['name']] for m in c['metrics']]]);candidate['CST_pass']=bool(passes(a,c)[0])
            if candidate['prediction']:candidate['error']={k:result['metrics'][k]-candidate['prediction'][k] for k in result['metrics']}
            write(folder/'batch.json',batch);event(t,f'{candidate["id"]} 全波完成：'+('合格' if candidate['CST_pass'] else '未全部达标'))
        except Exception as exc:
            candidate['last_error']=str(exc);write(folder/'batch.json',batch);event(t,'当前任务停止，现场保留：'+str(exc));raise
    if all(r['state']=='complete' for r in batch['candidates']):
        state['phase']='validated'
        if batch['model'] and 'trust_update' not in batch:
            snapshot=read(t/'models'/batch['model']/'data_snapshot.json')['rows'];xx,yy=arrays(snapshot,c);ci=int(np.argmin(loss(yy,c)));first=batch['candidates'][0];model=joblib.load(t/'models'/batch['model']/'model.joblib');pred0=model.predict(normalize(xx[ci:ci+1],c))[0]
            predicted=np.array([[first['prediction'][m['name']] for m in c['metrics']]]);actual=np.array([[first['actual'][m['name']] for m in c['metrics']]])
            dp=float(loss(pred0,c)[0]-loss(predicted,c)[0]);da=float(loss(yy[ci:ci+1],c)[0]-loss(actual,c)[0]);rho=da/dp if dp>1e-9 else None
            state['radius']=max(.002,state['radius']*.5) if rho is None or rho<.25 else min(.2,state['radius']*1.5) if rho>.75 else state['radius']
            batch['trust_update']=dict(predicted_improvement=dp,actual_improvement=da,ratio=rho,next_radius=state['radius']);write(folder/'batch.json',batch)
        write(t/'state.json',state)
    report(t);return batch

def rebaseline(task):
    """Re-register a source that gained saved runs: allowed only when its geometry matches the registration or the agent's latest frozen copy."""
    t=Path(task);c=read(t/'config.json');state=read(t/'state.json');source=Path(c['project'])
    if state.get('batch'):assert all(r['state'] in ('complete','discarded') for r in read(t/'batches'/state['batch']/'batch.json')['candidates']),'存在未完成候选，请先resume或discard'
    assert not (source.with_suffix('')/'Model.lok').exists(),'源工程正在打开，请先保存关闭'
    current=cst.geometry_signature(source)
    if c.get('geometry_sha256'):assert current==c['geometry_sha256'],'几何或求解属性已变化，不能重新登记；请新建任务'
    else:
        copies=sorted(t.glob('batches/*/cst/*/project.cst'),key=lambda p:p.stat().st_mtime)
        assert copies,'本任务没有几何登记，也没有可比对的验证副本，无法审计'
        structure=cst.GEOMETRY_FILES[:2]  # job copies get MaxCPUs/AcceleratedRestart written by the agent macro, so solver properties cannot be compared against them
        assert cst.geometry_signature(copies[-1],structure)==cst.geometry_signature(source,structure),'源工程结构（Model.mod/ModelHistory.json）与本任务最近一次验证时的副本不一致，不能重新登记'
        assert cst.fixed_parameters_signature(copies[-1],c['parameters'])==cst.fixed_parameters_signature(source,c['parameters']),'源工程固定参数取值与本任务最近一次验证时的副本不一致，不能重新登记'
    fixed=cst.fixed_parameters_signature(source,c['parameters'])
    if c.get('fixed_parameters_sha256'):assert fixed==c['fixed_parameters_sha256'],'源工程固定参数取值已改变，不能重新登记；请新建任务'
    old=c['source_sha256'];c['source_sha256']=digest(source);c['geometry_sha256']=current;c['fixed_parameters_sha256']=fixed;write(t/'config.json',validate_config(c))
    event(t,f'源工程重新登记：结构未变，文件哈希{old[:12]}→{c["source_sha256"][:12]}；几何与求解属性登记为{current[:12]}，此后按它校验。')
    return c

def database_best(task):
    t=Path(task);c=read(t/'config.json');rows=read(t/'data.json')['rows']
    if not rows:return float('inf')
    _,y=arrays(rows,c);return float(fitness(y,c).min())

def materialize(task,design=None):
    """Re-solve one already verified design inside the source project and save it there for inspection in CST.

    Adds a saved run to the user's own project; the isolated batch copies stay untouched. Returns None when the
    design was materialised before. The task keeps working afterwards because the geometry signature is unchanged.
    """
    t=Path(task);c=read(t/'config.json');rows=read(t/'data.json')['rows'];assert rows,'没有可写回的设计'
    ids=[r['id'] for r in rows];x,y=arrays(rows,c)
    i=ids.index(design) if design else int(np.argmin(fitness(y,c)))
    row=rows[i];log=read(t/'materialized.json') if (t/'materialized.json').exists() else []
    if any(e['design']==row['id'] for e in log):event(t,f'{row["id"]} 之前已写回源工程，跳过。');return None
    event(t,f'写回源工程：{row["id"]}（罚函数适应度{float(fitness(y,c)[i]):.3f}）')
    source=Path(c['project'])  # backup beside the project, like the earlier manual ones: shorter paths and easy to find
    r=cst.materialize(c,row['parameters'],source.parent/f'backup_{version_tag(source)}_{row["id"]}_{now()[:10].replace("-","")}',lambda m:event(t,m))
    entry=dict(design=row['id'],created=now(),agent_metrics=row['metrics'],difference={k:r['metrics'][k]-row['metrics'][k] for k in r['metrics']},**r)
    log.append(entry);write(t/'materialized.json',log)
    c['source_sha256']=r['sha256_after'];c['geometry_sha256']=r['geometry_sha256'];write(t/'config.json',validate_config(c))
    event(t,f'已写回源工程 Run {r["run_id"]}，与Agent独立工程最大差异{max(abs(v) for v in entry["difference"].values()):.4f}；几何未变，任务仍可验证；备份见 {r["backup"]}。')
    return entry

def auto_materialize(task,done):
    """Run-loop rule: save an improved design into the source project. A failure here never stops the optimisation."""
    t=Path(task);c=read(t/'config.json');policy=c.get('materialize',{});mode=policy.get('on','improvement')
    if mode=='never' or done>=policy.get('max',3):return 0
    rows=read(t/'data.json')['rows'];_,y=arrays(rows,c)
    if mode=='pass' and not bool(passes(y[int(np.argmin(fitness(y,c)))][None],c)[0]):return 0
    try:return 1 if materialize(t) else 0
    except Exception as exc:event(t,'写回源工程未成功，优化继续：'+str(exc));return 0

def near_bound(z,tol=.01):return (z<=tol)|(z>=1-tol)

def scatter_svg(path,xv,yv,agent,ok,xlab,ylab,xlim,ylim):
    """Dependency-free scatter: grey imported runs, orange agent runs, green ring for designs passing every constraint, dashed limits."""
    W,H,P=600,440,58;x0,x1=min(xv.min(),xlim),max(xv.max(),xlim);y0,y1=min(yv.min(),ylim),max(yv.max(),ylim)
    x0,x1=x0-.05*((x1-x0) or 1),x1+.05*((x1-x0) or 1);y0,y1=y0-.05*((y1-y0) or 1),y1+.05*((y1-y0) or 1)
    sx=lambda v:P+(v-x0)/(x1-x0)*(W-2*P);sy=lambda v:H-P-(v-y0)/(y1-y0)*(H-2*P)
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="12">','<rect width="100%" height="100%" fill="white"/>',f'<line x1="{P}" y1="{H-P}" x2="{W-P}" y2="{H-P}" stroke="#444"/>',f'<line x1="{P}" y1="{P}" x2="{P}" y2="{H-P}" stroke="#444"/>']
    for k in range(5):
        vx=x0+(x1-x0)*k/4;vy=y0+(y1-y0)*k/4;s.append(f'<text x="{sx(vx):.0f}" y="{H-P+16}" text-anchor="middle">{vx:.3g}</text><text x="{P-6}" y="{sy(vy)+4:.0f}" text-anchor="end">{vy:.3g}</text>')
    s.append(f'<line x1="{sx(xlim):.1f}" y1="{P}" x2="{sx(xlim):.1f}" y2="{H-P}" stroke="#c33" stroke-dasharray="4 3"/><line x1="{P}" y1="{sy(ylim):.1f}" x2="{W-P}" y2="{sy(ylim):.1f}" stroke="#c33" stroke-dasharray="4 3"/>')
    for i in np.argsort(agent.astype(int),kind='stable'):s.append(f'<circle cx="{sx(xv[i]):.1f}" cy="{sy(yv[i]):.1f}" r="3.2" fill="{"#e8842c" if agent[i] else "#9aa4b2"}" fill-opacity="0.85" stroke="{"#1b8a3c" if ok[i] else "none"}" stroke-width="2"/>')
    s.append(f'<text x="{W/2:.0f}" y="{H-14}" text-anchor="middle">{html.escape(xlab)}</text><text transform="translate(16,{H/2:.0f}) rotate(-90)" text-anchor="middle">{html.escape(ylab)}</text>')
    s.append(f'<text x="{W-P}" y="{P-8}" text-anchor="end" fill="#555">灰=导入的全波数据，橙=Agent验证，绿圈=全部达标，虚线=门限</text></svg>')
    Path(path).write_text('\n'.join(s),encoding='utf-8')

def tradeoff_sections(t,c,rows,x,y):
    """Constraint trade-offs, bound pressure and the agent's own convergence; the most conflicting constraint pair is drawn to tradeoff.svg."""
    lines=[];names=[m['name'] for m in c['metrics']];lim=[i for i,m in enumerate(c['metrics']) if 'limit' in m];z=normalize(x,c);L=loss(y,c);ids=[r['id'] for r in rows]
    if len(lim)<2 or len(rows)<12:return lines
    pm=np.column_stack([{'<=':y[:,i]<=m['limit'],'<':y[:,i]<m['limit'],'>=':y[:,i]>=m['limit'],'>':y[:,i]>m['limit']}[m['op']] for i,m in ((i,c['metrics'][i]) for i in lim)])
    lines+=['','## 约束取舍','','|指标|全体最优|对应设计|其他约束都满足时的最优|对应设计|','|---|---:|---|---:|---|']
    for k,i in enumerate(lim):
        pick=np.argmin if c['metrics'][i]['op'] in ['<=','<'] else np.argmax;b=int(pick(y[:,i]));ok=np.flatnonzero(np.delete(pm,k,axis=1).all(1))
        lines.append(f'|{names[i]}|{y[b,i]:.3f}|{ids[b]}|'+(f'{y[ok[pick(y[ok,i])],i]:.3f}|{ids[ok[pick(y[ok,i])]]}|' if len(ok) else '—|无设计满足其余约束|'))
    top=np.argsort(L)[:max(12,len(L)//4)];mg=margins(y[top],c);pairs=[]
    for a in range(len(lim)):
        for b in range(a+1,len(lim)):
            if mg[:,a].std()>0 and mg[:,b].std()>0:
                r=float(np.corrcoef(mg[:,a],mg[:,b])[0,1])
                if r<=-.3:pairs.append((a,b,r))
    lines+=['',f'较优的{len(top)}组设计中，余量相关系数≤−0.3的约束对（一项变好另一项变差）：'+('；'.join(f'{names[lim[a]]}–{names[lim[b]]}：r={r:+.2f}' for a,b,r in pairs) if pairs else '无')]
    a,b=min(pairs,key=lambda p:p[2])[:2] if pairs else (0,1);ia,ib=lim[a],lim[b];agent=np.array([i.startswith('batch_') for i in ids])
    scatter_svg(t/'tradeoff.svg',y[:,ia],y[:,ib],agent,pm.all(1),names[ia],names[ib],c['metrics'][ia]['limit'],c['metrics'][ib]['limit'])
    lines+=['',f'![{names[ia]} 与 {names[ib]} 的取舍散点](tradeoff.svg)']
    nb=near_bound(z[np.argsort(L)[:10]]);lines+=['','## 参数边界','','前10组设计中压在范围边界（±1%）的参数：'+(', '.join(f'{k}×{int(v)}' for k,v in zip(c['parameters'],nb.sum(0)) if v) or '无')]
    ag=np.flatnonzero(agent)
    if len(ag):
        nba=near_bound(z[ag]);lines.append(f'Agent验证过的{len(ag)}组中，{int(nba.any(1).sum())}组至少一个参数压边界：'+(', '.join(f'{k}×{int(v)}' for k,v in zip(c['parameters'],nba.sum(0)) if v) or '无')+'。若较优设计持续压边界，应考虑放宽范围并新建任务。')
        order=sorted(ag,key=lambda i:rows[i].get('imported',''));best=np.minimum.accumulate(L[order]);k=int(np.argmin(L[order]));rest=np.setdiff1d(np.arange(len(rows)),ag)
        lines+=['','## 收敛','',f'Agent已验证{len(order)}组：最差归一化余量从起始最优{f"{float(L[rest].min()):.3f}" if len(rest) else "—"}到{best[-1]:.3f}（越小越好，≤0为全部达标）；最近一次改进在第{k+1}组，此后{len(order)-k-1}组无改进。']
    return lines

def report(task):
    t=Path(task);c=read(t/'config.json');state=read(t/'state.json');db=read(t/'data.json');rows=db['rows'];lines=['# 仿真替代模型Agent运行报告','',f'有效参数组：{len(rows)}；重复记录：{len(db["repeats"])}；本任务启动CST次数：{state["simulations_started"]}。','']
    if state['model']:
        m=read(t/'models'/state['model']/'selection.json');lines += [f'当前模型：**{m["label"]}**。数据池{m["available_samples"]}组，实际拟合{m["fit_samples"]}组，{m["inputs"]}个输入，{m["outputs"]}个输出。', '',m['selection_rule'],'','离线留出误差（不等于前瞻全波验证）：','','|指标|MAE|P90|','|---|---:|---:|']
        for name,a,b in zip(m['metrics'],m['test']['MAE'],m['test']['P90']):lines.append(f'|{name}|{a:.4f}|{b:.4f}|')
        lines += ['',f'测试集{m["test"]["n"]}组，实际合格{m["test"]["actual_pass"]}组，误判合格{m["test"]["false_pass"]}组。若没有真实合格样本，不能据此估计合格检出率。','','算法选择比较（验证集，不是测试集）：','','|算法|验证评分|训练与预测秒数|','|---|---:|---:|']
        for r in m['candidate_results']:
            if 'score' in r:lines.append(f'|{r["label"]}|{r["score"]:.4f}|{r["seconds"]:.2f}|')
        if max(np.array(m['test']['MAE'])/np.array([v.get('scale',1) for v in c['metrics']]))>1:
            lines += ['','**区域留出误差较大：此模型用于引导补点，不能据其预测认证新设计达标。**']
        if m['data_sha256']!=digest(t/'data.json'):lines+=['','**已有新全波数据尚未回填模型；请train更新后再进行下一批代理优化。**']
    if rows:
        x,y=arrays(rows,c);order=np.argsort(loss(y,c));passmask=passes(y,c);lines += ['',f'当前已读全波数据中全部达标：{int(passmask.sum())}组。以下是已验证的前10组方案，排序依据最差归一化约束余量：','','|序号|'+ '|'.join(m['name'] for m in c['metrics'])+'|达标|','|---|'+ '|'.join('---:' for m in c['metrics'])+'|---|']
        for i in order[:10]:lines.append('|'+rows[i]['id']+'|'+'|'.join(f'{v:.4f}' for v in y[i])+'|'+('是' if passmask[i] else '否')+'|')
        write(t/'verified_designs.json',[dict(**rows[i],CST_pass=bool(passmask[i])) for i in order])
        lines+=tradeoff_sections(t,c,rows,x,y)
    if state['batch']:
        batch=read(t/'batches'/state['batch']/'batch.json');lines+=['','## 当前候选批次','','|候选|状态|预测达标|全波达标|预警|','|---|---|---|---|---|']
        for r in batch['candidates']:lines.append(f'|{r["id"]}|{r["state"]}|{r["predicted_pass"]}|{r.get("CST_pass","未验证")}|{"；".join(r.get("warnings",[])) or "—"}|')
        for r in batch['candidates']:
            if r.get('actual') and r.get('prediction'):
                lines += ['',f'候选 {r["id"]} 前瞻对照：','','|指标|冻结预测|CST实际|实际减预测|','|---|---:|---:|---:|']
                for name,v in r['actual'].items():lines.append(f'|{name}|{r["prediction"][name]:.4f}|{v:.4f}|{v-r["prediction"][name]:+.4f}|')
        lines+=['','完整参数、预测/实际对照、工程路径和原始Run ID见批次batch.json。']
    lines+=['','## 适用边界','','首版支持CST参数化频域模型，指标来自指定结果树及已保存的φ切面。更换结构需新建任务，不能复用旧结构模型。不确定度未校准，不能当成合格概率。频率离散点上的通过不等于未采样频率全带保证。没有找到合格方案只表示当前范围和预算内未找到。','', '贝叶斯末层网络只对末层进行近似贝叶斯推断；不是完整权重BNN，也不是Liu算法的完整复现。','']
    text='\n'.join(lines);svg=(t/'tradeoff.svg').read_text(encoding='utf-8') if (t/'tradeoff.svg').exists() else '';(t/'报告.md').write_text(text,encoding='utf-8');(t/'报告.html').write_text('<!doctype html><meta charset="utf-8"><title>仿真替代模型Agent</title><style>body{max-width:1100px;margin:40px auto;font:16px system-ui;background:#f7f9fc;color:#172b4d}pre{white-space:pre-wrap;line-height:1.7;background:white;padding:30px;border-radius:12px}svg{display:block;margin:20px auto;background:white;border-radius:12px}</style><pre>'+html.escape(text)+'</pre>'+svg,encoding='utf-8');return text

def run_loop(task,rounds,batch_size,budget):
    t=Path(task);before=read(t/'state.json')['simulations_started'];saved=0
    for _ in range(rounds):
        if (t/'PAUSE').exists():break
        remaining=budget-(read(t/'state.json')['simulations_started']-before)
        if remaining<=0:break
        state=read(t/'state.json');pending=False
        if state['batch']:pending=any(r['state']!='complete' for r in read(t/'batches'/state['batch']/'batch.json')['candidates'])
        if not pending:train(t);propose(t,batch_size)
        best_before=database_best(t)
        validate(t,remaining)
        state=read(t/'state.json')
        if state['phase']!='validated':break
        train(t)
        if database_best(t)<best_before-1e-9:saved+=auto_materialize(t,saved)
        _,y=arrays(read(t/'data.json')['rows'],read(t/'config.json'))
        if passes(y,read(t/'config.json')).any():event(t,'已找到全波达标方案，停止自动迭代。');break
    report(t)
