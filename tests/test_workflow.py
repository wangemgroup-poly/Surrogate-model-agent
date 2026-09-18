import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from simagent.core import read,write,ingest,passes,signature,config_signature,validate_config,fitness,digest
from simagent import engine,cst
from simagent.models import Predictor,eligible

class Workflow(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.t=Path(self.tmp.name)/'task';p=Path(self.tmp.name)/'source.cst';p.write_bytes(b'original')
        self.c=dict(schema_version=1,project=str(p),parameters={'x':[0,1]},metrics=[dict(name='s',kind='curve',tree='S11',reduce='max',op='<=',limit=-13),dict(name='g',kind='curve',tree='gain',reduce='ripple',op='<',limit=2)],simulation_budget=1)
        engine.init(self.t,self.c)
    def tearDown(self):self.tmp.cleanup()
    def test_strict_and_inclusive_boundaries(self):
        self.assertEqual(passes([[-13,1.99],[-13,2],[-12.99,1]],self.c).tolist(),[True,False,False])
    def test_duplicate_reimports_and_nonfinite_are_not_independent(self):
        r=dict(id='a',parameters={'x':.5},metrics={'s':-14,'g':1})
        ingest(self.t,[r,r],{});ingest(self.t,[r,dict(id='b',parameters={'x':.5},metrics={'s':-13.9,'g':1}),dict(id='nan',parameters={'x':.3},metrics={'s':float('nan'),'g':1})],{})
        d=read(self.t/'data.json');self.assertEqual((len(d['rows']),len(d['repeats'])),(1,1));self.assertEqual(len(read(self.t/'last_import.json')['rejected']),1)
    def test_zero_budget_and_pause_never_launch_solver(self):
        engine.propose(self.t,seed_count=2)
        with patch.object(cst,'prepare') as prepare:
            engine.validate(self.t,0);prepare.assert_not_called()
            (self.t/'PAUSE').touch();engine.validate(self.t,1);prepare.assert_not_called()
    def test_budget_resume_collects_existing_job_without_new_launch(self):
        b=engine.propose(self.t,seed_count=2)
        result=dict(parameters=b['candidates'][0]['parameters'],metrics={'s':-12,'g':1},provenance={'run_id':1})
        def prepared(c,d,parameters):write(d/'job.json',dict(state='prepared'))
        with patch.object(cst,'prepare',side_effect=prepared),patch.object(cst,'run',return_value=result) as run:
            engine.validate(self.t,1);self.assertEqual(run.call_count,1)
            engine.validate(self.t,1);self.assertEqual(run.call_count,1)
        self.assertEqual(read(self.t/'state.json')['simulations_started'],1)
    def test_changed_config_rejects_frozen_candidates(self):
        engine.propose(self.t,seed_count=1);c=read(self.t/'config.json');c['metrics'][0]['limit']=-15;write(self.t/'config.json',c)
        with self.assertRaises(AssertionError):engine.validate(self.t,1)
    def test_source_version_change_rejected_before_copy(self):
        c=read(self.t/'config.json');Path(c['project']).write_bytes(b'new')
        with self.assertRaises(AssertionError):cst.prepare(c,self.t/'job',{'x':.5})
        self.assertFalse((self.t/'job/project.cst').exists())
    def test_scale_policy_and_bayesian_last_layer_shapes(self):
        models,reasons=eligible(10000,60,40);self.assertNotIn('krr',models);self.assertIn('bayes_last_layer',models)
        x=np.random.default_rng(8).random((40,3));y=np.column_stack([x[:,0]**2,x[:,1]])
        with engine.threadpoolctl_context():
            m=Predictor('bayes_last_layer').fit(x,y);mu,std=m.predict(x[:3])
        self.assertEqual(mu.shape,(3,2));self.assertTrue(np.isfinite(mu).all());self.assertTrue((std>=0).all())
    def test_online_addition_preserves_evaluation_region_membership(self):
        z=np.random.default_rng(11).random((60,1))
        with engine.threadpoolctl_context():
            before=engine.region_labels(self.t,z,self.c)
            after=engine.region_labels(self.t,np.r_[z,[[.55],[.99]]],self.c)
        np.testing.assert_array_equal(before,after[:len(z)])
    def test_opt_in_global_gp_learns_distant_new_sample(self):
        self.assertNotIn('global_gp',eligible(40,1,1)[0])
        self.assertIn('global_gp',eligible(40,1,1,True)[0])
        x=np.linspace(0,.4,24)[:,None];y=np.sin(x*4)
        with engine.threadpoolctl_context():
            before=Predictor('global_gp').fit(x,y).predict([[1.]])[0][0,0]
            updated=Predictor('global_gp').fit(np.r_[x,[[1.]]],np.r_[y,[[-2.]]])
            after=updated.predict([[1.]])[0][0,0]
        self.assertLess(abs(after+2),abs(before+2)/2)
        self.assertLess(abs(after+2),.1)

    def test_surrogate_optimization_respects_domain_and_distance_rules(self):
        rng=np.random.default_rng(5);z=rng.random((40,2));y=np.column_stack([-15+10*((z-.7)**2).sum(1),1+((z-.3)**2).sum(1)])
        c=dict(self.c,parameters={'a':[0,1],'b':[0,1]},min_candidate_distance=.03,candidate_separation=.05,optimizer_iterations=60)
        with engine.threadpoolctl_context():model=Predictor('global_gp').fit(z,y)
        center=np.array([.2,.2])
        q=engine.optimize_point(model,dict(c,search_domain='trust_region'),z,np.empty((0,2)),center,.1,explore=False,seed=1)
        self.assertLessEqual(np.linalg.norm(q-center),.1+1e-9);self.assertGreaterEqual(np.linalg.norm(z-q,axis=1).min(),.03-1e-9)
        taken=engine.optimize_point(model,dict(c,search_domain='full'),z,np.empty((0,2)),center,.1,explore=False,seed=1)
        self.assertTrue(((taken>=0)&(taken<=1)).all())
        second=engine.optimize_point(model,dict(c,search_domain='full'),z,taken[None],center,.1,explore=True,seed=2)
        self.assertGreaterEqual(np.linalg.norm(second-taken),.05-1e-9)
        with self.assertRaises(AssertionError):validate_config(dict(self.c,proposal_method='bogus'))

    def test_liu_penalty_fitness(self):
        np.testing.assert_allclose(fitness([[-14,1.5],[-12,2.5],[-13,1.9]],self.c),[0,1.5,0])
    def test_sb_sadea_de_children_and_adaptive_lcb_switch(self):
        rng=np.random.default_rng(3);z=rng.random((30,2));y=np.column_stack([-10-3*z[:,0],1+z[:,1]])
        c=dict(self.c,sb_sadea={'lambda':8,'tau':10})
        U=engine.de_children(z[:8],np.random.default_rng(1));self.assertEqual(U.shape,(8,2));self.assertTrue(((U>=0)&(U<=1)).all())
        seen={}
        def fake(tx,ty,q):
            seen['shape']=tx.shape;return np.column_stack([-12-q[:,0],1.5+0*q[:,0]]),np.column_stack([.1+q[:,1],.1+0*q[:,1]]),{}
        self.assertEqual(engine.sb_sadea_select(z,y,c,[],np.random.default_rng(2),fake)['role'],'sb_sadea_mean')
        self.assertEqual(seen['shape'][1:],(10,2))
        self.assertEqual(engine.sb_sadea_select(z,y,c,[5.0]*9+[5.1],np.random.default_rng(2),fake)['role'],'sb_sadea_lcb')
        self.assertEqual(engine.sb_sadea_select(z,y,c,[],np.random.default_rng(2),fake)['omega'],2)
        self.assertEqual(engine.sb_sadea_select(z,y,dict(c,sb_sadea={'lambda':8,'tau':10,'surrogate':'bnn'}),[],np.random.default_rng(2),fake)['omega'],14)
        with self.assertRaises(AssertionError):validate_config(dict(self.c,sb_sadea={'surrogate':'svm'}))
    def test_batched_bnn_shapes(self):
        try:import torch
        except ImportError:self.skipTest('torch未安装')
        from simagent.models import batched_bnn_predict
        rng=np.random.default_rng(0);tx=rng.random((3,40,2));ty=np.stack([np.column_stack([np.sin(3*a[:,0]),a[:,1]]) for a in tx])
        mu,sd,info=batched_bnn_predict(tx,ty,rng.random((3,2)),max_steps=300,samples=50,device='cpu')
        self.assertEqual(mu.shape,(3,2));self.assertTrue(np.isfinite(mu).all() and (sd>=0).all())

    def test_config_signature_ignores_execution_settings(self):
        c=read(self.t/'config.json');a=config_signature(c)
        self.assertEqual(a,config_signature(dict(c,simulation_budget=99,cpus=8,timeout_minutes=5,source_sha256='x'*64)))
        self.assertNotEqual(a,config_signature(dict(c,metrics=[dict(c['metrics'][0],limit=-15),c['metrics'][1]])))
        engine.propose(self.t,seed_count=1);c['simulation_budget']=5;write(self.t/'config.json',c)
        with patch.object(cst,'prepare') as prepare:engine.validate(self.t,0);prepare.assert_not_called()
    def _fake_project(self,name,mod=b'geometry',hist=b'history'):
        p=Path(self.tmp.name)/f'{name}.cst';p.write_bytes(b'cst-'+name.encode());d=p.with_suffix('')/'Model/3D';d.mkdir(parents=True);(d/'Model.mod').write_bytes(mod);(d/'ModelHistory.json').write_bytes(hist);return p
    def test_geometry_identity_survives_saved_runs(self):
        a=self._fake_project('A');b=self._fake_project('B');other=self._fake_project('C',mod=b'different')
        self.assertTrue(cst.same_structure(a,b));self.assertFalse(cst.same_structure(a,other))
        c=dict(self.c,project=str(a),source_sha256=digest(a),geometry_sha256=cst.geometry_signature(a))
        a.write_bytes(b'cst-A plus new saved runs');cst.check_source(c)
        with self.assertRaises(AssertionError):cst.check_source(dict(c,geometry_sha256=None))
        (a.with_suffix('')/'Model/3D/Model.mod').write_bytes(b'changed')
        with self.assertRaises(AssertionError):cst.check_source(c)
        t=Path(self.tmp.name)/'geo_task';engine.init(t,dict(self.c,project=str(b)));self.assertEqual(len(read(t/'config.json')['geometry_sha256']),64)
        (b.with_suffix('')/'Model/simulationproperties.docstore').write_bytes(b'cpus changed by macro')
        self.assertFalse(cst.same_structure(b,self._fake_project('D')));self.assertEqual(cst.geometry_signature(b,cst.GEOMETRY_FILES[:2]),cst.geometry_signature(Path(self.tmp.name)/'D.cst',cst.GEOMETRY_FILES[:2]))

    def test_multi_project_import_requires_identical_geometry(self):
        src=self._fake_project('S');sib=self._fake_project('T');alien=self._fake_project('U',hist=b'other')
        t=Path(self.tmp.name)/'merge';engine.init(t,dict(self.c,project=str(src)))
        rows=lambda c,p:([dict(id='CST_Run_1',parameters={'x':.4 if Path(p).stem=='S' else .6},metrics={'s':-12,'g':1},provenance=dict(project=str(p),run_id=1))],[])
        with patch.object(cst,'import_existing',side_effect=rows):
            engine.import_cst(t,[src,sib]);ids=[r['id'] for r in read(t/'data.json')['rows']]
            self.assertEqual(ids,['CST_Run_1','T_CST_Run_1'])
            with self.assertRaises(AssertionError):engine.import_cst(t,[alien])
        self.assertEqual(engine.version_tag('example_array_2stage_V5_2.cst'),'V5_2')
        outside=lambda c,p:([dict(id='CST_Run_9',parameters={'x':1.5},metrics={'s':-12,'g':1},provenance=dict(project=str(p),run_id=9))],[])
        with patch.object(cst,'import_existing',side_effect=outside):engine.import_cst(t,[src])
        self.assertEqual(read(t/'bounds_advice.json')['suggested']['x'],[0,1.5]);self.assertEqual(read(t/'import_summary.json')[str(src.resolve())]['rejected'],1)
    def test_candidate_warnings_flag_extrapolation_and_disagreement(self):
        rng=np.random.default_rng(4);x=rng.random((50,1))*.3;rows=[dict(id=f'r{i}',parameters={'x':float(v)},metrics={'s':float(-10-5*v),'g':float(1+v)}) for i,v in enumerate(x[:,0])]
        ingest(self.t,rows,{});c=read(self.t/'config.json');xx,yy=engine.arrays(read(self.t/'data.json')['rows'],c);z=engine.normalize(xx,c)
        v0=float(xx[0,0]);near=dict(parameters={'x':v0},prediction={'s':-10-5*v0,'g':1+v0});far=dict(parameters={'x':.95},prediction={'s':-40.,'g':1.})
        with engine.threadpoolctl_context():flags=engine.annotate(self.t,c,z,yy,[near,far])
        self.assertEqual(near['warnings'],[]);self.assertEqual(flags,dict(extrapolation=1,unsupported=1));self.assertEqual(len(far['warnings']),2)
        self.assertIn('extra_trees',far['cross_check']['algorithms'])
    def test_materialize_writes_best_design_into_source_project(self):
        ingest(self.t,[dict(id='a',parameters={'x':.5},metrics={'s':-14,'g':1}),dict(id='b',parameters={'x':.6},metrics={'s':-12,'g':1})],{})
        fake=dict(run_id=7,parameters={'x':.5},metrics={'s':-13.9,'g':1.05},backup='B',sha256_before='0'*64,sha256_after='1'*64,geometry_sha256='2'*64,seconds=1.)
        with patch.object(cst,'materialize',return_value=fake) as mat:
            e=engine.materialize(self.t)
            self.assertEqual(mat.call_args[0][1],{'x':.5})          # 取适应度最优的一组
            self.assertEqual(e['design'],'a');self.assertAlmostEqual(e['difference']['s'],.1)
            self.assertEqual(read(self.t/'config.json')['source_sha256'],'1'*64)
            self.assertIsNone(engine.materialize(self.t));self.assertEqual(mat.call_count,1)   # 已写回过不重复求解
        c=read(self.t/'config.json');c['materialize']={'on':'never'};write(self.t/'config.json',c)
        with patch.object(cst,'materialize') as skip:
            self.assertEqual(engine.auto_materialize(self.t,0),0);skip.assert_not_called()
        c['materialize']={'on':'improvement','max':1};write(self.t/'config.json',c)
        with patch.object(cst,'materialize',side_effect=RuntimeError('源工程正在打开')):
            self.assertEqual(engine.auto_materialize(self.t,0),0)   # 失败不中断优化
            self.assertEqual(engine.auto_materialize(self.t,1),0)   # 达到上限不再写回
        with self.assertRaises(AssertionError):validate_config(dict(self.c,materialize={'on':'always'}))
    def test_fixed_parameter_values_are_part_of_structure(self):
        import json
        def params(p,**kv):(p.with_suffix('')/'Model/Parameters.json').write_text(json.dumps(dict(parameters=[dict(name=k,expr=str(v)) for k,v in kv.items()])),encoding='utf-8')
        a=self._fake_project('PA');b=self._fake_project('PB');params(a,x=.3,b=7.8);params(b,x=.9,b=7.8)
        self.assertEqual(cst.structure_differences(a,b,['x']),[])                          # 只有优化参数不同：同一结构
        params(b,x=.9,b=7.5);self.assertTrue(any('取值不同' in r for r in cst.structure_differences(a,b,['x'])))   # 几何历史相同但固定尺寸不同
        params(b,x=.9,b=7.8,oy2=1.);self.assertTrue(any('定义不同' in r for r in cst.structure_differences(a,b,['x'])))
        t=Path(self.tmp.name)/'fixed_task';engine.init(t,dict(self.c,project=str(a)));c=read(t/'config.json')
        self.assertEqual(len(c['fixed_parameters_sha256']),64);cst.check_source(c)
        params(a,x=.5,b=7.8);cst.check_source(c)                                           # 写入优化参数不影响
        params(a,x=.5,b=7.5)
        with self.assertRaises(AssertionError):cst.check_source(c)                          # 改了固定尺寸被拒
    def test_setup_band_snapping_and_subband_advice(self):
        from simagent import setup
        freq=np.linspace(17.,21.,1001)
        self.assertEqual(setup.snap_band(freq,17.8,20.2),(17.8,20.2))
        lo,hi=setup.snap_band(freq,17.8013,20.1994);self.assertAlmostEqual(lo,17.8,3);self.assertAlmostEqual(hi,20.2,3)
        with self.assertRaises(AssertionError):setup.snap_band(freq,16.5,20.2)      # 超出仿真频率范围
        band=(17.8,20.2);rng=np.random.default_rng(3);n=60
        # 最差点在两处来回跳：应建议分段
        jump=np.array([[-20+10*np.exp(-((f-(18.0 if i%2 else 19.8))**2)/.02) for f in freq] for i in range(n)])
        worst=setup.worst_frequencies(freq,jump,band,'max')
        self.assertLess(abs(np.median(worst[::2])-19.8),.05);self.assertLess(abs(np.median(worst[1::2])-18.0),.05)
        segs=setup.suggest_segments(freq,band,worst)
        self.assertTrue(segs);self.assertEqual(segs[0][0],band[0]);self.assertEqual(segs[-1][1],band[1])
        self.assertTrue(all(a<b for a,b in segs) and all(segs[i][1]==segs[i+1][0] for i in range(len(segs)-1)))
        # 最差点始终在同一处：不建议分段
        fixed=np.array([[-20+10*np.exp(-((f-19.0)**2)/.02)+.01*i for f in freq] for i in range(n)])
        self.assertEqual(setup.suggest_segments(freq,band,setup.worst_frequencies(freq,fixed,band,'max')),[])
        # 峰高随参数连续变化：全带最差值在两峰交替处出现折点（难预测），分段后每段都是平滑的
        u=np.linspace(0,1,n);base=-25.
        peaks=lambda hA,hB:base+np.maximum(hA*np.exp(-((freq-18.0)**2)/.02),hB*np.exp(-((freq-19.8)**2)/.02))
        kinked=np.array([peaks(15-5*v,15-5*(1-v)) for v in u])
        ks=setup.suggest_segments(freq,band,setup.worst_frequencies(freq,kinked,band,'max'))
        self.assertTrue(ks)
        with engine.threadpoolctl_context():
            g=setup.segmentation_gain(u[:,None],freq,kinked,band,ks,'max',folds=3)
        self.assertEqual((g['n'],g['folds']),(n,3));self.assertTrue(np.isfinite([g['whole'],g['segmented']]).all())
        self.assertGreaterEqual(min(g['whole'],g['segmented']),0)
        self.assertLessEqual(g['segmented'],g['whole'])   # 正是分段建议针对的情形
    def test_backup_copy_survives_paths_beyond_max_path(self):
        import os,shutil
        if os.name!='nt':self.skipTest('仅Windows有MAX_PATH限制')
        src=Path(self.tmp.name)/'src';deep=src.joinpath(*['d'*48]*5);os.makedirs(cst.long_path(deep))
        with open(cst.long_path(deep/'Scaled.sig'),'wb') as f:f.write(b'x')
        dst=Path(self.tmp.name)/'backup_with_a_fairly_long_name'/'example_array_2stage_V5_2'
        self.assertGreater(len(str(dst/deep.relative_to(src)/'Scaled.sig')),260)
        shutil.copytree(cst.long_path(src),cst.long_path(dst))
        self.assertTrue(os.path.exists(cst.long_path(dst/deep.relative_to(src)/'Scaled.sig')))
        shutil.rmtree(cst.long_path(Path(self.tmp.name)/'backup_with_a_fairly_long_name'));shutil.rmtree(cst.long_path(src))
    def test_conservative_constraints_and_report_sections(self):
        np.testing.assert_allclose(engine.pessimistic(np.array([[-13.5,1.5]]),np.array([[1.,.2]]),self.c,2.),[[-11.5,1.9]])
        rng=np.random.default_rng(6);v=rng.random(30);rows=[dict(id=('batch_a%02d'%i if i%3==0 else 'CST_Run_%d'%i),parameters={'x':float(u)},metrics={'s':float(-15+6*u),'g':float(2.5-2*u)}) for i,u in enumerate(v)]
        ingest(self.t,rows,{});text=engine.report(self.t)
        for key in ['## 约束取舍','## 参数边界','## 收敛','tradeoff.svg']:self.assertIn(key,text)
        self.assertTrue((self.t/'tradeoff.svg').exists());self.assertIn('<svg',(self.t/'报告.html').read_text(encoding='utf-8'))

    def test_reimport_is_idempotent_despite_numpy_rounding(self):
        """A value where numpy's scaled round and Python's decimal round disagree at 1e-11.

        Before the fix the second import of the same project silently appended those samples
        again, quietly weighting them twice in training."""
        from simagent.core import sample_key
        v=0.912436941965
        self.assertNotEqual(round(np.float64(v),11),round(float(v),11))   # the trap this guards
        self.assertEqual(sample_key([np.float64(v)]),sample_key([v]))
        rows=[dict(id='R1',parameters={'x':v},metrics={'s':-14.,'g':1.})]
        ingest(self.t,rows,{});n=len(read(self.t/'data.json')['rows'])
        self.assertEqual(n,1)
        ingest(self.t,rows,{})
        self.assertEqual(len(read(self.t/'data.json')['rows']),n,'重复导入同一工程必须幂等')

class ShippedFiles(unittest.TestCase):
    """The files a new user copies must actually load; a broken example costs more than a broken test."""
    def test_examples_are_valid_json_and_config(self):
        import json
        root=Path(__file__).resolve().parents[1]
        for f in sorted((root/'examples').glob('*.json')):
            with self.subTest(file=f.name):
                c=json.loads(f.read_text(encoding='utf-8'))
                if c.get('schema_version')==1:validate_config(c)

    def test_no_dangling_example_paths(self):
        root=Path(__file__).resolve().parents[1]
        import re
        for f in sorted(root.glob('*.py'))+sorted((root/'simagent').glob('*.py')):
            for ref in re.findall(r"examples/[A-Za-z0-9_.-]+",f.read_text(encoding='utf-8')):
                with self.subTest(file=f.name,ref=ref):self.assertTrue((root/ref).exists(),ref)

if __name__=='__main__':unittest.main()
