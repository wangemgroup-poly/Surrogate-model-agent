"""Chinese console entry point; all writes serialized per task."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'4')
import argparse,json,sys
from pathlib import Path
from simagent import engine,cst
from simagent.core import read,write,lock,event

ROOT=Path(__file__).resolve().parent

def discard(task):
    t=Path(task);s=read(t/'state.json')
    if not s.get('batch'):return
    p=t/'batches'/s['batch']/'batch.json';b=read(p)
    if any(r['state']=='running' for r in b['candidates']):raise ValueError('存在已启动求解，不能放弃或重复启动；先恢复收集结果。')
    for r in b['candidates']:
        if r['state']=='pending':r['state']='discarded'
    write(p,b);s.update(batch=None,phase='trained' if s['model'] else 'initialized');write(t/'state.json',s);event(t,'未运行候选已放弃，历史批次保留。')

def guided(project=None,lang=None,config=None,task=None):
    """Wizard: inspect a project, define targets, advise on sub-band splitting, then create the task."""
    from simagent.setup import wizard,ask,t
    path,projects,lang=wizard(project,lang,config)
    if ask(lang,'create',default='Y').lower() not in ('y','yes',''):print(t(lang,'skip',path=path));return None
    t2=Path(task or ask(lang,'task_path',default=str(ROOT/'tasks'/Path(read(path)['project']).stem))).resolve()
    with lock(t2):
        engine.init(t2,read(path))
        engine.import_cst(t2,projects if len(projects)>1 else None)
        engine.train(t2);engine.report(t2)
    print(t(lang,'done',path=t2))
    return t2

def dispatch(a):
    if a.command=='bounds':print(json.dumps(engine.union_bounds(a.project),ensure_ascii=False,indent=2));return
    if a.command=='setup':guided(a.project[0] if a.project else None,a.lang,a.config,a.task);return
    t=Path(a.task).resolve()
    if a.command=='pause':(t/'PAUSE').touch();print('已请求在当前求解结束后暂停。');return
    with lock(t):
        if a.command=='init':engine.init(t,read(a.config))
        elif a.command=='inspect':write(t/'inspection.json',cst.inspect(read(t/'config.json')));print(t/'inspection.json')
        elif a.command=='import-cst':engine.import_cst(t,a.project)
        elif a.command=='import-json':engine.import_json(t,a.file)
        elif a.command=='train':engine.train(t,a.force);engine.report(t)
        elif a.command=='propose':engine.propose(t,a.count);engine.report(t)
        elif a.command=='seed':engine.propose(t,seed_count=a.count);engine.report(t)
        elif a.command=='validate':engine.validate(t,a.budget)
        elif a.command=='resume':
            (t/'PAUSE').unlink(missing_ok=True);engine.validate(t,a.budget)
        elif a.command=='run':engine.run_loop(t,a.rounds,a.count,a.budget)
        elif a.command=='discard':discard(t)
        elif a.command=='rebaseline':engine.rebaseline(t)
        elif a.command=='materialize':engine.materialize(t,a.design)
        elif a.command in ('status','report'):print(engine.report(t))

def wizard():
    print('\n仿真替代模型 Agent 0.4  Copyright (C) 2026 MiraDaddy, Wangemgroup')
    print('本程序不提供任何担保，以 GNU GPL v3 或更新版本发布，欢迎在同一许可下再分发；详见 LICENSE。')
    print('选择结构 → 审计数据 → 自动比较算法 → 代理优化 → CST验收 → 回填更新')
    print('1 新建任务（引导式：导入工程、设定目标、分段建议）   2 打开已有任务   0 退出')
    print('1 Guided setup (import a project, define targets, sub-band advice)   2 Open an existing task   0 Quit')
    first=input('选择 / choose: ').strip()
    if first=='0':return
    if first=='1':
        t=guided()
        if t is None:return
    else:
        name=input('任务文件夹 / task directory: ').strip().strip('"');t=Path(name).resolve()
    if not (t/'config.json').exists():
        print('该目录还没有任务。新任务需要结构文件、可调参数及范围、性能目标和总仿真预算。')
        print('This directory holds no task yet. Give a prepared configuration JSON, or press Enter to use the guided setup.')
        cfg=input('配置JSON路径（回车转引导式建任务）/ config JSON path: ').strip().strip('"')
        if not cfg or Path(cfg).suffix.lower()=='.cst':
            if guided(cfg or None,task=str(t)) is None:return
        else:
            with lock(t):engine.init(t,read(cfg))
    while True:
        print('\n1 读取CST已有结果（可合并同结构其他版本）  2 导入全波指标JSON  3 自动选模/训练\n4 代理优化候选  5 CST验证/恢复  6 自动闭环\n7 初始全波采样计划  8 查看报告  9 检查工程  10 源工程重新登记（几何未变时）\n11 把最优设计写回源工程求解保存  0 退出')
        choice=input('选择：').strip()
        if choice=='0':break
        try:
            cmd={'1':'import-cst','2':'import-json','3':'train','4':'propose','5':'resume','6':'run','7':'seed','8':'report','9':'inspect','10':'rebaseline','11':'materialize'}[choice]
            args=dict(command=cmd,task=str(t),force=False,project=None,design=None)
            if choice=='11':args['design']=input('设计ID（回车取当前最优）：').strip() or None
            if choice=='1':args['project']=[s.strip().strip('"') for s in input('同一结构的其他工程文件（逗号分隔，回车只读源工程）：').replace('，',',').split(',') if s.strip()] or None
            if choice=='2':args['file']=input('全波指标JSON路径：').strip().strip('"')
            if choice in ('4','6','7'):args['count']=int(input('每批候选数/初始采样数：'))
            if choice in ('5','6'):args['budget']=int(input('本次最多启动多少次CST仿真：'))
            if choice=='6':args['rounds']=int(input('最多迭代多少轮：'))
            dispatch(argparse.Namespace(**args))
        except (Exception,KeyboardInterrupt) as e:print('操作停止：',e)

def main():
    p=argparse.ArgumentParser(description='CST全波数据驱动的替代模型优化Agent');sub=p.add_subparsers(dest='command')
    for cmd in ('init','inspect','import-cst','import-json','train','propose','seed','validate','resume','run','pause','discard','rebaseline','materialize','status','report'):
        q=sub.add_parser(cmd);q.add_argument('--task',required=True)
        if cmd=='init':q.add_argument('--config',required=True)
        if cmd=='materialize':q.add_argument('--design',help='写回哪个设计的ID；不给则取罚函数适应度最优的一组')
        if cmd=='import-cst':q.add_argument('--project',action='append',help='同一结构的其他工程文件，可重复；不给则只读任务源工程')
        if cmd=='import-json':q.add_argument('--file',required=True)
        if cmd=='train':q.add_argument('--force',action='store_true')
        if cmd in ('propose','seed','run'):q.add_argument('--count',type=int,default=3)
        if cmd in ('validate','resume','run'):q.add_argument('--budget',type=int,required=True)
        if cmd=='run':q.add_argument('--rounds',type=int,default=3)
    q=sub.add_parser('bounds');q.add_argument('--project',action='append',required=True,help='读取各工程优化器范围并给出并集')
    q=sub.add_parser('setup');q.add_argument('--project',action='append',help='CST工程路径；不给则交互询问')
    q.add_argument('--lang',choices=['en','zh'],help='向导语言；不给则交互询问')
    q.add_argument('--config',help='生成的配置文件路径');q.add_argument('--task',help='任务目录')
    a=p.parse_args()
    if a.command:dispatch(a)
    else:wizard()

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('已中断Agent；已启动的CST可能继续运行。使用resume收集，勿重复启动。');sys.exit(130)
    except Exception as e:print('错误：'+str(e),file=sys.stderr);sys.exit(1)
