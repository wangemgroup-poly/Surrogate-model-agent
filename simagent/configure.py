"""Interactive structure, variables, objectives and budget setup."""
from pathlib import Path
from .core import read,validate_config

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

def interactive(project,example):
    p=Path(project).resolve();assert p.is_file() and p.suffix.lower()=='.cst','需要现有CST结构文件'
    bounds=active_bounds(p)
    print('从工程读取到的优化参数范围：',bounds or '无')
    print('输入需要优化的参数名，用逗号分隔；回车沿用上面全部参数。')
    names=input('参数名：').strip();selected=names.replace('，',',').split(',') if names else list(bounds)
    params={}
    for n in selected:
        n=n.strip();old=bounds.get(n);v=input(f'{n} 的下限,上限'+(f'（回车采用 {old}）' if old else '')+'：').strip()
        params[n]=[float(x) for x in v.replace('，',',').split(',')] if v else old
    print('指标方案：1 当前10缝目标（17.8–20.2GHz，S11≤−13，端点SLL≤−10，增益浮动<2）；2 自定义')
    profile=input('请选择1或2：').strip()
    template=read(example)
    if profile=='1':metrics=template['metrics']
    elif profile=='2':
        metrics=[]
        for i in range(int(input('输出指标数量：'))):
            m=dict(name=input(f'指标{i+1}名称：').strip(),kind=input('类型 curve 或 sll_phi_cut：').strip())
            if m['kind']=='curve':
                m['tree']=input('CST结果树路径：').strip();m['transform']=input('变换 real/abs/db20：').strip();m['reduce']=input('归约 max/min/ripple/at：').strip()
                if m['reduce']=='at':m['frequency']=float(input('频率GHz：'))
                else:m['band']=[float(v) for v in input('频带起止GHz，以逗号分隔：').split(',')]
            else:m.update(frequency=float(input('频率GHz：')),phi=float(input('phi平面角度：')),port=int(input('端口号：')))
            op=input('约束 <= / < / >= / >；回车仅报告：').strip()
            if op:m.update(op=op,limit=float(input('门限值：')))
            m['scale']=float(input('误差评分尺度（回车1）：') or 1);metrics.append(m)
    else:raise ValueError('请选择1或2')
    install=input(f'CST安装目录（回车{template["cst_install"]}）：').strip().strip('"') or template['cst_install']
    c=dict(schema_version=1,project=str(p),cst_install=install,solver='frequency',parameters=params,metrics=metrics,simulation_budget=int(input('任务全生命周期最多启动CST次数：')),cpus=int(input('每次仿真CPU数（回车16）：') or 16),timeout_minutes=45,trust_radius=.05,purge_distance=.01)
    return validate_config(c)
