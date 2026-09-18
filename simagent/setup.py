"""Guided task setup: inspect a CST project, define targets with the user, advise on sub-band splitting.

What this can check: whether the project is readable and closed, how many runs carry complete results,
which parameters actually vary, whether sibling versions share the same structure, which result trees and
far-field cuts exist, whether a target band lies on the stored frequency grid, and whether there is enough
data to train. What it cannot check: mesh convergence, port and boundary correctness, or whether the model
is the structure you intended — imported runs carry no solver log.
"""
import json
from pathlib import Path

import numpy as np

from . import cst
from .configure import active_bounds
from .core import read, write, validate_config
from .models import Predictor, eligible

TEXT = {
    'lang': ('Language / 语言:  1 English   2 中文', '语言 / Language:  1 English   2 中文'),
    'project': ('Path to the CST project (.cst)', 'CST 工程文件 (.cst) 路径'),
    'install': ('CST installation directory', 'CST 安装目录'),
    'not_cst': ('Not an existing .cst file', '不是现有的 .cst 文件'),
    'is_open': ('The project is open in CST (Model.lok exists). Save and close it first.',
                '工程正在 CST 中打开（存在 Model.lok），请先保存并关闭。'),
    'reading': ('Reading the result database …', '正在读取结果库 …'),
    'runs': ('Runs: {total} saved, {complete} with a readable parameter set', 'Run：共 {total} 个，其中 {complete} 个可读出参数'),
    'no_runs': ('This project has no saved runs. Use `seed` + `validate` to generate an initial sampling plan.',
                '该工程没有已保存的 Run。可用 `seed` + `validate` 在明确预算内生成初始样本。'),
    'siblings': ('Sibling projects of the same structure in the same folder:', '同目录下结构相同的其他工程：'),
    'no_siblings': ('No sibling project of the same structure found.', '同目录下没有找到结构相同的其他工程。'),
    'sib_diff': ('  {name}: different structure — {why}', '  {name}：结构不同 —— {why}'),
    'sib_ok': ('  [{i}] {name}: same structure, {runs} runs', '  [{i}] {name}：结构相同，{runs} 个 Run'),
    'merge': ('Merge which of them? Numbers separated by commas, empty = none',
              '要合并哪几个？用逗号分隔编号，回车表示都不合并'),
    'params_head': ('Parameters (optimiser range → observed in saved runs → suggested box):',
                    '参数（优化器范围 → 已保存样本的实际范围 → 建议的搜索范围）：'),
    'param_extra': ('  ! {name} varies across runs but is not an optimiser parameter — it must be included or the import fails',
                    '  ! {name} 在各 Run 之间变化，但不是优化器参数——必须纳入，否则导入会报错'),
    'accept_box': ('Use the suggested box? [Y/n], or type a parameter name to edit it',
                   '采用建议的搜索范围吗？[Y/n]，或输入参数名以修改'),
    'edit_param': ('{name} lower,upper', '{name} 的下限,上限'),
    'trees_head': ('1D result trees available in this project:', '本工程可用的一维结果树：'),
    'pick_tree': ('Result tree number for this target (or 0 to define a far-field side-lobe target)',
                  '本项目标使用哪条结果树（输入 0 则定义远场旁瓣类目标）'),
    'grid': ('Grid: {n} points, {lo:.4f}–{hi:.4f} GHz, step {step:.5f}', '网格：{n} 点，{lo:.4f}–{hi:.4f} GHz，步长 {step:.5f}'),
    'metric_name': ('Target name', '指标名称'),
    'transform': ('Value: 1 real   2 abs   3 dB (20·log10)', '取值方式：1 real   2 abs   3 dB（20·log10）'),
    'reduce': ('Reduce over the band: 1 max   2 min   3 ripple (max−min)   4 value at one frequency',
               '频带内归约：1 最大值   2 最小值   3 波动(max−min)   4 指定频点取值'),
    'band': ('Band lo,hi in GHz', '频带 起,止（GHz）'),
    'snapped': ('Endpoints snapped to the grid: {lo:.4f}, {hi:.4f}  (they must be exact samples)',
                '端点已对齐到采样点：{lo:.4f}、{hi:.4f}（端点必须正好是采样频点）'),
    'at_freq': ('Frequency in GHz', '频点（GHz）'),
    'op': ('Constraint: 1 ≤   2 <   3 ≥   4 >   5 report only', '约束：1 ≤   2 <   3 ≥   4 >   5 只报告不约束'),
    'limit': ('Limit value', '门限值'),
    'scale': ('Error scale for model selection [1]', '模型选择用的误差尺度 [1]'),
    'weight': ('Penalty weight — how bad a violation is [1]', '罚函数权重——违反它有多糟 [1]'),
    'sll_freq': ('Far-field cut frequency in GHz', '远场切面频率（GHz）'),
    'sll_phi': ('phi angle [0]', 'phi 角度 [0]'),
    'sll_port': ('port [1]', '端口 [1]'),
    'sll_missing': ('No stored cut found for {n} of {t} runs at this frequency/phi/port.',
                    '{t} 个 Run 中有 {n} 个在该频率/phi/端口下没有保存切面。'),
    'seg_offer': ('Analyse whether this target should be split into sub-bands? Reads the curve of every run [Y/n]',
                  '分析这项目标是否应该分段？需要读取每个 Run 的曲线 [Y/n]'),
    'seg_reading': ('Reading {n} curves …', '正在读取 {n} 条曲线 …'),
    'seg_spread': ('Worst point of each design sits between {lo:.3f} and {hi:.3f} GHz (90% of designs); '
                   'that is {frac:.0f}% of the band, in {k} cluster(s).',
                   '各设计的最差点落在 {lo:.3f}–{hi:.3f} GHz（90% 的设计），占频带的 {frac:.0f}%，形成 {k} 个聚集区。'),
    'seg_no': ('The worst point barely moves, so one whole-band target models well. No split suggested.',
               '最差点位置很集中，整带作为一个目标即可，不建议分段。'),
    'seg_yes': ('Suggested split: {segs}', '建议分段：{segs}'),
    'seg_why': ('A worst-over-band value whose location jumps between designs is hard for any surrogate to '
                'predict; modelling each sub-band separately usually cuts that error.',
                '"全带最差值"的位置在不同设计间跳动时，任何替代模型都难以预测；分段分别建模通常能降低这项误差。'),
    'seg_measure': ('Measure it on your data? Cross-validates whole-band vs segmented surrogates '
                    '({n} designs, may take a minute) [y/N]',
                    '要用你的数据实测验证吗？对整带与分段两种方式做交叉验证（{n} 组设计，可能需要一分钟）[y/N]'),
    'seg_result': ('Cross-validated error of the worst-in-band estimate: whole band {whole:.3f}, '
                   'segmented {seg:.3f} ({verdict})',
                   '"带内最差值"的交叉验证误差：整带 {whole:.3f}，分段 {seg:.3f}（{verdict}）'),
    'seg_better': ('segmenting is better', '分段更好'),
    'seg_worse': ('whole band is better', '整带更好'),
    'seg_adopt': ('Adopt the split? Each sub-band becomes its own target [Y/n]', '采用分段吗？每个子频带成为一项独立目标 [Y/n]'),
    'seg_limit': ('Limit for {lo:.3f}–{hi:.3f} GHz [{default}]', '{lo:.3f}–{hi:.3f} GHz 的门限 [{default}]'),
    'more': ('Define another target? [y/N]', '还要再定义一项目标吗？[y/N]'),
    'summary': ('--- Summary ---', '--- 摘要 ---'),
    'data_ok': ('{n} designs: enough to train (needs {need}) and to run SB-SADEA (needs {sb}).',
                '{n} 组设计：可以训练（需要 {need} 组），也可以跑 SB-SADEA（需要 {sb} 组）。'),
    'data_train_only': ('{n} designs: enough to train (needs {need}) but not for SB-SADEA (needs {sb}). '
                        'Use proposal_method "optimize", or add {gap} more designs.',
                        '{n} 组设计：够训练（需要 {need} 组），但不够跑 SB-SADEA（需要 {sb} 组）。'
                        '可改用 proposal_method "optimize"，或再补 {gap} 组。'),
    'data_short': ('{n} designs: not enough to train (needs {need}). Add {gap} more, e.g. `seed --count {gap}` '
                   'followed by `validate` within an explicit budget.',
                   '{n} 组设计：不够训练（需要 {need} 组）。还差 {gap} 组，可用 `seed --count {gap}` 再在明确预算内 `validate`。'),
    'already': ('{n} of the existing designs already meet every target.', '已有设计中有 {n} 组已经满足全部目标。'),
    'none_meet': ('No existing design meets every target yet.', '已有设计中还没有全部达标的。'),
    'cannot': ('Not checked by this wizard: mesh convergence, port and boundary correctness, and whether the '
               'model is the structure you intended. Imported runs carry no solver log.',
               '本向导查不了：网格收敛性、端口与边界设置是否正确、模型是否就是你想要的结构——导入的历史 Run 没有求解日志。'),
    'config_path': ('Write the configuration to', '配置文件写到'),
    'written': ('Configuration written: {path}', '配置已写入：{path}'),
    'create': ('Create the task and import the data now? No solver is started [Y/n]',
               '现在就建任务并导入数据吗？不会启动任何求解 [Y/n]'),
    'task_path': ('Task directory', '任务目录'),
    'done': ('Task ready: {path}\nNext: train, then propose — and only then decide a simulation budget.',
             '任务已就绪：{path}\n下一步：train，然后 propose——之后再决定仿真预算。'),
    'skip': ('Configuration saved. Create the task later with:\n  agent.py init --task <dir> --config {path}',
             '配置已保存。稍后可以这样建任务：\n  agent.py init --task <目录> --config {path}'),
}


def t(lang, key, **kw):
    return TEXT[key][0 if lang == 'en' else 1].format(**kw)


def ask(lang, key, default=None, **kw):
    prompt = t(lang, key, **kw)
    value = input(prompt + ('' if default is None else f' [{default}]') + ': ').strip().strip('"')
    return value or (default if default is not None else '')


def snap(freq, value):
    """Nearest stored sample: a band endpoint that is not an exact sample makes extraction fail."""
    freq = np.asarray(freq, float)
    return float(freq[int(np.argmin(abs(freq - value)))])


def snap_band(freq, lo, hi):
    """Band endpoints must be exact samples of the stored curve, otherwise extraction fails."""
    freq = np.asarray(freq, float)
    assert lo < hi, 'band must be increasing'
    assert freq.min() - 1e-9 <= lo and hi <= freq.max() + 1e-9, 'band lies outside the simulated frequency range'
    return snap(freq, lo), snap(freq, hi)


def worst_frequencies(freq, curves, band, reduce='max'):
    """Where in the band each design's worst value sits — the statistic behind the sub-band advice."""
    freq = np.asarray(freq, float); Y = np.asarray(curves, float)
    m = (freq >= band[0] - 1e-9) & (freq <= band[1] + 1e-9)
    pick = np.argmax if reduce == 'max' else np.argmin
    return freq[m][pick(Y[:, m], axis=1)]


def suggest_segments(freq, band, worst, max_segments=3, min_fraction=.15):
    """Split where the worst points cluster. Returns [] when they sit close together."""
    freq = np.asarray(freq, float); worst = np.sort(np.asarray(worst, float))
    width = band[1] - band[0]
    if len(worst) < 10 or width <= 0:
        return []
    spread = (np.quantile(worst, .95) - np.quantile(worst, .05)) / width
    gaps = np.diff(worst)
    order = np.argsort(gaps)[::-1]
    cuts = []
    for i in order:
        c = (worst[i] + worst[i + 1]) / 2
        if gaps[i] < min_fraction * width / 2:
            break
        if all(abs(c - x) >= min_fraction * width for x in cuts + [band[0], band[1]]):
            cuts.append(c)
        if len(cuts) >= max_segments - 1:
            break
    if spread < .25 or not cuts:
        return []
    edges = [band[0]] + sorted(snap(freq, c) for c in cuts) + [band[1]]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def segmentation_gain(x, freq, curves, band, segments, reduce='max', folds=5, seed=17):
    """Cross-validate both ways on the user's own data, estimating the same physical quantity:
    the worst value in the band, predicted directly vs as the worst of per-segment predictions."""
    freq = np.asarray(freq, float); Y = np.asarray(curves, float); x = np.asarray(x, float)
    lo, hi = x.min(0), x.max(0); z = (x - lo) / np.where(hi - lo > 0, hi - lo, 1)
    agg = np.max if reduce == 'max' else np.min
    band_mask = (freq >= band[0] - 1e-9) & (freq <= band[1] + 1e-9)
    whole = agg(Y[:, band_mask], axis=1)
    seg = np.column_stack([agg(Y[:, (freq >= a - 1e-9) & (freq <= b + 1e-9)], axis=1) for a, b in segments])
    algorithm = 'global_gp' if len(z) <= 600 else ('krr' if len(z) <= 3000 else 'extra_trees')
    rng = np.random.default_rng(seed); order = rng.permutation(len(z)); parts = np.array_split(order, folds)
    ew, es = [], []
    for k in range(folds):
        te = parts[k]; tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        if len(tr) < 10 or not len(te):
            continue
        pw = Predictor(algorithm).fit(z[tr], whole[tr][:, None]).predict(z[te])[0][:, 0]
        ps = Predictor(algorithm).fit(z[tr], seg[tr]).predict(z[te])[0]
        ew.append(abs(pw - whole[te])); es.append(abs(agg(ps, axis=1) - whole[te]))
    return dict(algorithm=algorithm, folds=len(ew), n=int(len(z)),
                whole=float(np.mean(np.concatenate(ew))), segmented=float(np.mean(np.concatenate(es))))


def _trees(mod):
    try:
        items = [str(i) for i in mod.get_tree_items()]
    except Exception:
        return []
    return [i for i in items if i.startswith(('1D Results', 'Tables')) and 'Adaptive' not in i]


def _observed(mod, runs, names):
    """Observed ranges of the optimiser parameters, plus any other numeric parameter that varies between
    runs — those must be added to the box or the import refuses the data."""
    box, missing, readable = {}, {}, 0
    for run in runs:
        try:
            p = mod.get_parameter_combination(run)
        except Exception:
            continue
        readable += 1
        for k, v in p.items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            d = box if k in names else missing
            d[k] = [min(v, d[k][0]), max(v, d[k][1])] if k in d else [v, v]
    return box, {k: v for k, v in missing.items() if v[1] - v[0] > 1e-12}, readable


def wizard(project=None, lang=None, config=None):
    """Interactive setup. Returns the path of the configuration written, or None if the user aborted."""
    if lang not in ('en', 'zh'):
        lang = 'en' if input(TEXT['lang'][0] + ': ').strip() != '2' else 'zh'
    project = Path(project or ask(lang, 'project'))
    assert project.is_file() and project.suffix.lower() == '.cst', t(lang, 'not_cst')
    assert not (project.with_suffix('') / 'Model.lok').exists(), t(lang, 'is_open')
    base = dict(schema_version=1, project=str(project.resolve()), solver='frequency', parameters={}, metrics=[],
                cst_install=ask(lang, 'install', default=r'C:\Program Files (x86)\CST Studio Suite 2025'))
    print(t(lang, 'reading'))
    probe = dict(base, parameters={}, metrics=[])
    mod = cst.module(probe, project)
    runs = [i for i in mod.get_all_run_ids() if i > 0]
    bounds = active_bounds(project)
    observed, missing, readable = _observed(mod, runs, set(bounds))
    print(t(lang, 'runs', total=len(runs), complete=readable))
    if not runs:
        print(t(lang, 'no_runs'))
    # ---- sibling projects of the same structure ----
    projects = [project]
    others = [p for p in sorted(project.parent.glob('*.cst')) if p != project and cst.has_geometry(p)]
    ok = []
    if others:
        print(t(lang, 'siblings'))
        for p in others:
            why = cst.structure_differences(project, p, set(bounds) | set(active_bounds(p)))
            if why:
                print(t(lang, 'sib_diff', name=p.name, why='; '.join(why)))
            else:
                ok.append(p)
                try:
                    n = len([i for i in cst.module(probe, p).get_all_run_ids() if i > 0])
                except Exception:
                    n = '?'
                print(t(lang, 'sib_ok', i=len(ok), name=p.name, runs=n))
        if ok:
            picked = ask(lang, 'merge')
            for s in picked.replace('，', ',').split(','):
                if s.strip().isdigit() and 1 <= int(s) <= len(ok):
                    projects.append(ok[int(s) - 1])
    else:
        print(t(lang, 'no_siblings'))
    for p in projects[1:]:
        m2 = cst.module(probe, p)
        r2 = [i for i in m2.get_all_run_ids() if i > 0]
        b2, miss2, _ = _observed(m2, r2, set(bounds))
        for k, v in b2.items():
            observed[k] = [min(v[0], observed.get(k, v)[0]), max(v[1], observed.get(k, v)[1])]
        missing.update(miss2)
        for k, v in active_bounds(p).items():
            bounds[k] = [min(v[0], bounds.get(k, v)[0]), max(v[1], bounds.get(k, v)[1])]
    # ---- parameter box ----
    box = {k: [min(v[0], observed.get(k, v)[0]), max(v[1], observed.get(k, v)[1])] for k, v in bounds.items()}
    for k, v in missing.items():
        box[k] = [min(v[0], box.get(k, v)[0]), max(v[1], box.get(k, v)[1])]
    print('\n' + t(lang, 'params_head'))
    for k, v in box.items():
        o = observed.get(k)
        print(f'  {k:18s} {bounds.get(k, ["-", "-"])} → {o if o else "-"} → [{v[0]:.5f}, {v[1]:.5f}]')
        if k in missing:
            print(t(lang, 'param_extra', name=k))
    while True:
        answer = ask(lang, 'accept_box', default='Y')
        if answer.lower() in ('y', 'yes', ''):
            break
        if answer in box:
            lo, hi = [float(s) for s in ask(lang, 'edit_param', name=answer).replace('，', ',').split(',')]
            box[answer] = [lo, hi]
        elif answer.lower() in ('n', 'no'):
            break
    base['parameters'] = box
    # ---- targets ----
    trees = _trees(mod)
    curves_cache = {}
    while True:
        print('\n' + t(lang, 'trees_head'))
        for i, tr in enumerate(trees, 1):
            print(f'  [{i}] {tr}')
        pick = ask(lang, 'pick_tree')
        if pick == '0':
            f = float(ask(lang, 'sll_freq')); phi = float(ask(lang, 'sll_phi', default='0')); port = int(ask(lang, 'sll_port', default='1'))
            name = ask(lang, 'metric_name', default=f'SLL_{f:g}GHz'.replace('.', 'p'))
            metric = dict(name=name, kind='sll_phi_cut', frequency=f, phi=phi, port=port)
            bad = 0
            for run in runs[:50]:
                try:
                    cst.extract(dict(base, metrics=[metric], parameters={}), project, run, mod)
                except Exception:
                    bad += 1
            if bad:
                print(t(lang, 'sll_missing', n=bad, t=min(50, len(runs))))
        elif pick.isdigit() and 1 <= int(pick) <= len(trees):
            tree = trees[int(pick) - 1]
            transform = {'1': 'real', '2': 'abs', '3': 'db20'}.get(ask(lang, 'transform', default='3'), 'db20')
            key = (tree, transform)
            if key not in curves_cache:
                print(t(lang, 'seg_reading', n=len(runs)))
                curves_cache[key] = cst.read_curves(probe, tree, transform, project=project)
            freq, Y, kept = curves_cache[key]
            step = float(np.median(np.diff(freq))) if len(freq) > 1 else 0.
            print(t(lang, 'grid', n=len(freq), lo=freq.min(), hi=freq.max(), step=step))
            reduce = {'1': 'max', '2': 'min', '3': 'ripple', '4': 'at'}.get(ask(lang, 'reduce', default='1'), 'max')
            metric = dict(name='', kind='curve', tree=tree, transform=transform, reduce=reduce)
            if reduce == 'at':
                metric['frequency'] = snap(freq, float(ask(lang, 'at_freq')))
            else:
                lo, hi = [float(s) for s in ask(lang, 'band').replace('，', ',').split(',')]
                lo, hi = snap_band(freq, lo, hi)
                print(t(lang, 'snapped', lo=lo, hi=hi))
                metric['band'] = [lo, hi]
            metric['name'] = ask(lang, 'metric_name', default=f'{Path(tree).name}_{metric.get("band", [0])[0]:g}'.replace('.', 'p'))
            segments = []
            if reduce in ('max', 'min') and len(Y) >= 10 and ask(lang, 'seg_offer', default='Y').lower() in ('y', 'yes', ''):
                worst = worst_frequencies(freq, Y, metric['band'], reduce)
                segs = suggest_segments(freq, metric['band'], worst)
                w = metric['band'][1] - metric['band'][0]
                q5, q95 = np.quantile(worst, .05), np.quantile(worst, .95)
                print(t(lang, 'seg_spread', lo=q5, hi=q95, frac=100 * (q95 - q5) / w, k=len(segs) or 1))
                if not segs:
                    print(t(lang, 'seg_no'))
                else:
                    print(t(lang, 'seg_yes', segs=', '.join(f'{a:.3f}–{b:.3f}' for a, b in segs)))
                    print(t(lang, 'seg_why'))
                    if ask(lang, 'seg_measure', default='N', n=len(Y)).lower() in ('y', 'yes'):
                        X = np.array([[float(mod.get_parameter_combination(r)[k]) for k in box] for r in kept])
                        g = segmentation_gain(X, freq, Y, metric['band'], segs, reduce)
                        verdict = t(lang, 'seg_better' if g['segmented'] < g['whole'] else 'seg_worse')
                        print(t(lang, 'seg_result', whole=g['whole'], seg=g['segmented'], verdict=verdict))
                    if ask(lang, 'seg_adopt', default='Y').lower() in ('y', 'yes', ''):
                        segments = segs
            op = {'1': '<=', '2': '<', '3': '>=', '4': '>'}.get(ask(lang, 'op', default='1'))
            limit = float(ask(lang, 'limit')) if op else None
            scale = float(ask(lang, 'scale', default='1'))
            weight = float(ask(lang, 'weight', default='1'))
            targets = []
            if segments:
                for a, b in segments:
                    m = dict(metric, name=f'{metric["name"]}_{a:g}_{b:g}'.replace('.', 'p'), band=[a, b])
                    lim = float(ask(lang, 'seg_limit', lo=a, hi=b, default=limit))
                    targets.append((m, lim))
            else:
                targets.append((metric, limit))
            for m, lim in targets:
                if op:
                    m.update(op=op, limit=lim)
                m['scale'] = scale
                if weight != 1:
                    m['penalty_weight'] = weight
                base['metrics'].append(m)
            if ask(lang, 'more', default='N').lower() not in ('y', 'yes'):
                break
            continue
        else:
            continue
        base['metrics'].append(metric)
        if ask(lang, 'more', default='N').lower() not in ('y', 'yes'):
            break
    # ---- summary ----
    base.update(simulation_budget=0, cpus=16, timeout_minutes=45, proposal_method='sb_sadea',
                sb_sadea={'surrogate': 'gp'}, include_global_gp=True, materialize={'on': 'improvement', 'max': 3})
    validate_config(base)
    d = len(base['parameters']); need = max(30, 3 * d); sb = 4 * d; n = len(runs)
    print('\n' + t(lang, 'summary'))
    if n >= sb and n >= need:
        print(t(lang, 'data_ok', n=n, need=need, sb=sb))
    elif n >= need:
        print(t(lang, 'data_train_only', n=n, need=need, sb=sb, gap=sb - n))
    else:
        print(t(lang, 'data_short', n=n, need=need, gap=need - n))
    print(t(lang, 'cannot'))
    path = Path(config or ask(lang, 'config_path', default=str(Path.cwd() / f'config_{project.stem}.json')))
    write(path, base)
    print(t(lang, 'written', path=path))
    return path, projects, lang
