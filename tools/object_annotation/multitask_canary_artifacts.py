"""Package the completed canary for an offline desktop browser and reproducible review."""
import csv
import html
import json
from pathlib import Path
import zipfile

from annotate import BASE, PROJECT, sha, write_json
from multitask_reseed_canary import ROOT


def main():
    metrics=json.loads((ROOT/'metrics.json').read_text(encoding='utf-8'))
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    checked=json.loads((ROOT/'validation.json').read_text(encoding='utf-8'))
    assert checked['status']=='PASS'
    recovery=json.loads((ROOT/'recovery/validation.json').read_text(encoding='utf-8'))
    assert recovery['status']=='PASS'
    repaired={r['case_id']:r for r in recovery['outputs']}
    candidate=[]
    for c in cfg['cases']:
        path=Path(repaired[c['case_id']]['path']) if c['case_id'] in repaired else ROOT/c['case_id']/'predictions/multi_seed.jsonl.gz'
        candidate.append(dict(case_id=c['case_id'],start=c['start'],end=c['end'],objects=str(path),
            objects_sha256=sha(path),robot_parts=c['robot_parts'],semantic_acceptance=False))
    write_json(ROOT/'candidate_manifest.json',dict(status='development_candidate',cases=candidate,
        policy='Two streams have post-score local repairs; fixed benchmark metrics describe unrepaired outputs. No corpus promotion.'))
    with (ROOT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics['per_label'][0]));writer.writeheader();writer.writerows(metrics['per_label'])
    names={'frozen_production':'原有全库结果','single_seed':'一次辅助播种','multi_seed':'两次辅助播种'}
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>跨任务重播种检查</title>',
        '<style>body{background:#171a1d;color:#eee;font:16px system-ui;max-width:1350px;margin:24px auto;padding:16px}a{color:#8fd3ff}td,th{padding:9px;border:1px solid #555}table{border-collapse:collapse}video,img{width:100%;height:auto}section{margin:40px 0;border-top:1px solid #555;padding-top:15px}</style>',
        '<h1>跨任务 SAM2 重播种 canary</h1><p>9 个窗口，3 个任务，三路相机；已有人工 GT 只在评分帧显示。解压后直接打开本页，不需要服务器或联网。</p>',
        '<p>视频六格：原 RGB / 原全库结果 / 冻结机器人层；单次 seed / 两次 seed / 评分帧人工 GT。机器人层未在本轮修复或验收。</p>',
        '<p>UNKNOWN 是弃权，不等于物体已确认不可见。种子由助手看图后给点框，SAM2 精分；不是新增人工 GT，也不是无人干预自动化。</p>',
        '<table><tr><th>方案</th><th>可见目标平均 IoU</th><th>平均 Dice</th><th>IoU≥0.5</th><th>不可见帧假阳性</th></tr>']
    for mode,m in metrics['overall'].items():
        parts.append(f'<tr><td>{names[mode]}</td><td>{m["mean_iou_visible"]:.4f}</td><td>{m["mean_dice_visible"]:.4f}</td><td>{m["iou_ge_05"]}/{m["visible"]}</td><td>{m["absent_false_positives"]}/{m["absent"]}</td></tr>')
    parts+=['</table><p>这是反复使用的困难开发集，数值不代表全库准确率。详见 <a href="report.md">报告</a>、<a href="metrics.json">JSON</a>、<a href="metrics.csv">CSV</a>。</p>']
    for c in cfg['cases']:
        cid=c['case_id'];target=c['eval_frame']
        parts.append(f'<section><h2>{html.escape(cid)}</h2><p>范围 {c["start"]}–{c["end"]}；seeds {c["seed_frames"]}；GT {target}。评分距最近 seed 45 帧。</p><img loading="lazy" src="qa/{cid}/{target:05d}.jpg"><video controls preload="none" src="qa/{cid}/comparison.mp4"></video></section>')
        if cid in repaired:
            parts.append(f'<h3>评分后局部修复</h3><p>右格为修复候选；本结果没有回填上面的独立播种对照分数。UNKNOWN 终止可能舍弃真实可见像素。</p><video controls preload="none" src="recovery/{cid}/comparison.mp4"></video>')
    parts.append('</html>');(ROOT/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    report=PROJECT/'docs/multitask_reseed_canary_report_20260919.md';assert report.exists()
    outputs={}
    for name in ['index.html','config.json','metrics.json','metrics.csv','validation.json','visual_review.json','candidate_manifest.json','bank_probe.json']:
        outputs[name]=ROOT/name
    outputs['report.md']=report
    for path in (ROOT/'qa').rglob('*'):
        if path.is_file():outputs[str(path.relative_to(ROOT))]=path
    for path in (ROOT/'seeds').rglob('*'):
        if path.is_file():outputs[str(path.relative_to(ROOT))]=path
    for path in (ROOT/'recovery').rglob('*'):
        if path.is_file():outputs[str(path.relative_to(ROOT))]=path
    for c in cfg['cases']:
        for path in (ROOT/c['case_id']/'predictions').iterdir():outputs[str(path.relative_to(ROOT))]=path
    for path in ROOT.glob('run_*.log'):outputs[path.name]=path
    for name in ['multitask_reseed_canary.py','multitask_canary_artifacts.py','probe_multitask_canary_bank.py',
                 'repair_multitask_canary.py','config/multitask_reseed_canary_prompts.json','config/multitask_canary_repair.json']:
        outputs['code/'+name]=BASE/name
    hashes={name:sha(path) for name,path in sorted(outputs.items())}
    target=PROJECT/'deliverables/astribot_multitask_reseed_canary_20260919.zip';assert not target.exists()
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for name,path in sorted(outputs.items()):z.write(path,name)
        z.writestr('SHA256SUMS',''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    import hashlib
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    write_json(ROOT/'package.json',dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),verified_files=len(outputs)))
    print(json.dumps(json.loads((ROOT/'package.json').read_text()),indent=2))


if __name__=='__main__':main()
