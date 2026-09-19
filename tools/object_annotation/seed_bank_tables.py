"""Export auditable ranking/abstention breakdowns from frozen pilot queries."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from annotate import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(s) for s in (a.pilot/'queries.jsonl').read_text().splitlines()]
    groups=defaultdict(list)
    for r in rows:
        kind='objects' if r['object_id']<1000 else 'robot_parts'
        for key in (kind,kind+'/'+r['camera'],str(r['object_id'])+'/'+r['camera']):groups[key].append(r)
    results=[]
    for group,rs in groups.items():
        visible=[r for r in rs if r['gt_visible']]
        for mode in ('detector','rgb','masked','hybrid'):
            accepted=[r for r in rs if r['methods'][mode]['accepted']]
            correct=sum(r['gt_visible'] and r['methods'][mode]['iou']>=.5 for r in accepted)
            improved=sum(r['methods']['detector']['iou']<.5<=r['methods'][mode]['iou'] for r in visible)
            regressed=sum(r['methods'][mode]['iou']<.5<=r['methods']['detector']['iou'] for r in visible)
            results.append(dict(group=group,mode=mode,queries=len(rs),visible=len(visible),
                absent=len(rs)-len(visible),oracle_valid=sum(r['oracle_iou']>=.5 for r in visible),
                top1_valid=sum(r['methods'][mode]['iou']>=.5 for r in visible),
                mean_iou=sum(r['methods'][mode]['iou'] for r in visible)/len(visible) if visible else None,
                rescued=improved,regressed=regressed,accepted=len(accepted),accepted_valid=correct,
                accepted_bad_mask=sum(r['gt_visible'] and r['methods'][mode]['iou']<.5 for r in accepted),
                absent_false_accepts=sum(not r['gt_visible'] for r in accepted),
                accepted_valid_fraction=correct/len(accepted) if accepted else None))
    write_json(a.pilot/'comparison_tables.json',dict(queries_sha256=sha(a.pilot/'queries.jsonl'),rows=results))
    with (a.pilot/'comparison_tables.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    for r in results:
        if r['group'] in ('objects','robot_parts'):print(json.dumps(r))


if __name__=='__main__':main()
