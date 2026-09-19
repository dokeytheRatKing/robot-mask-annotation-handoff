"""Apply explicit visual suitability decisions into a new immutable bank version."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

from annotate import BASE, sha, write_json
from seed_bank import CAMERAS, PARTS, SeedBank, compatible


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); bank=SeedBank(a.bank); review=json.loads(a.review.read_text())
    assert review['bank_sha256']==sha(a.bank/'bank.json')
    ids={e['exemplar_id'] for e in bank.entries}
    assert set(review['exclude'])<=ids and set(review['tags'])<=ids
    shutil.copytree(a.bank,a.output)
    for e in bank.entries:
        eid=e['exemplar_id']; reason=review['exclude'].get(eid)
        e['bank_status']='retrieval_excluded' if reason else 'active'
        e['bank_selection_status']='assistant_inspected_pending_user_bank_review'
        e['bank_review']=dict(reviewer=review['reviewer'],reason=reason or review['default_reason'],
                              review_sha256=sha(a.review),human_bank_review='pending')
        if eid in review['tags']:
            e['assistant_tags']=review['tags'][eid]
    write_json(a.output/'bank.json',dict(schema='astribot.identity_seed_bank.v1',entries=bank.entries))
    shutil.copy2(a.review,a.output/'assistant_visual_review.json')
    coverage=[]
    all_ids=sorted({r['object_id'] for r in json.loads((BASE/'config/objects.json').read_text())}|set(PARTS))
    for oid in all_ids:
        for cam in CAMERAS:
            own=[e for e in bank.entries if e['object_id']==oid and e['camera']==cam and e['bank_status']=='active']
            usable=[e for e in bank.entries if compatible(e,oid) and e['camera']==cam and e['bank_status']=='active']
            coverage.append(dict(object_id=oid,camera=cam,active_direct=len(own),active_with_explicit_aliases=len(usable),
                episodes=len({e['episode_id'] for e in usable}),
                status='target_met' if len(usable)>=3 else 'sparse' if usable else 'missing'))
    write_json(a.output/'coverage.json',coverage)
    report=json.loads((a.bank/'build_report.json').read_text())
    report.update(parent_bank=str(a.bank),parent_bank_sha256=sha(a.bank/'bank.json'),
                  bank_sha256=sha(a.output/'bank.json'),curation_code_sha256=sha(__file__),
                  review_sha256=sha(a.review),active_exemplars=sum(e['bank_status']=='active' for e in bank.entries),
                  excluded_exemplars=len(review['exclude']),coverage_summary=dict(Counter(r['status'] for r in coverage)))
    write_json(a.output/'build_report.json',report)
    with (a.output/'review_checklist.csv').open('w') as f:
        keys=['exemplar_id','object_id','identity_name','camera','episode_id','frame_idx','bank_status','reason','user_decision','user_note']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for e in bank.entries:
            writer.writerow({k:e[k] for k in keys[:7]}|dict(reason=e['bank_review']['reason'],user_decision='',user_note=''))
    print(json.dumps(dict(active=report['active_exemplars'],excluded=report['excluded_exemplars'],coverage=report['coverage_summary']),indent=2))


if __name__=='__main__':main()
