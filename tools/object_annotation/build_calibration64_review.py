"""Materialize recorded assistant decisions; never creates an automatic semantic PASS."""
import json
from annotate import sha, write_json
from calibration64 import ROOT, load, read


def main():
    cfg=load(ROOT);notes=read(ROOT,'assistant_review_notes.json');clips={}
    decisions=read(ROOT/'repairs','review_decisions.json');repair_reviews={}
    for number,r in decisions.items():
        c=cfg['clips'][int(number)];folder=r.get('repair_folder','repairs');path=ROOT/folder/c['clip_id']
        meta=read(path,'complete.json');assert sha(path/'objects.jsonl.gz')==meta['output_sha256']
        repair_reviews[c['clip_id']]=dict(r,output_sha256=meta['output_sha256'],reviewed_frames=meta['checks'],
            evidence_sha256={p.name:sha(p) for p in sorted(path.glob('qa_*.jpg'))},human_confirmed=False,
            reviewer='assistant_visual_review',scope='All rendered repair check frames inspected, not every pixel/frame.')
    assert len(repair_reviews)==12
    write_json(ROOT/'repairs'/'propagation_review.json',dict(clips=repair_reviews))
    for i,c in enumerate(cfg['clips']):
        n=notes[str(i)];e=dict(status='reviewed_for_bounded_calibration',reviewer='assistant_visual_review',human_confirmed=False,
            reliable_object_ids=n['objects'],reliable_ee_ids=n['ee'],reviewed_absent_ids=n['absent'],exclude_ids=n.get('exclude',[]),
            note=n['note'],reviewed_frames=read(ROOT/'qa',c['clip_id']+'.json')['checks'],
            evidence_sha256=sha(ROOT/'qa'/f'{c["clip_id"]}.jpg'),review_scope='Sparse temporal visual review plus all-frame engineering checks; not pixel GT')
        if 'override' in n:e['object_ids_override']=n['override']
        if n.get('repair_needed'):
            # Separate explicit repair review, written only after observing rendered results.
            r=read(ROOT/'repairs','propagation_review.json')['clips'][c['clip_id']]
            folder=r.get('repair_folder','repairs')
            meta=read(ROOT/folder/c['clip_id'],'complete.json')
            p=ROOT/folder/c['clip_id']/'objects.jsonl.gz';assert sha(p)==meta['output_sha256']
            assert r['output_sha256']==sha(p)
            e.update(repair_objects=str(p.relative_to(ROOT)),repair_objects_sha256=sha(p),repair_review=r)
            e['reliable_object_ids']=sorted(set(n['objects']+r['reliable_object_ids']))
            e['exclude_ids']=sorted(set(e['exclude_ids']+r.get('exclude_ids',[])))
            e['frame_exclusions']=r.get('frame_exclusions',[])
            e['incomplete_object_ids']=r.get('incomplete_object_ids',[])
        clips[c['clip_id']]=e
    assert len(clips)==64
    write_json(ROOT/'review.json',dict(schema='astribot.calibration64.assistant_review.v1',clips=clips,
        new_human_requests=0,selection_unchanged=True,all_pixels_manually_reviewed=False,
        selection_manifest_sha256=sha(ROOT/'manifest.json'),notes_sha256=sha(ROOT/'assistant_review_notes.json')))
    print('REVIEW',len(clips))


if __name__=='__main__':main()
