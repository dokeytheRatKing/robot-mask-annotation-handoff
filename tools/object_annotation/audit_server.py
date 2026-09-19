"""Portable loopback manual audit UI. Python standard library only."""
import argparse
from datetime import datetime,timezone
import hashlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse,parse_qs
import webbrowser
from audit_rle import encode_counts,validate_counts

BASE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def line(file,value):file.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def current_label(root, sample_id, object_id):
    """Human edits always take precedence over unconfirmed model proposals."""
    name=f'{sample_id}__{object_id}.json'
    for folder in ['human_labels','proposed_labels']:
        path=root/folder/name
        if path.exists():
            row=json.loads(path.read_text(encoding='utf-8'))
            row['label_source']='human' if folder=='human_labels' else 'proposal'
            return row
    return None


def validate_annotation(body,sample,allowed_conditions):
    assert body['human_confirmed'] is True,'Human confirmation is required'
    assert isinstance(body['annotator'],str) and body['annotator'].strip(),'Annotator required'
    oid=int(body['object_id']);assert oid in [o['object_id'] for o in sample['objects']]
    status=body['visibility'];assert status in ['visible','partial_occlusion','fully_occluded','out_of_view','unknown']
    assert set(body.get('conditions',[]))<=set(allowed_conditions)
    instance=body['instance_id'].strip();assert instance and len(instance)<160
    counts=body['mask_rle_counts'];height=sample['image_height'];width=sample['image_width']
    area=validate_counts(counts,height,width)
    if status in ['visible','partial_occlusion']:assert area>0,'Visible object needs a manually drawn mask'
    else:assert area==0,'Invisible/unknown object must have empty visible-surface mask'
    return dict(sample_id=sample['sample_id'],episode_id=sample['episode_id'],camera=sample['camera'],
        frame_idx=sample['frame_idx'],object_id=oid,instance_id=instance,visibility=status,
        conditions=body.get('conditions',[]),mask=encode_counts(counts,height,width),annotator=body['annotator'].strip(),
        human_confirmed=True,annotation_method='manual_pixel_editor',prediction_prefill=False,
        source_image_sha256=sample['image_sha256'],notes=body.get('notes',''),
        saved_utc=datetime.now(timezone.utc).isoformat())


def main():
    p=argparse.ArgumentParser();p.add_argument('audit',type=Path,nargs='?',default=BASE/'audit')
    p.add_argument('--port',type=int,default=8766);p.add_argument('--open-browser',action='store_true');a=p.parse_args()
    root=a.audit.resolve();manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));samples={s['sample_id']:s for s in manifest['samples']}
    labels=root/'human_labels';labels.mkdir(exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def respond(self,data,kind='application/json',status=200):
            content=json.dumps(data).encode() if kind=='application/json' else data
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(content)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content)
        def do_GET(self):
            parsed=urlparse(self.path);q=parse_qs(parsed.query)
            if parsed.path=='/':return self.respond((BASE/'audit_ui.html').read_bytes(),'text/html; charset=utf-8')
            if parsed.path=='/manifest':return self.respond(manifest)
            if parsed.path=='/status':
                entries=[json.loads(p.read_text(encoding='utf-8')) for p in labels.glob('*.json')]
                completed={f'{r["sample_id"]}:{r["object_id"]}' for r in entries}
                proposed={f'{r["sample_id"]}:{r["object_id"]}' for p in (root/'proposed_labels').glob('*.json') for r in [json.loads(p.read_text(encoding='utf-8'))]}
                return self.respond(dict(labels=len(entries),frames=len(samples),required_labels=sum(len(s['objects']) for s in samples.values()),completed=sorted(completed),proposals=len(proposed-completed),available=len(proposed|completed)))
            sample=samples.get(q.get('id',[''])[0])
            if sample is None:return self.respond(dict(error='unknown sample'),status=404)
            if parsed.path=='/image':return self.respond((root/sample['image']).read_bytes(),'image/png')
            if parsed.path=='/labels':
                return self.respond([r for o in sample['objects'] for r in [current_label(root,sample['sample_id'],o['object_id'])] if r is not None])
            if parsed.path=='/label':
                oid=int(q.get('object',['-1'])[0])
                return self.respond(current_label(root,sample['sample_id'],oid))
            return self.respond(dict(error='not found'),status=404)
        def do_POST(self):
            try:
                assert self.path=='/save'
                origin=self.headers.get('Origin')
                assert origin is None or urlparse(origin).netloc==self.headers.get('Host'),'Cross-origin write refused'
                assert self.headers.get('Sec-Fetch-Site','same-origin') in ['same-origin','none'],'Cross-site write refused'
                size=int(self.headers.get('Content-Length','0'));assert 0<size<8_000_000
                body=json.loads(self.rfile.read(size));sample=samples[body['sample_id']]
                assert sha(root/sample['image'])==sample['image_sha256'],'Audit image changed'
                row=validate_annotation(body,sample,manifest['required_conditions'])
                previous=current_label(root,row['sample_id'],row['object_id'])
                if previous and (previous.get('label_source')=='proposal' or previous.get('prediction_prefill')):
                    row['prediction_prefill']=True
                    row['proposal_provenance']=previous.get('proposal_provenance',{
                        'annotation_method':previous.get('annotation_method'),
                        'proposal_id':previous.get('proposal_id'),
                        'model':previous.get('model'),
                    })
                path=labels/f'{row["sample_id"]}__{row["object_id"]}.json'
                row['revision']=json.loads(path.read_text(encoding='utf-8'))['revision']+1 if path.exists() else 1
                # Keep every revision even when current annotation is corrected.
                with (root/'human_label_revisions.jsonl').open('a',encoding='utf-8') as f:line(f,row)
                write_json(path,row);self.respond(dict(saved=True,revision=row['revision']))
            except Exception as exc:self.respond(dict(error=str(exc) or type(exc).__name__),status=400)
        def log_message(self,format,*args):pass
    server=HTTPServer(('127.0.0.1',a.port),Handler)
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'Open in your local browser: {url}\n{len(samples)} frames; labels saved in {labels}\nKeep this terminal open. Ctrl+C stops the server; saved labels are retained.',flush=True)
    if a.open_browser:
        timer=threading.Timer(.3,webbrowser.open,args=(url,));timer.daemon=True;timer.start()
    try:server.serve_forever()
    except KeyboardInterrupt:print('\nStopped; saved labels retained.',flush=True)
    finally:server.server_close()


if __name__=='__main__':main()
