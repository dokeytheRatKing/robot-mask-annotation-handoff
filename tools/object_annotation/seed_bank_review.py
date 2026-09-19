"""Build a self-contained local-file seed review page; no server or CDN needed."""
import argparse
import json
from pathlib import Path

from annotate import sha


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Identity Seed Bank Review</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f5;color:#202622;font:14px Arial,sans-serif;letter-spacing:0}
header{padding:16px 20px;background:white;border-bottom:1px solid #ccc}h1{font-size:22px;margin:0 0 12px}
nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap}select,button,input{font:inherit;padding:7px;max-width:100%}
button{cursor:pointer}#count{font-variant-numeric:tabular-nums}main{padding:18px;display:grid;grid-template-columns:repeat(auto-fill,minmax(min(350px,100%),1fr));gap:16px}
article{background:white;border:1px solid #bfc9c1;border-radius:6px;padding:12px;min-width:0}
article.excluded{border-color:#bd7542}h2{font-size:16px;overflow-wrap:anywhere;margin:0 0 8px}
.images{display:grid;grid-template-columns:1fr 1fr;gap:6px}.images img{width:100%;height:180px;object-fit:contain;background:#e8ebea}
.meta{font-size:12px;line-height:1.6;overflow-wrap:anywhere;margin:8px 0}.reason{min-height:44px;font-size:12px;color:#535b55}
.controls{display:grid;grid-template-columns:125px minmax(0,1fr);gap:8px}a{color:#16624d}#notice{color:#98392b}
</style><header><h1>Identity Seed Bank</h1><nav>
<select id="identity" aria-label="Identity"><option value="">All identities</option></select>
<select id="camera" aria-label="Camera"><option value="">All cameras</option><option>head</option><option>left_wrist</option><option>right_wrist</option></select>
<select id="status" aria-label="Bank status"><option value="">All statuses</option><option value="active">Active</option><option value="retrieval_excluded">Excluded</option></select>
<button id="accept">Accept Visible</button><button id="export">Export Review</button>
<label>Import Review <input type="file" id="import" accept="application/json"></label>
<span id="count"></span><span id="notice" role="status"></span></nav></header><main id="grid"></main>
<script>
const DATA=__DATA__, key='seed-review-'+DATA.bank_sha256, decisions={};
const $=id=>document.getElementById(id);
try{Object.assign(decisions,JSON.parse(localStorage.getItem(key)||'{}'));}catch(e){}
function save(){try{localStorage.setItem(key,JSON.stringify(decisions));}catch(e){$('notice').textContent='Local storage unavailable';}}
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
const identities=new Map(DATA.entries.map(e=>[e.object_id,e.identity_name]));
for(const [id,name] of identities){const o=el('option',id+' '+name);o.value=id;$('identity').append(o);}
function visible(){return DATA.entries.filter(e=>(!$('identity').value||String(e.object_id)===$('identity').value)&&(!$('camera').value||e.camera===$('camera').value)&&(!$('status').value||e.bank_status===$('status').value));}
function render(){const rows=visible();$('grid').replaceChildren();$('count').textContent=rows.length+' / '+DATA.entries.length;
for(const e of rows){const card=el('article',undefined,e.bank_status==='active'?'':'excluded');card.dataset.exemplar=e.exemplar_id;
card.append(el('h2',e.exemplar_id+' | '+e.identity_name));const imgs=el('div',undefined,'images');
for(const kind of ['rgb.png','masked_rgb.png']){const link=el('a');link.href=e.assets[kind];link.target='_blank';const im=el('img');im.src=e.assets[kind];im.alt=e.exemplar_id+' '+kind;im.loading='lazy';link.append(im);imgs.append(link);}card.append(imgs);
card.append(el('div',e.camera+' | '+e.episode_id+' | frame '+e.frame_idx+' | '+e.bank_status,'meta'));
card.append(el('div',e.bank_review.reason,'reason'));const controls=el('div',undefined,'controls');const select=el('select');select.setAttribute('aria-label',e.exemplar_id+' review');
for(const [v,t] of [['','Pending'],['accept','Accept decision'],['include','Include'],['exclude','Exclude'],['uncertain','Uncertain']]){const o=el('option',t);o.value=v;select.append(o);}
select.value=(decisions[e.exemplar_id]||{}).decision||'';const note=el('input');note.placeholder='Review note';note.setAttribute('aria-label',e.exemplar_id+' note');note.value=(decisions[e.exemplar_id]||{}).note||'';
const update=()=>{decisions[e.exemplar_id]={decision:select.value,note:note.value};save();};select.onchange=update;note.oninput=update;controls.append(select,note);card.append(controls);$('grid').append(card);}}
for(const name of ['identity','camera','status'])$(name).onchange=render;
$('accept').onclick=()=>{for(const e of visible())if(!(decisions[e.exemplar_id]||{}).decision)decisions[e.exemplar_id]={decision:'accept',note:''};save();render();};
$('export').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'astribot.seed_bank_user_review.v1',bank_sha256:DATA.bank_sha256,exported_at:new Date().toISOString(),decisions},null,2)],{type:'application/json'});const a=el('a');a.href=URL.createObjectURL(blob);a.download='seed_bank_user_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000);};
$('import').onchange=async()=>{try{const file=$('import').files[0];if(!file)return;const data=JSON.parse(await file.text());if(data.bank_sha256!==DATA.bank_sha256)throw Error('Bank hash mismatch');const known=new Set(DATA.entries.map(e=>e.exemplar_id));for(const [id,d] of Object.entries(data.decisions)){if(!known.has(id)||!['','accept','include','exclude','uncertain'].includes(d.decision))throw Error('Invalid review');}Object.assign(decisions,data.decisions);save();render();$('notice').textContent='Review imported';}catch(e){$('notice').textContent=e.message;}};
render();
</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True,type=Path)
    a=p.parse_args()
    data=json.loads((a.bank/'bank.json').read_text())
    data['bank_sha256']=sha(a.bank/'bank.json')
    payload=json.dumps(data,ensure_ascii=True).replace('<','\\u003c')
    (a.bank/'review.html').write_text(PAGE.replace('__DATA__',payload))
    print(a.bank/'review.html')


if __name__=='__main__':main()
