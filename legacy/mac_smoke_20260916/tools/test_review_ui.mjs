// Run after creating work/segmentation/browser_test_config.json with an isolated audit.
import fs from 'node:fs';
import assert from 'node:assert/strict';
const config=JSON.parse(fs.readFileSync('work/segmentation/browser_test_config.json','utf8'));
const targets=await fetch(`http://127.0.0.1:${config.devtools_port}/json/list`).then(r=>r.json());
const ws=new WebSocket(targets.find(t=>t.type==='page'&&t.url.startsWith(config.url)).webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
let serial=0;const pending=new Map(),errors=[];
ws.onmessage=e=>{const msg=JSON.parse(e.data);if(msg.id){const callback=pending.get(msg.id);pending.delete(msg.id);msg.error?callback.reject(msg.error):callback.resolve(msg.result)}else if(msg.method==='Runtime.exceptionThrown')errors.push(msg.params.exceptionDetails)};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
const waitFor=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,50))}throw Error('Timeout: '+expression)};
const binary='Array.from(m.getImageData(0,0,mc.width,mc.height).data).filter((_,i)=>i%4===3).map(n=>n>127?1:0).join("")';
try{
 await call('Runtime.enable');await waitFor('typeof manifest!=="undefined"&&manifest&&!loading&&layers.size===5');
 assert.equal(await evaluate('$("all").checked'),true);
 assert.equal(await evaluate('document.querySelectorAll(".object-card").length'),5);
 const all=await evaluate('v.toDataURL()');await evaluate('$("all").checked=false;render()');assert.notEqual(await evaluate('v.toDataURL()'),all);
 await evaluate('$("all").checked=true;render()');assert.equal(await evaluate('v.toDataURL()'),all);
 assert.equal(new Set(await evaluate('[...layers.keys()].map(color)')).size,5);
 await evaluate('$("pending").click()');await waitFor('pos===1&&!loading');
 assert.equal(await evaluate('activeObject'),2);assert.match(await evaluate('$("notice").textContent'),/待复核/);assert.equal(await evaluate('$("confirm").checked'),false);
 const before=await evaluate(binary);assert.ok(before.includes('1'));
 await evaluate('$("annotator").value="BROWSER_TEST_REVIEWER";$("confirm").checked=true;$("save").click()');
 await waitFor('$("notice").textContent.includes("已保存 revision 1")&&!saving');
 const currentSample=await evaluate('sample().sample_id');
 const row=await fetch(config.url+'/label?id='+currentSample+'&object=2').then(r=>r.json());assert.equal(row.label_source,'human');assert.equal(row.prediction_prefill,true);assert.equal(row.proposal_provenance.proposal_id,'test-proposal');
 assert.equal(await evaluate('document.querySelectorAll(".object-card .pending").length'),4);
 await evaluate('selectObject(4)');assert.equal(await evaluate('activeObject'),4);assert.equal(await evaluate('layers.size'),5);
 await evaluate('selectObject(2)');assert.equal(await evaluate(binary),before);
 await evaluate('$("clear").click()');assert.ok(!(await evaluate(binary)).includes('1'));
 assert.equal(await evaluate('$("confirm").checked'),false);await evaluate('$("save").click()');assert.match(await evaluate('$("notice").textContent'),/请先画出/);
 await evaluate('$("undo").click()');assert.equal(await evaluate(binary),before);
 await evaluate('dirty=false;$("pending").click()');await waitFor('activeObject===4&&!loading');
 assert.equal(await evaluate('pos'),1);assert.equal(await evaluate('$("all").checked'),true);
 await evaluate('$("visibility").value="fully_occluded";$("clear").click();$("confirm").checked=true;$("save").click()');await waitFor('$("notice").textContent.includes("已保存 revision 1")&&!saving');
 const basket=await fetch(config.url+'/label?id='+currentSample+'&object=4').then(r=>r.json());assert.equal(basket.visibility,'fully_occluded');
 await evaluate('move(2)');await waitFor('pos===2&&!loading');await evaluate('selectObject(2)');
 // Changing the current polygon leaves the other four object masks intact.
 const other=await evaluate('layers.get(10).toDataURL()');const rect=await evaluate('(()=>{const r=v.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}})()');
 for(const [dx,dy] of [[.4,.4],[.45,.4],[.43,.45]]){const x=rect.x+rect.width*dx,y=rect.y+rect.height*dy;await call('Input.dispatchMouseEvent',{type:'mousePressed',x,y,button:'left',clickCount:1});await call('Input.dispatchMouseEvent',{type:'mouseReleased',x,y,button:'left',clickCount:1})}
 await evaluate('$("polygon").click()');assert.equal(await evaluate('points.length'),0);assert.equal(await evaluate('layers.get(10).toDataURL()'),other);
 await evaluate('$("confirm").checked=true;$("save").click()');await waitFor('$("notice").textContent.includes("已保存 revision 1")&&!saving');
 await evaluate('move(1)');await waitFor('pos===1&&!loading');await evaluate('selectObject(2)');assert.equal(await evaluate(binary),before);
 // A failed frame load must never enable saving an old mask to the new frame.
 await evaluate('window.savedReadJSON=readJSON;readJSON=async(url,...args)=>{if(url.startsWith("/labels"))throw Error("TEST_FRAME_LOAD_FAILURE");return window.savedReadJSON(url,...args)};move(2)');
 await waitFor('!loading&&!frameReady');assert.equal(await evaluate('$("save").disabled'),true);assert.equal(await evaluate('$("jump").disabled'),false);
 assert.equal(await evaluate('layers.size'),0);
 await evaluate('readJSON=window.savedReadJSON;move(1)');await waitFor('!loading&&frameReady');assert.equal(await evaluate(binary),before);
 assert.equal(errors.length,0,JSON.stringify(errors));
 const screenshot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});fs.writeFileSync('work/segmentation/review_ui_test.png',Buffer.from(screenshot.data,'base64'));
 console.log('PASS: all masks/colors, legend, pending navigation, pixel persistence, source provenance, independent editing, empty-mask review, failed-load save protection, reload, no runtime exceptions');
}finally{ws.close()}
