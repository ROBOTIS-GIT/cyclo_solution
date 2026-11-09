"""Browser layout; contains no ROS or robot logic."""

HTML = """<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cup perception test</title>
<style>body{background:#151b25;color:#e6edf3;font:18px system-ui;margin:24px}
img{max-width:100%;border:1px solid #526071}pre{white-space:pre-wrap}
.note{color:#a9bacb;font-size:15px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:16px}
article{min-width:0}article img{width:100%;object-fit:contain}h2{font-size:19px}small{color:#a9bacb}</style>
<h1>컵 인식 테스트</h1><p class="note">영상·인식 결과만 표시합니다. 로봇 동작 명령은 없습니다.</p>
<pre id="status">연결 중…</pre>
<div class="grid"><article><h2>헤드 컵 인식 RGB</h2><img id="image" alt="RGB 입력을 기다립니다"></article>
<article><h2>헤드 depth</h2><small id="depth_head_info">대기 중</small><img id="depth_head" alt="헤드 depth 입력 대기"></article>
<article><h2>왼손 depth</h2><small id="depth_left_info">대기 중</small><img id="depth_left" alt="왼손 depth 입력 대기"></article>
<article><h2>왼손 depth — 거리 필터</h2>
<p><label>최소(cm) <input id="near" type="number" value="22" min="1" max="499" step="1" style="width:65px"></label>
<label>최대(cm) <input id="far" type="number" value="38" min="2" max="500" step="1" style="width:65px"></label>
<button onclick="setFilter()">적용</button></p><small id="filter_info">대기 중</small>
<img id="depth_filtered" alt="원본 숫자 depth 입력 대기"><p class="note">검정: 범위 밖 또는 깊이 없음. 남은 영역은 컵 확정 결과가 아닙니다.</p></article>
<article><h2>왼손 depth — 컵 후보</h2><small id="candidate_info">대기 중</small>
<img id="candidate" alt="컵 후보 입력 대기"><p class="note">빨간 사각형: 그리퍼 사이 제외 영역 · 노랑: 크기 조건을 통과한 후보 · 초록: 연속 관측된 선택 후보.
십자와 XYZ는 보이는 표면 점들의 중앙값입니다. 입구 중심이나 이동 목표가 아닙니다.</p></article>
<article><h2>왼손 입구 중심 추정</h2><small id="rim_info">대기 중</small><img id="rim" alt="입구 추정 대기">
<p class="note">파랑 점: 사용한 테두리 · 자홍 원/십자: 입구 추정. VALID일 때만 보정 액션에서 사용합니다.</p></article>
<article><h2>오른손 depth</h2><small id="depth_right_info">대기 중</small><img id="depth_right" alt="오른손 depth 입력 대기"></article></div>
<p class="note">원본 숫자 depth를 흑백으로 표시합니다. 가까울수록 밝고, 검정은 깊이 없음입니다.
필터 화면은 선택한 거리 범위만 남기며, 처리 결과의 거리 숫자는 그대로 유지합니다.
필터 전 화면의 표시 범위는 0.1~2.0m입니다.<br>
각 depth는 해당 카메라 좌표계의 영상이며 RGB와 픽셀 정렬이 보장되지 않습니다.<br>
초록 십자는 depth로 계산한 몸체 표면점입니다. 입구나 중심을 측정한 값은 아닙니다.<br>
WAITING: 입력 없음 · UNSYNCED: RGB/depth 시간 불일치 · INVALID: 인식 조건 미충족</p>
<script>
async function setFilter(){try{
 const r=await fetch('/filter',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({min_m:Number(document.getElementById('near').value)/100,max_m:Number(document.getElementById('far').value)/100})});
 const s=await r.json();if(!r.ok) throw new Error(s.error);
 document.getElementById('filter_info').textContent='적용됨';
 }catch(e){document.getElementById('filter_info').textContent=e.message;}}
async function update(){try{
 const s=await (await fetch('/status',{cache:'no-store'})).json();
 document.getElementById('status').textContent=s.status+
 '\\n상태 수신 경과: '+(s.status_age===null?'없음':s.status_age.toFixed(1)+'초')+
 '\\n카메라 영상 경과: '+(s.image_age===null?'없음':s.image_age.toFixed(1)+'초');
 if(s.status_age===null || s.status_age>2 || (s.image_age!==null && s.image_age>2))
  document.getElementById('status').textContent+='\\n⚠ 입력이 없거나 오래되었습니다.';
 if(s.has_image) document.getElementById('image').src='/image.jpg?t='+Date.now();
 for(const name of ['head','left','right']){
   const d=s.depth[name], age=d.age===null?'없음':d.age.toFixed(1)+'초';
   document.getElementById('depth_'+name+'_info').textContent=
     d.has_image?(d.encoding+' · 수신 '+age+(d.age>2?' ⚠ 오래됨':'')):'수신 대기 · '+d.topic;
   if(d.has_image) document.getElementById('depth_'+name).src='/depth/'+name+'.jpg?t='+Date.now();
 }
 const rim=s.rim;
 document.getElementById('rim_info').textContent=rim.detail+(rim.age!==null && rim.age>2?' ⚠ 오래됨':'');
 if(rim.has_image) document.getElementById('rim').src='/depth/left_rim.jpg?t='+Date.now();
 const c=s.candidate;
 document.getElementById('candidate_info').textContent=c.detail+
   (c.age!==null && c.age>2?' ⚠ 오래된 영상':'');
 if(c.has_image) document.getElementById('candidate').src='/depth/left_candidate.jpg?t='+Date.now();
 const f=s.filter;
 document.getElementById('filter_info').textContent=
   (f.min_m*100).toFixed(0)+'~'+(f.max_m*100).toFixed(0)+'cm · '+f.detail+
   (f.age!==null && f.age>2?' ⚠ 오래된 영상':'');
 if(f.has_image) document.getElementById('depth_filtered').src='/depth/left_filtered.jpg?t='+Date.now();
 }catch(e){document.getElementById('status').textContent='뷰어 연결 끊김';}
 setTimeout(update,500);}update();</script></html>"""
