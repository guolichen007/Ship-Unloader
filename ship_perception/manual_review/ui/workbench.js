const IDENTITY = ["session_id","frame_instance_id","view_id","manifest_sha256","wire_sha256","semantic_sha256","frame_id"];
const SVG = "http://www.w3.org/2000/svg";
const STATE = {
  RUNNING:"正在自动识别", LOADING:"正在核验自动结果", READY:"自动结果已加载",
  FAILED:"自动识别或结果校验失败", DISABLED:"自动入口未启用",
  AUTO_PENDING:"待审自动框", ACCEPTED_AUTO:"已确认自动框", ACCEPTED_REFINED:"已确认局部重测",
  ACCEPTED_GUIDED_CHOICE:"已确认所选 Raw 候选", GUIDED_CHOICES_PENDING:"存在竞争，未选择",
  SEARCH_INCOMPLETE:"搜索未完成", MANUAL_ONLY:"纯人工坐标", VOID:"误检已作废", GUIDED_REFINED_PENDING:"局部结果待确认",
  AMBIGUOUS:"存在竞争，请沿可见边重新框选", PARTIAL:"证据不足，请重新框选",
  NO_MATCH:"局部未匹配", BUDGET_EXCEEDED:"超出预算，请缩小搜索范围",
  TIMEOUT:"测量超时", CANCELLED:"已取消", REVIEW_PENDING:"场景待审核",
  REVIEW_COMPLETE:"场景审核完成", REVIEW_REOPENED:"修订后场景待重审",
};

// The same component mounts on the standalone page and the host integration example.
export function mount(root, options) {
  const base = options.basePath || "";
  if (!/^(?:\/[A-Za-z0-9_-]+)*$/.test(base)) throw Error("INVALID_BASE_PATH");
  const v3 = options.protocol === 3;
  const refinePath = v3 ? "/api/v3/refine" : "/api/v2/refine";
  root.innerHTML = `<main><section class="workspace" aria-label="点云地图">
    <div class="toolbar"><label>图层 <select data-id="layer"><option value="height">高度</option>
    <option value="density">密度</option><option value="mask">有效点</option></select></label>
    <button data-id="reset">重置视图</button><label><input data-id="original" type="checkbox" checked>原框</label>
    <label><input data-id="rough" type="checkbox" checked>粗框</label></div>
    <div data-id="viewport" class="mhr-viewport" tabindex="0" aria-label="地图画布。圈选时拖动，浏览时拖动平移。滚轮缩放。">
    <canvas aria-hidden="true"></canvas><svg aria-hidden="true"></svg></div></section>
    <aside><div data-id="auto" role="status"></div>
    <label class="field">船舱<select data-id="target"></select></label>
    <div data-id="selectionStatus" role="status"></div>
    <div data-id="source"></div><div data-id="mapChoices"></div>
    <div data-id="choices"></div><div data-id="profiles"></div>
    <table class="mhr-table" aria-label="四角 XYZ 米"><thead><tr><th>角点</th><th>X</th><th>Y</th><th>Z</th></tr></thead>
    <tbody data-id="coordinates"></tbody></table><p data-id="dimensions"></p>
    <div class="actions"><button data-id="confirm" class="primary">确认标定</button>
      <button data-id="edit">重新框选</button><button data-id="add">新增船舱</button></div>
    <div data-id="editor" hidden><label class="field">粗框方向（度）<input data-id="angle" type="number" value="0" min="-180" max="180" step="0.25"></label>
      <div class="actions"><button data-id="orient">两点定向</button><button data-id="measure">重新标定</button><button data-id="cancel">取消</button></div></div>
    <div data-id="status" role="status" aria-live="polite"></div>
    <details><summary>其他操作</summary><div class="actions">
       <button data-id="void">作废误检</button><button data-id="undo">撤销</button>
       <button data-id="manual">仅保存粗框</button></div><ul data-id="evidence"></ul></details>
    <hr><div data-id="scene" role="status"></div>
    <div class="actions"><button data-id="sceneConfirm">确认整景</button><button data-id="export">导出复核</button></div></aside></main>`;
  const el = id => root.querySelector('[data-id="'+id+'"]');
  const viewport = el("viewport"), canvas = viewport.querySelector("canvas"), svg = viewport.querySelector("svg");
  const life = new AbortController();
  const on = (target, name, fun, opts = {}) => target.addEventListener(name, fun, {...opts, signal:life.signal});
  let view, image, target, draft, preview, drag, timer, inFlight, orienting, baseForm;
  let profileTicket = 0;
  let epoch = 0, navigation = 0, layerEpoch = 0, zoom = 1, pan = [0,0], editing = false, saving = false, alive = true;
  const manifest = () => view.manifest;
  const identity = () => Object.fromEntries(IDENTITY.map(k=>[k,manifest()[k]]));
  const status = value => { el("status").textContent = value; };
  const fail = error => { if (alive) { status("操作失败 · "+error.message); options.onError?.(error); } };
  const active = () => has(view?.auto.targets,target) ? view.auto.targets[target] : null;
  const head = () => has(view?.heads,target) ? view.heads[target] : null;
  const has = (object, key) => object && key != null && Object.prototype.hasOwnProperty.call(object,key);
  const inMap = p => p.every((v,i)=>v>=0&&v<=[manifest().nx,manifest().ny][i]);
  const xy = ([u,v]) => [manifest().x0+u*manifest().s,manifest().y0+(manifest().ny-v)*manifest().s];
  const uv = ([x,y]) => [(x-manifest().x0)/manifest().s,manifest().ny-(y-manifest().y0)/manifest().s];

  async function fetchAPI(path,data,signal) {
    const init = {headers: {"X-MHR-Token": options.token || ""}, signal};
    if (data !== undefined) {
      init.method="POST"; init.headers["Content-Type"]="application/json"; init.body=JSON.stringify(data);
    }
    const r=await fetch(base+path,init);
    if (!r.ok) throw Error((await r.json()).error || "HTTP_"+r.status);
    return r;
  }

  function invalidate() {
    epoch++;
    if (inFlight) {
      const op=inFlight;
      op.abort.abort();
      fetchAPI(refinePath+"/cancel",{...op.identity,annotation_id:op.target,request_id:op.id}).catch(()=>{});
    }
    if (preview && view && target===preview.annotation_id && !inFlight) {
      fetchAPI(refinePath+"/cancel",{...identity(),annotation_id:target,request_id:preview.request_id}).catch(()=>{});
    }
    inFlight=undefined; preview=null;
    profileTicket++; el("profiles").replaceChildren();
  }

  function resetDraft() { invalidate(); editing=false; draft=null; drag=null; orienting=null; baseForm=null; }

  async function openFrame() {
    const ticket=++navigation;
    clearTimeout(timer);
    try {
      const next=await (await fetchAPI(v3?"/api/v3/view":"/api/v2/view",undefined,life.signal)).json();
      if (!alive || ticket!==navigation) return;
      const changed=view?.manifest.view_id!==next.manifest.view_id;
      if (changed) { resetDraft(); target=null; zoom=1; pan=[0,0]; }
      view=next;
      if (!target) target=Object.keys(next.auto.targets)[0] || Object.keys(next.heads)[0] || null;
      renderTargets();
      if (changed) await openLayer();
      if (!alive || ticket!==navigation) return;
      render();
      options.onReviewChange?.(view);
      if (["RUNNING","LOADING"].includes(view.auto.status)) timer=setTimeout(openFrame,650);
    } catch (error) { if (alive && ticket===navigation) fail(error); }
  }

  function setTarget(id) {
    if (!view || (!has(view.auto.targets,id) && !has(view.heads,id))) return false;
    resetDraft(); target=id; el("target").value=id; el("mapChoices").replaceChildren(); status(""); render();
    return true;
  }

  function renderTargets() {
    const list=el("target");
    list.replaceChildren();
    const ids=[...new Set([...Object.keys(view.auto.targets),...Object.keys(view.heads)])];
    if (target && !ids.includes(target)) ids.push(target);
    if (!ids.length) ids.push("");
    for (const id of ids) {
      const opt=document.createElement("option"); opt.value=id;
      opt.textContent=has(view.auto.targets,id)?id:(id ? "新增 "+id.slice(0,8) : "暂无船舱");
      list.append(opt);
    }
    list.value=target || "";
  }

  async function openLayer() {
    const ticket=++layerEpoch;
    try {
      const r=await fetchAPI("/layer/"+el("layer").value+".png",undefined,life.signal);
      const bitmap=await createImageBitmap(await r.blob());
      if (!alive || ticket!==layerEpoch) { bitmap.close(); return; }
      image?.close(); image=bitmap; draw();
    } catch (error) { if (alive && ticket===layerEpoch) fail(error); }
  }

  function camera() {
    const {width:w,height:h}=viewport.getBoundingClientRect();
    const {nx,ny}=manifest();
    const s=Math.min(w/nx,h/ny)*zoom;
    return {w,h,s,x:(w-nx*s)/2+pan[0],y:(h-ny*s)/2+pan[1]};
  }
  const project = ([u,v], c) => [c.x+u*c.s,c.y+v*c.s];
  const unproject = ([x,y],c) => [(x-c.x)/c.s,(y-c.y)/c.s];

  function polygon(points,c,stroke,dash,label) {
    const mapped=points.map(p=>project(p,c));
    const outline=document.createElementNS(SVG,"polygon");
    outline.setAttribute("points",mapped.map(p=>p.join(",")).join(" "));
    outline.setAttribute("stroke",stroke); outline.setAttribute("stroke-dasharray",dash);
    outline.dataset.label=label; svg.append(outline);
    const text=document.createElementNS(SVG,"text");
    text.textContent=label; text.setAttribute("fill",stroke);
    text.setAttribute("x",mapped[0][0]+3); text.setAttribute("y",mapped[0][1]-5); svg.append(text);
  }

  function vertices() {
    if (!draft) return [];
    const theta=Number(el("angle").value)*Math.PI/180;
    const cos=Math.cos(theta), sin=Math.sin(theta), [u,v]=draft.center;
    return [[-draft.w,-draft.h],[draft.w,-draft.h],[draft.w,draft.h],[-draft.w,draft.h]]
      .map(([x,y])=>[u+cos*x-sin*y,v+sin*x+cos*y]);
  }

  function draw() {
    if (!view || !alive) return;
    const c=camera(), dpr=devicePixelRatio||1;
    if (!c.w || !c.h) return;
    const w=Math.max(1,Math.round(c.w*dpr)),h=Math.max(1,Math.round(c.h*dpr));
    if (canvas.width!==w||canvas.height!==h) { canvas.width=w; canvas.height=h; }
    const ctx=canvas.getContext("2d");
    ctx.setTransform(w/c.w,0,0,h/c.h,0,0); ctx.clearRect(0,0,c.w,c.h);
    ctx.imageSmoothingEnabled=false;
    if (image) ctx.drawImage(image,c.x,c.y,manifest().nx*c.s,manifest().ny*c.s);
    svg.replaceChildren();
    for (const [id,row] of Object.entries(view.auto.targets)) {
      const revision=view.heads[id];
      if (revision?.state==="VOID" && revision.baseline_digest===row.geometry_digest) continue;
      const changed=id===target && (preview?.geometry||revision?.geometry);
      if (!changed || el("original").checked)
        polygon(row.geometry.xy.map(uv),c,changed?"#b7c1c8":id===target?"#a6c8ff":"#81a5c0",changed?"5 5":"",id);
    }
    for (const [id,revision] of Object.entries(view.heads)) {
      if (!revision.geometry||id===target||revision.state==="ACCEPTED_AUTO") continue;
      if (has(view.auto.targets,id) && view.pending.includes(id)) continue;
      polygon(revision.geometry.xy.map(uv),c,"#74c2a4","",id);
    }
    const saved=head();
    if (target && saved?.geometry && !view.pending.includes(target) &&
        (saved.state!=="ACCEPTED_AUTO" || preview?.geometry))
      polygon(saved.geometry.xy.map(uv),c,"#b5a6d1",preview?.geometry?"5 5":"","上次确认");
    if (draft && el("rough").checked) polygon(vertices(),c,"#ffc96c","9 5","粗框");
    if (v3 && preview?.match_state==="GUIDED_CHOICES_PENDING") {
      preview.candidates.forEach((p,i)=>{
        if (p.confirmable && (p.candidate_id===preview.candidate_id || p.candidate_id===preview.recommendation_id))
          polygon(p.geometry.xy.map(uv),c,"#70edb4",p.candidate_id===preview.candidate_id?"":"6 3","候选 "+(i+1));
        else if (p.confirmable) polygon(p.geometry.xy.map(uv),c,"#ffc96c","4 3","候选 "+(i+1));
      });
    } else if (preview?.geometry) polygon(preview.geometry.xy.map(uv),c,"#70edb4","","重测候选");
  }

  function render() {
    if (!view || !alive) return;
    draw();
    const auto=active(), saved=head(), candidate=preview?.geometry;
    const usable=saved && !view.pending.includes(target) ? saved : null;
    const geometry=candidate || usable?.geometry || (usable?.state==="VOID"?null:auto?.geometry);
    el("auto").textContent=(STATE[view.auto.status] || view.auto.status)+(view.auto.reason?" · "+view.auto.reason:"");
    const id = view.source_identity;
    el("source").textContent=id ? (id.status==="VERIFIED" ? "构建 "+id.build_sha.slice(0,12) : "构建来源未核验") : "";
    el("selectionStatus").textContent=candidate?(preview.match_state==="GUIDED_CHOICES_PENDING"?
       (preview.selection_id?"已选 Raw 候选，待确认":STATE.GUIDED_CHOICES_PENDING):STATE.GUIDED_REFINED_PENDING):
       usable?(STATE[usable.state]||usable.state):auto?STATE.AUTO_PENDING+" · "+auto.state:"";
    el("coordinates").replaceChildren();
    const fmt=n=>n==null?"未观测":n.toFixed(3);
    if (geometry) geometry.xyz.forEach((row,i)=>{
      const tr=document.createElement("tr");
      for(const value of [geometry.corner_ids[i],...row.map(fmt)]) {
        const td=document.createElement("td"); td.textContent=value; tr.append(td);
      }
      el("coordinates").append(tr);
    });
    el("dimensions").textContent=geometry?"水平长 "+fmt(geometry.length_m)+" m · 宽 "+fmt(geometry.width_m)+" m":"";
    const valid=draft && vertices().every(inMap) && 2*Math.min(draft.w,draft.h)*manifest().s>=1 &&
      2*Math.max(draft.w,draft.h)*manifest().s<=250;
    el("editor").hidden=!editing;
    el("measure").disabled=!valid||saving;
    el("confirm").disabled=saving||!!inFlight||(!candidate&&(!auto||editing||!!usable))
      || (v3 && !!candidate && (!preview.selection_id || view.source_identity?.status!=="VERIFIED"));
    el("orient").hidden=!v3;
    el("choices").replaceChildren();
    if (v3 && preview?.match_state==="GUIDED_CHOICES_PENDING") preview.candidates.forEach((c,i)=>{
      if (!c.confirmable) return;
      const button=document.createElement("button");
      button.textContent=(preview.candidate_id===c.candidate_id?"已选 ":"选择 ")+"候选 "+(i+1)+" · 四边 Raw";
      button.disabled=saving || !!inFlight;
      button.onclick=()=>choose(c);
      el("choices").append(button);
    });
    el("edit").disabled=saving||!target;
    el("add").disabled=saving;
    el("manual").disabled=saving||!valid||!!inFlight;
    el("void").disabled=saving||!auto||usable?.state==="VOID";
    el("undo").disabled=saving||!saved;
    el("sceneConfirm").disabled=saving||view.pending.length>0||["RUNNING","LOADING"].includes(view.auto.status);
    el("scene").textContent=(STATE[view.scene_state]||view.scene_state)+" · 待审 "+view.pending.length;
    el("evidence").replaceChildren(...(preview?.side_summary || []).map(row=>{
      const li=document.createElement("li"); li.textContent=row.side_id+" "+row.reason+" ("+row.mode_count+")";
      if(v3 && preview && ["PARTIAL","AMBIGUOUS","GUIDED_CHOICES_PENDING","SEARCH_INCOMPLETE"].includes(preview.match_state)){
        const b=document.createElement("button");
        b.textContent="剖面";
        b.onclick=()=>openProfile(row.side_id);
        li.append(b);
      }
      return li;
    }));
  }

  function position(event) {
    const r=viewport.getBoundingClientRect(); return [event.clientX-r.left,event.clientY-r.top];
  }

  function startEdit(isNew) {
    if (!view || saving) return;
    resetDraft();
    if (isNew) target=crypto.randomUUID();
    editing=true;
    const g = isNew ? null : active()?.geometry || head()?.geometry;
    const [a,b]=g ? g.xy.slice(0,2).map(uv) : [[0,0],[1,0]];
    el("angle").value=String(v3 ? Math.atan2(b[1]-a[1],b[0]-a[0])*180/Math.PI : 0);
    renderTargets(); render();
    status("沿可见内沿拖动圈选。斜舱可先设置粗框方向。");
    viewport.focus();
  }

  on(viewport,"pointerdown",event=>{
    if(!view||saving||event.button!==0||drag) return;
    const at=position(event), p=unproject(at,camera());
    if(editing&&!inMap(p)) return;
    if (orienting) {
      if (!orienting.start) { orienting.start=p; status("点击方向上的第二点"); }
      else {
        const [u,v]=p.map((x,i)=>x-orienting.start[i]);
        if (Math.hypot(u,v)>1) {
          el("angle").value=String(Math.atan2(v,u)*180/Math.PI);
          orienting=null; status("拖动绘制旋转框");
        }
      }
      return;
    }
    viewport.setPointerCapture(event.pointerId);
    if(editing) { invalidate(); draft={center:p,w:0,h:0}; }
    drag={id:event.pointerId,at,p,pan:[...pan],mode:editing?"draw":"pan"};
    draw();
  });
  function updatePointer(event) {
    if(!drag||drag.id!==event.pointerId) return;
    const at=position(event);
    if(drag.mode==="pan") pan=drag.pan.map((v,i)=>v+at[i]-drag.at[i]);
    else {
      const p=unproject(at,camera()),a=drag.p;
      const theta=v3 ? Number(el("angle").value)*Math.PI/180 : 0;
      const dx=p[0]-a[0], dy=p[1]-a[1];
      draft={center:p.map((v,i)=>(v+a[i])/2),w:Math.abs(dx*Math.cos(theta)+dy*Math.sin(theta))/2,
        h:Math.abs(-dx*Math.sin(theta)+dy*Math.cos(theta))/2};
    }
    draw();
  }
  on(viewport,"pointermove", updatePointer);
  on(viewport,"pointerup", event=>{
    if(!drag||drag.id!==event.pointerId) return;
    const shouldMeasure=drag.mode==="draw";
    const isClick=drag.mode==="pan" && Math.hypot(...position(event).map((v,i)=>v-drag.at[i])) < 4;
    const p=drag.p;
    updatePointer(event); drag=null; render();
    if(v3 && isClick) pickTarget(p);
    if(shouldMeasure&&!el("measure").disabled) measure();
  });
  on(viewport,"pointercancel",()=>{if(drag?.mode==="draw") draft=null; drag=null; render();});
  on(viewport,"wheel",event=>{
    if(!view||drag) return;
    event.preventDefault();
    const at=position(event), pinned=unproject(at,camera());
    zoom=Math.max(1,Math.min(60,zoom*Math.exp(-event.deltaY*.002)));
    const after=project(pinned,camera()); pan=pan.map((v,i)=>v+at[i]-after[i]); draw();
  }, {passive:false});

  const mutation = () => ({...identity(),annotation_id:target,expected_parent_revision:head()?.revision||0,
    mutation_id:crypto.randomUUID(),bundle_digest:view.auto.bundle_digest});
  function selection() {
    const points=vertices(),a=active();
    return {schema:"manual_selection.1", request_id:crypto.randomUUID(),operation:a?"REFINE":"CREATE",
      target_opening_id:a?target:null,target_vessel_id:a?.vessel_id||null,vertices_uv:points,client_xy:points.map(xy)};
  }
  function pickTarget(p) {
    const ids=[...new Set([...Object.keys(view.auto.targets),...Object.keys(view.heads)])].filter(id=>{
      const record=view.heads[id];
      if(record?.state==="VOID"&&!view.pending.includes(id)) return false;
      const g=(!view.pending.includes(id)&&record?.geometry)||view.auto.targets[id]?.geometry;
      if(!g) return false;
      const poly=g.xy.map(uv);
      const cross=poly.map((a,i)=>{
        const b=poly[(i+1)%4];
        return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]);
      });
      return cross.every(x=>x>=0)||cross.every(x=>x<=0);
    });
    const list=el("mapChoices");
    list.replaceChildren();
    if(ids.length===1) return setTarget(ids[0]);
    if(ids.length>1) status("此处多个船舱重叠，请明确选择");
    for(const id of ids) {
      const b=document.createElement("button"); b.textContent=id;
      b.onclick=()=>setTarget(id); list.append(b);
    }
  }
  function binding(p) {
    const {mutation_id,...base}=mutation();
    return {...base,request_id:p.request_id,draft_digest:p.draft_digest,list_digest:p.list_digest};
  }
  async function choose(c) {
    if(saving||inFlight||!preview) return;
    const p=preview;
    saving=true; render();
    try {
      const next=await(await fetchAPI(refinePath+"/select",{...binding(p),candidate_id:c.candidate_id,
        geometry_digest:c.geometry_digest},life.signal)).json();
      if(preview===p && alive) { preview=next; options.onPreview?.(next); }
    } catch(error){if(preview===p) fail(error);}
    finally{saving=false;render();}
  }
  async function openProfile(side_id) {
    if (!preview || !baseForm) return;
    const p=preview, ticket=++profileTicket;
    try {
      const data=await(await fetchAPI(refinePath+"/profile",{...binding(p),side_id},life.signal)).json();
      if(!alive || p!==preview || ticket!==profileTicket) return;
      const panel=el("profiles"), chart=document.createElement("canvas");
      const title=document.createElement("p");
      title.textContent=side_id+" 中段 Raw 高度剖面 · 横轴向舱内为正";
      panel.replaceChildren(title);
      if (!data.points.length) {
        const empty=document.createElement("p"); empty.textContent="中段没有可显示的训练观测";
        panel.append(empty); return;
      }
      chart.width=320; chart.height=130; chart.className="mhr-profile";
      chart.setAttribute("aria-label","点击表示该边的位置提示");
      const ctx=chart.getContext("2d"), w=data.offset_window_m;
      const zs=data.points.map(q=>q[1]), lo=Math.min(...zs), hi=Math.max(...zs);
      ctx.fillStyle="#a6c8ff";
      for(const [d,z] of data.points) ctx.fillRect((d+w)/(2*w)*320,124-118*(z-lo)/Math.max(hi-lo,.01),2,2);
      panel.append(chart);
      on(chart,"click",event=>{
        if(saving || preview!==p) return;
        const r=chart.getBoundingClientRect(), d=((event.clientX-r.left)/r.width*2-1)*w;
        const point=data.midpoint_uv.map((v,i)=>v+d*data.normal_uv[i]);
        measure({request_id:p.request_id,side_id,uv:point});
      },{once:true});
    }catch(error){if(preview===p) fail(error);}
  }
  async function measure(hint) {
    if(!draft||el("measure").disabled) return;
    if(!hint) invalidate();
    else { epoch++; preview=null; el("profiles").replaceChildren(); }
    const {mutation_id,...base}=mutation();
    const form=hint?{...baseForm,request_id:crypto.randomUUID(),edge_hint:hint}:selection();
    if(!hint) baseForm=form;
    const issued=epoch;
    const op={abort:new AbortController(),identity:identity(),target,id:form.request_id};
    inFlight=op; status("正在局部重测"); render();
    try {
      const result=await (await fetchAPI(refinePath,{...base,...form},op.abort.signal)).json();
      if(!alive||issued!==epoch||inFlight!==op||result.view_id!==manifest().view_id||
         result.annotation_id!==target||result.request_id!==form.request_id) return;
      preview=result;
      status((STATE[result.match_state||result.execution_state]||"测量未完成")+(result.geometry?"":" · "+result.reason));
      options.onPreview?.(result);
    } catch(error) { if(issued===epoch&&inFlight===op&&error.name!=="AbortError") fail(error); }
    finally { if(inFlight===op){inFlight=null;render();} }
  }

  async function commit(path,body) {
    if(saving) return;
    saving=true; render();
    try {
      const result=await (await fetchAPI(path,body,life.signal)).json();
      // The committed candidate has already been consumed by the CAS.
      preview=null; resetDraft();
      await openFrame();
      if(alive){status("已保存"); options.onCommit?.(result);}
    } catch(error){fail(error);}
    finally{saving=false;render();}
  }

  const actions = {
    reset:()=>{zoom=1;pan=[0,0];draw();},
    edit:()=>startEdit(false), add:()=>startEdit(true),
    cancel:()=>{resetDraft();status("已取消本次编辑");render();},
    orient:()=>{invalidate(); draft=null; orienting={}; status("点击方向上的第一点");render();},
    measure:()=>measure(),
    confirm:()=>{
      const p=preview;
      if(p?.geometry && v3 && p.selection_id) commit(refinePath+"/confirm",{...binding(p),
        mutation_id:crypto.randomUUID(),selection_id:p.selection_id,candidate_id:p.candidate_id,
        geometry_digest:p.geometry_digest});
      else if(p?.geometry && !v3) commit(refinePath+"/confirm",{...mutation(),request_id:p.request_id,
        candidate_id:p.candidate_id,geometry_digest:p.geometry_digest,draft_digest:p.draft_digest});
      else if(!editing && active()) commit("/api/v2/auto/confirm",{...mutation(),geometry_digest:active().geometry_digest});
    },
    void:()=>{if(active()&&window.confirm("将该舱标记为误检作废？"))
      commit("/api/v2/auto/void",{...mutation(),geometry_digest:active().geometry_digest});},
    undo:()=>commit("/api/v2/undo",mutation()),
    manual:()=>{if(window.confirm("仅保存人工粗框，所有 Z 标为未观测。是否继续？"))
      commit("/api/v2/manual/commit",{...mutation(),...selection()});},
    sceneConfirm:()=>{
      const empty=!Object.values(view.heads).some(h=>h.geometry);
      if(!empty||window.confirm("明确确认本次复核结果为空场？")) commit("/api/v2/scene/confirm",
        {...identity(),mutation_id:crypto.randomUUID(),scene_digest:view.scene_digest,confirm_empty:empty});
    },
    export:async()=>{
      try {
        const data=await(await fetchAPI("/api/v2/export",{...identity(),scene_digest:view.scene_digest},life.signal)).json();
        const {csv,...summary}=data;
        for(const [ext,type,content] of [["json","application/json",JSON.stringify(summary,null,2)],["csv","text/csv;charset=utf-8",csv]]){
          const link=document.createElement("a"); link.download="船舱复核-R"+(v3?3:2)+"."+ext;
          link.href=URL.createObjectURL(new Blob([content],{type})); link.click();
          setTimeout(()=>URL.revokeObjectURL(link.href),5000);
        }
      }catch(error){fail(error);}
    },
  };
  for(const [id,action] of Object.entries(actions)) on(el(id),"click",action);
  on(el("target"),"change",()=>setTarget(el("target").value));
  on(el("layer"),"change",openLayer);
  on(el("angle"),"input",()=>{invalidate();render();});
  for(const id of ["original","rough"]) on(el(id),"change",draw);
  on(root,"keydown",event=>{if(event.key==="Escape"&&!saving)actions.cancel();});
  const observer=new ResizeObserver(draw); observer.observe(viewport);
  return {
    openFrame,setTarget,resize:draw,
    destroy(){resetDraft();alive=false;life.abort();observer.disconnect();clearTimeout(timer);image?.close();root.replaceChildren();}
  };
}
