#!/usr/bin/env python3
"""Build a prediction-blind interactive GT labeling page."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camera-manifest", type=Path, required=True)
    p.add_argument("--review-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.camera_manifest.read_text(encoding="utf-8"))
    rows = [row for row in manifest["rows"] if int(row["view_count"]) > 0]
    blocks = []
    for index, row in enumerate(rows, 1):
        sid = str(row["slot_id"])
        frames = ";".join(str(x) for x in row["frames"])
        blocks.append(
            f"""
<section class="case" data-slot="{html.escape(sid)}" data-frames="{frames}">
  <div class="casehead"><h2>{index:02d}. {html.escape(sid)}</h2><span class="badge pending">未标注</span></div>
  <div class="images">
    <figure><img src="location_maps/{html.escape(sid)}.png"><figcaption>大位置图：紫色目标；红 X 是 LiDAR/map 位姿；青色是左侧 Camera/FOV</figcaption></figure>
    <figure><img src="../camera_views_pose_tuned/{html.escape(sid)}.jpg"><figcaption>严格因果多帧 Camera；目标多边形仅用于人工确认车位身份</figcaption></figure>
  </div>
  <div class="form">
    <div><label>GT 状态</label><div class="choices state">
      <button data-value="free">Free</button><button data-value="occupied">Occupied</button><button data-value="unknown">Unknown</button>
    </div></div>
    <div><label>可观测性</label><div class="choices observable">
      <button data-value="visible">可清楚判断</button><button data-value="occluded">被遮挡</button><button data-value="ambiguous">身份/画面不清</button>
    </div></div>
    <div><label>车位身份</label><div class="choices identity">
      <button data-value="yes">已确认</button><button data-value="no">未确认</button>
    </div></div>
    <div><label>置信度</label><input class="confidence" type="number" min="0" max="1" step="0.05" value="0.90"></div>
    <div class="wide"><label>备注</label><input class="notes" placeholder="遮挡物、关键帧、判断依据"></div>
  </div>
</section>"""
        )

    payload = json.dumps(
        {
            "camera_origin_lidar_m": manifest["camera_origin_lidar_m"],
            "camera_yaw_deg": manifest["camera_yaw_deg"],
            "slot_count": len(rows),
        },
        ensure_ascii=False,
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Frame 6241 交互式人工GT标注</title>
<style>
:root{{--ink:#14213d;--muted:#52647b;--line:#ccd6e3;--free:#0f9d71;--occ:#d9485f;--unk:#7c899b}}
*{{box-sizing:border-box}}body{{margin:0;background:#edf2f7;color:var(--ink);font-family:Arial,"Microsoft YaHei",sans-serif}}
header{{position:sticky;top:0;z-index:20;background:#14213df2;color:white;padding:14px 4vw;display:flex;gap:18px;align-items:center;flex-wrap:wrap}}
header h1{{font-size:20px;margin:0}}header p{{margin:0;color:#dbe7f5;font-size:13px}}header button{{margin-left:auto;background:#1d76db;color:white;border:0;border-radius:8px;padding:10px 16px;font-weight:bold;cursor:pointer}}
main{{max-width:1550px;margin:auto;padding:18px}}.intro{{background:#fff8df;border-left:5px solid #e6a700;padding:14px 18px;border-radius:8px}}
.case{{background:white;border-radius:14px;padding:20px;margin:20px 0;box-shadow:0 4px 18px #22334d18}}.casehead{{display:flex;align-items:center;gap:12px}}h2{{margin:0}}
.badge{{border-radius:20px;padding:4px 10px;font-size:12px}}.pending{{background:#e8edf4}}.done{{background:#dff8ee;color:#087451}}
.images{{display:grid;grid-template-columns:1fr 1.25fr;gap:18px;align-items:start;margin-top:15px}}figure{{margin:0}}img{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px}}figcaption{{padding:7px;color:var(--muted);font-size:13px}}
.form{{display:grid;grid-template-columns:1.25fr 1.4fr 1fr .55fr;gap:15px;margin-top:18px;padding-top:15px;border-top:1px solid var(--line)}}label{{display:block;font-size:12px;font-weight:bold;margin-bottom:7px;color:var(--muted)}}.choices{{display:flex;gap:7px;flex-wrap:wrap}}
.choices button{{border:1px solid #aab8ca;background:white;border-radius:7px;padding:8px 11px;cursor:pointer}}.choices button.selected{{color:white;border-color:transparent;background:#315a88}}
.state button[data-value=free].selected{{background:var(--free)}}.state button[data-value=occupied].selected{{background:var(--occ)}}.state button[data-value=unknown].selected{{background:var(--unk)}}
input{{width:100%;padding:9px;border:1px solid #aab8ca;border-radius:7px}}.wide{{grid-column:1/-1}}.progress{{font-weight:bold}}
@media(max-width:950px){{.images,.form{{grid-template-columns:1fr}}.wide{{grid-column:auto}}header button{{margin-left:0}}}}
</style></head><body>
<header><h1>Frame 6241 人工 GT</h1><p class="progress">0 / {len(rows)} 已标注</p><p>预测盲：页面不显示 Part1 或 Agent 结果</p><button id="export">导出 GT CSV</button></header>
<main><div class="intro"><b>规则：</b>只有目标车位身份明确且内部足够可见时填 Free / Occupied；被车辆、柱子遮挡或身份不确定时填 Unknown。页面选择自动保存在本机浏览器。</div>
{''.join(blocks)}</main>
<script>
const META={payload}; const KEY='parking_gt_frame6241_pose_tuned_v1';
let saved=JSON.parse(localStorage.getItem(KEY)||'{{}}');
const cases=[...document.querySelectorAll('.case')];
function select(group,value){{group.querySelectorAll('button').forEach(b=>b.classList.toggle('selected',b.dataset.value===value));}}
function collect(c){{return{{slot_id:c.dataset.slot,gt_state:c.querySelector('.state .selected')?.dataset.value||'',gt_observability:c.querySelector('.observable .selected')?.dataset.value||'',identity_verified:c.querySelector('.identity .selected')?.dataset.value||'',state_confidence:c.querySelector('.confidence').value,annotator:'human',evidence_frames:c.dataset.frames,notes:c.querySelector('.notes').value}}}}
function refresh(){{let n=0;cases.forEach(c=>{{let d=collect(c),done=!!d.gt_state;if(done)n++;let b=c.querySelector('.badge');b.textContent=done?d.gt_state.toUpperCase():'未标注';b.className='badge '+(done?'done':'pending')}});document.querySelector('.progress').textContent=`${{n}} / ${{cases.length}} 已标注`;localStorage.setItem(KEY,JSON.stringify(Object.fromEntries(cases.map(c=>[c.dataset.slot,collect(c)]))))}}
cases.forEach(c=>{{let d=saved[c.dataset.slot]||{{}};[['state','gt_state'],['observable','gt_observability'],['identity','identity_verified']].forEach(([g,k])=>{{if(d[k])select(c.querySelector('.'+g),d[k])}});if(d.state_confidence)c.querySelector('.confidence').value=d.state_confidence;if(d.notes)c.querySelector('.notes').value=d.notes;c.querySelectorAll('.choices button').forEach(b=>b.onclick=()=>{{select(b.parentElement,b.dataset.value);refresh()}});c.querySelectorAll('input').forEach(x=>x.oninput=refresh)}});
document.querySelector('#export').onclick=()=>{{let rows=cases.map(collect);let missing=rows.filter(r=>!r.gt_state);if(missing.length&&!confirm(`还有 ${{missing.length}} 个未标注，仍然导出吗？`))return;let fields=['slot_id','gt_state','gt_observability','identity_verified','state_confidence','annotator','evidence_frames','notes'];let esc=v=>'\"'+String(v??'').replaceAll('\"','\"\"')+'\"';let csv='\\ufeff'+fields.join(',')+'\\n'+rows.map(r=>fields.map(k=>esc(r[k])).join(',')).join('\\n');let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='frame6241_camera_visible_gt.csv';a.click();URL.revokeObjectURL(a.href)}};refresh();
</script></body></html>"""
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(a.output), "cases": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
