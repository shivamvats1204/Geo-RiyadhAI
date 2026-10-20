/* Geo-RiyadhAI: one workspace = the all-Riyadh district map + the analyst.
   Pick a district / neighbourhood (or use the map view), ask a question, read the answer in plain words, see the places on the map.
   Every number comes from the API (computed server-side). */
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const state = { session: null, scope: null, ops: [], addMode: false, weights: {}, lastTop: null };

/* ----------------------------------------------------------------- config */
const TASKS = [
  { icon: '📍', title: 'Best place for a new shop', text: 'Find five suitable commercial locations for a supermarket. Avoid highways, prefer crossroads, compare nearby competitors and parking, and explain the recommendation.' },
  { icon: '📊', title: 'Count places by district', text: 'Count shops in each district' },
  { icon: '🅿️', title: 'Where should parking go?', text: 'Where should three new parking facilities be located to improve access?' },
  { icon: '✅', title: 'Find places that meet rules', text: 'Find shops located at crossroads but away from highways, and show nearby parking.' },
  { icon: '📖', title: 'Explain a planning term', text: 'What does maxBuildingCoefficient mean?' },
];
const CRITERIA = { demand: ['Customers nearby', 0.30], junction_access: ['Easy to reach (crossroads)', 0.20], road_visibility: ['Visible from the road', 0.10], competition: ['Few competitors', 0.25], agglomeration: ['Other shops nearby', 0.10], parking: ['Parking', 0.05] };
const STEP_WORDS = { get_geographic_coverage: 'checking what data is available', rank_site_suitability: 'ranking possible places', optimize_facility_locations: 'finding the best spots', compare_scenarios: 'testing what-ifs', aggregate_spatial_statistics: 'counting', run_geoquery: 'checking every place against your rules', filter_features: 'filtering places', retrieve_reference_documents: 'looking up explanations', generate_evidence_report: 'writing the report', find_intersections: 'finding crossroads', measure_proximity: 'measuring distances', calculate_network_accessibility: 'measuring road access' };
const LAYERS = [
  { id: 'urbi_building', label: 'Buildings', minzoom: 14, zooms: [14], color: '#f79009' },
  { id: 'parcel', label: 'Land parcels', minzoom: 15, zooms: [14], color: '#7a5af8' },
  { id: 'urbi_parking', label: 'Parking', minzoom: 12, zooms: [12, 14], color: '#12b76a' },
  { id: 'street', label: 'Streets', minzoom: 13, zooms: [13, 14], color: '#98a2b3' },
  { id: 'attraction', label: 'Attractions', minzoom: 12, zooms: [12, 14], color: '#ee46bc' },
  { id: 'metro', label: 'Metro', minzoom: 12, zooms: [12, 14], color: '#06aed4' },
];
const BOUNDS = [
  { id: 'urbi_district', source: 'urbi', kind: 'district', label: 'Districts', color: '#2e90fa', on: true },
  { id: 'suhail_nbh', source: 'suhail', label: 'Neighbourhoods (Suhail)', color: '#f79009', on: false },
  { id: 'overture_nbh', source: 'overture', label: 'Neighbourhoods (Overture)', color: '#12b76a', on: false },
  { id: 'urbi_living_area', source: 'urbi', kind: 'living_area', label: 'Living areas', color: '#ee46bc', on: true },
  { id: 'urbi_settlement', source: 'urbi', kind: 'settlement', label: 'Settlements', color: '#f04438', on: false },
  { id: 'urbi_district_area', source: 'urbi', kind: 'district_area', label: 'District areas', color: '#d0d5dd', on: false },
];
const PART = { d: { source: 'urbi', kind: 'district' }, n: { source: 'suhail' }, o: { source: 'overture' } };
const hue = (f) => ['case', ['<', ['to-number', f], 0], '#667085', ['to-color', ['concat', 'hsl(', ['to-string', ['%', ['*', ['to-number', f], 47], 360]], ',65%,55%)']]];
const PURPOSE_COLOR = ['match', ['get', 'purpose_class'], 'residential', '#2e90fa', 'commercial', '#f79009', 'industrial', '#7a5af8', 'public', '#12b76a', 'education', '#06aed4', 'religious', '#ee46bc', 'health', '#f04438', '#98a2b3'];
const FRIENDLY = {
  rank_site_suitability: { rank: '#', score: ['Score /100', (v) => Math.round(v * 100)], competitors_500m: 'Competitors within 500 m', dist_four_way_m: 'To crossroads (m)', retail_500m: 'Shops within 500 m', parking_300m: 'Parking within 300 m', dist_highway_m: 'From highway (m)' },
  aggregate_spatial_statistics: { name: 'Area', count: 'Count', density_per_km2: 'Per km²', area_km2: ['Area km²', (v) => +v.toFixed(1)] },
  optimize_facility_locations: { strategy: 'Option', mean_distance_m: 'Average trip (m)', coverage_within_radius: ['Within 500 m', (v) => Math.round(v * 100) + '%'] },
  compare_scenarios: { scenario: 'What-if', delta_mean_distance_m: 'Trip change (m)', delta_coverage_within_radius: ['Coverage change', (v) => (v * 100).toFixed(1) + ' pts'] },
};

let map, reg = [], regByIdx = {}, summary = {}, sel = null, partition = 'd', mapReadyResolve;
const mapReady = new Promise((r) => { mapReadyResolve = r; });
const gj = {};
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) { let t = await r.text(); try { t = JSON.parse(t).detail || t; } catch (e) {} throw new Error(t); }
  return r.json();
};
const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

/* -------------------------------------------------------------------- map */
async function init() {
  summary = await api('/v1/city/summary');
  reg = await api('/v1/city/registry.json');
  reg.forEach((r) => regByIdx[r.idx] = r);
  map = new maplibregl.Map({
    container: 'map', center: [46.72, 24.71], zoom: 10.6, maxZoom: 20, hash: true,
    style: { version: 8, glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
      sources: { osm: { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256, attribution: '© OpenStreetMap contributors · Urbi/2GIS · Suhail · Overture' } },
      layers: [{ id: 'osm', type: 'raster', source: 'osm', paint: { 'raster-opacity': 0.55 } }] },
  });
  map.addControl(new maplibregl.NavigationControl());
  map.addControl(new maplibregl.ScaleControl());
  await new Promise((r) => map.on('load', r));
  for (const b of BOUNDS) await addBoundary(b);
  for (const l of LAYERS) addDataLayer(l);
  addResultLayers();
  buildLayerBar(); buildList(); coverage();
  map.on('click', onClick); map.on('zoom', zoomHint); map.on('moveend', updateChips); zoomHint();
  mapReadyResolve();
  try {
    const h = await api('/v1/health');
    $('#badge').textContent = `${reg.length} areas · ${(summary.layers.urbi_building || 0).toLocaleString()} buildings`;
    $('#badge').className = 'badge ' + (h.geoacquirer.available ? 'ok' : 'off');
    $('#badge').title = h.geoacquirer.available ? 'live data access available' : 'offline: using stored data only';
  } catch (e) { /* badge only */ }
}

async function addBoundary(b) {
  const data = await api(`/v1/city/registry.geojson?source=${b.source}` + (b.kind ? `&kind=${b.kind}` : ''));
  gj[b.id] = data;
  map.addSource(b.id, { type: 'geojson', data });
  const vis = b.on ? 'visible' : 'none', poly = ['==', ['get', 'has_polygon'], true];
  map.addLayer({ id: b.id + '-fill', type: 'fill', source: b.id, filter: poly, layout: { visibility: vis }, paint: { 'fill-color': hue(['get', 'idx']), 'fill-opacity': 0.14 } });
  map.addLayer({ id: b.id + '-line', type: 'line', source: b.id, filter: poly, layout: { visibility: vis }, paint: { 'line-color': b.color, 'line-width': 1.4 } });
  map.addLayer({ id: b.id + '-pt', type: 'circle', source: b.id, filter: ['!=', ['get', 'has_polygon'], true], layout: { visibility: vis }, paint: { 'circle-radius': 6, 'circle-color': b.color, 'circle-stroke-color': '#fff', 'circle-stroke-width': 1.5 } });
  map.addLayer({ id: b.id + '-label', type: 'symbol', source: b.id, minzoom: 10, layout: { visibility: vis, 'text-field': ['coalesce', ['get', 'name'], ''], 'text-font': ['Open Sans Regular', 'Arial Unicode MS Regular'], 'text-size': 11 },
    paint: { 'text-color': '#101828', 'text-halo-color': '#fff', 'text-halo-width': 1.5 } });
}

const variants = (l) => l.zooms.map((z, i) => ({ z, key: `${l.id}@${z}`, minzoom: i === 0 ? l.minzoom : z, maxzoom: i < l.zooms.length - 1 ? l.zooms[i + 1] : 24 }));
const allLayerIds = (lid) => { const l = LAYERS.find((x) => x.id === lid); return variants(l).flatMap((v) => ['-fill', '-line', '-pt', '-c'].map((s) => v.key + s)); };

function addDataLayer(l) {
  for (const v of variants(l)) {
    map.addSource('c_' + v.key, { type: 'vector', tiles: [location.origin + `/v1/city/tiles/${l.id}/{z}/{x}/{y}.pbf`], minzoom: v.z, maxzoom: v.z, promoteId: 'i' });
    const col = l.id === 'urbi_building' ? ['step', ['zoom'], hue(['get', partition]), 16, PURPOSE_COLOR] : hue(['get', partition]);
    const common = { source: 'c_' + v.key, minzoom: v.minzoom, maxzoom: v.maxzoom };
    const fillOp = { urbi_building: 0.75, parcel: 0.04, street: 0.05, urbi_parking: 0.55, attraction: 0.45, metro: 0.6 }[l.id];
    map.addLayer({ ...common, id: v.key + '-fill', 'source-layer': l.id, type: 'fill', filter: ['==', ['geometry-type'], 'Polygon'], paint: { 'fill-color': col, 'fill-opacity': fillOp } });
    map.addLayer({ ...common, id: v.key + '-line', 'source-layer': l.id, type: 'line', filter: ['any', ['==', ['geometry-type'], 'Polygon'], ['==', ['geometry-type'], 'LineString']],
      paint: { 'line-color': { urbi_building: '#7a4a00', parcel: '#7a5af8', street: '#667085' }[l.id] || l.color, 'line-width': l.id === 'urbi_building' ? 0.6 : l.id === 'street' ? 0.9 : 0.5, 'line-opacity': l.id === 'parcel' ? 0.7 : 0.9 } });
    map.addLayer({ ...common, id: v.key + '-pt', 'source-layer': l.id, type: 'circle', filter: ['==', ['geometry-type'], 'Point'], paint: { 'circle-radius': l.id === 'urbi_building' ? 3 : 5, 'circle-color': l.color, 'circle-stroke-color': '#fff', 'circle-stroke-width': 1 } });
    map.addLayer({ ...common, id: v.key + '-c', 'source-layer': l.id + '_c', type: 'circle', paint: { 'circle-radius': 2.5, 'circle-color': l.color } });
  }
}

function addResultLayers() {
  for (const id of ['results', 'selected', 'scenario']) map.addSource(id, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addLayer({ id: 'results-fill', type: 'fill', source: 'results', filter: ['==', '$type', 'Polygon'], paint: { 'fill-color': '#f04438', 'fill-opacity': 0.45 } });
  map.addLayer({ id: 'results-line', type: 'line', source: 'results', filter: ['any', ['==', '$type', 'LineString'], ['==', '$type', 'Polygon']], paint: { 'line-color': '#b42318', 'line-width': 2.5 } });
  map.addLayer({ id: 'results', type: 'circle', source: 'results', filter: ['==', '$type', 'Point'], paint: { 'circle-color': '#f04438', 'circle-radius': 11, 'circle-stroke-width': 2, 'circle-stroke-color': '#fff' } });
  map.addLayer({ id: 'results-label', type: 'symbol', source: 'results', filter: ['==', '$type', 'Point'], layout: { 'text-field': ['to-string', ['coalesce', ['get', 'rank'], '']], 'text-font': ['Open Sans Regular', 'Arial Unicode MS Regular'], 'text-size': 12, 'text-allow-overlap': true },
    paint: { 'text-color': '#fff' } });
  map.addLayer({ id: 'selected', type: 'circle', source: 'selected', paint: { 'circle-color': 'rgba(0,0,0,0)', 'circle-radius': 17, 'circle-stroke-width': 3, 'circle-stroke-color': '#111' } });
  map.addLayer({ id: 'scenario', type: 'circle', source: 'scenario', paint: { 'circle-color': '#7a5af8', 'circle-radius': 8, 'circle-stroke-width': 2, 'circle-stroke-color': '#fff' } });
}

/* ------------------------------------------------ compact layers bar (above the map) */
function buildLayerBar() {
  $('#boundToggles').innerHTML = BOUNDS.map((b) => `<div class="tog"><input type="checkbox" id="b_${b.id}" ${b.on ? 'checked' : ''}><i style="background:${b.color}"></i><label for="b_${b.id}">${b.label} (${gj[b.id].features.length})</label></div>`).join('');
  BOUNDS.forEach((b) => $('#b_' + b.id).onchange = (e) => { ['-fill', '-line', '-pt', '-label'].forEach((s) => map.setLayoutProperty(b.id + s, 'visibility', e.target.checked ? 'visible' : 'none')); updateChips(); });
  $('#dataToggles').innerHTML = LAYERS.map((l) => `<div class="tog"><input type="checkbox" id="d_${l.id}" checked><i style="background:${l.color}"></i><label for="d_${l.id}">${l.label} <span class="muted">(${(summary.layers[l.id] || 0).toLocaleString()}, from zoom ${l.minzoom})</span></label></div>`).join('');
  LAYERS.forEach((l) => $('#d_' + l.id).onchange = (e) => { allLayerIds(l.id).forEach((i) => map.setLayoutProperty(i, 'visibility', e.target.checked ? 'visible' : 'none')); updateChips(); });
  $('#osm').onchange = (e) => map.setLayoutProperty('osm', 'visibility', e.target.checked ? 'visible' : 'none');
  $('#byType').onchange = recolor;
  $('#layersBtn').onclick = () => { const p = $('#layerPanel'); p.hidden = !p.hidden; $('#layersBtn').setAttribute('aria-expanded', String(!p.hidden)); $('#layersBtn').textContent = p.hidden ? '▦ Layers ▾' : '▦ Layers ▴'; };
  $('#verifyBtn').onclick = verify;
  updateChips();
}
function updateChips() {
  const on = [...BOUNDS.filter((b) => $('#b_' + b.id)?.checked).map((b) => [b.label, b.color]), ...LAYERS.filter((l) => $('#d_' + l.id)?.checked).map((l) => [l.label, l.color])];
  const z = map ? map.getZoom() : 0;
  $('#layerChips').innerHTML = on.slice(0, 7).map(([t, c]) => `<span class="pill"><i style="background:${c}"></i>${esc(t)}</span>`).join('') + (on.length > 7 ? `<span class="pill">+${on.length - 7}</span>` : '') + (z < 14 ? '<span class="pill hint">zoom in to see buildings</span>' : '');
}
function zoomHint() {
  const z = map.getZoom();
  $('#zoomHint').textContent = `Zoom ${z.toFixed(1)} · ` + LAYERS.map((l) => `${l.label} ${z >= l.minzoom ? '✓' : 'from zoom ' + l.minzoom}`).join(' · ');
  updateChips();
}
function recolor() {
  const byType = $('#byType').checked;
  LAYERS.forEach((l) => variants(l).forEach((v) => map.setPaintProperty(v.key + '-fill', 'fill-color', l.id === 'urbi_building' ? (byType ? PURPOSE_COLOR : ['step', ['zoom'], hue(['get', partition]), 16, PURPOSE_COLOR]) : hue(['get', partition]))));
}
function applyFilter() {
  LAYERS.forEach((l) => {
    const poly = ['==', ['geometry-type'], 'Polygon'], line = ['any', poly, ['==', ['geometry-type'], 'LineString']], pt = ['==', ['geometry-type'], 'Point'];
    const f = sel === null ? null : ['==', ['get', partition], sel], w = (b) => (f ? ['all', b, f] : b);
    variants(l).forEach((v) => { map.setFilter(v.key + '-fill', w(poly)); map.setFilter(v.key + '-line', w(line)); map.setFilter(v.key + '-pt', w(pt)); map.setFilter(v.key + '-c', f); });
  });
  BOUNDS.forEach((b) => map.setPaintProperty(b.id + '-line', 'line-width', ['case', ['==', ['get', 'idx'], sel === null ? -2 : sel], 4, 1.4]));
}

/* ---------------------------------------- where: district / neighbourhood chooser (replaces the old "area" box) */
function rowsForPartition() { const p = PART[partition]; return reg.filter((r) => r.source === p.source && (!p.kind || r.kind === p.kind)); }
function buildList() {
  const p = PART[partition], q = $('#search').value.trim().toLowerCase();
  const rows = rowsForPartition().filter((r) => !q || (r.name || '').toLowerCase().includes(q) || (r.name_ar || '').includes(q))
    .map((r) => ({ r, n: (summary.district_counts[r.uid] || {}).urbi_building || 0 })).sort((a, b) => (a.r.name || a.r.name_ar || '').localeCompare(b.r.name || b.r.name_ar || ''));
  $('#distList').innerHTML = rows.map(({ r, n }) => `<div data-i="${r.idx}" class="${r.idx === sel ? 'sel' : ''}"><span>${esc(r.name || r.name_ar || r.uid)}${r.name && r.name_ar ? ` <span class="muted ar">${esc(r.name_ar)}</span>` : ''}${r.has_polygon ? '' : ' ·'}</span><span class="muted">${n ? n.toLocaleString() : ''}</span></div>`).join('') || '<div class="muted">no match</div>';
  $('#distList').querySelectorAll('div[data-i]').forEach((d) => d.onclick = () => selectArea(+d.dataset.i));
}
async function selectArea(idx) {
  sel = idx; applyFilter(); buildList();
  const r = regByIdx[idx];
  const feat = Object.values(gj).flatMap((g) => g.features).find((f) => f.properties.idx === idx);
  if (feat) {
    if (feat.geometry.type === 'Point') map.flyTo({ center: feat.geometry.coordinates, zoom: 15 });
    else { let w = 1e9, s = 1e9, e = -1e9, n = -1e9; const walk = (c) => { if (typeof c[0] === 'number') { w = Math.min(w, c[0]); e = Math.max(e, c[0]); s = Math.min(s, c[1]); n = Math.max(n, c[1]); } else c.forEach(walk); }; walk(feat.geometry.coordinates); map.fitBounds([[w, s], [e, n]], { padding: 40 }); }
  }
  showSelected(r);
  await setScope({ uid: r.uid }, r.name || r.name_ar);
}
$('#mapViewBtn')?.addEventListener('click', async () => { sel = null; applyFilter(); buildList(); $('#selected').hidden = true; state.scope = null; await ensureScope(true); });

function viewBBox() {
  const b = map.getBounds(); let [w, s, e, n] = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
  const cx = (w + e) / 2, cy = (s + n) / 2, hw = Math.min((e - w) / 2, 0.15), hh = Math.min((n - s) / 2, 0.15);  // an analysis area is at most ~0.3 deg wide
  return [cx - hw, cy - hh, cx + hw, cy + hh].map((v) => +v.toFixed(5));
}
async function setScope(body, label) {
  $('#scopeChip').textContent = `Area: ${label || 'map view'}…`;
  $('#progress').innerHTML = '<span class="muted">getting the data for this area ready…</span>';
  state.session = null; clearResults();
  try {
    let r = await post('/v1/scope', body);
    while (r.state === 'building') {
      $('#progress').innerHTML = `<span class="muted">${esc(r.message)}</span>`;
      const b = r.build_id;
      for (;;) { await new Promise((x) => setTimeout(x, 4000)); const s = await api('/v1/scope/build/' + b); if (s.state === 'ready') { r = s.result; break; } if (s.state === 'failed') throw new Error(s.message); }
    }
    state.scope = { name: r.scope, uid: r.uid, share: r.coverage_share };
    $('#scopeChip').textContent = `Area: ${r.scope}`;
    $('#progress').innerHTML = '';
    const n = r.layers || {};
    const note = r.coverage_share < 0.9 ? `<div class="warnbox">The prepared map data covers about ${Math.round(r.coverage_share * 100)}% of ${esc(r.scope)}, so answers describe that part only.</div>` : '';
    $('#answer').innerHTML = `<div class="headline">Ready: ${esc(r.scope)}</div><div class="muted">${(n.poi || 0).toLocaleString()} places · ${((n.urbi_building || n.building) || 0).toLocaleString()} buildings · ${(n.parking || 0).toLocaleString()} parking spots in this area's data. Ask a question on the left.</div>${note}`;
  } catch (e) {
    state.scope = null; $('#scopeChip').textContent = 'Area: map view'; $('#progress').innerHTML = `<span class="s-error">${esc(e.message)}</span>`;
  }
}
async function ensureScope(force) {
  if (state.scope && !force) return;
  await setScope({ bbox: viewBBox() }, 'map view');
}

/* ------------------------------------------------------------------ inspector (clicked area / feature) */
function showSelected(r) {
  const c = summary.district_counts[r.uid] || {};
  const lab = { urbi_building: 'buildings', parcel: 'land parcels', urbi_parking: 'parking', street: 'streets', attraction: 'attractions', metro: 'metro' };
  $('#selected').hidden = false;
  $('#selected').innerHTML = `<b>${esc(r.name || r.name_ar)}</b>${r.name_ar && r.name ? ` <span class="muted ar">${esc(r.name_ar)}</span>` : ''}
    <span class="muted small"> · ${esc(r.source)} ${esc(r.kind)} · ${r.has_polygon ? 'outline on map' : 'point only (no outline in the source)'}</span>
    <div class="small">${Object.entries(lab).filter(([k]) => c[k]).map(([k, t]) => `${c[k].toLocaleString()} ${t}`).join(' · ') || 'no stored features'}</div>
    <button id="useArea" class="ghost small">Ask about this area</button>`;
  $('#useArea').onclick = () => selectArea(r.idx);
}
function onClick(e) {
  if (state.addMode) { state.ops.push({ type: 'add_facility', lon: +e.lngLat.lng.toFixed(6), lat: +e.lngLat.lat.toFixed(6) }); renderOps(); return; }
  const rl = map.queryRenderedFeatures(e.point, { layers: ['results', 'results-fill'] })[0];
  if (rl) { inspect('result', rl); return; }
  const dl = LAYERS.flatMap((l) => allLayerIds(l.id)).filter((i) => map.getLayer(i));
  const f = map.queryRenderedFeatures(e.point, { layers: dl })[0];
  if (f) {
    const p = f.properties, nm = (k) => (regByIdx[p[k]] ? (regByIdx[p[k]].name || regByIdx[p[k]].name_ar) : '-');
    $('#selected').hidden = false;
    $('#selected').innerHTML = `<b>${esc(LAYERS.find((l) => f.layer.id.startsWith(l.id))?.label || 'feature')}</b> <span class="muted small">${esc(f.geometry.type)}</span><br>${Object.entries(p).filter(([k]) => !['i', 'd', 'n', 'o'].includes(k)).map(([k, v]) => `${esc(k)}: ${esc(v)}`).join('<br>')}
      <div class="small muted">District: <b>${esc(nm('d'))}</b> · Neighbourhood: <b>${esc(nm('n'))}</b></div>`;
    return;
  }
  const bl = BOUNDS.flatMap((b) => [b.id + '-fill', b.id + '-pt']).filter((i) => map.getLayer(i) && map.getLayoutProperty(i, 'visibility') !== 'none');
  const b = map.queryRenderedFeatures(e.point, { layers: bl })[0];
  if (b) { const r = regByIdx[b.properties.idx]; if (r) showSelected(r); }
}
function inspect(kind, f) {
  const p = f.properties || {};
  $('#selected').hidden = false;
  let html = `<b>${esc(kind === 'result' ? 'Recommended place' : kind)}</b><table>` + Object.entries(p).filter(([k]) => !['evidence_json', 'explanation', 'geometry'].includes(k) && typeof p[k] !== 'object').slice(0, 12).map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(typeof v === 'number' ? +v.toFixed(3) : v)}</td></tr>`).join('') + '</table>';
  if (p.explanation) html += `<div class="small"><b>Why it ranks here</b><br>${esc(p.explanation).split(' | ').join('<br>')}</div>`;
  $('#selected').innerHTML = html;
  if (f.geometry?.type === 'Point') map.getSource('selected').setData({ type: 'FeatureCollection', features: [f] });
}
function coverage() {
  $('#coverage').innerHTML = `Data on the map: ${Object.entries(summary.layers || {}).map(([k, v]) => `${v.toLocaleString()} ${({ urbi_building: 'buildings', parcel: 'parcels', urbi_parking: 'parking', street: 'streets', attraction: 'attractions', metro: 'metro' })[k]}`).join(' · ')}. Sources: Urbi/2GIS, Suhail, Overture Maps, OpenStreetMap.`;
}

/* --------------------------------------------------------------------- ask */
async function ask(text) {
  if (!text.trim()) return;
  $('#askBtn').disabled = true; $('#q').value = text;
  try {
    await ensureScope();
    $('#progress').innerHTML = '<span class="muted">understanding your question…</span>';
    const r = await post('/v1/ask', { question: text, session_id: state.session, mode: $('#mode').value });
    if (r.session_id) state.session = r.session_id;
    saveQuery(text);
    if (r.mode === 'code') await pollAnalysis(r.job_id); else await poll(r.job_id);
  } catch (e) { $('#progress').innerHTML = `<span class="s-error">${esc(e.message)}</span>`; }
  $('#askBtn').disabled = false;
}
async function poll(id) {
  for (let i = 0; i < 900; i++) {
    const j = await api('/v1/jobs/' + id);
    renderProgress(j);
    if (['done', 'partial', 'failed', 'unsupported', 'awaiting_approval'].includes(j.status)) { renderJob(j); return; }
    await new Promise((r) => setTimeout(r, 700));
  }
}
async function pollAnalysis(id) {
  for (let i = 0; i < 900; i++) {
    const j = await api('/v1/analysis/' + id);
    const last = (j.steps || []).slice(-1)[0];
    $('#progress').innerHTML = j.status === 'running' ? `<span class="muted">Analysing… ${esc(last ? ({ data: 'loading the data', thought: 'thinking', plan: 'planning', act: 'running code on the data', observe: 'checking the result', format: 'retrying' }[last.kind] || last.kind) : 'starting')}</span>` : '';
    if (j.status !== 'running') { renderAnalysis(j); return; }
    await new Promise((r) => setTimeout(r, 1200));
  }
}
function renderAnalysis(j) {
  $('#selected').hidden = true; clearResults();
  let h = '';
  if (j.status === 'unsupported') h = `<div class="warnbox"><b>Custom analysis is not available for this question.</b><br>${esc(j.error)}</div><div class="muted small">Tip: the built-in questions (cards on the left) work without it.</div>`;
  else if (j.status === 'needs_approval') h = `<div class="warnbox"><b>This needs your OK.</b> To answer, the system wants to ${esc(j.pending?.why || 'download more data')}.<br><span class="muted small">Open map data, cached afterwards. Nothing is downloaded until you agree.</span><br><button id="approveAnalysis">Yes, download it and continue</button></div>`;
  else if (j.status === 'needs_input') h = `<div class="headline">One question first</div><div>${esc(j.question_back)}</div><div class="muted small">Type your answer in the question box and press Get answer.</div>`;
  else if (j.status === 'failed') h = `<div class="warnbox"><b>The analysis could not be completed.</b> ${esc(j.error)}</div>`;
  else {
    const f = j.final || {};
    h = `<div class="headline">${esc(f.headline || 'Result')}</div>`;
    if (j.entities?.length) h += `<div class="muted small">Places I understood from your question: ${j.entities.map((e) => `<b>${esc(e.name)}</b> (${esc(e.kind)}${e.score < 0.8 ? ', uncertain match' : ''})`).join(', ')}</div>`;
    if (j.figure) h += `<img class="fig" src="/v1/figures/${esc(j.figure)}" alt="chart of the result">`;
    if (f.points?.length) h += `<ul class="points">${f.points.map((t) => `<li>${esc(t)}</li>`).join('')}</ul>`;
    if (j.unverified_numbers?.length) h += `<div class="warnbox small"><b>Check these numbers:</b> ${j.unverified_numbers.map(esc).join(', ')} could not be traced to the computed result.</div>`;
    else h += `<div class="conf high small"><b>Numbers checked.</b> Every number in this answer was found in what the code computed.</div>`;
    const lim = [...(f.caveats || []), ...(j.notes || [])];
    if (lim.length) h += `<div class="limits"><b>Good to know</b><ul>${[...new Set(lim)].map((t) => `<li>${esc(t)}</li>`).join('')}</ul></div>`;
    if (j.checks?.length) h += `<div class="warnbox small">Automatic checks flagged: ${j.checks.map(esc).join('; ')}</div>`;
    if (f.next_questions?.length) h += `<div class="next"><span class="muted small">Try next:</span> ${f.next_questions.map((t) => `<button class="ghost small nq">${esc(t)}</button>`).join(' ')}</div>`;
  }
  $('#answer').className = ''; $('#answer').innerHTML = h;
  document.querySelectorAll('.nq').forEach((b) => b.onclick = () => ask(b.textContent));
  if ($('#approveAnalysis')) $('#approveAnalysis').onclick = async () => { await post(`/v1/analysis/${j.id}/approve`, {}); $('#progress').innerHTML = '<span class="muted">downloading and continuing…</span>'; pollAnalysis(j.id); };
  if (j.table?.length) $('#tables').innerHTML = '<h3>Result table</h3>' + friendlyTable('custom', j.table);
  if (j.result_id) { showResult(j.result_id); $('#downloads').innerHTML = `<h3>Save or share</h3><a href="/v1/results/${j.result_id}.csv">Download table (CSV)</a> · <a href="/v1/results/${j.result_id}.geojson">Map file (GeoJSON)</a>`; }
  const steps = (j.steps || []).map((s) => `<div class="step"><b>${esc(s.kind)}</b> ${esc(s.text || s.interpretation || '')}${s.steps ? '<ol>' + s.steps.map((x) => `<li>${esc(x)}</li>`).join('') + '</ol>' : ''}${s.assumptions?.length ? '<div class="muted small">assumptions: ' + s.assumptions.map(esc).join('; ') + '</div>' : ''}</div>`).join('');
  const lc = j.llm_calls && j.llm_calls.calls ? `<h3>Model calls made by the code (${j.llm_calls.calls}, ${j.llm_calls.cached} cached; budget ${j.llm_calls.budget_calls})</h3>` + (j.llm_calls.log || []).map((c) => `<div class="small muted">${esc(c.prompt)}… → ${esc(c.answer)}</div>`).join('') : '';
  const aq = (j.acquisition || []).length ? '<h3>Data acquisition</h3>' + j.acquisition.map((a) => `<div class="small">${esc(a.tool)} ${esc(JSON.stringify(a.args || {}))}<pre>${esc(JSON.stringify(a.result, null, 1).slice(0, 900))}</pre></div>`).join('') : '';
  $('#tech').innerHTML = lc + aq + `<div class="muted small">Method: ${esc(j.mode)} · ${j.seconds}s. The question was turned into Python, run on the data in an isolated process, then explained from what it produced.</div><h3>Steps</h3>${steps}<h3>The code that was run</h3><pre>${esc(j.code || '')}</pre>`;
  $('#techBox').hidden = false;
}
function renderProgress(j) {
  const steps = j.plan?.steps || [], done = steps.filter((s) => j.trace.find((t) => t.step === s.id && ['ok', 'error'].includes(t.status))).length;
  const cur = steps.find((s) => !j.trace.find((t) => t.step === s.id && ['ok', 'error'].includes(t.status)));
  $('#progress').innerHTML = ['done', 'partial', 'failed', 'unsupported'].includes(j.status) ? '' : `<span class="muted">Working… step ${Math.min(done + 1, steps.length)} of ${steps.length}${cur ? ': ' + (STEP_WORDS[cur.tool] || cur.tool) : ''}</span>`;
}
function friendlyTable(tool, rows) {
  const spec = FRIENDLY[tool];
  const cols = spec ? Object.keys(spec).filter((c) => c in rows[0]) : Object.keys(rows[0]).filter((c) => !/geometry|_id$|^id$|json|wkb/i.test(c)).slice(0, 6);
  const head = (c) => (spec ? (Array.isArray(spec[c]) ? spec[c][0] : spec[c]) : c.replace(/_/g, ' '));
  const fmt = (c, v) => { const f = spec && Array.isArray(spec[c]) ? spec[c][1] : null; if (v == null) return ''; if (f) return f(v); return typeof v === 'number' ? +v.toFixed(2) : v; };
  return `<table><tr>${cols.map((c) => `<th>${esc(head(c))}</th>`).join('')}</tr>${rows.map((r, i) => `<tr data-i="${i}">${cols.map((c) => `<td>${esc(fmt(c, r[c]))}</td>`).join('')}</tr>`).join('')}</table>`;
}
function plainHtml(p) {
  let h = `<div class="headline">${esc(p.headline)}</div>`;
  if (p.cards?.length) h += `<div class="cards">${p.cards.map((c) => `<div class="card"><div class="big">${esc(c.value)}</div><div>${esc(c.label)}</div><div class="muted small">${esc(c.hint || '')}</div></div>`).join('')}</div>`;
  if (p.points?.length) h += `<ul class="points">${p.points.map((t) => `<li>${esc(t)}</li>`).join('')}</ul>`;
  if (p.confidence) h += `<div class="conf ${esc(p.confidence.level)}"><b>How sure are we? ${({ high: 'Quite sure', medium: 'Fairly sure', low: 'Not very sure' })[p.confidence.level] || ''}</b><br>${esc(p.confidence.text)}</div>`;
  if (p.limits?.length) h += `<div class="limits"><b>Good to know</b><ul>${p.limits.map((t) => `<li>${esc(t)}</li>`).join('')}</ul></div>`;
  if (p.how_to_read) h += `<div class="howto muted small">${esc(p.how_to_read)}</div>`;
  if (p.next_questions?.length) h += `<div class="next"><span class="muted small">Try next:</span> ${p.next_questions.map((t) => `<button class="ghost small nq">${esc(t)}</button>`).join(' ')}</div>`;
  return h;
}
function renderJob(j) {
  $('#selected').hidden = true;
  let html = '';
  if (j.status === 'awaiting_approval') html += `<div class="warnbox"><b>This needs your OK:</b> ${esc(j.pending_approval.why)}<br><button id="approveBtn">Yes, go ahead</button></div>`;
  if (j.status === 'failed') html += `<div class="warnbox"><b>Something went wrong.</b> ${esc(j.error || '')} Try rephrasing, or pick a different area.</div>`;
  if (j.plain) html += plainHtml(j.plain);
  else if (j.status !== 'failed') html += (j.answer || []).slice(0, 4).map((f) => `<div class="fact">${esc(f.text)}</div>`).join('');
  for (const w of [...new Set(j.warnings || [])].slice(0, 3)) html += `<div class="warnbox small">${esc(w)}</div>`;
  $('#answer').className = ''; $('#answer').innerHTML = html || '<span class="muted">no output</span>';
  document.querySelectorAll('.nq').forEach((b) => b.onclick = () => ask(b.textContent));
  if ($('#approveBtn')) $('#approveBtn').onclick = async () => { await post(`/v1/jobs/${j.id}/approve`, { tools: [j.pending_approval.tool] }); poll(j.id); };
  let th = '';
  for (const [sid, rows] of Object.entries(j.tables || {})) {
    if (!rows.length) continue; const tool = j.results?.[sid]?.tool || sid;
    if (['retrieve_reference_documents', 'get_geographic_coverage'].includes(tool) && (j.plain?.headline || '')) continue;
    th += `<h3>${esc({ rank_site_suitability: 'Ranked places', aggregate_spatial_statistics: 'Counts', optimize_facility_locations: 'Options compared', compare_scenarios: 'What-ifs', run_geoquery: 'Places found' }[tool] || tool.replace(/_/g, ' '))}</h3>` + friendlyTable(tool, rows.slice(0, 25));
  }
  $('#tables').innerHTML = th;
  if (j.primary_result_id) showResult(j.primary_result_id);
  let dl = '';
  for (const r of Object.values(j.results || {})) if (r.result_id) dl += `<a href="/v1/results/${r.result_id}.csv">Download table (CSV)</a> · <a href="/v1/results/${r.result_id}.geojson">Map file (GeoJSON)</a><br>`;
  if (j.report) dl += `<a href="/v1/report/${j.id}" target="_blank">Full report with evidence (HTML)</a>`;
  $('#downloads').innerHTML = dl ? '<h3>Save or share</h3>' + dl : '';
  renderTech(j);
}
function renderTech(j) {
  const names = { observed: 'measured', model: 'model', inferred: 'inferred', reference: 'reference note' };
  const facts = (j.answer || []).map((f) => `<div class="fact ${f.type}"><span class="tag">${names[f.type] || f.type}</span> ${esc(f.text)} <code>${esc(f.calc)}</code></div>`).join('');
  const steps = (j.plan?.steps || []).map((s) => { const t = j.trace.find((x) => x.step === s.id); return `<div class="step"><b>${esc(s.tool)}</b> <span class="muted">${esc(s.why)}</span> · ${t ? esc(t.status) : 'pending'}${t?.seconds != null ? ' · ' + t.seconds + 's' : ''}</div>`; }).join('');
  const audit = j.claim_audit ? `<div class="muted small">Number check: ${j.claim_audit.numbers_checked} numbers in the answer traced to a calculation, ${j.claim_audit.unsupported.length} not traced.</div>` : '';
  $('#tech').innerHTML = facts + audit + '<h3>Steps taken</h3>' + steps;
  $('#techBox').hidden = false;
}
async function showResult(rid) {
  await mapReady;
  const g = await fetch(`/v1/results/${rid}.geojson`).then((r) => r.json());
  map.getSource('results').setData(g); state.lastTop = g;
  if (g.features.length) {
    const b = new maplibregl.LngLatBounds();
    for (const f of g.features) { const c = f.geometry.type === 'Point' ? [f.geometry.coordinates] : f.geometry.type === 'Polygon' ? f.geometry.coordinates[0] : []; c.forEach((x) => b.extend(x)); }
    if (!b.isEmpty() && g.features.length < 200) map.fitBounds(b, { padding: 90, maxZoom: 16, duration: 600 });
  }
  document.querySelectorAll('#tables tr[data-i]').forEach((tr) => { tr.classList.add('hl'); tr.onclick = () => { const f = g.features[+tr.dataset.i]; if (f) { inspect('result', f); if (f.geometry.type === 'Point') map.flyTo({ center: f.geometry.coordinates, zoom: 16 }); } }; });
}
function clearResults() { if (map?.getSource('results')) map.getSource('results').setData({ type: 'FeatureCollection', features: [] }); $('#tables').innerHTML = ''; $('#downloads').innerHTML = ''; $('#techBox').hidden = true; $('#selected').hidden = true; }

/* ------------------------------------------------------ weights / rerank / what-if */
function buildWeights() {
  const w = $('#weights');
  for (const [k, [label, v]] of Object.entries(CRITERIA)) {
    state.weights[k] = v;
    w.insertAdjacentHTML('beforeend', `<label class="w">${esc(label)}<input type="range" min="0" max="100" value="${v * 100}" data-k="${k}"><span id="wv_${k}">${v.toFixed(2)}</span></label>`);
  }
  w.querySelectorAll('input').forEach((el) => el.oninput = () => { state.weights[el.dataset.k] = el.value / 100; $('#wv_' + el.dataset.k).textContent = (el.value / 100).toFixed(2); });
  $('#rerank').onclick = async () => {
    $('#rerank').disabled = true;
    try {
      await ensureScope();
      const r = await post('/v1/tool', { tool: 'rank_site_suitability', args: { facility: 'supermarket', k: 5, region: { kind: 'all' }, weights: state.weights } });
      await showResult(r.result_id);
      $('#answer').innerHTML = `<div class="headline">Re-ranked with your choices</div><ul class="points">${r.facts.filter((f) => f.type === 'inferred').slice(0, 4).map((f) => `<li>${esc(f.text)}</li>`).join('')}</ul>`;
      $('#tables').innerHTML = '<h3>Ranked places</h3>' + friendlyTable('rank_site_suitability', r.table);
      document.querySelectorAll('#tables tr[data-i]').forEach((tr) => { tr.classList.add('hl'); tr.onclick = () => { const f = state.lastTop.features[+tr.dataset.i]; if (f) { inspect('result', f); map.flyTo({ center: f.geometry.coordinates, zoom: 16 }); } }; });
    } catch (e) { $('#answer').innerHTML = `<div class="warnbox">${esc(e.message)}</div>`; }
    $('#rerank').disabled = false;
  };
}
function renderOps() {
  $('#ops').innerHTML = state.ops.map((o, i) => `<li data-i="${i}">parking lot at ${o.lat}, ${o.lon}  ✕</li>`).join('');
  $('#ops').querySelectorAll('li').forEach((li) => li.onclick = () => { state.ops.splice(+li.dataset.i, 1); renderOps(); });
  map.getSource('scenario').setData({ type: 'FeatureCollection', features: state.ops.map((o) => ({ type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: [o.lon, o.lat] } })) });
}
async function simulate() {
  if (!state.ops.length) { $('#answer').innerHTML = '<div class="warnbox">Click the map to add at least one parking lot first.</div>'; return; }
  try {
    await ensureScope();
    const r = await post('/v1/tool', { tool: 'compare_scenarios', args: { scenarios: [{ name: 'your what-if', operations: state.ops }] } });
    const row = r.table?.[r.table.length - 1] || {};
    const d = row.delta_mean_distance_m;
    $('#answer').innerHTML = `<span class="sim">SIMULATED - nothing real changes</span><div class="headline">${d == null ? 'Result' : `Average trip to parking would be ${Math.abs(d).toFixed(1)} m ${d < 0 ? 'shorter' : 'longer'}`}</div>` + r.warnings.map((w) => `<div class="warnbox small">${esc(w)}</div>`).join('');
    $('#tables').innerHTML = '<h3>What-if</h3>' + friendlyTable('compare_scenarios', r.table);
  } catch (e) { $('#answer').innerHTML = `<div class="warnbox">${esc(e.message)}</div>`; }
}

/* ------------------------------------------------- check the picture against the data */
async function verify() {
  $('#verifyOut').textContent = 'checking…';
  await new Promise((r) => { if (map.loaded() && !map.isMoving()) r(); else map.once('idle', r); });
  const b = map.getBounds(), z = map.getZoom();
  const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(6)).join(',');
  let exp;
  try { exp = await api(`/v1/city/count?bbox=${bbox}&ids=true`); } catch (e) { $('#verifyOut').textContent = 'Zoom in closer to check (the view is too large for an exact count).'; return; }
  const rows = LAYERS.map((l) => {
    const visible = z >= l.minzoom && map.getLayoutProperty(variants(l)[0].key + '-fill', 'visibility') !== 'none';
    const rendered = new Set(map.queryRenderedFeatures({ layers: allLayerIds(l.id) }).map((f) => String(f.properties.i)));
    const stored = new Set(exp[l.id].ids || []);
    return { l, visible, stored: exp[l.id].count, rendered: rendered.size, missing: [...stored].filter((i) => !rendered.has(i)).length, extra: [...rendered].filter((i) => !stored.has(i)).length };
  });
  window.__verify = { bbox, zoom: z, rows: rows.map((r) => ({ layer: r.l.id, visible: r.visible, stored: r.stored, rendered: r.rendered, missing: r.missing, extra: r.extra })) };
  $('#verifyOut').innerHTML = `<table><tr><th>layer</th><th>stored</th><th>drawn</th><th>missing</th><th>extra</th></tr>${rows.map((r) => `<tr><td>${r.l.label}${r.visible ? '' : ' (zoom in)'}</td><td>${r.stored}</td><td>${r.rendered}</td><td class="${r.visible && r.missing ? 'bad' : 'ok'}">${r.visible ? r.missing : '-'}</td><td class="${r.extra ? 'bad' : 'ok'}">${r.extra}</td></tr>`).join('')}</table>`;
}
window.cityVerify = verify;

/* -------------------------------------------------------------------- misc */
function saveQuery(t) { const a = JSON.parse(localStorage.getItem('saved') || '[]').filter((x) => x !== t); a.unshift(t); localStorage.setItem('saved', JSON.stringify(a.slice(0, 12))); renderSaved(); }
function renderSaved() { const a = JSON.parse(localStorage.getItem('saved') || '[]'); $('#saved').innerHTML = a.map((t, i) => `<li data-i="${i}">${esc(t.slice(0, 70))}</li>`).join('') || '<li class="muted">nothing yet</li>'; $('#saved').querySelectorAll('li[data-i]').forEach((li) => li.onclick = () => { $('#q').value = a[+li.dataset.i]; }); }
function help() {
  $('#answer').innerHTML = `<div class="headline">How it works</div><ol class="points"><li>Pick what you want to know (a card, or type it).</li><li>Pick a district or neighbourhood on the left, or just zoom the map to where you care about.</li><li>Press <b>Get answer</b>. You get a short answer, the key numbers, and the places on the map.</li><li>Click a row in the table to see that place. Open <b>Technical details</b> only if you want to see how a number was computed.</li></ol><div class="muted small">Everything is estimated from open map data (Urbi/2GIS, Suhail, Overture, OpenStreetMap). There is no sales, traffic, population or land-price data.</div>`;
}
window.addEventListener('DOMContentLoaded', () => {
  $('#taskCards').innerHTML = TASKS.map((t, i) => `<button class="task" data-i="${i}"><span class="ico">${t.icon}</span><span>${esc(t.title)}</span></button>`).join('');
  $('#taskCards').querySelectorAll('.task').forEach((b) => b.onclick = () => ask(TASKS[+b.dataset.i].text));
  $('#askBtn').onclick = () => ask($('#q').value);
  $('#q').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask($('#q').value); });
  $('#clearBtn').onclick = () => { state.session = null; $('#q').value = ''; clearResults(); $('#answer').className = 'muted'; $('#answer').textContent = 'Starting fresh. Choose a question on the left.'; };
  $('#addFac').onclick = () => { state.addMode = !state.addMode; $('#addFac').textContent = state.addMode ? 'Click the map… (click here to stop)' : 'Click the map to add a parking lot'; };
  $('#simulate').onclick = simulate;
  $('#partition').onchange = (e) => { partition = e.target.value; sel = null; recolor(); applyFilter(); buildList(); };
  $('#search').oninput = buildList;
  $('#helpLink').onclick = (e) => { e.preventDefault(); help(); };
  buildWeights(); renderSaved(); init().catch((e) => { $('#badge').textContent = 'error: ' + e.message; console.error(e); });
});
