"""Look and feel for the dashboard: one CSS layer for the whole app and an animated hero banner
(a spinning globe with planes on orbit, plus live status chips for the selected airport/moment).
The banner is plain page HTML, so it grows with its content, and it uses container queries to adapt
to the width it is given. Pure presentation: no numbers are computed here."""
import html

import streamlit as st

FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&family=Space+Grotesk:wght@500;700&display=swap"
LEVEL_COLOR = {"NORMAL": "#10b981", "WATCH": "#f59e0b", "ACTION": "#ef4444"}
LEVEL_ICON = {"NORMAL": "🟢", "WATCH": "🟠", "ACTION": "🔴"}

APP_CSS = """
<style>
@import url('__FONTS__');
:root { --c1:#6366f1; --c2:#06b6d4; --c3:#ec4899; --glass:rgba(99,102,241,.08); --line:rgba(99,102,241,.28); }
html, body, [class*="st-"], .stMarkdown, .stCaption { font-family:'Inter', system-ui, sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family:'Material Symbols Rounded' !important; }  /* keep icon glyphs */
h1, h2, h3, h4 { font-family:'Space Grotesk', 'Inter', sans-serif !important; letter-spacing:-.01em; }
.block-container { padding-top:3.4rem; }

/* tabs as rounded pills (the selected one matches the green status chip); they wrap instead of scrolling */
.stTabs [role="tablist"] { display:flex; flex-wrap:wrap; gap:6px; width:fit-content; max-width:100%; padding:5px;
  background:var(--glass); border:1px solid var(--line); border-radius:22px; }
.stTabs [role="tab"] { border-radius:999px; padding:7px 16px; height:auto; font-weight:600; transition:background .25s, transform .25s, box-shadow .25s; }
.stTabs [role="tab"]:hover { background:rgba(16,185,129,.14); transform:translateY(-1px); }
.stTabs [role="tab"][aria-selected="true"] { background:linear-gradient(135deg,#10b981,#14b8a6) !important;
  box-shadow:0 6px 18px rgba(16,185,129,.38), inset 0 0 0 1px rgba(255,255,255,.18); }
.stTabs [role="tab"][aria-selected="true"] p { color:#fff !important; }
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }
.stTabs [role="tabpanel"] { animation:fadeUp .45s ease both; }
@keyframes fadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }

/* metric cards: gradient glass, lift on hover, gradient numbers */
[data-testid="stMetric"] { background:linear-gradient(135deg,rgba(99,102,241,.12),rgba(6,182,212,.08) 60%,rgba(236,72,153,.06));
  border:1px solid var(--line); border-radius:18px; padding:14px 18px; transition:transform .25s ease, box-shadow .25s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-4px); box-shadow:0 14px 30px rgba(99,102,241,.28); }
[data-testid="stMetricValue"] > div, [data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] p {
  white-space:normal !important; overflow:visible !important; text-overflow:clip !important; }
[data-testid="stMetricValue"] { font-size:clamp(1.15rem, 2.2vw, 1.6rem) !important; line-height:1.2; font-family:'Space Grotesk', sans-serif; font-weight:700;
  background:linear-gradient(90deg,var(--c1),var(--c2) 55%,var(--c3)); -webkit-background-clip:text; background-clip:text; color:transparent; }

/* expanders, bordered containers, tables */
[data-testid="stExpander"] details { border-radius:14px; border:1px solid var(--line); transition:box-shadow .25s ease, border-color .25s ease; }
[data-testid="stExpander"] details:hover { border-color:rgba(6,182,212,.6); box-shadow:0 8px 22px rgba(6,182,212,.18); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }

/* sidebar: soft aurora */
/* gradient as an image layer only: Streamlit's own opaque, theme-aware background colour stays underneath,
   so the sidebar never turns see-through when it slides over the page on narrow screens */
[data-testid="stSidebar"] { background-image:linear-gradient(180deg,rgba(99,102,241,.16),rgba(6,182,212,.07) 55%,rgba(236,72,153,.10)); }
[data-testid="stSidebar"] h1 { background:linear-gradient(90deg,var(--c1),var(--c2)); -webkit-background-clip:text; background-clip:text; color:transparent; }

/* buttons */
.stButton button, [data-testid="stFormSubmitButton"] button { border-radius:12px; border:none; color:#fff;
  background:linear-gradient(135deg,var(--c1),var(--c2)); box-shadow:0 6px 16px rgba(99,102,241,.35); transition:transform .2s, box-shadow .2s; }
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover { transform:translateY(-2px); box-shadow:0 10px 24px rgba(99,102,241,.45); color:#fff; }

/* alert level badge + status banner */
.badge { display:inline-block; color:#fff; padding:3px 12px; border-radius:999px; font-weight:700; font-size:.8rem; letter-spacing:.04em; }
.badge.lvl-ACTION { animation:pulseR 1.8s infinite; }
.badge.lvl-WATCH { animation:pulseA 2.2s infinite; }
@keyframes pulseR { 0% { box-shadow:0 0 0 0 rgba(239,68,68,.65); } 70% { box-shadow:0 0 0 12px rgba(239,68,68,0); } 100% { box-shadow:0 0 0 0 rgba(239,68,68,0); } }
@keyframes pulseA { 0% { box-shadow:0 0 0 0 rgba(245,158,11,.6); } 70% { box-shadow:0 0 0 10px rgba(245,158,11,0); } 100% { box-shadow:0 0 0 0 rgba(245,158,11,0); } }
.status-card { position:relative; overflow:hidden; animation:slideIn .55s cubic-bezier(.2,.8,.2,1) both; transition:transform .25s ease; }
.status-card:hover { transform:scale(1.005); }
.status-card::after { content:""; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(110deg,transparent 30%,rgba(255,255,255,.10) 50%,transparent 70%); background-size:250% 100%; animation:sheen 4.5s ease-in-out infinite; }
@keyframes slideIn { from { opacity:0; transform:translateX(-14px); } to { opacity:1; transform:none; } }
@keyframes sheen { 0% { background-position:150% 0; } 60%, 100% { background-position:-100% 0; } }

/* landing: feature cards */
.feats { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:18px; margin:8px 0 6px; }
.feat { position:relative; padding:22px 22px 18px; border-radius:22px; border:1px solid var(--line); overflow:hidden;
  background:linear-gradient(160deg,rgba(99,102,241,.12),rgba(6,182,212,.06) 55%,rgba(236,72,153,.07));
  transition:transform .3s cubic-bezier(.2,.8,.2,1), box-shadow .3s, border-color .3s; animation:fadeUp .6s ease both; }
.feat:nth-child(2) { animation-delay:.1s; } .feat:nth-child(3) { animation-delay:.2s; }
.feat:hover { transform:translateY(-6px) rotate(-.3deg); box-shadow:0 18px 40px rgba(99,102,241,.28); border-color:rgba(6,182,212,.6); }
.feat::before { content:""; position:absolute; width:180px; height:180px; right:-60px; top:-60px; border-radius:50%;
  background:radial-gradient(circle,rgba(6,182,212,.35),transparent 70%); transition:transform .5s; }
.feat:hover::before { transform:scale(1.4); }
.feat .ico { font-size:30px; width:54px; height:54px; display:grid; place-items:center; border-radius:16px;
  background:linear-gradient(135deg,var(--c1),var(--c2)); box-shadow:0 8px 20px rgba(99,102,241,.4); margin-bottom:12px; }
.feat h4 { margin:0 0 6px; font-size:1.15rem; }
.feat p { margin:0 0 12px; opacity:.85; font-size:.93rem; line-height:1.45; }
.feat .kpi { display:inline-block; font-weight:700; font-size:.85rem; padding:5px 12px; border-radius:999px;
  background:rgba(16,185,129,.15); color:#10b981; border:1px solid rgba(16,185,129,.35); }

/* landing: how it works */
.flow { display:flex; align-items:stretch; gap:6px; flex-wrap:wrap; margin:6px 0 10px; }
.step { flex:1 1 170px; position:relative; padding:18px 16px 14px; border-radius:18px; border:1px solid var(--line);
  background:var(--glass); transition:transform .25s, box-shadow .25s; animation:fadeUp .6s ease both; }
.step:hover { transform:translateY(-4px); box-shadow:0 12px 26px rgba(6,182,212,.22); }
.step .num { position:absolute; top:10px; right:12px; font-family:'Space Grotesk'; font-weight:700; font-size:.8rem; opacity:.5; }
.step .ico { font-size:26px; margin-bottom:6px; }
.step h5 { margin:0 0 4px; font-size:1rem; }
.step p { margin:0; font-size:.85rem; opacity:.8; line-height:1.4; }
.arrow { flex:0 0 26px; display:flex; align-items:center; }
.arrow span { display:block; width:100%; height:3px; border-radius:2px;
  background:linear-gradient(90deg,var(--c1),var(--c2),var(--c1)); background-size:200% 100%; animation:flowline 1.6s linear infinite; }
@keyframes flowline { to { background-position:-200% 0; } }
@media (max-width:900px) { .arrow { display:none; } }
.landing-foot { text-align:center; opacity:.65; font-size:.82rem; margin-top:18px; }

/* at-risk flights table: flagged rows tinted red with a blinking red warning sign */
.rk-wrap { overflow-x:auto; border-radius:14px; border:1px solid var(--line); margin:4px 0 6px; }
table.rk { width:100%; border-collapse:collapse; font-size:.88rem; }
.rk th { text-align:left; font-weight:600; font-size:.74rem; text-transform:uppercase; letter-spacing:.05em; opacity:.7;
  padding:10px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }
.rk td { padding:9px 12px; border-bottom:1px solid rgba(127,127,127,.15); vertical-align:middle; }
.rk tr:last-child td { border-bottom:none; }
.rk tbody tr { transition:background .2s; }
.rk tbody tr:hover td { background:rgba(99,102,241,.08); }
.rk tr.on td { background:rgba(239,68,68,.09); }
.rk tr.on:hover td { background:rgba(239,68,68,.15); }
.rk tr.on td:first-child { box-shadow:inset 4px 0 0 #ef4444; }
.rk-flag { width:22px; height:22px; display:block; animation:rk-blink 1.1s ease-in-out infinite;
  filter:drop-shadow(0 0 6px rgba(239,68,68,.95)); }
@keyframes rk-blink { 0%, 100% { opacity:1; transform:scale(1); } 50% { opacity:.2; transform:scale(.82); } }
.rk-nowrap { white-space:nowrap; }
.rk-bar { position:relative; height:20px; min-width:110px; border-radius:999px; background:rgba(127,127,127,.18); overflow:hidden; }
.rk-bar i { position:absolute; left:0; top:0; bottom:0; border-radius:999px; background:linear-gradient(90deg,#6366f1,#06b6d4); }
.rk tr.on .rk-bar i { background:linear-gradient(90deg,#f97316,#ef4444); }
.rk-bar span { position:relative; display:block; text-align:center; font-weight:700; font-size:.78rem; line-height:20px; }
.rk-why { font-size:.82rem; opacity:.88; min-width:220px; }
.rk-yes { color:#ef4444; font-weight:700; } .rk-no { opacity:.7; }
@media (prefers-reduced-motion: reduce) { .rk-flag { animation:none; } }

/* hero banner (dashboard + landing). container queries: it adapts to its own width, not the window's */
.hx { position:relative; overflow:hidden; border-radius:22px; color:#e2e8f0; container-type:inline-size; margin:0 0 6px;
  background:radial-gradient(900px 260px at 78% -30%,rgba(99,102,241,.6),transparent 60%),
             radial-gradient(600px 300px at 5% 130%,rgba(6,182,212,.45),transparent 60%),
             radial-gradient(400px 200px at 50% 120%,rgba(236,72,153,.25),transparent 60%),
             linear-gradient(135deg,#070b1f 0%,#121741 45%,#1e1b4b 100%);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.08); }
.hx-row { position:relative; display:flex; align-items:center; gap:12px; padding:22px 26px; }
.hx-text { position:relative; z-index:5; flex:1 1 auto; min-width:0; }
.hx-star { position:absolute; width:2px; height:2px; border-radius:50%; background:#fff; animation:hx-tw 3.2s infinite ease-in-out; }
@keyframes hx-tw { 0%,100% { opacity:.12; } 50% { opacity:.95; } }
.hx-shoot { position:absolute; top:18%; left:-10%; width:120px; height:1px; background:linear-gradient(90deg,transparent,#fff);
  transform:rotate(-12deg); animation:hx-shoot 7s 2s infinite ease-in; opacity:0; }
@keyframes hx-shoot { 0% { left:-10%; opacity:0; } 3% { opacity:1; } 12% { left:60%; opacity:0; } 100% { left:60%; opacity:0; } }
.hx-eyebrow { font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:#a5b4fc; font-weight:600; }
.hx-title { font-family:'Space Grotesk',Inter,sans-serif; font-weight:700; font-size:clamp(22px,3.6cqi,32px); line-height:1.12; margin:6px 0 8px;
  background:linear-gradient(90deg,#fff 0%,#a5f3fc 30%,#c4b5fd 60%,#f9a8d4 80%,#fff 100%); background-size:200% auto;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation:hx-shine 7s linear infinite; }
@keyframes hx-shine { to { background-position:200% center; } }
.hx-h2 { font-family:'Space Grotesk',Inter,sans-serif; font-size:clamp(16px,2.6cqi,24px); color:#a5f3fc; font-weight:500; margin:0 0 12px; }
.hx-sub { color:#cbd5e1; font-size:13.5px; line-height:1.5; margin:0 0 14px; max-width:640px; }
.hx-chips, .hx-stats { display:flex; flex-wrap:wrap; gap:8px; }
.hx-chip { font-size:12.5px; font-weight:600; padding:5px 12px; border-radius:999px; background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.16); backdrop-filter:blur(6px); transition:transform .2s, background .2s;
  animation:hx-pop .5s cubic-bezier(.2,.8,.2,1) both; }
.hx-chip:nth-child(2) { animation-delay:.08s; } .hx-chip:nth-child(3) { animation-delay:.16s; }
.hx-chip:nth-child(4) { animation-delay:.24s; } .hx-chip:nth-child(5) { animation-delay:.32s; }
.hx-chip:hover { transform:translateY(-3px); background:rgba(255,255,255,.18); }
@keyframes hx-pop { from { opacity:0; transform:translateY(8px) scale(.96); } to { opacity:1; transform:none; } }
.hx-lvl { color:#fff; border:none; }
.hx-lvl.ACTION { background:#ef4444; animation:hx-pop .5s both, pulseR 1.8s .6s infinite; }
.hx-lvl.WATCH { background:#f59e0b; animation:hx-pop .5s both, pulseA 2.2s .6s infinite; }
.hx-lvl.NORMAL { background:#10b981; }
.hx-stat { padding:10px 14px; border-radius:16px; background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14);
  backdrop-filter:blur(6px); transition:transform .25s, background .25s; animation:hx-pop .6s cubic-bezier(.2,.8,.2,1) both; }
.hx-stat:nth-child(2) { animation-delay:.1s; } .hx-stat:nth-child(3) { animation-delay:.2s; } .hx-stat:nth-child(4) { animation-delay:.3s; }
.hx-stat:hover { transform:translateY(-4px); background:rgba(255,255,255,.14); }
.hx-stat b { display:block; font-family:'Space Grotesk',sans-serif; font-size:clamp(20px,3cqi,28px);
  background:linear-gradient(90deg,#a5f3fc,#c4b5fd,#f9a8d4); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hx-stat span { font-size:12px; color:#cbd5e1; }

/* globe + planes: drawn at 320x220 and scaled as a whole */
.hx-scene { position:relative; flex:0 0 auto; width:320px; height:220px; margin:-12px -14px -12px 0; }
.hx-stage { position:absolute; left:0; top:0; width:320px; height:220px; transform-origin:0 0; }
.hx-tilt { position:absolute; inset:0; transform:rotate(-16deg); }
.hx-tilt.b { transform:rotate(22deg); }
.hx-globe { position:absolute; left:85px; top:35px; width:150px; height:150px; border-radius:50%; z-index:2; overflow:hidden;
  background:radial-gradient(circle at 50% 50%,#1e40af,#0b1e5b 75%); box-shadow:0 0 50px rgba(96,165,250,.55),0 0 110px rgba(99,102,241,.35); }
.hx-land { position:absolute; inset:0; background-repeat:repeat-x; background-size:150px 150px; animation:hx-spin 14s linear infinite; opacity:.9;
  background-image:radial-gradient(ellipse 22px 30px at 22px 52px,#34d399 55%,transparent 58%),
   radial-gradient(ellipse 14px 22px at 34px 98px,#10b981 55%,transparent 58%),
   radial-gradient(ellipse 30px 20px at 92px 44px,#2dd4bf 55%,transparent 58%),
   radial-gradient(ellipse 16px 24px at 98px 92px,#34d399 55%,transparent 58%),
   radial-gradient(ellipse 18px 10px at 126px 118px,#5eead4 55%,transparent 58%),
   radial-gradient(ellipse 10px 8px at 66px 20px,#a7f3d0 55%,transparent 58%); }
.hx-grid { position:absolute; inset:0; animation:hx-spin 14s linear infinite;
  background-image:repeating-linear-gradient(90deg,rgba(255,255,255,.16) 0 1px,transparent 1px 25px),
                   repeating-linear-gradient(0deg,rgba(255,255,255,.10) 0 1px,transparent 1px 25px); }
@keyframes hx-spin { from { background-position:0 0; } to { background-position:150px 0; } }
.hx-shade { position:absolute; inset:0; border-radius:50%;
  background:radial-gradient(circle at 32% 28%,rgba(255,255,255,.35),transparent 35%),
             radial-gradient(circle at 70% 75%,rgba(2,6,23,.75),transparent 60%),
             radial-gradient(circle at 50% 50%,transparent 58%,rgba(2,6,23,.65) 72%); }
.hx-pin { position:absolute; left:58%; top:38%; width:8px; height:8px; border-radius:50%; background:#f472b6; z-index:3; box-shadow:0 0 10px #f472b6; }
.hx-pin::after { content:""; position:absolute; inset:-4px; border-radius:50%; border:2px solid #f472b6; animation:hx-ping 1.8s infinite; }
@keyframes hx-ping { from { transform:scale(.6); opacity:1; } to { transform:scale(2.6); opacity:0; } }
.hx-orbit { position:absolute; inset:0; overflow:visible; }
.hx-orbit ellipse { fill:none; stroke:rgba(165,180,252,.45); stroke-width:1.2; stroke-dasharray:3 7; animation:hx-dash 2.5s linear infinite; }
.hx-tilt.b .hx-orbit ellipse { stroke:rgba(244,114,182,.35); }
@keyframes hx-dash { to { stroke-dashoffset:-20; } }
.hx-plane { position:absolute; left:0; top:0; width:36px; height:36px; offset-rotate:auto;
  filter:drop-shadow(0 0 6px rgba(165,243,252,.9)) drop-shadow(-14px 0 6px rgba(255,255,255,.25));
  offset-path:path("M 20 110 A 140 46 0 1 1 300 110 A 140 46 0 1 1 20 110"); animation:hx-fly 9s linear infinite; }
.hx-tilt.b .hx-plane { width:22px; height:22px; offset-path:path("M 50 110 A 110 34 0 1 1 270 110 A 110 34 0 1 1 50 110");
  animation:hx-fly 12s linear infinite reverse; filter:drop-shadow(0 0 5px rgba(249,168,212,.9)); }
.hx-plane svg { display:block; width:100%; height:100%; }
@keyframes hx-fly { 0% { offset-distance:0%; z-index:1; transform:scale(.85); opacity:.85; }
  25% { transform:scale(.6); opacity:.55; } 49.9% { z-index:1; } 50% { z-index:3; transform:scale(.85); opacity:.9; }
  75% { transform:scale(1.15); opacity:1; } 100% { offset-distance:100%; z-index:3; transform:scale(.85); opacity:.85; } }

/* landing (big) version */
.hx.big { border-radius:28px; }
.hx.big .hx-row { padding:44px 48px; }
.hx.big .hx-title { font-size:clamp(30px,6cqi,56px); margin:10px 0 4px; }
.hx.big .hx-sub { font-size:15px; margin-bottom:22px; }
.hx.big .hx-scene { width:480px; height:330px; margin:0 -20px 0 0; }
.hx.big .hx-stage { transform:scale(1.5); }

/* medium width: the globe becomes a faded backdrop behind the text, so nothing is squeezed or cut */
@container (max-width: 860px) {
  .hx-scene, .hx.big .hx-scene { position:absolute; right:-60px; top:50%; width:320px; height:220px; margin:0;
    transform:translateY(-50%); opacity:.45; pointer-events:none; }
  .hx.big .hx-stage { transform:scale(1); }
  .hx-row, .hx.big .hx-row { padding:24px 26px; }
}
/* phone width: no globe, tighter spacing */
@container (max-width: 520px) {
  .hx-scene, .hx.big .hx-scene { display:none; }
  .hx-row, .hx.big .hx-row { padding:18px 18px; }
  .hx-sub, .hx.big .hx-sub { font-size:12.5px; }
  .hx-chip { font-size:11.5px; padding:4px 10px; }
}
</style>
""".replace("__FONTS__", FONTS)


def inject_css():
    st.markdown(APP_CSS, unsafe_allow_html=True)


def badge(level):
    return f"<span class='badge lvl-{level}' style='background:{LEVEL_COLOR[level]}'>{level}</span>"


_PLANE = ("<svg viewBox='0 0 24 24' width='100%' height='100%'><path fill='#fff' d='M21 16v-2l-8-5V3.5a1.5 1.5 0 0 0-3 0V9l-8 5v2"
          "l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5z' transform='rotate(90 12 12)'/></svg>")

_SCENE = ("<div class='hx-scene'><div class='hx-stage'>"
          "<div class='hx-tilt b'><svg class='hx-orbit' viewBox='0 0 320 220'><ellipse cx='160' cy='110' rx='110' ry='34'/></svg>"
          "<div class='hx-plane'>__PLANE__</div></div>"
          "<div class='hx-tilt'><svg class='hx-orbit' viewBox='0 0 320 220'><ellipse cx='160' cy='110' rx='140' ry='46'/></svg>"
          "<div class='hx-globe'><div class='hx-land'></div><div class='hx-grid'></div><div class='hx-shade'></div>"
          "<div class='hx-pin'></div></div><div class='hx-plane'>__PLANE__</div></div>"
          "</div></div>").replace("__PLANE__", _PLANE)


def _stars(n=42):
    # deterministic scatter so the hero doesn't jump between reruns
    out = []
    for i in range(n):
        x, y = (i * 37) % 100, (i * 53 + 11) % 100
        out.append(f"<span class='hx-star' style='left:{x}%;top:{y}%;animation-delay:{(i % 7) * .45:.2f}s'></span>")
    return "".join(out)


def _render(text_html, big=False, stars=42):
    # one line, no indentation: markdown would treat indented lines as a code block
    page = (f"<div class='hx{' big' if big else ''}'>{_stars(stars)}<div class='hx-shoot'></div>"
            f"<div class='hx-row'><div class='hx-text'>{text_html}</div>{_SCENE}</div></div>")
    st.markdown(page, unsafe_allow_html=True)


def hero(ap, when, horizon, level, meaning, peak, normal):
    if peak is None or normal != normal:  # no forecast / NaN
        peak_txt = "📊 no congestion forecast in this window"
    else:
        peak_txt = f"📊 up to {peak*100:.0f} of 100 flights disrupted (normal ≤ {normal*100:.0f})"
    e = html.escape
    _render("<div class='hx-eyebrow'>Autonomous SG 2026 · Airbus Challenge #5</div>"
            "<div class='hx-title'>Airport Capacity &amp; Delay-Risk</div>"
            "<div class='hx-chips'>"
            f"<span class='hx-chip hx-lvl {level}'>{LEVEL_ICON[level]} {level} · {e(meaning)}</span>"
            f"<span class='hx-chip'>📍 {e(ap)}</span><span class='hx-chip'>🕕 {e(when)}</span>"
            f"<span class='hx-chip'>⏱ next {horizon} h</span><span class='hx-chip'>{e(peak_txt)}</span></div>")


def landing_hero(stats):
    """Big landing version. stats: list of (number, decimals, suffix, label)."""
    cards = "".join(f"<div class='hx-stat'><b>{v:,.{d}f}{html.escape(suf)}</b><span>{html.escape(label)}</span></div>"
                    for v, d, suf, label in stats)
    _render("<div class='hx-eyebrow'>Autonomous SG 2026 · Airbus Challenge #5 · Atlanta &amp; Dallas/Fort Worth</div>"
            "<div class='hx-title'>See the squeeze coming.</div>"
            "<div class='hx-h2'>Airport capacity &amp; delay risk, 3–12 hours ahead.</div>"
            "<div class='hx-sub'>For airline ops duty managers and ATC flow coordinators. It forecasts how many flights will be "
            "disrupted and which ones will run more than 45 minutes late, from weather and flight data, and shows the evidence "
            "behind every call. The human decides.</div>"
            f"<div class='hx-stats'>{cards}</div>", big=True, stars=70)


def feature_cards(cards):
    """cards: list of (icon, title, text, stat_line). Rendered as hover cards in the main page."""
    items = "".join(f"<div class='feat'><div class='ico'>{i}</div><h4>{html.escape(t)}</h4><p>{html.escape(x)}</p>"
                    f"<div class='kpi'>{html.escape(k)}</div></div>" for i, t, x, k in cards)
    st.markdown(f"<div class='feats'>{items}</div>", unsafe_allow_html=True)


def how_it_works(steps):
    """steps: list of (icon, title, text), drawn as a flow with animated connectors."""
    parts = []
    for n, (i, t, x) in enumerate(steps):
        if n:
            parts.append("<div class='arrow'><span></span></div>")
        parts.append(f"<div class='step'><div class='num'>{n + 1}</div><div class='ico'>{i}</div>"
                     f"<h5>{html.escape(t)}</h5><p>{html.escape(x)}</p></div>")
    st.markdown(f"<div class='flow'>{''.join(parts)}</div>", unsafe_allow_html=True)


LANDING_ONLY_CSS = """<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display:none !important; }
.block-container { max-width:1240px; padding-top:3rem; }
div[data-testid="stButton"] button { font-size:1.05rem; padding:.8rem 1.4rem; border-radius:16px; width:100%; }
div[data-testid="stButton"] button p { font-size:1.05rem; font-weight:700; white-space:normal; }
[data-testid="stBaseButton-secondary"] { background:transparent !important; border:2px solid #06b6d4 !important; box-shadow:none !important; }
[data-testid="stBaseButton-secondary"] p { color:#06b6d4 !important; }
[data-testid="stBaseButton-secondary"]:hover { background:rgba(6,182,212,.12) !important; }
</style>"""


def landing_mode_css():
    st.markdown(LANDING_ONLY_CSS, unsafe_allow_html=True)


def info_cards(cards):
    """cards: list of (icon, title, html_text). Same look as the landing feature cards, without the KPI pill."""
    items = "".join(f"<div class='feat'><div class='ico'>{i}</div><h4>{html.escape(ti)}</h4><p>{x}</p></div>"
                    for i, ti, x in cards)
    st.markdown(f"<div class='feats'>{items}</div>", unsafe_allow_html=True)


_FLAG = ("<svg class='rk-flag' viewBox='0 0 24 24' role='img' aria-label='flagged'><path d='M12 2 1 21h22z' fill='#ef4444'/>"
         "<path d='M12 9v5.2' stroke='#fff' stroke-width='2.4' stroke-linecap='round'/><circle cx='12' cy='17.6' r='1.4' fill='#fff'/></svg>")


def risk_table(rows, hindsight=False):
    """rows: dicts with flagged, when, side, flight, other, p (0-1), why, made and, with hindsight, late ('yes'/'no')."""
    e = html.escape
    head = ["", "scheduled (local)", "dep/arr", "flight", "to / from", "chance of &gt;45 min delay", "why (top 3 factors)", "forecast made"]
    if hindsight:
        head.append("actually &gt;45 min late?")
    body = []
    for r in rows:
        late = ""
        if hindsight:
            late = f"<td class='rk-{r['late'] or 'no'}'>{e(r['late'] or '–')}</td>"
        body.append(
            f"<tr class='{'on' if r['flagged'] else ''}'><td>{_FLAG if r['flagged'] else ''}</td>"
            f"<td class='rk-nowrap'>{e(r['when'])}</td><td>{e(r['side'])}</td><td class='rk-nowrap'><b>{e(r['flight'])}</b></td>"
            f"<td>{e(r['other'])}</td><td><div class='rk-bar'><i style='width:{max(0.0, min(1.0, r['p'])) * 100:.0f}%'></i>"
            f"<span>{r['p']:.0%}</span></div></td><td class='rk-why'>{e(r['why'])}</td>"
            f"<td class='rk-nowrap'>{e(r['made'])}</td>{late}</tr>")
    st.markdown(f"<div class='rk-wrap'><table class='rk'><thead><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table></div>", unsafe_allow_html=True)
