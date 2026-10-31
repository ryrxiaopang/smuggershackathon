"""Look and feel for the dashboard: one CSS layer for the whole app and an animated landing hero
(a spinning globe with planes on orbit, plus live status chips for the selected airport/moment).
Pure presentation: no numbers are computed here."""
import html

import streamlit as st
import streamlit.components.v1 as components

FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&family=Space+Grotesk:wght@500;700&display=swap"
LEVEL_COLOR = {"NORMAL": "#10b981", "WATCH": "#f59e0b", "ACTION": "#ef4444"}
LEVEL_ICON = {"NORMAL": "🟢", "WATCH": "🟠", "ACTION": "🔴"}

APP_CSS = """
<style>
@import url('__FONTS__');
:root { --c1:#6366f1; --c2:#06b6d4; --c3:#ec4899; --glass:rgba(99,102,241,.08); --line:rgba(99,102,241,.28); }
html, body, [class*="st-"], .stMarkdown, .stCaption { font-family:'Inter', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family:'Space Grotesk', 'Inter', sans-serif !important; letter-spacing:-.01em; }
.block-container { padding-top:3.4rem; }

/* tabs as glowing pills */
.stTabs [data-baseweb="tab-list"] { gap:6px; background:var(--glass); padding:6px; border-radius:16px; border:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius:11px; padding:8px 18px; height:auto; font-weight:600; transition:all .25s ease; }
.stTabs [data-baseweb="tab"]:hover { background:rgba(99,102,241,.16); transform:translateY(-1px); }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,var(--c1),var(--c2)) !important; box-shadow:0 8px 22px rgba(99,102,241,.38); }
.stTabs [aria-selected="true"] p { color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs [data-baseweb="tab-panel"] { animation:fadeUp .45s ease both; }
@keyframes fadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }

/* metric cards: gradient glass, lift on hover, gradient numbers */
[data-testid="stMetric"] { background:linear-gradient(135deg,rgba(99,102,241,.12),rgba(6,182,212,.08) 60%,rgba(236,72,153,.06));
  border:1px solid var(--line); border-radius:18px; padding:14px 18px; transition:transform .25s ease, box-shadow .25s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-4px); box-shadow:0 14px 30px rgba(99,102,241,.28); }
[data-testid="stMetricValue"] > div { white-space:normal !important; overflow:visible !important; text-overflow:clip !important; }
[data-testid="stMetricValue"] { font-size:1.6rem !important; line-height:1.2; font-family:'Space Grotesk', sans-serif; font-weight:700;
  background:linear-gradient(90deg,var(--c1),var(--c2) 55%,var(--c3)); -webkit-background-clip:text; background-clip:text; color:transparent; }

/* expanders, bordered containers, tables */
[data-testid="stExpander"] details { border-radius:14px; border:1px solid var(--line); transition:box-shadow .25s ease, border-color .25s ease; }
[data-testid="stExpander"] details:hover { border-color:rgba(6,182,212,.6); box-shadow:0 8px 22px rgba(6,182,212,.18); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }

/* sidebar: soft aurora */
[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(99,102,241,.16),rgba(6,182,212,.07) 55%,rgba(236,72,153,.10)); }
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
</style>
""".replace("__FONTS__", FONTS)


def inject_css():
    st.markdown(APP_CSS, unsafe_allow_html=True)


def badge(level):
    return f"<span class='badge lvl-{level}' style='background:{LEVEL_COLOR[level]}'>{level}</span>"


_PLANE = ("<svg viewBox='0 0 24 24' width='100%' height='100%'><path fill='#fff' d='M21 16v-2l-8-5V3.5a1.5 1.5 0 0 0-3 0V9l-8 5v2"
          "l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5z' transform='rotate(90 12 12)'/></svg>")

HERO = """<!doctype html><html><head><meta charset="utf-8"><link href="__FONTS__" rel="stylesheet"><style>
*{box-sizing:border-box;margin:0}
body{font-family:Inter,system-ui,sans-serif;background:transparent;overflow:hidden}
.hero{position:relative;overflow:hidden;height:256px;border-radius:22px;padding:24px 30px;color:#e2e8f0;
  display:flex;align-items:center;justify-content:space-between;gap:16px;
  background:radial-gradient(900px 260px at 78% -30%,rgba(99,102,241,.6),transparent 60%),
             radial-gradient(600px 300px at 5% 130%,rgba(6,182,212,.45),transparent 60%),
             radial-gradient(400px 200px at 50% 120%,rgba(236,72,153,.25),transparent 60%),
             linear-gradient(135deg,#070b1f 0%,#121741 45%,#1e1b4b 100%);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.08)}
.star{position:absolute;width:2px;height:2px;border-radius:50%;background:#fff;animation:tw 3.2s infinite ease-in-out}
@keyframes tw{0%,100%{opacity:.12}50%{opacity:.95}}
.shoot{position:absolute;top:18%;left:-10%;width:120px;height:1px;background:linear-gradient(90deg,transparent,#fff);
  transform:rotate(-12deg);animation:shoot 7s 2s infinite ease-in;opacity:0}
@keyframes shoot{0%{left:-10%;opacity:0}3%{opacity:1}12%{left:60%;opacity:0}100%{left:60%;opacity:0}}
.text{position:relative;z-index:5;max-width:62%}
.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#a5b4fc;font-weight:600}
h1{font-family:'Space Grotesk',Inter,sans-serif;font-size:30px;line-height:1.1;margin:6px 0 8px;
  background:linear-gradient(90deg,#fff 0%,#a5f3fc 30%,#c4b5fd 60%,#f9a8d4 80%,#fff 100%);background-size:200% auto;
  -webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 7s linear infinite}
@keyframes shine{to{background-position:200% center}}
.sub{color:#cbd5e1;font-size:13.5px;line-height:1.45;margin-bottom:14px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{font-size:12px;font-weight:600;padding:5px 11px;border-radius:999px;background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.16);backdrop-filter:blur(6px);transition:transform .2s,background .2s;cursor:default;
  animation:pop .5s cubic-bezier(.2,.8,.2,1) both}
.chip:nth-child(2){animation-delay:.08s}.chip:nth-child(3){animation-delay:.16s}.chip:nth-child(4){animation-delay:.24s}.chip:nth-child(5){animation-delay:.32s}
.chip:hover{transform:translateY(-3px);background:rgba(255,255,255,.18)}
@keyframes pop{from{opacity:0;transform:translateY(8px) scale(.96)}to{opacity:1;transform:none}}
.lvl{color:#fff;border:none}
.lvl.ACTION{background:#ef4444;animation:pop .5s both,pr 1.8s .6s infinite}
.lvl.WATCH{background:#f59e0b;animation:pop .5s both,pa 2.2s .6s infinite}
.lvl.NORMAL{background:#10b981}
@keyframes pr{0%{box-shadow:0 0 0 0 rgba(239,68,68,.7)}70%{box-shadow:0 0 0 12px rgba(239,68,68,0)}100%{box-shadow:0 0 0 0 rgba(239,68,68,0)}}
@keyframes pa{0%{box-shadow:0 0 0 0 rgba(245,158,11,.7)}70%{box-shadow:0 0 0 12px rgba(245,158,11,0)}100%{box-shadow:0 0 0 0 rgba(245,158,11,0)}}

.scene{position:relative;width:320px;height:220px;flex:none}
.tilt{position:absolute;inset:0;transform:rotate(-16deg)}
.tilt.b{transform:rotate(22deg)}
.globe{position:absolute;left:85px;top:35px;width:150px;height:150px;border-radius:50%;z-index:2;overflow:hidden;
  background:radial-gradient(circle at 50% 50%,#1e40af,#0b1e5b 75%);
  box-shadow:0 0 50px rgba(96,165,250,.55),0 0 110px rgba(99,102,241,.35)}
.land{position:absolute;inset:0;background-repeat:repeat-x;background-size:150px 150px;animation:spin 14s linear infinite;opacity:.9;
  background-image:
   radial-gradient(ellipse 22px 30px at 22px 52px,#34d399 55%,transparent 58%),
   radial-gradient(ellipse 14px 22px at 34px 98px,#10b981 55%,transparent 58%),
   radial-gradient(ellipse 30px 20px at 92px 44px,#2dd4bf 55%,transparent 58%),
   radial-gradient(ellipse 16px 24px at 98px 92px,#34d399 55%,transparent 58%),
   radial-gradient(ellipse 18px 10px at 126px 118px,#5eead4 55%,transparent 58%),
   radial-gradient(ellipse 10px 8px at 66px 20px,#a7f3d0 55%,transparent 58%)}
.grid{position:absolute;inset:0;animation:spin 14s linear infinite;
  background-image:repeating-linear-gradient(90deg,rgba(255,255,255,.16) 0 1px,transparent 1px 25px),
                   repeating-linear-gradient(0deg,rgba(255,255,255,.10) 0 1px,transparent 1px 25px)}
@keyframes spin{from{background-position:0 0}to{background-position:150px 0}}
.shade{position:absolute;inset:0;border-radius:50%;
  background:radial-gradient(circle at 32% 28%,rgba(255,255,255,.35),transparent 35%),
             radial-gradient(circle at 70% 75%,rgba(2,6,23,.75),transparent 60%),
             radial-gradient(circle at 50% 50%,transparent 58%,rgba(2,6,23,.65) 72%)}
.pin{position:absolute;left:58%;top:38%;width:8px;height:8px;border-radius:50%;background:#f472b6;z-index:3;box-shadow:0 0 10px #f472b6}
.pin::after{content:"";position:absolute;inset:-4px;border-radius:50%;border:2px solid #f472b6;animation:ping 1.8s infinite}
@keyframes ping{from{transform:scale(.6);opacity:1}to{transform:scale(2.6);opacity:0}}
svg.orbit{position:absolute;inset:0;overflow:visible}
svg.orbit ellipse{fill:none;stroke:rgba(165,180,252,.45);stroke-width:1.2;stroke-dasharray:3 7;animation:dash 2.5s linear infinite}
.tilt.b svg.orbit ellipse{stroke:rgba(244,114,182,.35)}
@keyframes dash{to{stroke-dashoffset:-20}}
.plane{position:absolute;left:0;top:0;width:36px;height:36px;offset-rotate:auto;
  filter:drop-shadow(0 0 6px rgba(165,243,252,.9)) drop-shadow(-14px 0 6px rgba(255,255,255,.25));
  offset-path:path("M 20 110 A 140 46 0 1 1 300 110 A 140 46 0 1 1 20 110");animation:fly 9s linear infinite}
.tilt.b .plane{width:22px;height:22px;offset-path:path("M 50 110 A 110 34 0 1 1 270 110 A 110 34 0 1 1 50 110");
  animation:fly 12s linear infinite reverse;filter:drop-shadow(0 0 5px rgba(249,168,212,.9))}
@keyframes fly{0%{offset-distance:0%;z-index:1;transform:scale(.85);opacity:.85}
  25%{transform:scale(.6);opacity:.55}49.9%{z-index:1}50%{z-index:3;transform:scale(.85);opacity:.9}
  75%{transform:scale(1.15);opacity:1}100%{offset-distance:100%;z-index:3;transform:scale(.85);opacity:.85}}
@media (max-width:980px){.scene{transform:scale(.8);margin-right:-40px}}
@media (max-width:760px){.scene{display:none}.text{max-width:100%}h1{font-size:24px}.sub{display:none}}
</style></head><body>
<div class="hero">
  __STARS__<div class="shoot"></div>
  <div class="text">
    <div class="eyebrow">Autonomous SG 2026 · Airbus Challenge #5</div>
    <h1>Airport Capacity &amp; Delay-Risk</h1>
    <div class="sub">Early warning for ops duty managers: congestion and &gt;45-minute delay risk 3–12 hours ahead,
      with the weather evidence behind every call.</div>
    <div class="chips">
      <span class="chip lvl __LEVEL__">__ICON__ __LEVEL__ · __MEANING__</span>
      <span class="chip">📍 __AP__</span>
      <span class="chip">🕕 __WHEN__</span>
      <span class="chip">⏱ next __H__ h</span>
      <span class="chip">__PEAK__</span>
    </div>
  </div>
  <div class="scene">
    <div class="tilt b"><svg class="orbit" viewBox="0 0 320 220"><ellipse cx="160" cy="110" rx="110" ry="34"/></svg>
      <div class="plane">__PLANE__</div></div>
    <div class="tilt"><svg class="orbit" viewBox="0 0 320 220"><ellipse cx="160" cy="110" rx="140" ry="46"/></svg>
      <div class="globe"><div class="land"></div><div class="grid"></div><div class="shade"></div><div class="pin"></div></div>
      <div class="plane">__PLANE__</div></div>
  </div>
</div></body></html>"""


def _stars(n=42):
    # deterministic scatter so the hero doesn't jump between reruns
    out = []
    for i in range(n):
        x, y = (i * 37) % 100, (i * 53 + 11) % 100
        out.append(f"<span class='star' style='left:{x}%;top:{y}%;animation-delay:{(i % 7) * .45:.2f}s;"
                   f"opacity:.{2 + i % 7}'></span>")
    return "".join(out)


def hero(ap, when, horizon, level, meaning, peak, normal):
    if peak is None or normal != normal:  # no forecast / NaN
        peak_txt = "📊 no congestion forecast in this window"
    else:
        peak_txt = f"📊 up to {peak*100:.0f} of 100 flights disrupted (normal ≤ {normal*100:.0f})"
    page = (HERO.replace("__FONTS__", FONTS).replace("__STARS__", _stars()).replace("__PLANE__", _PLANE)
            .replace("__LEVEL__", level).replace("__ICON__", LEVEL_ICON[level]).replace("__MEANING__", html.escape(meaning))
            .replace("__AP__", html.escape(ap)).replace("__WHEN__", html.escape(when)).replace("__H__", str(horizon))
            .replace("__PEAK__", html.escape(peak_txt)))
    components.html(page, height=262)
