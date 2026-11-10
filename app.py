"""Streamlit interface:  streamlit run app.py"""
from __future__ import annotations

import base64
import html
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from pydantic import ValidationError

from copywriter.config import ConfigError, Settings
from copywriter.evaluation import evaluate
from copywriter.exports import outcome_row, to_csv_text, to_json_text, to_markdown
from copywriter.generator import CopyGenerator, GenerationOutcome
from copywriter.llm import create_backend
from copywriter.prompts import PROMPT_VERSION
from copywriter.schemas import CORE_PLATFORMS, CORE_TONES, CopyRequest, Platform, Tone
from copywriter.specs import PLATFORM_SPECS
from copywriter.usage_log import UsageLogger

st.set_page_config(page_title="Copywriting & Tone Transformer", page_icon="✍️", layout="wide")

# --------------------------------------------------------------------------- #
# Look & feel
# --------------------------------------------------------------------------- #

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--ink:#0F1E2E;--canvas:#EEF1F5;--surface:#FFFFFF;--brand:#2D3A8C;--marker:#FFD23F;
      --ok:#1F8A5B;--warn:#8F5600;--bad:#B3261E;--line:#D5DBE3;--muted:#5B6878;}
html,body,[class*="css"]{font-family:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--ink);}
h1,h2,h3{font-family:'Bricolage Grotesque','IBM Plex Sans',system-ui,sans-serif;letter-spacing:-0.01em;}
.block-container{max-width:1180px;padding-top:2.2rem;}
.ctt-lede{color:var(--muted);margin:-0.4rem 0 1.2rem;max-width:62ch;}
.ctt-frame{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;
           line-height:1.55;overflow-wrap:anywhere;}
.ctt-head{display:flex;gap:10px;align-items:center;margin-bottom:10px;}
.ctt-avatar{width:40px;height:40px;border-radius:50%;background:var(--brand);color:#fff;display:flex;
            align-items:center;justify-content:center;font-weight:600;flex:none;}
.ctt-name{font-weight:600;line-height:1.2;} .ctt-sub{color:var(--muted);font-size:.82rem;}
.ctt-hook{font-weight:600;margin-bottom:.6rem;} .ctt-cta{margin-top:.7rem;font-weight:500;}
.ctt-tags{margin-top:.7rem;color:var(--brand);} .ctt-foot{margin-top:.9rem;padding-top:.6rem;
            border-top:1px solid var(--line);color:var(--muted);font-size:.85rem;}
.ctt-ig-img{aspect-ratio:1/1;background:var(--brand);color:#fff;display:flex;align-items:flex-end;
            padding:18px;border-radius:6px;margin-bottom:12px;}
.ctt-ig-img span{background:var(--marker);color:var(--ink);padding:4px 10px;font-family:'Bricolage Grotesque',sans-serif;
            font-weight:700;font-size:1.25rem;}
.ctt-mail-bar{display:flex;gap:6px;margin-bottom:10px;} .ctt-mail-bar i{width:10px;height:10px;border-radius:50%;
            background:var(--line);display:block;}
.ctt-mail-row{color:var(--muted);font-size:.85rem;} .ctt-mail-subject{font-weight:600;font-size:1.05rem;
            margin:6px 0 12px;padding-bottom:10px;border-bottom:1px solid var(--line);}
.ctt-meter{margin:.9rem 0 .2rem;} .ctt-meter-track{position:relative;height:10px;background:#E3E8EE;border-radius:5px;}
.ctt-meter-band{position:absolute;top:0;bottom:0;background:var(--marker);opacity:.55;border-radius:5px;}
.ctt-meter-fill{position:absolute;top:0;bottom:0;left:0;background:var(--brand);border-radius:5px;}
.ctt-meter-cap{display:flex;justify-content:space-between;color:var(--muted);font-size:.8rem;margin-top:4px;}
.ctt-empty{border:1px dashed var(--line);border-radius:10px;padding:28px;color:var(--muted);background:var(--surface);}
.ctt-ok{color:var(--ok);} .ctt-warn{color:var(--warn);} .ctt-bad{color:var(--bad);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EXAMPLE = {
    "product_name": "NovaDesk",
    "product_description": (
        "An electric standing desk with four memory presets, a quiet dual motor and a bamboo top. "
        "It lifts up to 120 kg and ships flat for easy assembly."
    ),
    "platform": "linkedin",
    "tone": "professional",
}


def load_example() -> None:
    for key, value in EXAMPLE.items():
        st.session_state[key] = value


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

env_settings = Settings.from_env()


@st.cache_resource(show_spinner=False)
def get_engine(backend: str, model: str) -> tuple[CopyGenerator, Settings]:
    settings = Settings.from_env(backend=backend, model=model or None)
    generator = CopyGenerator(create_backend(settings), settings, usage_logger=UsageLogger(settings.log_path))
    return generator, settings


with st.sidebar:
    st.subheader("Engine")
    backend_name = st.selectbox(
        "Backend", ["openai", "mock"], index=["openai", "mock"].index(env_settings.backend), key="engine_backend",
        help="'mock' returns placeholder text for demos; it does not call a model.",
    )
    model_name = st.text_input("Model", value=env_settings.model, key="engine_model", disabled=backend_name == "mock")

try:
    generator, settings = get_engine(backend_name, model_name.strip())
except ConfigError as exc:
    st.error(str(exc))
    st.stop()

caps = generator.backend.capabilities
missing_key = backend_name == "openai" and not (settings.api_key or settings.base_url)

with st.sidebar:
    if backend_name == "mock":
        st.warning("Demo engine: placeholder text built from templates, not written by a language model.")
    elif missing_key:
        st.error("OPENAI_API_KEY is not set. Copy .env.example to .env, add your key, then restart the app.")
    else:
        st.success(f"Using {generator.backend.model}")
    sampling_text = {
        "always": "Temperature and Top-P are sent.",
        "needs_reasoning_none": "Sent together with reasoning_effort = none.",
        "never": "This model ignores Temperature and Top-P.",
        "unknown": "Unknown model: sent if the API accepts them.",
    }[caps.sampling]
    st.caption(f"Sampling: {sampling_text}")
    st.caption(f"Prompt version {PROMPT_VERSION}")
    if generator.usage_logger:
        usage = generator.usage_logger.summary()
        st.caption(f"Logged calls: {int(usage['calls'])} | tokens: {int(usage['prompt_tokens'] + usage['completion_tokens'])}")

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

st.session_state.setdefault("outcomes", [])
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_request", None)

# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def esc(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def initials(name: str) -> str:
    letters = [w[0] for w in name.split() if w]
    return (("".join(letters[:2])) or name[:1] or "?").upper()


def preview_html(o: GenerationOutcome) -> str:
    c, r = o.copy, o.request
    tags = esc(" ".join(c.hashtags))
    hook = f'<div class="ctt-hook">{esc(c.headline)}</div>' if c.headline else ""
    cta = f'<div class="ctt-cta">{esc(c.call_to_action)}</div>' if c.call_to_action else ""
    tag_html = f'<div class="ctt-tags">{tags}</div>' if tags else ""
    avatar = f'<div class="ctt-avatar">{esc(initials(r.product_name))}</div>'
    if r.platform is Platform.EMAIL:
        return (
            '<div class="ctt-frame"><div class="ctt-mail-bar"><i></i><i></i><i></i></div>'
            f'<div class="ctt-mail-row">From: Your Brand &lt;hello@yourbrand.com&gt;</div>'
            f'<div class="ctt-mail-row">To: Customer</div>'
            f'<div class="ctt-mail-subject">{esc(c.headline) or "(no subject)"}</div>'
            f'<div>{esc(c.body)}</div>{cta}</div>'
        )
    if r.platform is Platform.INSTAGRAM:
        return (
            f'<div class="ctt-frame"><div class="ctt-head">{avatar}<div><div class="ctt-name">yourbrand</div>'
            f'<div class="ctt-sub">Sponsored</div></div></div>'
            f'<div class="ctt-ig-img"><span>{esc(r.product_name)}</span></div>'
            f'{hook}<div>{esc(c.body)}</div>{cta}{tag_html}</div>'
        )
    if r.platform is Platform.TWITTER:
        return (
            f'<div class="ctt-frame"><div class="ctt-head">{avatar}<div><div class="ctt-name">Your Brand</div>'
            f'<div class="ctt-sub">@yourbrand</div></div></div><div>{esc(o.final_text)}</div></div>'
        )
    return (  # LinkedIn
        f'<div class="ctt-frame"><div class="ctt-head">{avatar}<div><div class="ctt-name">Your Brand</div>'
        f'<div class="ctt-sub">Promoted</div></div></div>{hook}<div>{esc(c.body)}</div>{cta}{tag_html}'
        '<div class="ctt-foot">Like &nbsp;·&nbsp; Comment &nbsp;·&nbsp; Repost &nbsp;·&nbsp; Send</div></div>'
    )


def meter_html(o: GenerationOutcome) -> str:
    spec = PLATFORM_SPECS[o.request.platform]
    chars = o.report.char_count
    pct = lambda n: max(0.0, min(100.0, 100.0 * n / spec.hard_max_chars))  # noqa: E731
    band_l, band_w = pct(spec.target_min_chars), pct(spec.target_max_chars) - pct(spec.target_min_chars)
    return (
        '<div class="ctt-meter"><div class="ctt-meter-track">'
        f'<div class="ctt-meter-band" style="left:{band_l:.1f}%;width:{band_w:.1f}%"></div>'
        f'<div class="ctt-meter-fill" style="width:{pct(chars):.1f}%"></div></div>'
        f'<div class="ctt-meter-cap"><span>{chars} characters</span>'
        f'<span>target {spec.target_min_chars}-{spec.target_max_chars} · limit {spec.hard_max_chars}</span></div></div>'
    )


def copy_button(text: str) -> None:
    """Copy-to-clipboard button. The text travels base64-encoded, so nothing the model
    wrote can ever be interpreted as HTML or script inside the iframe."""
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    st.iframe(
        f"""
<button id="b" style="font:600 14px system-ui;padding:8px 14px;border:1px solid #2D3A8C;border-radius:8px;
background:#fff;color:#2D3A8C;cursor:pointer">Copy to clipboard</button>
<script>
const bytes = Uint8Array.from(atob("{payload}"), c => c.charCodeAt(0));
const t = new TextDecoder("utf-8").decode(bytes); const b = document.getElementById('b');
b.onclick = async () => {{
  try {{ await navigator.clipboard.writeText(t); }}
  catch (e) {{ const a = document.createElement('textarea'); a.value = t; document.body.appendChild(a);
               a.select(); document.execCommand('copy'); a.remove(); }}
  b.textContent = 'Copied'; setTimeout(() => b.textContent = 'Copy to clipboard', 1600);
}};
</script>""",
        height=48,
    )


def render_outcome(o: GenerationOutcome, key: str) -> None:
    if not o.ok:
        st.error(o.error or "Generation failed.")
    if o.copy is None:
        return
    st.markdown(preview_html(o), unsafe_allow_html=True)
    st.markdown(meter_html(o), unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    a.metric("Characters", o.report.char_count)
    b.metric("Words", o.report.word_count)
    c.metric("Hashtags", o.report.hashtag_count)
    d.metric("Tokens", o.prompt_tokens + o.completion_tokens)

    st.caption("Plain text (use the copy icon at the top right of the box):")
    st.code(o.final_text, language=None)
    copy_button(o.final_text)

    for issue in o.report.issues:
        css = "ctt-bad" if issue.severity == "error" else "ctt-warn"
        st.markdown(f'<span class="{css}">{issue.severity.capitalize()}: {esc(issue.message)}</span>', unsafe_allow_html=True)
    for note in o.notes:
        st.info(note)

    with st.expander("Quality heuristics"):
        ev = evaluate(o)
        st.write({k: round(v, 2) for k, v in ev.scores.items()})
        st.caption("Rule-based checks, useful for comparison. They do not replace reading the copy yourself.")
        if ev.flags:
            st.write("Flags: " + "; ".join(ev.flags))
    with st.expander("Prompt X-ray: what the model actually received"):
        st.code(o.prompt.as_text(), language=None)
        st.json(o.params_sent or {"temperature": o.request.temperature, "top_p": o.request.top_p})


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

st.title("Automated Copywriting & Tone Transformer")
st.markdown(
    '<p class="ctt-lede">Describe a product once. Get copy shaped for LinkedIn, Instagram or email, '
    "in the tone you choose.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([5, 7], gap="large")

with left:
    st.subheader("Brief")
    st.button("Load example", on_click=load_example, key="example")
    product_name = st.text_input("Product name", key="product_name", max_chars=80)
    product_description = st.text_area(
        "Product description", key="product_description", height=170, max_chars=1500,
        help="Facts only. The copy will not add features, prices or statistics that are not here.",
    )
    platform_value = st.radio(
        "Platform", [p.value for p in (*CORE_PLATFORMS, Platform.TWITTER)],
        format_func=lambda v: Platform(v).label, horizontal=True, key="platform",
    )
    tone_value = st.selectbox(
        "Tone", [t.value for t in (*CORE_TONES, Tone.WITTY)], format_func=lambda v: Tone(v).label, key="tone",
    )

    st.markdown("**Creativity controls**")
    locked = caps.sampling == "never" or backend_name == "mock"
    temperature = st.slider(
        "Temperature", 0.0, 2.0, 0.7, 0.05, key="temperature", disabled=locked,
        help="Low = predictable and consistent. High = more varied, less predictable wording.",
    )
    top_p = st.slider(
        "Top-P", 0.05, 1.0, 1.0, 0.05, key="top_p", disabled=locked,
        help="Only the most likely words that together make up this share of probability are considered.",
    )
    if backend_name == "mock":
        st.caption("The demo engine ignores these settings.")
    elif caps.sampling == "never":
        st.caption(f"{caps.model} does not accept Temperature or Top-P, so these are disabled.")
    else:
        profile = "Consistent" if temperature < 0.3 else "Balanced" if temperature <= 0.9 else "Exploratory"
        st.caption(f"Profile: {profile}. Most guides suggest changing Temperature or Top-P, not both.")
    variations = st.number_input("Variations", 1, 3, 1, key="variations")

    go = st.button("Generate copy", type="primary", key="generate", disabled=missing_key, width="stretch")
    again = st.button(
        "Regenerate", key="regenerate", disabled=missing_key or st.session_state.last_request is None,
        width="stretch",
    )


def run(request: CopyRequest, count: int, regenerate: bool) -> None:
    avoid = [o.copy.headline for o in st.session_state.outcomes if o.copy and o.copy.headline] if regenerate else []
    offset = 1 if regenerate else 0

    def one(i: int) -> GenerationOutcome:
        return generator.generate(request, variation=i + offset, avoid=avoid)

    with st.spinner("Writing…"):
        if count == 1:
            outcomes = [one(0)]
        else:
            with ThreadPoolExecutor(max_workers=count) as pool:
                outcomes = list(pool.map(one, range(count)))
    st.session_state.outcomes = outcomes
    st.session_state.last_request = request
    st.session_state.history = ([outcome_row(o) for o in outcomes] + st.session_state.history)[:30]
    st.rerun()  # refresh widgets that depend on the new state (e.g. the Regenerate button)


if go or again:
    try:
        if go:
            req = CopyRequest(
                product_name=product_name, product_description=product_description,
                platform=platform_value, tone=tone_value, temperature=temperature, top_p=top_p,
            )
        else:
            req = st.session_state.last_request
        run(req, int(variations), regenerate=again)
    except ValidationError as exc:
        st.session_state.outcomes = []
        with left:
            for err in exc.errors():
                field = str(err["loc"][0]).replace("_", " ") if err["loc"] else "input"
                st.error(f"{field.capitalize()}: {err['msg']}")

with right:
    st.subheader("Copy")
    outcomes: list[GenerationOutcome] = st.session_state.outcomes
    if not outcomes:
        st.markdown(
            '<div class="ctt-empty">Your copy appears here, framed the way it would look on the platform. '
            "Fill in the brief and choose <b>Generate copy</b>, or load the example first.</div>",
            unsafe_allow_html=True,
        )
    elif len(outcomes) == 1:
        render_outcome(outcomes[0], "v0")
    else:
        for tab, (i, o) in zip(st.tabs([f"Variation {n + 1}" for n in range(len(outcomes))]), enumerate(outcomes)):
            with tab:
                render_outcome(o, f"v{i}")

    if outcomes and any(o.copy for o in outcomes):
        st.markdown("**Download**")
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Text", "\n\n---\n\n".join(o.final_text for o in outcomes if o.copy), "copy.txt", key="dl_txt")
        d2.download_button("JSON", to_json_text(outcomes), "copy.json", "application/json", key="dl_json")
        d3.download_button("CSV", to_csv_text(outcomes), "copy.csv", "text/csv", key="dl_csv")
        d4.download_button("Markdown", to_markdown(outcomes), "copy.md", key="dl_md")

    if st.session_state.history:
        with st.expander(f"History ({len(st.session_state.history)})"):
            st.dataframe(
                [{k: r[k] for k in ("product_name", "platform", "tone", "temperature", "top_p", "chars", "ok")}
                 for r in st.session_state.history],
                width="stretch",
            )
