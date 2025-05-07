import streamlit as st
import pandas as pd
from io import BytesIO

from sprint_engine import run_sprint
from tmf_synth_utils import load_personas

# ─── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Burgo's Campaign Messaging Tester",
    layout="centered",
    page_icon="🧪",
)
st.title("🧪 Quick Copy Pulse - Synthetic Focus Group")

# ─── About block with blue background ──────────────────────────────────
st.markdown(
    """
<div style="background:#E7EBF8;padding:20px 25px;border-radius:10px;margin-bottom:25px">
<h3 style="margin-top:0">ℹ️ About This Tool</h3>

This tool lets you drop in draft copy and
instantly hear from 50 AI personas that attempt to approximate real Australian investor segments.

**What you get**

* Qualitative reactions (plain English)  
* 0-10 “intent to act” score per persona  
* Sentiment clusters + executive summary  
* One-click Excel export (responses + clusters)

<span style="font-size:0.9em"><em>Accepted formats: txt, html, md, doc, docx, pdf</em></span>
</div>
""",
    unsafe_allow_html=True,
)

# ─── How-it-works expander ─────────────────────────────────────────────
with st.expander("🔬 How the synthetic focus group works"):
    st.markdown(
        """
1. **Seed personas** – As part of our ["Persona Portal tool"](https://burgo-ai-persona-project.streamlit.app/), we previously developed 10 richly detailed investor archetypes, based on real Australian survey data.  
2. **Variant cloning** – the app jitters age/income/traits to create ~50 look-alike respondents.  
3. **Persona prompting** – each variant reads your copy and reacts in character.  
4. **K-means clustering** – similar reactions are grouped; GPT labels the theme.  
5. **Results** – you get raw comments, intent scores, cluster summaries & an Excel export.

**Caveat Emptors**

* Although every respondent is a variant of research-based seed personas, treat the clusters as directional themes, not statistically significant strata.   
* Think of this as a cheap, quick first filter – kill weak copy quickly, then run real A/B tests to validate.
"""
    )

# ─── Load personas & build segment list ───────────────────────────────
personas_data = load_personas("personas.json")
segments = sorted({p["segment"] for p in personas_data})
SEG_ALL = "All Segments"
segment_options = segments + [SEG_ALL]        # All Segments last

# ─── Inputs ────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload creative copy",
    type=["txt", "html", "md", "doc", "docx", "pdf"],
)
segment = st.selectbox("Audience segment", segment_options)

# ─── Run sprint with progress bar ─────────────────────────────────────
run = st.button("🚀 Run Synthetic Feedback Session", type="primary")
if run and uploaded:
    progress_bar = st.progress(0, text="Contacting personas…")

    with st.spinner("Gathering synthetic reactions…"):
        summary, df, fig, cluster_df = run_sprint(
            uploaded,
            segment,
            personas_data,
            return_cluster_df=True,
            progress_cb=progress_bar,      # pass bar to engine
        )

    progress_bar.empty()  # remove bar when finished

    st.session_state["last_result"] = {
        "summary": summary,
        "df": df,
        "fig": fig,
        "clusters": cluster_df,
    }

# ─── Display & download section ───────────────────────────────────────
if "last_result" in st.session_state:
    res = st.session_state["last_result"]

    st.plotly_chart(res["fig"], use_container_width=True)

    st.dataframe(
        res["df"],
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    st.markdown(res["summary"], unsafe_allow_html=True)

    # Excel export helper
    def _to_excel(responses: pd.DataFrame, clusters: pd.DataFrame) -> BytesIO:
        out = BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            responses.to_excel(writer, sheet_name="responses", index=False)
            clusters.to_excel(writer, sheet_name="clusters", index=False)
        out.seek(0)
        return out

    st.download_button(
        "Download results (Excel)",
        data=_to_excel(res["df"], res["clusters"]),
        file_name="concept_sprint_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
