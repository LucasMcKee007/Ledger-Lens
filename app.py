import streamlit as st
import requests
from bs4 import BeautifulSoup
import anthropic
import re

headers = {"User-Agent": "YourName your_email@example.com"}

st.set_page_config(page_title="Ledger Lens", page_icon="🔎", layout="centered")


def get_cik(ticker):
    """Look up a company's CIK number from its ticker symbol."""
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=headers)
    companies = response.json()

    for key, company in companies.items():
        if company["ticker"] == ticker:
            return str(company["cik_str"]).zfill(10)

    return None


def get_last_two_10k_filings(cik):
    """Find the two most recent 10-K filings: (accession, date, doc) for each."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    data = response.json()

    recent = data["filings"]["recent"]
    forms = recent["form"]
    dates = recent["filingDate"]
    accession_numbers = recent["accessionNumber"]
    primary_documents = recent["primaryDocument"]

    ten_k_indexes = []
    for i in range(len(forms)):
        if forms[i] == "10-K":
            ten_k_indexes.append(i)

    if len(ten_k_indexes) < 2:
        return None, None

    current_index = ten_k_indexes[0]
    prior_index = ten_k_indexes[1]

    current_filing = (accession_numbers[current_index], dates[current_index], primary_documents[current_index])
    prior_filing = (accession_numbers[prior_index], dates[prior_index], primary_documents[prior_index])

    return current_filing, prior_filing


def download_filing(cik, accession, primary_doc):
    """Download a filing and return its raw HTML as text."""
    accession_no_dashes = accession.replace("-", "")
    cik_no_zeros = str(int(cik))

    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_dashes}/{primary_doc}"
    response = requests.get(doc_url, headers=headers)
    return response.text


def extract_section(html, section_key):
    """Extract a 10-K section by name, robust to formatting variations across companies."""
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text()

    # Normalize non-breaking spaces and other whitespace variants
    # (some companies embed Item numbers with special space characters instead of regular spaces)
    full_text = full_text.replace("\xa0", " ").replace("\u2009", " ").replace("\u202f", " ")
    search_text = full_text.lower()

    # For each section, try multiple start-marker patterns and multiple end-marker options
    if section_key == "mda":
        start_candidates = ["item 7.", "item 7 ", "management's discussion and analysis"]
        end_candidates = ["item 7a.", "item 7a ", "item 8.", "item 8 ", "quantitative and qualitative disclosures", "financial statements and supplementary data"]
    elif section_key == "risk_factors":
        start_candidates = ["item 1a.", "item 1a ", "risk factors"]
        end_candidates = ["item 1b.", "item 1b ", "item 1c.", "item 1c ", "item 2.", "item 2 ", "unresolved staff comments", "properties"]
    else:
        return None

    # Find the last occurrence of any start-marker candidate (skips table of contents entries)
    start_index = -1
    for candidate in start_candidates:
        found = search_text.rfind(candidate)
        if found > start_index:
            start_index = found

    if start_index == -1:
        return None

    # Find the earliest end-marker candidate that appears AFTER start_index
    end_index = len(full_text)
    for candidate in end_candidates:
        found = search_text.find(candidate, start_index + 50)  # skip 50 chars past the header itself
        if found != -1 and found < end_index:
            end_index = found

    section = full_text[start_index:end_index].strip()

    # Sanity check -- a real section should be at least a few thousand characters
    if len(section) < 2000:
        return None

    return section


def get_instructions_for_level(level, task):
    """Return prompt instructions tailored to the user's experience level."""

    if level == "beginner":
        tone = "Explain this in plain, everyday English for someone with NO financial background. Avoid jargon entirely -- if you must use a financial term, immediately explain it in simple words, like you're talking to a friend."
    elif level == "medium":
        tone = "Explain this for someone who understands basic investing concepts (like revenue, profit, and stock prices) but isn't a financial professional. You can use moderate financial terminology, but briefly clarify anything more advanced (like specific accounting or valuation terms)."
    else:
        tone = "Write for an experienced analyst or investor. You can use financial and accounting terminology freely without explaining basic terms."

    if task == "mda":
        return f"""{tone} Please provide:
1. A 3-4 sentence summary of the key points
2. The 3 most important takeaways for an investor
3. Any notable changes, concerns, or red flags mentioned"""

    elif task == "risk":
        return f"""{tone} Risk factor sections often repeat similar boilerplate language year after year. Please provide:
1. A brief overview of the main risk categories covered (2-3 sentences)
2. The 3-5 risks that seem most significant or specific to this company right now (not generic boilerplate like "we face competition")
3. Any risk language that sounds new, unusually specific, or more urgent than typical corporate risk disclosures"""

    elif task == "compare":
        return f"""{tone} Please cover:
1. What are the most significant changes between the two years (new topics, removed topics, changed tone or severity)?
2. Are there any risks or issues that got worse or more urgent?
3. Are there any risks or issues from last year that seem to have gotten better or gone away?
4. Overall, does the trajectory look improving, worsening, or stable for this company?"""


def analyze_with_claude(text, section_name, prompt_instructions):
    """Send a section of filing text to Claude and return the analysis."""
    client = anthropic.Anthropic()

    prompt = f"""Here is the {section_name} section from a company's 10-K filing:

{text}

{prompt_instructions}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def compare_with_claude(current_text, prior_text, section_name, prompt_instructions):
    """Send two years of the same section to Claude and ask for a comparison."""
    client = anthropic.Anthropic()

    prompt = f"""Here are two versions of the {section_name} section from the same company's 10-K filings, one year apart.

CURRENT YEAR:
{current_text}

PRIOR YEAR:
{prior_text}

{prompt_instructions}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def rate_with_claude(mda_analysis, risk_analysis, mda_comparison, risk_comparison, ticker, level):
    """Ask Claude to synthesize everything into a score and verdict, tailored to experience level."""
    client = anthropic.Anthropic()

    if level == "beginner":
        tone = "Write for someone with NO financial background. Use plain, everyday English and avoid jargon completely."
    elif level == "medium":
        tone = "Write for someone who understands basic investing concepts but isn't a financial professional. Light financial terminology is fine, briefly clarified."
    else:
        tone = "Write for an experienced investor or analyst. Financial terminology is fine without explanation."

    prompt = f"""You've analyzed {ticker}'s most recent 10-K filing across four dimensions. Here is all of that analysis:

MD&A ANALYSIS:
{mda_analysis}

RISK FACTORS ANALYSIS:
{risk_analysis}

MD&A YEAR-OVER-YEAR COMPARISON:
{mda_comparison}

RISK FACTORS YEAR-OVER-YEAR COMPARISON:
{risk_comparison}

{tone} Based ONLY on the fundamentals described above (not on stock price, market sentiment, or anything outside this filing), write a verdict using this exact structure:

**Bottom Line:** One short paragraph explaining the overall picture.

**Fundamental Health Score:** Write the number on its own line first, in this exact format: SCORE: X/10 -- then one sentence on why.

**Trajectory:** One word -- Improving, Stable, or Deteriorating -- plus one short sentence why.

**What's Good:** 3 short bullet points.

**What's Concerning:** 3 short bullet points.

**Important Disclaimer:** A brief note that this score is based only on the company's own filing, is NOT a stock price prediction, and should never be the sole basis for an investment decision.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


@st.cache_data(show_spinner=False)
def run_full_analysis(ticker, level):
    """Run the entire pipeline for a ticker + experience level. Cached so repeat lookups are instant."""

    cik = get_cik(ticker)
    if cik is None:
        return {"error": f"Could not find a CIK for ticker {ticker}. Check the spelling and try again."}

    current_filing, prior_filing = get_last_two_10k_filings(cik)
    if current_filing is None:
        return {"error": f"Could not find two annual filings (10-Ks) for {ticker}. This tool works best with established public companies."}

    current_accession, current_date, current_doc = current_filing
    prior_accession, prior_date, prior_doc = prior_filing

    current_html = download_filing(cik, current_accession, current_doc)
    prior_html = download_filing(cik, prior_accession, prior_doc)

    # --- Extraction now uses simple section keys with the more robust helper ---
    current_mda = extract_section(current_html, "mda")
    prior_mda = extract_section(prior_html, "mda")

    mda_analysis = None
    mda_comparison = None

    if current_mda:
        mda_instructions = get_instructions_for_level(level, "mda")
        mda_analysis = analyze_with_claude(current_mda, "Management's Discussion and Analysis (MD&A)", mda_instructions)

    if current_mda and prior_mda:
        compare_instructions = get_instructions_for_level(level, "compare")
        mda_comparison = compare_with_claude(current_mda, prior_mda, "Management's Discussion and Analysis (MD&A)", compare_instructions)

    current_risk = extract_section(current_html, "risk_factors")
    prior_risk = extract_section(prior_html, "risk_factors")

    risk_analysis = None
    risk_comparison = None

    if current_risk:
        risk_instructions = get_instructions_for_level(level, "risk")
        risk_analysis = analyze_with_claude(current_risk, "Risk Factors (Item 1A)", risk_instructions)

    if current_risk and prior_risk:
        compare_instructions = get_instructions_for_level(level, "compare")
        risk_comparison = compare_with_claude(current_risk, prior_risk, "Risk Factors (Item 1A)", compare_instructions)

    if not (mda_analysis and risk_analysis and mda_comparison and risk_comparison):
        missing = []
        if not current_mda:
            missing.append("current year MD&A")
        if not prior_mda:
            missing.append("prior year MD&A")
        if not current_risk:
            missing.append("current year Risk Factors")
        if not prior_risk:
            missing.append("prior year Risk Factors")
        return {"error": f"Could not extract these sections: {', '.join(missing)}. This company's filing may use different formatting than expected."}

    rating = rate_with_claude(mda_analysis, risk_analysis, mda_comparison, risk_comparison, ticker, level)

    return {
        "error": None,
        "current_date": current_date,
        "prior_date": prior_date,
        "rating": rating,
        "mda_analysis": mda_analysis,
        "mda_comparison": mda_comparison,
        "risk_analysis": risk_analysis,
        "risk_comparison": risk_comparison,
    }


def extract_score(rating_text):
    """Pull the X/10 score out of the rating text, if present."""
    match = re.search(r"SCORE:\s*(\d+)\s*/\s*10", rating_text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def extract_trajectory(rating_text):
    """Pull the trajectory word out of the rating text, if present."""
    match = re.search(r"Trajectory:?\*{0,2}\s*(Improving|Stable|Deteriorating)", rating_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None


# ============================================================
# CUSTOM DESIGN SYSTEM -- fonts, colors, and the signature stamp
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    color: #14213D !important;
    font-weight: 600 !important;
}

.ledger-subtitle {
    font-family: 'Inter', sans-serif;
    color: #6B6B63;
    font-size: 1.05rem;
    margin-top: -0.6rem;
}

.ledger-rule {
    border: none;
    border-top: 1px solid #D8D4C7;
    margin: 1.6rem 0;
}

.ticker-mono {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.03em;
}

.stamp-wrapper {
    display: flex;
    justify-content: center;
    margin: 1.5rem 0 2rem 0;
}

.stamp {
    width: 148px;
    height: 148px;
    border-radius: 50%;
    border: 3px double #B08D57;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #F6F5F1;
    box-shadow: 0 1px 3px rgba(20, 33, 61, 0.08);
}

.stamp-score {
    font-family: 'Fraunces', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #14213D;
    line-height: 1;
}

.stamp-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #B08D57;
    margin-top: 4px;
}

.trajectory-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 3px;
    margin-bottom: 1rem;
}

.trajectory-improving { background: #E7EEE7; color: #3A5A40; }
.trajectory-stable { background: #EDEBE3; color: #6B6B63; }
.trajectory-deteriorating { background: #F2E1DF; color: #8C1C13; }

.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    border-radius: 3px;
    border: 1px solid #14213D;
}

.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# STREAMLIT PAGE STARTS HERE
# ============================================================

st.markdown("# 🔎 Ledger Lens")
st.markdown('<p class="ledger-subtitle">Understand any public company\'s financial health in one click — no finance background, no digging through 100-page filings required.</p>', unsafe_allow_html=True)
st.caption("Ledger Lens automatically finds, reads, and compares a company's official SEC filings year-over-year, then explains what it means in plain English.")

st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)

st.write("**Try an example:**")
example_col1, example_col2, example_col3, example_col4 = st.columns(4)

if "ticker_value" not in st.session_state:
    st.session_state.ticker_value = ""

with example_col1:
    if st.button("AAPL"):
        st.session_state.ticker_value = "AAPL"
with example_col2:
    if st.button("MSFT"):
        st.session_state.ticker_value = "MSFT"
with example_col3:
    if st.button("TSLA"):
        st.session_state.ticker_value = "TSLA"
with example_col4:
    if st.button("NVDA"):
        st.session_state.ticker_value = "NVDA"

st.write("")

col1, col2 = st.columns(2)

with col1:
    ticker_input = st.text_input("Stock ticker", value=st.session_state.ticker_value, placeholder="e.g. AAPL, MSFT, TSLA")

with col2:
    level_choice = st.selectbox(
        "Your investing experience",
        ["Beginner", "Medium", "Advanced"]
    )

level = level_choice.lower()

analyze_clicked = st.button("Analyze", type="primary")
st.caption("Takes about 30–45 seconds — we're reading two years of annual filings for you.")

if analyze_clicked:
    ticker = ticker_input.upper().strip()

    if ticker == "":
        st.error("Please enter a ticker symbol first.")
    else:
        try:
            with st.spinner(f"Analyzing {ticker}... this takes about 30-45 seconds"):
                result = run_full_analysis(ticker, level)

        except requests.exceptions.RequestException:
            st.error("We couldn't connect to the SEC's website right now. Please check your internet connection and try again.")
            result = None
        except anthropic.APIError:
            st.error("We couldn't reach Claude to generate the analysis. Please try again in a moment.")
            result = None
        except Exception:
            st.error("Something unexpected went wrong. Please double check the ticker and try again.")
            result = None

        if result is not None:
            if result["error"]:
                st.error(result["error"])
            else:
                st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
                st.info(f"✅ Found and compared {ticker}'s two most recent annual filings (10-Ks): {result['current_date']} vs. {result['prior_date']}")

                st.markdown(f'<h2><span class="ticker-mono">{ticker}</span> — Quick Verdict</h2>', unsafe_allow_html=True)

                score = extract_score(result["rating"])
                trajectory = extract_trajectory(result["rating"])

                if score:
                    st.markdown(f"""
                    <div class="stamp-wrapper">
                        <div class="stamp">
                            <div class="stamp-score">{score}/10</div>
                            <div class="stamp-label">Fundamental Health</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if trajectory:
                    css_class = f"trajectory-{trajectory.lower()}"
                    st.markdown(f'<div style="text-align:center;"><span class="trajectory-badge {css_class}">{trajectory}</span></div>', unsafe_allow_html=True)

                st.markdown(result["rating"])

                st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
                st.markdown("## MD&A: What Management Said")

                with st.expander("This Year's MD&A Analysis"):
                    st.markdown(result["mda_analysis"])

                with st.expander("What Changed From Last Year"):
                    st.markdown(result["mda_comparison"])

                st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
                st.markdown("## Risk Factors: What Could Go Wrong")

                with st.expander("This Year's Risk Factors Analysis"):
                    st.markdown(result["risk_analysis"])

                with st.expander("What Changed From Last Year"):
                    st.markdown(result["risk_comparison"])

                st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
                st.caption("This analysis is based only on the company's own public SEC filing. It is not a stock price prediction and should not be the sole basis for an investment decision.")
