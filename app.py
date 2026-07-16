import streamlit as st
import requests
from bs4 import BeautifulSoup
import anthropic
import re

headers = {"User-Agent": "YourName your_email@example.com"}


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


def extract_section(html, start_marker, end_marker):
    """Pull out a section of text between two markers (e.g. 'item 7.' and 'item 8.')."""
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text()
    search_text = full_text.lower()

    start_index = search_text.rfind(start_marker)
    if start_index == -1:
        return None

    end_index = search_text.find(end_marker, start_index + len(start_marker))
    return full_text[start_index:end_index]


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

**Fundamental Health Score:** A number from 1-10 (10 = excellent, 1 = severe concerns), plus one sentence on why. Write the number on its own line first, in this exact format: SCORE: X/10

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


# ============================================================
# CACHED WRAPPER FUNCTIONS
# ============================================================
# @st.cache_data remembers the result for a given set of inputs.
# If someone looks up the same ticker + experience level again,
# Streamlit returns the saved result instantly instead of
# re-downloading filings and re-calling Claude from scratch.

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

    current_mda = extract_section(current_html, "item 7.", "item 8.")
    prior_mda = extract_section(prior_html, "item 7.", "item 8.")

    mda_analysis = None
    mda_comparison = None

    if current_mda:
        mda_instructions = get_instructions_for_level(level, "mda")
        mda_analysis = analyze_with_claude(current_mda, "Management's Discussion and Analysis (MD&A)", mda_instructions)

    if current_mda and prior_mda:
        compare_instructions = get_instructions_for_level(level, "compare")
        mda_comparison = compare_with_claude(current_mda, prior_mda, "Management's Discussion and Analysis (MD&A)", compare_instructions)

    current_risk = extract_section(current_html, "item 1a.", "item 1b.")
    prior_risk = extract_section(prior_html, "item 1a.", "item 1b.")

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
    """Pull the X/10 score out of the rating text, if present, for display as a metric."""
    match = re.search(r"SCORE:\s*(\d+)\s*/\s*10", rating_text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


# ============================================================
# STREAMLIT PAGE STARTS HERE -- everything below is the UI
# ============================================================

st.title("🔎 Ledger Lens")
st.write("Understand any public company's financial health in one click — no finance background, no digging through 100-page filings required.")
st.caption("Ledger Lens automatically finds, reads, and compares a company's official SEC filings year-over-year, then explains what it means in plain English.")

st.write("")  # small spacer

# Example ticker buttons
st.write("**Try an example:**")
example_col1, example_col2, example_col3, example_col4 = st.columns(4)

# session_state lets us "remember" a value across reruns -- here, which ticker was clicked
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
                st.info(f"✅ Found and compared {ticker}'s two most recent annual filings (10-Ks): {result['current_date']} vs. {result['prior_date']}")

                # --- Verdict, with a visual score metric ---
                st.header(f"Quick Verdict: {ticker}")

                score = extract_score(result["rating"])
                if score:
                    st.metric(label="Fundamental Health Score", value=f"{score}/10")

                st.markdown(result["rating"])

                # --- MD&A ---
                st.header("MD&A: What Management Said")

                with st.expander("This Year's MD&A Analysis"):
                    st.markdown(result["mda_analysis"])

                with st.expander("What Changed From Last Year"):
                    st.markdown(result["mda_comparison"])

                # --- Risk Factors ---
                st.header("Risk Factors: What Could Go Wrong")

                with st.expander("This Year's Risk Factors Analysis"):
                    st.markdown(result["risk_analysis"])

                with st.expander("What Changed From Last Year"):
                    st.markdown(result["risk_comparison"])

                st.caption("This analysis is based only on the company's own public SEC filing. It is not a stock price prediction and should not be the sole basis for an investment decision.")
