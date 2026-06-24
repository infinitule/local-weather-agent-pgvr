"""
PGVR: a Parameterize → Ground → Verify → Refine agentic loop.

Why this exists
---------------
The plain ReAct app has two blind spots a small model cannot see:

  1. The 0.8B model is a *thinking* model. Asked for schema-constrained JSON it
     runs away in its reasoning trace and returns EMPTY content. So the usual
     "extract structured params with format=<schema>" trick does not work here.
  2. The geocoder fuzzy-matches. "Kamand, Himachal Pradesh" silently resolved to
     "Gamand, Iran" — wrong country, no warning. The model happily reported it.

What we discovered the small model CAN do reliably
--------------------------------------------------
Tool-calling. When weather is exposed as a tool with rich, well-described
parameters, the 0.8B model fills them correctly every time and does NOT melt
down into a thinking loop:

    "weather in Kamand, Himachal Pradesh"
      -> city='Kamand', region='Himachal Pradesh', country='India'

So we use TOOL-CALLING as the structured-output channel (PARAMETERIZE), then a
deterministic loop does the part the model is bad at:

    GROUND  : fetch up to 10 geocoder candidates for the city
    VERIFY  : score each candidate against the user's stated region/country;
              never accept a wrong-country match silently
    REFINE  : if nothing matches the stated region, retry with broader queries;
              if still ambiguous, hand candidates back for disambiguation

The result: the small model gets location-grounding it could never do alone,
and the Kamand→Iran class of bug is impossible.

Run:
    python agentic.py "how's the weather in Kamand, Himachal Pradesh?"
    python agentic.py            # interactive
"""

import re
import sys
import time
import unicodedata

import ollama
import requests

from weather_app import _WMO, MODEL


def _chat(**kwargs):
    """ollama.chat with one retry — the local server occasionally drops the
    connection (e.g. when it auto-updates), which we don't want to crash on."""
    for attempt in range(2):
        try:
            return ollama.chat(**kwargs)
        except Exception:
            if attempt == 1:
                raise
            time.sleep(1.5)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# country aliases the model/users use vs. what the geocoder returns
_COUNTRY_ALIASES = {
    "usa": "united states", "us": "united states", "u.s.": "united states",
    "uk": "united kingdom", "uae": "united arab emirates", "uk.": "united kingdom",
}


# ── GROUND: candidate geocoding ───────────────────────────────────────────────
def geocode_candidates(name, count=10):
    r = requests.get(
        GEOCODE_URL,
        params={"name": name, "count": count, "language": "en", "format": "json"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("results", []) or []


def _typo_variants(name):
    """Edit-distance-1 spellings (one deletion or one adjacent swap).

    The geocoder does prefix matching but no typo correction: 'Gadansk' finds
    nothing, yet deleting the stray 'a' gives 'Gdansk'. We try these only as a
    fallback when the exact spelling returns no candidates.
    """
    out, seen = [], set()
    for i in range(len(name)):                       # deletions
        v = name[:i] + name[i + 1:]
        if len(v) >= 3 and v.lower() != name.lower() and v not in seen:
            seen.add(v); out.append(v)
    for i in range(len(name) - 1):                   # adjacent transpositions
        v = name[:i] + name[i + 1] + name[i] + name[i + 2:]
        if v.lower() != name.lower() and v not in seen:
            seen.add(v); out.append(v)
    return out[:25]


def _norm_country(c):
    c = (c or "").strip().lower()
    return _COUNTRY_ALIASES.get(c, c)


def _deaccent(s):
    """'Punjābpura' → 'punjabpura' so name matching ignores diacritics."""
    return "".join(c for c in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(c)).lower().strip()


# Indian states / UTs are not point locations — the geocoder returns random
# villages for them. Map each to a well-known, unambiguously geocodable city so
# "weather in Punjab" means a real place instead of "Punjābpura, Uttar Pradesh".
_REGION_CITY = {
    "andhra pradesh": "Visakhapatnam", "arunachal pradesh": "Itanagar",
    "assam": "Guwahati", "bihar": "Patna", "chhattisgarh": "Raipur",
    "goa": "Panaji", "gujarat": "Ahmedabad", "haryana": "Gurugram",
    "himachal pradesh": "Shimla", "jharkhand": "Ranchi", "karnataka": "Bengaluru",
    "kerala": "Thiruvananthapuram", "madhya pradesh": "Indore",
    "maharashtra": "Mumbai", "manipur": "Imphal", "meghalaya": "Shillong",
    "mizoram": "Aizawl", "nagaland": "Kohima", "odisha": "Bhubaneswar",
    "punjab": "Ludhiana", "rajasthan": "Jaipur", "sikkim": "Gangtok",
    "tamil nadu": "Chennai", "telangana": "Hyderabad", "tripura": "Agartala",
    "uttar pradesh": "Lucknow", "uttarakhand": "Dehradun", "west bengal": "Kolkata",
    "delhi": "New Delhi", "jammu and kashmir": "Srinagar", "ladakh": "Leh",
    "puducherry": "Puducherry", "chandigarh": "Chandigarh",
    "andaman and nicobar islands": "Port Blair",
}


def _region_to_city(city, region, country):
    """If the request points at an Indian state/UT rather than a city, return its
    representative city (and the state name), else (None, None).

    Handles both shapes the model produces:
      city='Punjab'                          → the city field is a state
      city='Himachal', region='Himachal …'   → city is just a fragment of the state
    """
    if _norm_country(country) not in ("", "india"):
        return None, None  # don't apply the India gazetteer to other countries
    c = _deaccent(city)
    if c in _REGION_CITY:                       # city field is itself a state
        return _REGION_CITY[c], city
    rg = _deaccent(region)
    if rg in _REGION_CITY and (not c or c in rg):  # state in region, city is a fragment
        return _REGION_CITY[rg], region
    return None, None


def _score(cand, city, region, country):
    """How well a geocoder candidate matches the user's request.

    Rewards an exact name match and place importance, not just region/country —
    otherwise a fuzzy village ('Punjabpura') ties with the real city.
    """
    s, why = 0.0, []
    name = _deaccent(cand.get("name"))
    qcity = _deaccent(city)
    admin1 = _deaccent(cand.get("admin1"))
    admin2 = _deaccent(cand.get("admin2"))
    cname = (cand.get("country") or "").lower()
    ccode = (cand.get("country_code") or "").lower()

    # exact name match is the strongest signal; a partial one barely counts
    if qcity and name == qcity:
        s += 6; why.append("name✓")
    elif qcity and qcity in name:
        s += 1

    region = _deaccent(region)
    if region:
        if region in admin1 or admin1 in region:
            s += 5; why.append("region✓")
        elif region in admin2:
            s += 3; why.append("district✓")

    country = _norm_country(country)
    if country:
        if country in cname or cname in country or country == ccode:
            s += 5; why.append("country✓")

    # gentle tiebreakers: prefer populous, administratively-significant places
    pop = cand.get("population") or 0
    s += min(pop, 5_000_000) / 5_000_000          # up to +1.0
    if (cand.get("feature_code") or "").startswith(("PPLA", "PPLC")):
        s += 0.5                                    # admin seat / capital
    return s, why


# ── VERIFY + REFINE: grounded resolution ──────────────────────────────────────
def resolve_location(city, region, country, trace):
    """Return (best_candidate, status, ranked) where status is one of
    'high' / 'medium' / 'low' / 'conflict' / 'not_found'."""
    # REFINE step 0: if the "city" is really a state/UT, swap in a real city.
    rep, state = _region_to_city(city, region, country)
    if rep:
        trace.append(f"  ↪ '{city}' is a region → using {rep} (major city)")
        region = region or state
        city = rep

    has_hint = bool((region or "").strip() or _norm_country(country))

    # REFINE step 1: progressively broader queries until the geocoder answers.
    queries = [city]
    if region:
        queries += [f"{city} {region}", region]
    cands = []
    for q in queries:
        cands = geocode_candidates(q)
        trace.append(f"  geocode({q!r}) → {len(cands)} candidates")
        if cands:
            break

    # REFINE step 2: still nothing? the name may be misspelled — try edit-distance-1
    # variants (handles 'Gadansk' → 'Gdansk') and adopt the first spelling that hits.
    if not cands:
        for v in _typo_variants(city):
            hits = geocode_candidates(v)
            if hits:
                trace.append(f"  ✎ '{city}' not found — corrected to '{v}' → {len(hits)} candidates")
                city, cands = v, hits
                break
    if not cands:
        return None, "not_found", []

    ranked = sorted(
        ((_score(c, city, region, country), c) for c in cands),
        key=lambda x: x[0][0], reverse=True,
    )
    (best_score, best_why), best = ranked[0]

    if has_hint and best_score == 0:
        # VERIFY failed: the user named a region/country and NOTHING matched it.
        # This is exactly the Kamand→Iran trap — refuse to guess.
        trace.append(f"  ⚠ no candidate matches region/country → CONFLICT")
        return best, "conflict", ranked

    if not has_hint:
        status = "medium"  # took the geocoder's top hit, user gave no hint
    elif best_score >= 5:
        status = "high"
        trace.append(f"  ✓ matched {best_why} → {best.get('name')}, "
                     f"{best.get('admin1')}, {best.get('country')}")
    else:
        status = "low"
    return best, status, ranked


def _fmt_place(c):
    return ", ".join(p for p in (c.get("name"), c.get("admin1"), c.get("country")) if p)


# ── ACT: fetch weather for resolved coordinates ───────────────────────────────
def fetch_weather(cand, mode, days):
    lat, lon = cand["latitude"], cand["longitude"]
    if mode == "forecast":
        days = max(1, min(int(days or 3), 7))
        r = requests.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max",
            "forecast_days": days, "timezone": "auto"}, timeout=15)
        r.raise_for_status()
        d = r.json()["daily"]
        lines = []
        for i in range(len(d["time"])):
            sky = _WMO.get(d["weather_code"][i], "unknown")
            lines.append(f"{d['time'][i]}: {d['temperature_2m_min'][i]}–"
                         f"{d['temperature_2m_max'][i]}°C, {sky}, "
                         f"{d['precipitation_probability_max'][i]}% rain")
        return "\n".join(lines)

    r = requests.get(FORECAST_URL, params={
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                   "weather_code,wind_speed_10m"}, timeout=15)
    r.raise_for_status()
    c = r.json()["current"]
    sky = _WMO.get(c["weather_code"], "unknown")
    return (f"{c['temperature_2m']}°C (feels like {c['apparent_temperature']}°C), "
            f"{sky}, humidity {c['relative_humidity_2m']}%, "
            f"wind {c['wind_speed_10m']} km/h")


# ── PARAMETERIZE: the tool the model fills via tool-calling ────────────────────
def _weather_tool(city: str, region: str = "", country: str = "",
                  mode: str = "current", days: int = 3) -> str:
    """Get current weather or a forecast for a place.

    Args:
        city: city/town/village name ONLY, e.g. 'Kamand' — never include region or country here.
        region: the state or province if the user named it, e.g. 'Himachal Pradesh', else ''.
        country: the country if the user named it, e.g. 'India', else ''.
        mode: 'current' for now, 'forecast' for upcoming days.
        days: number of forecast days, 1 to 7.
    """
    # body is unused — we intercept the call to run the PGVR loop ourselves.
    return ""


# ── The agentic loop ──────────────────────────────────────────────────────────
# words that signal the user actually wants a multi-day forecast
_FORECAST_HINTS = ("forecast", "tomorrow", "next ", "coming", "upcoming", "week",
                   "weekend", "later", "days", "rest of")


def _intended_mode(query, model_mode):
    """Decide current vs forecast from the user's words, not the model's guess.

    The 0.8B model defaults to mode='forecast' even for 'how is the weather now',
    so we override it: forecast only if the query actually mentions the future.
    """
    q = query.lower()
    return "forecast" if any(h in q for h in _FORECAST_HINTS) else "current"


def _intended_days(query, model_days):
    """How many forecast days the user actually asked for (1–7, capped).

    The model mis-parses phrases like 'next one week' as days=1, so we read the
    span from the query: 'week' → 7, 'N days' → N, 'tomorrow' → 2.
    """
    q = query.lower()
    if "week" in q or "fortnight" in q:
        return 7
    m = re.search(r"(\d+)\s*day", q)
    if m:
        return max(1, min(int(m.group(1)), 7))
    if "tomorrow" in q:
        return 2
    return max(1, min(int(model_days or 3), 7))


def _city_in_query(city, query):
    """True if the model's extracted city actually appears in the user's text.

    The 0.8B model will fabricate a city (e.g. 'Kamand') for a question that
    names no place at all. We require the city's words to be present in the
    query so we can U-turn out instead of grounding a hallucinated location.
    """
    q = query.lower()
    return any(len(t) >= 3 and t in q for t in re.split(r"[\s,]+", city.lower()))


# Shown on a U-turn (no real place in the query) — also answers "what are you?".
_WHAT_I_AM = (
    "I'm a weather tool, not a chatbot — ask me about a place's weather.\n"
    "What makes me different from a plain weather app:\n"
    "  • Region-grounded — 'Kamand, Himachal Pradesh' resolves to India, never Iran.\n"
    "  • I verify the match against your stated region/country and refuse to guess.\n"
    "  • Runs fully local (qwen3.5:0.8b) on live Open-Meteo data, no API key.\n"
    "Try:  weather in Tokyo   ·   forecast for Pune next 3 days"
)


# phrasing that implies the user is comparing two or more places
_COMPARE_HINTS = (" vs ", " versus ", "compare", "compared to", "different from",
                  "difference between", "how is it different", "warmer", "cooler",
                  "hotter", "colder", "between ")


def _looks_like_comparison(query):
    q = f" {query.lower()} "
    return any(h in q for h in _COMPARE_HINTS)


def _extract(query, emphasize_all=False):
    """Ask the model to turn the query into one get_weather call per place."""
    sysp = (
        ("Call get_weather SEPARATELY for EVERY place named in the question. "
         "A comparison ('A vs B', 'how is A different from B', 'compare A and B') "
         "names TWO places — you MUST call get_weather for BOTH. "
         if emphasize_all else
         "Call get_weather once per place the user asks about. ")
        + "Put ONLY the city in `city`; put any state/province in `region` and any "
          "country in `country`. Use mode='forecast' only if they ask about upcoming days."
    )
    resp = _chat(
        model=MODEL,
        messages=[{"role": "system", "content": sysp},
                  {"role": "user", "content": query}],
        tools=[_weather_tool],
        options={"temperature": 0},
    )
    return resp.message.tool_calls or []


def smart_weather(query, verbose=True):
    """Parameterize → Ground → Verify → Refine → synthesize an answer."""
    trace = []

    # PARAMETERIZE — let the model turn free text into structured tool calls.
    calls = _extract(query)

    # REFINE: a comparison ('A vs B', 'how is A different from B') needs two
    # places, but the 0.8B model intermittently drops the second one. If we see
    # comparison phrasing yet got fewer than two places, retry insisting on all.
    n_places = len({_deaccent(dict(c.function.arguments or {}).get("city", "")) for c in calls})
    if _looks_like_comparison(query) and n_places < 2:
        retry = _extract(query, emphasize_all=True)
        if len(retry) > len(calls):
            trace.append(f"  ↻ refine: comparison detected — re-extracted {len(retry)} places")
            calls = retry

    # ── U-TURN ────────────────────────────────────────────────────────────────
    # Before grounding anything, keep only cities that actually appear in the
    # user's text. A made-up city (model hallucination on a non-weather question)
    # makes us turn back here instead of running the full Ground→Verify→Refine
    # pipeline on a place the user never mentioned.
    valid, seen = [], set()
    for call in calls:
        a = dict(call.function.arguments or {})
        city = a.get("city", "").strip()
        if not city or _deaccent(city) in seen:
            continue  # skip blanks and duplicate places
        if _city_in_query(city, query):
            seen.add(_deaccent(city))
            valid.append(a)
        else:
            trace.append(f"↩ U-turn: '{city}' isn't in your question — ignored (hallucinated)")
    if not valid:
        if verbose and trace:
            print("\n".join(trace))
        return _WHAT_I_AM

    # GROUND + VERIFY + REFINE for each requested place.
    findings = []
    current_temps = []  # (place_name, temp°C) for the comparison line
    for a in valid:
        city = a.get("city", "").strip()
        region, country = a.get("region", ""), a.get("country", "")
        mode = _intended_mode(query, a.get("mode", "current"))  # override model's guess
        days = _intended_days(query, a.get("days", 3))           # ditto for span
        trace.append(f"▸ {city!r} (region={region!r}, country={country!r}, mode={mode})")

        best, status, ranked = resolve_location(city, region, country, trace)

        if status == "not_found":
            findings.append(f"'{city}': no such place found.")
            continue
        if status == "conflict":
            # VERIFY caught a wrong-region match — disambiguate instead of lying.
            opts = "; ".join(_fmt_place(c) for (_, c) in ranked[:3])
            asked = ", ".join(x for x in (region, country) if x)
            findings.append(
                f"'{city}' has no match in {asked}. Did you mean one of: {opts}? "
                f"(I did NOT report weather to avoid giving you the wrong place.)")
            continue

        # If the user named a country but the real place is elsewhere, correct
        # them rather than failing or silently reporting the wrong country.
        note = ""
        ch = _norm_country(country)
        if ch:
            bc = (best.get("country") or "").lower()
            bcc = (best.get("country_code") or "").lower()
            if not (ch in bc or bc in ch or ch == bcc):
                note = (f"⚠ {best.get('name')} is in {best.get('country')}, "
                        f"not {country.strip().title()}.\n")

        place = _fmt_place(best)
        data = fetch_weather(best, mode, days)
        conf = {"high": "", "medium": " (no region given — best guess)",
                "low": " (weak region match — verify the place)"}[status]
        label = "Forecast" if mode == "forecast" else "Now"
        findings.append(f"{note}{place}{conf}\n{label}: {data}")

        if mode == "current":
            m = re.match(r"\s*(-?\d+(?:\.\d+)?)°C", data)
            if m:
                current_temps.append((best.get("name", city), float(m.group(1))))

    # SYNTHESIZE the comparison the user asked for, deterministically: with two+
    # current readings, state the actual temperature gap instead of leaving the
    # reader to eyeball it. No model call → no hallucinated numbers.
    if _looks_like_comparison(query) and len(current_temps) >= 2:
        warm = max(current_temps, key=lambda t: t[1])
        cool = min(current_temps, key=lambda t: t[1])
        gap = round(warm[1] - cool[1], 1)
        if gap == 0:
            findings.append(f"Difference: {warm[0]} and {cool[0]} are the same temperature.")
        else:
            findings.append(
                f"Difference: {warm[0]} is {gap}°C warmer than {cool[0]} "
                f"({warm[1]}°C vs {cool[1]}°C).")

    if verbose and trace:
        print("\n".join(trace))

    return "\n\n".join(findings)


# ── Banner / CLI ──────────────────────────────────────────────────────────────
# truecolor ANSI; degrades gracefully to plain text if the terminal ignores it
_O  = "\033[38;2;215;119;87m"      # orange (matches the Claude Code accent)
_OB = "\033[1;38;2;215;119;87m"    # bold orange
_W  = "\033[1;38;2;236;232;225m"   # bold off-white
_D  = "\033[38;2;150;145;138m"     # dim grey
_B  = "\033[38;2;122;162;204m"     # blue
_R  = "\033[0m"


def _boxed(title, width=62):
    line = "─" * width
    pad = width - 2 - len(title)
    return (f"{_O}╭{line}╮{_R}\n"
            f"{_O}│{_R} {_W}{title}{_R}{' ' * pad}{_O}│{_R}\n"
            f"{_O}╰{line}╯{_R}")


def print_banner():
    print(_boxed("✳ Welcome to Smart Weather — PGVR agentic edition"))
    print(f"""
       {_O}\\   |   /{_R}
        {_O}.--·--.{_R}        {_OB}SMART WEATHER{_R}
    {_O}──=( {_OB} ☀ {_R}{_O} )=──{_R}     {_D}grounded · verified · fully local{_R}
        {_O}`--·--'{_R}
       {_O}/   |   \\{_R}

  {_OB}Model{_R}   {_W}{MODEL:<14}{_R}{_D}local · ~1 GB · runs on CPU{_R}
  {_OB}Data{_R}    {_W}{'Open-Meteo':<14}{_R}{_D}live · free · no API key{_R}
  {_OB}Loop{_R}    {_W}Parameterize → Ground → Verify → Refine{_R}
  {_OB}Guard{_R}   {_D}region-checked — 'Kamand, HP' won't drift to Iran{_R}
  {_OB}Skills{_R}  {_D}current weather · 7-day forecast · multi-city compare{_R}

  {_D}Try:{_R}  {_B}weather in Tokyo{_R}   {_D}·{_R}   {_B}forecast for Pune next 3 days{_R}
        {_B}is it warmer in Mumbai or Delhi?{_R}

  {_D}Type your question, or {_R}{_W}quit{_R}{_D} to exit.{_R}
""")


def interactive():
    print_banner()
    while True:
        try:
            q = input(f"{_OB}you ▸{_R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_D}bye 👋{_R}"); break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "bye"}:
            print(f"{_D}bye 👋{_R}"); break
        print(f"\n{smart_weather(q)}\n")


def main():
    if len(sys.argv) > 1:
        print(smart_weather(" ".join(sys.argv[1:])))
        return
    interactive()


if __name__ == "__main__":
    main()
