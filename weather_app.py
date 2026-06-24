"""
Weather Agent — a small local-model weather app.

Same idea as the Day-6 notebook (Ollama + a ReAct tool loop), but the
`get_weather` tool now returns REAL data from the free Open-Meteo API
instead of the hardcoded demo dictionary.

Stack:
  • Local model  : qwen3.5:0.8b  (a small ~1 GB model, runs on CPU)
  • Weather data : Open-Meteo    (free, no API key needed)
  • Agent style  : ReAct loop with tools (weather + forecast + calculator)

Run:
  python weather_app.py                       # interactive chat
  python weather_app.py "weather in Mumbai?"  # one-shot question
"""

import ast
import operator
import sys

import ollama
import requests

from memory import VectorMemory

# ── Config ────────────────────────────────────────────────────────────────────
# A small model you already have pulled. Change to any model `ollama list` shows.
MODEL = "qwen3.5:0.8b"

SYSTEM = (
    "You are a friendly weather assistant. "
    "Use the get_weather tool for current conditions and get_forecast for the "
    "days ahead. Use calculate for any arithmetic (e.g. comparing temperatures). "
    "Always call a tool to get real numbers — never guess the weather. "
    "Once you have the data, answer in one or two short, natural sentences."
)

# WMO weather interpretation codes → human text
# https://open-meteo.com/en/docs
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


# ── Weather helpers ───────────────────────────────────────────────────────────
def _geocode(city: str):
    """Turn a city name into (display_name, latitude, longitude).

    Tries the full string, then just the part before the first comma — small
    models sometimes pass 'Pune, Maharashtra, India' which the API won't match.
    """
    for query in (city, city.split(",")[0].strip()):
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 1, "language": "en", "format": "json"},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("results")
        if results:
            top = results[0]
            name = ", ".join(p for p in (top.get("name"), top.get("admin1"), top.get("country")) if p)
            return name, top["latitude"], top["longitude"]
    return None


# ── Tools ─────────────────────────────────────────────────────────────────────
def get_weather(city: str) -> str:
    """Get the CURRENT weather for a city.

    Args:
        city: The city name, for example 'Pune'.
    """
    loc = _geocode(city)
    if loc is None:
        return f"No location found for '{city}'."
    name, lat, lon = loc
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                       "weather_code,wind_speed_10m",
        },
        timeout=15,
    )
    r.raise_for_status()
    c = r.json()["current"]
    sky = _WMO.get(c["weather_code"], "unknown conditions")
    return (
        f"{name}: {c['temperature_2m']}°C (feels like {c['apparent_temperature']}°C), "
        f"{sky}, humidity {c['relative_humidity_2m']}%, "
        f"wind {c['wind_speed_10m']} km/h."
    )


def get_forecast(city: str, days: int = 3) -> str:
    """Get a daily weather FORECAST for the next few days for a city.

    Args:
        city: The city name, for example 'Delhi'.
        days: How many days ahead, 1 to 7 (default 3).
    """
    loc = _geocode(city)
    if loc is None:
        return f"No location found for '{city}'."
    name, lat, lon = loc
    days = max(1, min(int(days), 7))
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": days, "timezone": "auto",
        },
        timeout=15,
    )
    r.raise_for_status()
    d = r.json()["daily"]
    lines = [f"Forecast for {name}:"]
    for i in range(len(d["time"])):
        sky = _WMO.get(d["weather_code"][i], "unknown")
        lines.append(
            f"  {d['time'][i]}: {d['temperature_2m_min'][i]}–{d['temperature_2m_max'][i]}°C, "
            f"{sky}, {d['precipitation_probability_max'][i]}% chance of rain."
        )
    return "\n".join(lines)


# Safe arithmetic (no eval) — carried over from the notebook.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval_expr(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_expr(node.operand))
    raise ValueError("Unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '33 - 31'.

    Args:
        expression: The arithmetic expression to evaluate.
    """
    return str(_eval_expr(ast.parse(expression, mode="eval").body))


TOOLS = {"get_weather": get_weather, "get_forecast": get_forecast, "calculate": calculate}


# ── ReAct loop ────────────────────────────────────────────────────────────────
def run_tool(name, args):
    fn = TOOLS.get(name)
    if fn is None:
        return f"Error: no tool called '{name}'."
    try:
        return str(fn(**(args or {})))
    except Exception as e:
        return f"Tool error: {e}"


def ask(question: str, memory: VectorMemory | None = None, verbose: bool = True) -> str:
    """Run the ReAct loop for one question and return the final answer.

    If `memory` is given, relevant past turns are pulled from the vector store
    and injected into the system prompt BEFORE the loop, and this turn is saved
    AFTER it — so the agent remembers things like your home city or unit choice.
    """
    system_prompt = SYSTEM
    if memory is not None:
        past = memory.recall(question)
        if past:
            if verbose:
                print(f"  💭 recalled: {past[0][:80]}...")
            system_prompt += "\n\nWhat the user told you earlier:\n" + "\n".join(f"- {p}" for p in past)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    answer = "Reached the step limit without an answer."
    for _ in range(8):  # step cap to avoid infinite loops
        resp = ollama.chat(model=MODEL, messages=messages, tools=list(TOOLS.values()))
        messages.append(resp.message)

        if not resp.message.tool_calls:
            answer = resp.message.content
            break

        for call in resp.message.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            if verbose:
                print(f"  🔧 {call.function.name}({dict(call.function.arguments or {})}) → {result}")
            messages.append({"role": "tool", "content": result})

    if memory is not None:
        memory.save_context(question, answer)
        if verbose:
            print(f"  💾 saved turn — memory now has {memory.count()} entries")
    return answer


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    # --smart routes to the PGVR agentic loop (region-grounded, see agentic.py)
    if "--smart" in sys.argv:
        import agentic
        rest = [a for a in sys.argv[1:] if a != "--smart"]
        if rest:
            print(agentic.smart_weather(" ".join(rest)))
        else:
            agentic.interactive()
        return

    memory = VectorMemory()  # persists to ./weather_memory.json across runs

    if len(sys.argv) > 1:  # one-shot mode
        print(ask(" ".join(sys.argv[1:]), memory=memory, verbose=False))
        return

    print(f"🌤️  Weather Agent (model: {MODEL})  —  type 'quit' to exit, 'forget' to wipe memory")
    print("It remembers across turns. Try: 'my home city is Pune'  then later  'how's the weather?'")
    print("Other asks: 'forecast for Pune next 5 days', 'is it warmer in Mumbai or Delhi?'\n")
    while True:
        try:
            q = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye 👋")
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "bye"}:
            print("bye 👋")
            break
        if q.lower() == "forget":
            memory.clear()
            print("🧠 memory cleared.\n")
            continue
        print(f"\n{ask(q, memory=memory)}\n")


if __name__ == "__main__":
    main()
