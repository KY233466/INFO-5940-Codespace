# app.py
"""
Multi-Agent Travel Planner

Highlights:
- Clear separation of concerns (tools, agents, orchestration, UI)
- Simple global logger to display tool calls live in the sidebar
- Planner → Reviewer pipeline enforced before rendering any answer
- Minimal dependencies and straightforward control flow
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Callable, Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Globals
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Loads variables from a local .env if present
os.environ.setdefault("OPENAI_LOG", "error")
os.environ.setdefault("OPENAI_TRACING", "false")

# Tool call logger: the UI sets this per request. The tool checks it and logs.
# Using a simple global makes this easy to teach and reason about.
TOOL_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None


def set_tool_logger(logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install or remove the UI logger used by tools to report activity."""
    global TOOL_LOGGER
    TOOL_LOGGER = logger


def log_tool_event(event: Dict[str, Any]) -> None:
    """If a logger is installed, send the event to the UI."""
    if TOOL_LOGGER is not None:
        try:
            TOOL_LOGGER(event)
        except Exception:
            # Logging should never break the app or the tool itself
            pass


def redact_for_logs(value: Any) -> Any:
    """
    Make sure we don't leak secrets and keep logs small.
    This is deliberately simple for teaching.
    """
    if isinstance(value, str):
        low = value.lower()
        if any(k in low for k in ("api_key", "token", "secret", "password")):
            return "[redacted]"
        return value if len(value) <= 300 else value[:120] + "… [truncated]"
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("key", "token", "secret", "password"))
                    else redact_for_logs(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_for_logs(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Agent Framework Imports (provided by you)
# ──────────────────────────────────────────────────────────────────────────────
# These come from your own framework. We assume:
# - Agent: defines a model + instructions + optional tools
# - Runner.run(agent, input): executes an agent and returns an object with text
from agents import Agent, Runner, function_tool  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@function_tool
def internet_search(query: str) -> str:
    """
    Internet search backed by Tavily.
    - Reads TAVILY_API_KEY from environment.
    - Sends simple log events before/after the call so the UI can show activity.
    """
    log_tool_event({"type": "call", "tool": "internet_search", "args": {"query": redact_for_logs(query)}})

    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            msg = "missing TAVILY_API_KEY in environment."
            log_tool_event({"type": "error", "tool": "internet_search", "error": msg})
            return f"Search error: {msg}"

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)

        items = response.get("results", [])
        lines = [f"- {it.get('title', 'N/A')}: {it.get('content', 'N/A')}" for it in items]
        output = "\n".join(lines) if lines else "No results found."

        log_tool_event({
            "type": "result",
            "tool": "internet_search",
            "preview": redact_for_logs(output[:400] + ("…" if len(output) > 400 else "")),
        })
        return output

    except Exception as e:
        log_tool_event({"type": "error", "tool": "internet_search", "error": str(e)})
        return f"Search error: {e}"

    finally:
        log_tool_event({"type": "end", "tool": "internet_search"})


# ──────────────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────────────

# BEGIN SOLUTION
REVIEWER_INSTRUCTIONS = """
Your job is to review, validate, and improve a travel plan.

Your responsibilities:
1. Check feasibility and realism
   - Opening days / hours of key attractions and museums.
   - Whether activities scheduled on the same day are geographically reasonable.
   - Travel times and logistics between cities and neighborhoods (train/flight/bus time, transfers).
   - Very early or very late activities that are unrealistic (e.g., 7am museum visits when they open at 9am).
   - Typical price ranges for major tickets and transportation so that the overall budget is plausible.
   - Spot conflicting, impossible, or highly impractical items (closed attractions, too many hours of transit, unrealistic day pacing, wildly off ticket prices, etc.).

2. Propose fixes to found issues
   - When you find an issue, propose a specific fix, not just a vague comment.
   - Always connect the fix explicitly to the reason (e.g., “museum closed on Mondays”, “budget exceeded”, “travel time too long”).

3. Use tools correctly
   - You CAN and SHOULD use the `internet_search` tool for fact-checking.
   - Keep queries short and targeted: `[city] [attraction] opening hours`, `[city A] to [city B] train time`, `[attraction] ticket price`.
   - Prioritize calling tools for most important checks, whether than repeatedly for trivial information like every single restaurant's opening hours.

Your output should be in markdown with these sections:

1. `### Delta List (required changes)`
   - A concise list of **concrete edits** to the draft.
   - For each change, follow this pattern:
     - `- Day X - [What to change] → [New suggestion] (Reason: …)`
   - Include only changes that materially improve feasibility, safety, or coherence.

2. `### Revised Itinerary (after fixes)`
   - DO NOT change the structure of the plan
   - Rewrite the full itinerary **after applying all deltas**.

Additional guidelines:
- If the original plan is already strong, keep your Delta List short and say so.
- If some information cannot be fully verified, make a **best-effort judgment** and clearly label it as an approximation.
- Never ignore serious feasibility problems just to preserve the original plan.
- Be concise but precise: prioritize clarity and usefulness over excessive detail.
"""

PLANNER_INSTRUCTIONS = """
You are a professional trip planner that helps users with their trip planning! You are friendly and positive, sometimes using
emojis. Your job is to take a vague travel prompt from the user and turn it into a **clear, day-by-day itinerary. 
Focus on making a coherent, exciting, and reasonably realistic plan.

You need to consider:
1. Understand user needs
     - Duration (number of days).
     - Budget (total and—if useful—per day).
     - Main destinations or regions (if specified or strongly implied).
     - Interests (e.g., history, art, food, nightlife, nature, shopping).
     - Pacing preferences (e.g., relaxed vs. packed days, solo traveler vs. family).
   - If any constraint is missing, make a reasonable assumption and clearly state it.

2. Plan around a city or region
   - Decide which city or area the user should be in on each day.
   - Avoid excessive city hopping: cluster nearby cities and minimize backtracking.
   - For multi-city trips, include short notes on how they move between cities (train, bus, flight, etc.).

3. Produce a detailed day-by-day itinerary
   - For each day, include:
     - A short title, like “Day 3 - Rome (Ancient history focus)”, with appropriate emoji
     - A breakdown by time block:
       - Morning: 1-2 main activities (with approximate times).
       - Afternoon: 1-2 main activities.
       - Evening: 1-2 lighter activities (dinner, stroll, viewpoint, etc.).
     - Mention specific neighborhoods / areas and well-known attractions when appropriate.
   - Keep the pacing exlaxing: no more than 3-4 substantial activities per day.

4. Budget and cost estimates
   - Provide rough cost estimates:
     - Major tickets (museums, landmarks, day trips).
     - Inter-city transportation (train/bus/flight).
     - Daily food & local transport estimates.
   - Add a budget summary:
     - Estimated total cost vs. user's stated budget.
     - If the plan is tight or slightly over budget, explain trade-offs and possible savings.

5. Logistics and practical notes
   - Always briefly describe how to get between major points (e.g., “metro + short walk”, “2.5h train from Paris to Lyon”).
   - Call out when it's wise to pre-book tickets (e.g., very popular attractions).
   - Note any important patterns (e.g., “Many museums close on Mondays; schedule museums on other days”).

Your output should be in markdown with these sections:

1. `### Trip Overview`
   - 4-8 bullet points summarizing:
     - Duration and destinations.
     - Main themes (history, food, art, etc.).
     - Overall pacing and style.
     - High-level budget commentary.

2. `### User Constraints & Assumptions`
   - Bullet list of:
     - Stated constraints (copied / paraphrased from the prompt).
     - Any assumptions you had to make (clearly labeled as assumptions).

3. `### Day-by-Day Itinerary`
   - For each day:
     - `#### Day X - City / Area (short theme)`
     - Sub-bullets for Morning / Afternoon / Evening:
       - Time window + activity + neighborhood/area.
       - Short motivation (why it fits the user's interests).
   - Include light logistics notes in-line (e.g., “Walk 15 min to…”, “Take metro line 2 for 10 min”).

4. `### Budget & Logistics Summary`
   - Rough per-day or per-city cost breakdown (only estimates).
   - Highlight big-ticket items (e.g., major day trips or flights).
   - Call out any places where the user may want to swap in cheaper options.
"""

reviewer_agent = Agent(
    name="Reviewer Agent",
    model="openai.gpt-4o",
    instructions=REVIEWER_INSTRUCTIONS.strip(),
    tools=[internet_search],
)

planner_agent = Agent(
    name="Planner Agent",
    model="openai.gpt-4o",
    instructions=PLANNER_INSTRUCTIONS.strip(),
)

# END SOLUTION


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_text(result_obj: Any) -> str:
    """
    Pull a usable string from the Runner result in a tolerant way.
    Your Runner may expose final_output, text, or __str__.
    """
    return (
        getattr(result_obj, "final_output", None)
        or getattr(result_obj, "text", None)
        or str(result_obj)
    )


def run_planner(user_text: str) -> str:
    """Run the Planner and return its itinerary text."""
    result = asyncio.run(Runner.run(planner_agent, user_text))
    return extract_text(result)


def run_reviewer(plan_text: str) -> str:
    """Run the Reviewer on the planner's output and return validated text."""
    result = asyncio.run(Runner.run(reviewer_agent, plan_text))
    return extract_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Travel Planner", page_icon="✈️")

st.title("✈️ Multi-Agent Travel Planner")
st.caption("Planner → Reviewer (with live tool calls in the sidebar)")

# Sidebar: session controls + examples + dev panel
with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Try these prompts")
    st.code("Plan a week-long Europe trip for a student on a $1,500 budget who loves history and food")
    st.code("3-day Paris trip for art lovers with $800 budget")

    st.subheader("Developer view")
    show_tools = st.toggle("Show tool activity (live)", value=True)
    if show_tools:
        tool_expander = st.expander("🔧 Tool activity", expanded=True)
        tool_panel = tool_expander.container()
    else:
        tool_panel = st.container()  # inert sink

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]
if "meta" not in st.session_state:
    st.session_state.meta = []      # list[dict(trace)]

# Render history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.meta):
            meta = st.session_state.meta[i]
            if meta:
                st.caption(meta.get("trace", ""))

# Chat input
user_input = st.chat_input("Describe your travel (destination, duration, budget, interests)…")

if user_input:
    # Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.meta.append(None)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant output block
    with st.chat_message("assistant"):
        # Live “working…” text and progress bar
        live_msg = st.empty()
        progress = st.progress(0)

        # Per-request tool log (shown in the sidebar)
        tool_events: List[Dict[str, Any]] = []

        def ui_tool_logger(event: Dict[str, Any]) -> None:
            """Append an event and re-render the sidebar log."""
            tool_events.append(event)
            with tool_panel:
                st.markdown("**Recent tool calls**")
                for ev in tool_events[-60:]:  # last N entries
                    t = ev.get("tool", "unknown")
                    et = ev.get("type", "event")
                    if et == "call":
                        st.write(f"• **{t}** called with `{ev.get('args')}`")
                    elif et == "result":
                        st.write(f"• **{t}** result preview:\n\n> {ev.get('preview')}")
                    elif et == "error":
                        st.error(f"• **{t}** error: {ev.get('error')}")
                    elif et == "end":
                        st.write(f"• **{t}** finished")

        # Install the logger so tools can report to the sidebar
        set_tool_logger(ui_tool_logger)

        try:
            # Optional: clear sidebar panel on each run
            with tool_panel:
                st.empty()

            # Step 1: Planner
            with st.status("🧭 Planner Agent: generating itinerary…", expanded=True) as status:
                live_msg.markdown("🧭 Planner Agent is creating your itinerary…")
                plan_text = run_planner(user_input)
                progress.progress(40)
                status.update(label="🔎 Reviewer Agent: validating with live searches…", state="running")

            # Step 2: Reviewer (tool calls will appear live in sidebar)
            live_msg.markdown("🔎 Reviewer Agent is validating the plan with live searches…")
            review_text = run_reviewer(plan_text)
            progress.progress(90)

            # Completed
            live_msg.markdown("✅ Validation complete. Rendering results…")
            time.sleep(0.2)
            progress.progress(100)

            # Final render: show only the validated result, with the raw plan expandable
            st.info("🤖 **Reviewer Agent** (validated)")
            st.markdown(review_text)
            with st.expander("See raw plan from Planner Agent"):
                st.markdown(plan_text)

            # Save only the validated result to history
            st.session_state.messages.append({"role": "assistant", "content": review_text})
            st.session_state.meta.append({"trace": "Planner Agent → Reviewer Agent"})
            st.caption("Planner Agent → Reviewer Agent")

        except Exception as e:
            # Friendly error box
            live_msg.markdown("❌ Something went wrong.")
            err = f"⚠️ Error while processing your request:\n\n```\n{e}\n```"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.session_state.meta.append({"trace": "Runtime error."})

        finally:
            # Always remove the logger so it doesn't leak into the next request
            set_tool_logger(None)
