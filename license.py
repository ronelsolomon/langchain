"""
License Corporation — LangGraph Multi-Agent System
====================================================
Architecture: StateGraph with Human-in-the-Loop (HITL)

Agents:
  Node A → Discovery Agent   : Finds businesses in a zip code
  Node B → Cross-Reference Agent : Checks licensing DB for compliance
  Node C → Human-in-the-Loop     : City official approval gate
  Node D → Enforcement Agent     : Generates violation notices

Requirements:
    pip install langgraph langchain langchain-ollama langchain-community \
                tavily-python sqlalchemy pydantic langsmith python-dotenv
"""

import os
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, Field

load_dotenv()

# ─────────────────────────────────────────────
# 1. SCHEMA — What a License object looks like
# ─────────────────────────────────────────────

class License(BaseModel):
    business_name: str
    jurisdiction: str
    license_type: str
    license_number: str | None = None
    expiry_date: str | None = None
    fee_owed: float = 0.0
    required_docs: list[str] = Field(default_factory=list)
    status: Literal["active", "expired", "missing", "pending"] = "pending"


class BusinessRecord(BaseModel):
    name: str
    address: str
    zip_code: str
    business_type: str
    detected_source: str  # e.g. "google_maps", "county_registry"


# ─────────────────────────────────────────────
# 2. SHARED STATE — Flows through every node
# ─────────────────────────────────────────────

class LicenseState(TypedDict):
    # Conversation / reasoning trace
    messages: Annotated[list, add_messages]

    # Inputs
    target_zip: str
    jurisdiction: str

    # Node A output
    detected_businesses: list[dict]

    # Node B output
    non_compliant: list[dict]
    compliant_count: int

    # Node C output
    human_approved: bool
    approver_notes: str

    # Node D output
    enforcement_notices: list[dict]

    # Meta
    audit_id: str
    run_timestamp: str


# ─────────────────────────────────────────────
# 3. TOOLS — Wrapped for LangChain agents
# ─────────────────────────────────────────────

@tool
def search_businesses_in_zip(zip_code: str, jurisdiction: str) -> list[dict]:
    """
    Discovers businesses operating in a given zip code.
    In production: swap mock data for TavilySearchResults + Google Maps API.
    """
    # --- MOCK DATA (replace with real API calls) ---
    mock_businesses = [
        {"name": "Sunrise Bakery LLC",     "address": f"101 Main St, {zip_code}", "zip_code": zip_code, "business_type": "Food Service",  "detected_source": "google_maps"},
        {"name": "QuickFix Auto Repair",   "address": f"245 Oak Ave, {zip_code}", "zip_code": zip_code, "business_type": "Automotive",    "detected_source": "county_registry"},
        {"name": "Greenleaf Landscaping",  "address": f"389 Pine Rd, {zip_code}", "zip_code": zip_code, "business_type": "Landscaping",   "detected_source": "google_maps"},
        {"name": "NightOwl Bar & Grill",   "address": f"512 Elm St, {zip_code}",  "zip_code": zip_code, "business_type": "Food/Beverage", "detected_source": "google_maps"},
        {"name": "TechNow Electronics",    "address": f"678 Cedar Blvd, {zip_code}", "zip_code": zip_code, "business_type": "Retail",    "detected_source": "county_registry"},
    ]
    return mock_businesses


@tool
def check_license_database(business_name: str, jurisdiction: str) -> dict:
    """
    Queries the city licensing database for a given business.
    In production: connect via SQLDatabaseChain to the real licensing DB.
    """
    # --- MOCK DATABASE (replace with SQLAlchemy query) ---
    licensed_businesses = {
        "Sunrise Bakery LLC":   {"status": "active",   "license_number": "FB-2023-0041", "expiry_date": "2026-01-15", "fee_owed": 0.0},
        "TechNow Electronics":  {"status": "expired",  "license_number": "RET-2021-0199", "expiry_date": "2023-06-30", "fee_owed": 350.0},
    }
    record = licensed_businesses.get(business_name)
    if record:
        return {**record, "business_name": business_name, "found": True}
    return {
        "business_name": business_name,
        "found": False,
        "status": "missing",
        "fee_owed": 500.0,   # Default penalty for unlicensed operation
        "license_number": None,
        "expiry_date": None,
    }


@tool
def generate_enforcement_notice(business: dict, jurisdiction: str) -> dict:
    """Generates a formal enforcement notice for a non-compliant business."""
    return {
        "notice_id": f"ENF-{datetime.now().strftime('%Y%m%d%H%M%S')}-{business['name'][:4].upper()}",
        "business_name": business["name"],
        "address": business.get("address", "Unknown"),
        "violation_type": "Unlicensed/Expired Business Operation",
        "fee_owed": business.get("fee_owed", 500.0),
        "issued_date": datetime.now().strftime("%Y-%m-%d"),
        "response_deadline": "30 days from issue date",
        "jurisdiction": jurisdiction,
        "status": "issued",
    }


# ─────────────────────────────────────────────
# 4. LLM SETUP
# ─────────────────────────────────────────────

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

tools = [search_businesses_in_zip, check_license_database, generate_enforcement_notice]
# Note: ChatOllama may not support bind_tools directly
# Tools will be used explicitly in the agent functions


# ─────────────────────────────────────────────
# 5. NODES — The agents
# ─────────────────────────────────────────────

def discovery_agent(state: LicenseState) -> dict:
    """
    Node A: Searches for businesses in the target zip code.
    Uses the search_businesses_in_zip tool.
    """
    print(f"\n[Node A] 🔍 Discovery Agent — scanning ZIP {state['target_zip']}...")

    result = search_businesses_in_zip.invoke({
        "zip_code": state["target_zip"],
        "jurisdiction": state["jurisdiction"],
    })

    message = AIMessage(
        content=f"Discovery complete. Found {len(result)} businesses in {state['target_zip']}."
    )
    print(f"[Node A] ✅ Found {len(result)} businesses.")
    return {
        "detected_businesses": result,
        "messages": [message],
    }


def cross_reference_agent(state: LicenseState) -> dict:
    """
    Node B: Cross-references detected businesses against the licensing DB.
    """
    print(f"\n[Node B] 🗂  Cross-Reference Agent — checking {len(state['detected_businesses'])} businesses...")

    non_compliant = []
    compliant_count = 0

    for biz in state["detected_businesses"]:
        db_result = check_license_database.invoke({
            "business_name": biz["name"],
            "jurisdiction": state["jurisdiction"],
        })
        if db_result["status"] in ("missing", "expired"):
            non_compliant.append({**biz, **db_result})
            print(f"[Node B]   ❌ NON-COMPLIANT: {biz['name']} ({db_result['status']})")
        else:
            compliant_count += 1
            print(f"[Node B]   ✅ Compliant: {biz['name']}")

    summary = (
        f"Cross-reference complete. "
        f"Non-compliant: {len(non_compliant)}, Compliant: {compliant_count}."
    )
    return {
        "non_compliant": non_compliant,
        "compliant_count": compliant_count,
        "messages": [AIMessage(content=summary)],
    }


def human_approval_gate(state: LicenseState) -> dict:
    """
    Node C: Human-in-the-Loop — pauses graph for city official approval.
    LangGraph interrupt() suspends execution; resume by passing human input.
    """
    print(f"\n[Node C] 🧑‍⚖️  Human Approval Gate — waiting for official review...")

    non_compliant_names = [b["name"] for b in state["non_compliant"]]
    prompt_text = (
        f"\n{'='*55}\n"
        f"  AUDIT ID   : {state['audit_id']}\n"
        f"  ZIP CODE   : {state['target_zip']}\n"
        f"  JURISDICTION: {state['jurisdiction']}\n"
        f"  NON-COMPLIANT BUSINESSES ({len(non_compliant_names)}):\n"
        + "\n".join(f"    • {n}" for n in non_compliant_names) +
        f"\n{'='*55}\n"
        f"Approve enforcement action? (yes/no) + optional notes:\n> "
    )

    # LangGraph interrupt — suspends the graph here until resumed
    human_response: str = interrupt(prompt_text)

    approved = human_response.strip().lower().startswith("yes")
    notes = human_response.strip()

    print(f"[Node C] {'✅ Approved' if approved else '❌ Rejected'} by official.")
    return {
        "human_approved": approved,
        "approver_notes": notes,
        "messages": [HumanMessage(content=f"Official decision: {notes}")],
    }


def enforcement_agent(state: LicenseState) -> dict:
    """
    Node D: Generates formal enforcement notices for approved non-compliant businesses.
    """
    print(f"\n[Node D] 📋 Enforcement Agent — generating notices...")

    notices = []
    for biz in state["non_compliant"]:
        notice = generate_enforcement_notice.invoke({
            "business": biz,
            "jurisdiction": state["jurisdiction"],
        })
        notices.append(notice)
        print(f"[Node D]   📄 Notice issued: {notice['notice_id']} → {biz['name']}")

    summary = f"Enforcement complete. {len(notices)} notices issued."
    return {
        "enforcement_notices": notices,
        "messages": [AIMessage(content=summary)],
    }


def rejection_handler(state: LicenseState) -> dict:
    """Handles cases where the human official rejects the enforcement action."""
    print("\n[Node C] ⛔ Enforcement action rejected by official. Audit closed.")
    return {
        "enforcement_notices": [],
        "messages": [AIMessage(content=f"Audit {state['audit_id']} closed without enforcement. Notes: {state['approver_notes']}")],
    }


# ─────────────────────────────────────────────
# 6. EDGES — Routing logic between nodes
# ─────────────────────────────────────────────

def route_after_cross_reference(state: LicenseState) -> str:
    """Skip HITL if no violations found."""
    if not state["non_compliant"]:
        print("[Router] No violations found — skipping to END.")
        return "end_clean"
    return "human_approval_gate"


def route_after_human_approval(state: LicenseState) -> str:
    """Branch based on official's decision."""
    return "enforcement_agent" if state["human_approved"] else "rejection_handler"


# ─────────────────────────────────────────────
# 7. BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_license_hunter_graph() -> StateGraph:
    graph = StateGraph(LicenseState)

    # Register nodes
    graph.add_node("discovery_agent",       discovery_agent)
    graph.add_node("cross_reference_agent", cross_reference_agent)
    graph.add_node("human_approval_gate",   human_approval_gate)
    graph.add_node("enforcement_agent",     enforcement_agent)
    graph.add_node("rejection_handler",     rejection_handler)

    # Wire the edges
    graph.add_edge(START, "discovery_agent")
    graph.add_edge("discovery_agent", "cross_reference_agent")

    graph.add_conditional_edges(
        "cross_reference_agent",
        route_after_cross_reference,
        {"human_approval_gate": "human_approval_gate", "end_clean": END},
    )

    graph.add_conditional_edges(
        "human_approval_gate",
        route_after_human_approval,
        {"enforcement_agent": "enforcement_agent", "rejection_handler": "rejection_handler"},
    )

    graph.add_edge("enforcement_agent", END)
    graph.add_edge("rejection_handler", END)

    return graph


# ─────────────────────────────────────────────
# 8. RUNNER — Persistence via MemorySaver
# ─────────────────────────────────────────────

def run_license_audit(
    target_zip: str,
    jurisdiction: str,
    human_decision: str = "yes — all violations confirmed by audit team",
) -> dict:
    """
    Runs the full License Hunter multi-agent pipeline.

    Args:
        target_zip    : ZIP code to audit (e.g. "90210")
        jurisdiction  : City/county name (e.g. "Los Angeles, CA")
        human_decision: Simulated official response (in prod, this comes from a UI)

    Returns:
        Final LicenseState dict
    """
    # MemorySaver = in-process checkpointer (swap for SqliteSaver/PostgresSaver in prod)
    checkpointer = MemorySaver()
    graph = build_license_hunter_graph()
    compiled = graph.compile(checkpointer=checkpointer, interrupt_before=["human_approval_gate"])

    audit_id = f"AUDIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    thread_config = {"configurable": {"thread_id": audit_id}}

    initial_state: LicenseState = {
        "messages": [SystemMessage(content="License Hunter audit initialized.")],
        "target_zip": target_zip,
        "jurisdiction": jurisdiction,
        "detected_businesses": [],
        "non_compliant": [],
        "compliant_count": 0,
        "human_approved": False,
        "approver_notes": "",
        "enforcement_notices": [],
        "audit_id": audit_id,
        "run_timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'='*55}")
    print(f"  License Hunter — Starting Audit {audit_id}")
    print(f"{'='*55}")

    # Phase 1: Run until HITL interrupt
    for event in compiled.stream(initial_state, config=thread_config):
        pass  # Nodes print their own status

    # Phase 2: Resume with human decision
    print(f"\n[System] Resuming with human decision: '{human_decision}'")
    for event in compiled.stream(
        {"human_approved": None, "messages": [HumanMessage(content=human_decision)]},
        config=thread_config,
        # Pass the human response to the interrupted node
    ):
        pass

    # Return final state snapshot
    final_state = compiled.get_state(config=thread_config).values

    # ── Print Summary ──
    print(f"\n{'='*55}")
    print(f"  AUDIT COMPLETE — {audit_id}")
    print(f"{'='*55}")
    print(f"  ZIP Scanned    : {final_state['target_zip']}")
    print(f"  Jurisdiction   : {final_state['jurisdiction']}")
    print(f"  Businesses Found: {len(final_state['detected_businesses'])}")
    print(f"  Compliant       : {final_state['compliant_count']}")
    print(f"  Non-Compliant   : {len(final_state['non_compliant'])}")
    print(f"  Notices Issued  : {len(final_state['enforcement_notices'])}")
    if final_state["enforcement_notices"]:
        print(f"\n  Notices:")
        for n in final_state["enforcement_notices"]:
            print(f"    • {n['notice_id']} | {n['business_name']} | ${n['fee_owed']:.2f}")
    print(f"{'='*55}\n")

    return final_state


# ─────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Example: audit ZIP 90210 in Los Angeles
    final = run_license_audit(
        target_zip="90210",
        jurisdiction="Los Angeles, CA",
        human_decision="yes — violations confirmed, proceed with enforcement",
    )