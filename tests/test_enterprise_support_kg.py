from __future__ import annotations

from pathlib import Path

from src.data.enterprise_support_loader import load_enterprise_support_dataset
from src.kg.builder import build_graph_from_enterprise_support_dataset
from src.kg.retriever import retrieve_graph_context, search_nodes
from src.kg.store import get_neighbors

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_enterprise_support"


def _graph():
    dataset = load_enterprise_support_dataset(DATA_DIR)
    return build_graph_from_enterprise_support_dataset(dataset)


def test_graph_builds_with_expected_node_types() -> None:
    graph = _graph()

    assert len(graph.nodes) >= 240
    assert len(graph.edges) >= 300
    assert graph.nodes["Customer:cust_001"].type == "Customer"
    assert graph.nodes["Account:acct_001"].label == "Northstar Analytics"
    assert graph.nodes["Product:prod_api"].label == "Developer API"
    assert graph.nodes["Ticket:tkt_001"].label == "API timeout during batch sync"
    assert graph.nodes["TicketMessage:msg_001"].type == "TicketMessage"
    assert graph.nodes["Policy:pol_sla"].label == "Enterprise SLA Policy"
    assert graph.nodes["Service:svc_api_gateway"].label == "API Gateway"
    assert graph.nodes["GitHubIssue:gh_001"].label == "Gateway timeout under batch sync load"
    assert graph.nodes["RiskEvent:risk_001"].type == "RiskEvent"
    assert graph.nodes["Team:eng_api_platform"].type == "Team"


def test_graph_builds_expected_edges() -> None:
    graph = _graph()

    assert graph.edge_exists("Customer:cust_001", "Account:acct_001", "HAS_ACCOUNT")
    assert graph.edge_exists("Customer:cust_001", "Ticket:tkt_001", "CREATED_TICKET")
    assert graph.edge_exists("Ticket:tkt_001", "TicketMessage:msg_001", "HAS_MESSAGE")
    assert graph.edge_exists("Ticket:tkt_001", "Product:prod_api", "MENTIONS_PRODUCT")
    assert graph.edge_exists("Ticket:tkt_001", "Service:svc_api_gateway", "AFFECTS_SERVICE")
    assert graph.edge_exists("Service:svc_api_gateway", "Team:eng_api_platform", "OWNED_BY_TEAM")
    assert graph.edge_exists("Ticket:tkt_001", "GitHubIssue:gh_001", "RELATED_TO_ISSUE")
    assert graph.edge_exists("Ticket:tkt_001", "RiskEvent:risk_001", "HAS_RISK_EVENT")
    assert graph.edge_exists("Ticket:tkt_003", "Policy:pol_refund", "HAS_RESOLUTION")
    assert graph.edge_exists("Ticket:tkt_003", "Policy:pol_refund", "REFERENCES_POLICY")


def test_traversal_returns_neighbors_across_incoming_and_outgoing_edges() -> None:
    _graph()

    neighbor_ids = {node.id for node in get_neighbors("Ticket:tkt_001", depth=1)}

    assert "Customer:cust_001" in neighbor_ids
    assert "Product:prod_api" in neighbor_ids
    assert "Service:svc_api_gateway" in neighbor_ids
    assert "TicketMessage:msg_001" in neighbor_ids
    assert "GitHubIssue:gh_001" in neighbor_ids
    assert "RiskEvent:risk_001" in neighbor_ids


def test_search_nodes_returns_relevant_matches() -> None:
    graph = _graph()

    results = search_nodes("api timeout batch sync", graph=graph, limit=5)
    result_ids = [node.id for node in results]

    assert "Ticket:tkt_001" in result_ids
    assert any(node.type in {"GitHubIssue", "Policy", "Service"} for node in results)


def test_retrieve_graph_context_returns_useful_text() -> None:
    graph = _graph()

    context = retrieve_graph_context("api timeout northstar escalation", graph=graph, depth=2)

    assert context.matched_nodes
    assert context.context_nodes
    assert context.context_edges
    assert "Graph Context" in context.text
    assert "Ticket:tkt_001" in context.text
    assert "Northstar Analytics" in context.text
    assert "API Gateway" in context.text
    assert "Enterprise SLA Policy" in context.text
