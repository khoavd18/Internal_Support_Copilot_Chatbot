# Routing Notes

## Why This Was Refactored

The original routing flow worked, but it was hard to extend safely because:

- single-agent routing signals lived inline inside `src/agent/router.py`
- supervisor keyword rules lived separately inside `src/agent/graph/nodes/route.py`
- matching, scoring, and final dispatch decisions were mixed together
- mode overrides and short-query fallback rules were repeated in different places

The current structure keeps the existing behavior largely intact while making route changes easier to review and test.

## Current Structure

- `src/agent/route_config.py`: route definitions, keyword lists, priorities, and default routing constants
- `src/agent/router.py`: normalization, rule matching, scoring, and final decision helpers
- `src/agent/service.py`: single-agent mode dispatch through `decide_route_for_mode(...)`
- `src/agent/graph/nodes/route.py`: supervisor node that calls `select_supervisor_agents(...)` and `decide_supervisor_dispatch(...)`
- `tests/test_routing.py`: regression coverage for ambiguous queries, forced modes, and short-query edge cases

## Single-Agent Routing Flow

1. `normalize_question(...)` prepares the raw, lowercased, and compact forms.
2. `match_route_rules(...)` finds matching route rules from `RETRIEVE_ONLY_ROUTE_RULES`.
3. `score_route_matches(...)` ranks matches by score, then priority.
4. `decide_route(...)` applies empty-question and short-question guards, then picks the highest-ranked route.
5. `decide_route_for_mode(...)` applies the explicit `answer` or `search` override without duplicating the base logic.

## Supervisor Routing Flow

1. `match_supervisor_agents(...)` checks each configured domain definition in `SUPERVISOR_DOMAIN_ROUTE_DEFINITIONS`.
2. `score_supervisor_agent_matches(...)` ranks domain matches by score, then priority.
3. `select_supervisor_agents(...)` returns the ordered agent list plus match metadata for logging and debugging.
4. `decide_supervisor_dispatch(...)` decides whether the final response should clarify, retrieve only, or answer from the merged evidence.

The supervisor state now also carries `route_scores` and `route_matches` so routing behavior is easier to inspect in logs and debug responses.

## Add A New Retrieve-Only Route

1. Add a `RetrieveOnlyRouteRule(...)` entry in `src/agent/route_config.py`.
2. Pick a stable `name` and a `priority`. Lower numbers win ties.
3. Add or update tests in `tests/test_routing.py` for the new signals and any collisions with existing rules.

If the route needs behavior beyond the current `retrieve_only` or `answer_from_kb` decisions, add the new route name in `RouteName` first, then update the callers that consume `RouteDecision`.

## Add A New Supervisor Sub-Agent

1. Extend `SupervisorAgentName` in `src/agent/route_config.py`.
2. Add a `SupervisorDomainRouteDefinition(...)` entry with keywords, description, and priority.
3. Wire the execution node into `src/agent/graph/supervisor.py`.
4. Extend `SupervisorState` in `src/agent/graph/state.py` if the new worker returns its own result field.
5. Update merge or synthesis code if the new worker contributes evidence in a different shape.
6. Add routing tests for ambiguous queries so the ordering stays stable.

Adding a config entry alone only affects matching. The new sub-agent still needs a graph node and result handling before it can execute.

## Practical Guardrails

- Keep route keywords small and high-signal. Broad keywords make ambiguous queries noisy.
- Prefer adding tests for ties and short queries whenever you change priorities.
- Preserve explicit mode overrides unless the API contract changes intentionally.
- Treat `route_config.py` as the source of truth for routing definitions, not the graph node.
