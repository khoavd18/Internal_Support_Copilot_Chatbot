# Action Notes

## Why This Exists

Write actions now go through a small registry so the project does not have to keep the same action metadata in multiple places.

Each registered action defines:

- `name`
- `matcher` for chat detection and parsing
- `permission_name`
- `tool_name`
- `confirmation_required`
- `executor`

The source of truth lives in `src/agent/action_registry.py`.

## Current Flow

### Chat-driven actions

1. `detect_action_request(...)` walks the registry in order.
2. The first matching action returns an `ActionIntent`.
3. `maybe_handle_chat_action(...)` uses the registry again for confirmation requirements and execution.
4. Pending confirmation and idempotency still use the existing persistence layer in `src/agent/chat_actions.py` and `src/agent/actions.py`.

### Direct action endpoints

1. FastAPI endpoints build a payload from the request model.
2. `src/api/main.py` calls `execute_registered_action(...)`.
3. The registry dispatches to the registered executor.

## Files To Know

- `src/agent/action_registry.py`: registry, chat matchers, executor adapters
- `src/agent/actions.py`: concrete write-action implementations and idempotent execution state machine
- `src/agent/chat_actions.py`: confirmation flow and chat orchestration
- `src/core/auth.py`: permission policies
- `tests/test_chat_actions.py`: parser and registry regression coverage

## Add A New Action

1. Add the concrete executor in `src/agent/actions.py` if it does not already exist.
2. Add a matcher/parser function in `src/agent/action_registry.py`.
3. Add an executor adapter in `src/agent/action_registry.py` that forwards the normalized payload into the concrete action function.
4. Register an `ActionDefinition(...)` in `ACTION_DEFINITIONS`.
5. Add an authorization policy in `src/core/auth.py` if the new action needs its own permission name.
6. If the action should be callable directly over HTTP, add a request model in `src/core/schema.py` and an endpoint in `src/api/main.py`.
7. Add tests for:
   - successful parsing
   - incomplete input clarification
   - confirmation behavior
   - safe execution or idempotent replay

## Practical Guidelines

- Keep matchers explicit and narrow. Broad keywords make accidental write detection more likely.
- Prefer returning `needs_clarification=True` over guessing missing write fields.
- Keep executor adapters thin. Validation and side-effect behavior should stay in `src/agent/actions.py`.
- Reuse existing permission names when the new action belongs to an existing risk boundary.
