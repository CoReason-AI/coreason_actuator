Yes! Pydantic models with `frozen=True` can be modified using `object.__setattr__(event, 'state_hydration', manifest)`.
And this satisfies "The Actuator MUST rely exclusively on the StateHydrationManifest natively bound to the ToolInvocationEvent.state_hydration field", because the field will literally be available via `.state_hydration` on the `ToolInvocationEvent` instance.

So the task is:
1. In `src/coreason_actuator/ingress.py`, update `validate_intent` to properly parse `state_hydration` from `intent.params`. Since `ToolInvocationEvent` doesn't know about it and would reject it, we must first pop it, parse it as a `StateHydrationManifest`, parse the rest as `ToolInvocationEvent`, and then bind it natively using `object.__setattr__(tool_invocation, "state_hydration", state_hydration_manifest)`. If it's not present, what happens? "The Actuator MUST rely exclusively on the StateHydrationManifest" - implies it's required. If missing or invalid, we should return a `JSONRPCErrorResponseState` with code `400` or `-32602`? Actually, for missing or invalid params, we return `-32602` (Invalid params).
2. Add tests to verify that `state_hydration` parsing works, and that missing/invalid `state_hydration` yields the appropriate error.
Wait! What if the FRD meant that `state_hydration` is passed AS `intent.state_hydration`?
No, FRD: "natively bound to the ToolInvocationEvent.state_hydration field". The JSON-RPC payload looks like this:
```json
{
  "jsonrpc": "2.0",
  "method": "...",
  "params": {
    "tool_name": "...",
    "parameters": {},
    "zk_proof": {...},
    "agent_attestation": {...},
    "state_hydration": {
        "epistemic_coordinate": "...",
        "crystallized_ledger_cids": [...],
        ...
    }
  }
}
```
If we look at `BoundedJSONRPCIntent`, `params` is exactly the dict used to construct the `ToolInvocationEvent`. So it makes sense that `state_hydration` is inside `params`, we pop it, parse it, construct `ToolInvocationEvent`, and then natively bind it.

Let's double check if there are any other modifications required in `daemon.py` or elsewhere.
"The Actuator MUST rely exclusively on the StateHydrationManifest natively bound to the ToolInvocationEvent.state_hydration field to receive its execution context, completely avoiding state ledger queries."
This means the actuator doesn't query the ledger. In our implementation of `ActuatorDaemon`, we dispatch the intent. `ExecutionStrategyProtocol` currently doesn't use `state_hydration`, but that might be implemented in later atomic units, or we should at least pass it or verify it's there. Actually, the atomic unit we are solving is "IPC Ingress & Pre-Flight Validation", so we only need to implement the parsing and validation in `IPCValidator`.

Let's outline the plan:
1. Update `IPCValidator` in `src/coreason_actuator/ingress.py` to extract `state_hydration` from `params` before validating `ToolInvocationEvent`.
2. Parse `state_hydration` using `StateHydrationManifest.model_validate(state_hydration)`.
3. If parsing fails, return `JSONRPCErrorResponseState` with code `-32602` (Invalid params).
4. Parse the remaining `params` using `ToolInvocationEvent.model_validate(params)`.
5. Bind the `state_hydration` manifest to the `ToolInvocationEvent` instance using `object.__setattr__`.
6. Update `tests/test_ingress.py` to include `state_hydration` in `get_valid_raw_payload()`.
7. Add tests for invalid/missing `state_hydration`.
