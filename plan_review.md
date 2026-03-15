# Plan Review
## Goal:
Implement Atomic Unit 1: IPC Ingress & Pre-Flight Validation for the Kinetic Actuator Engine (`coreason_actuator`).

## Atomic Unit 1 Scope:
1.  **IPC Daemon Subscription (FR-1.1):**
    *   Since the engine is designed as an IPC daemon, we need a service/class that subscribes to a designated IPC message broker. The `TRD` mentions `asyncio` polling an IPC message queue (`Redis`, `RabbitMQ`, or `gRPC stream`).
    *   For the scope of this atomic unit, we can create an abstract `IPCBroker` protocol and an `ActuatorDaemon` class that continuously polls the broker using an `asyncio` event loop. We can implement an in-memory or a basic mock queue for testing.
2.  **Backpressure & Load Shedding (FR-1.2):**
    *   The `ActuatorDaemon` must enforce `BackpressurePolicy.max_concurrent_tool_invocations`. If the internal execution pool is saturated, it MUST yield the payload back to the broker or return an immediate `JSONRPCErrorResponseState` indicating a `429 Too Many Requests` pre-flight rejection to prevent daemon crash.
    *   *Implementation detail:* Maintain a `current_concurrent_invocations` counter and compare it against `max_concurrent_invocations`.
3.  **Cryptographic Intent Authorization (FR-1.3):**
    *   Upon dequeuing a `BoundedJSONRPCIntent` containing a `ToolInvocationEvent`, mathematically verify the `zk_proof` and `agent_attestation`.
    *   *Implementation detail:* We'll need a verifier component. For this unit, we will create a `CryptographicVerifier` protocol/service with a `verify(intent)` method. Since we can't implement real ZK proof verification without actual cryptography libraries (and it might be out of scope for just this package), we will create a mock-able structure that enforces the presence and formatting, and allows injecting a real verifier later or simulating rejection for invalid proofs. Wait, `FR-1.3` says: "mathematically verify the zk_proof... If the proofs are invalid or missing, it MUST reject the payload and quarantine the event."
4.  **Topological Registry Verification (FR-1.4):**
    *   Verify that `tool_name` exists within its currently mounted `ActionSpaceManifest` (or a `ToolRegistry`).
    *   *Implementation detail:* Create a `ToolRegistry` that holds `ToolManifest` definitions. If absent, emit `JSONRPCErrorResponseState` indicating tool is missing.
5.  **Dual-Evaluation Permission Boundary (FR-1.5):**
    *   Verify `ToolManifest.permissions.network_access` AND `EphemeralNamespacePartitionState.allow_network_egress` evaluate to True for network socket authorization.
    *   *Implementation detail:* Create a validation step that checks these two conditions. *Correction*: This permission check happens *during* or *before* execution, but the prompt says it's part of Pre-Flight validation. However, `EphemeralNamespacePartitionState` is created *during* Module 2 (Sandbox Provisioning). The FR says: "To authorize a network socket, the Actuator MUST functionally verify that BOTH... evaluate to True. A conflict MUST result in mathematical blocking of the execution." This feels slightly more related to execution or sandbox provisioning. Let's include a validation function for this, but maybe it's called later. Let's focus on the initial Ingress components.

## Proposed Steps for Implementation:
1.  **Define Core Interfaces (`src/coreason_actuator/interfaces.py` or similar):**
    *   `IPCBrokerProtocol`
    *   `ToolRegistryProtocol`
    *   `CryptographicVerifierProtocol`
2.  **Implement the IPC Ingress Service (`src/coreason_actuator/ingress.py`):**
    *   Create a class `IPCValidator` that encapsulates FR-1.3 (Cryptographic) and FR-1.4 (Registry) logic.
    *   Implement methods to take a raw payload (from broker), deserialize it using `coreason-manifest` models (`ToolInvocationEvent`), and perform the validations.
3.  **Implement the Daemon Loop (`src/coreason_actuator/daemon.py`):**
    *   Create `ActuatorDaemon` class.
    *   Implement an asynchronous loop `async def run()`.
    *   Implement backpressure logic (FR-1.2).
4.  **Write Tests (`tests/test_ingress.py`, `tests/test_daemon.py`):**
    *   Test successful validation.
    *   Test failure on invalid ZK proof.
    *   Test failure on missing tool.
    *   Test backpressure (429 Too Many Requests response).
5.  **Pre-commit verification.**

I will now request a review of this plan.
