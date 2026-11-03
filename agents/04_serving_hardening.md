# 04. `S1` -- close the serving wrapper's real gaps (and refuse the fake ones)

## Goal

Make `deploy/server.py` honest under more than one client and under a client that asks for
something unreasonable, without turning a local measurement rig into a production web service.

## What is already closed (2026-11-02, do not redo)

- Inference is serialized by a lock: two requests on one CUDA context would otherwise interleave and
  report the second one's queueing as the policy's latency [V, `server.PolicyServer.inference_lock`].
- `Content-Length` is validated and capped at 32 MiB; a body above it is refused unread with 413
  [V, `MAX_BODY_BYTES`, `test_an_oversized_body_is_refused_from_its_length_header`].
- The wire contract has named cameras, a language slot, and 400-with-message on every shape error
  [V, 12 contract tests].

## Status (2026-11-02)

**Steps 1-3 done**, with socket-level tests in `tests/unit/test_deploy_server.py` (`steps: 0` and
`10**6` are 400s naming the bound, a met deadline is echoed, a missed one flags and still returns
the actions, and `shutdown()` waits 0.4 s of in-flight inference before closing). **Step 4 is open
by design**: a `LICENSE` and a version above `0.0.1` are the owner's call, so the repository still
has no licence and that is written down rather than assumed.

## Non-goals -- deliberately not done

- **Auth, TLS, multi-tenancy.** This server binds `127.0.0.1` for a measurement rig; adding auth
  would imply a deployment story the repo does not claim, and a half-done one is worse than none.
- **Batching.** No evidence anyone needs it, and it would change what `latency_ms` means -- which is
  the number the endpoint exists to report.
- **A web framework.** Standard library only, on the documented rule that dependencies are named
  before they land.

## Steps

| # | Work | Acceptance |
|---|---|---|
| 1 | Bound `steps` to `[1, max_steps]` -- the sampler's own schedule, 100 here -- with a 400 naming the bound, and advertise it from `/health`, instead of trusting the payload | unit test: `steps=0` and `steps=10**6` are 400s; a legal value still reaches the policy; a server constructed with a step count outside its own bound refuses to start |
| 2 | An optional `deadline_ms` in the payload: if the policy's own elapsed time passes it, the response carries `deadline_missed: true` (the action is still returned -- refusing it would be a different contract) | unit test with a stub that sleeps; `latency_ms` and the flag agree |
| 3 | Graceful shutdown: stop accepting, let the in-flight request finish, then close | test that `shutdown()` during a slow request returns after that request's response |
| 4 | Packaging decision, **owner's call, not the agent's**: a `LICENSE` file and a version above `0.0.1` | the file exists and the README says what it is; if the owner declines, record it in `docs/06` |

## Cost

0.5-1 day **estimated**; 0 GPU-hours **measured** (all four are CPU-testable).

## False if / kill

- Step 3 needs threads joined across a `ThreadingHTTPServer` shutdown path that the stdlib does not
  expose cleanly -> record the limitation and keep the current behaviour; a shutdown race is not
  worth a hand-rolled server.
- The deadline flag changes any measured latency row -> it must not; if it does, the flag is on the
  timed path and belongs outside it.

## Why this is smaller than it looks

The deployment claim in the README is already measured (`serve_load`, two clients queueing, the
JSON/socket share of the round trip). This plan is about not being embarrassed by an edge case, not
about adding capability -- and the review pass cut two items that would have pretended otherwise.
