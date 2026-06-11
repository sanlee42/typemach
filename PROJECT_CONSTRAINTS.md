# PROJECT_CONSTRAINTS.md

This file is the project-specific source of truth for typemach. It overrides
the parent `AGENTS.md` when the two conflict.

## 0. Style First

- Do not write Java-style APIs: no long ceremonial names, no needless layering,
  no boilerplate builders, and no `manager`/`service`/`factory`/`coordinator`
  objects unless a framework or stable external contract requires them.
- Use Linux/Rust style: short names, obvious data flow, explicit state, small
  hard types, functions, and enums. Put abstractions only on stable boundaries.
- Public APIs must be friendly to agents and humans: callers should express the
  business intent and should not need to know internal cursors, transactions,
  registries, lease tokens, or serde JSON mechanics.
- An abstraction belongs in typemach only if it covers real stonex paths and the
  future trade-agent path. A wrapper for one current caller stays out.
