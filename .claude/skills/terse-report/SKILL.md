---
name: terse-report
description: Use whenever the user asks for a summary, status, recap, or "what have you done", and at the end of every step. Enforces a short fixed format with no narrative.
---
# Terse report (max 8 lines, no intro, no outro)

```
Step: <id> <done|blocked|partial>
Did: <1 line, files/commands>
Evidence: <command -> decisive output, 1 line>
Not done: <1 line or none>
Deviations: <1 line or none>
Needs you: <numbered questions or "nothing">
Next (proposed, not started): <step id + title>
Green flag: <still on: run `! rm .claude/GREENLIGHT` | off>
```
Rules: no adjectives, no explanation of why unless a deviation, no code blocks over 5 lines, no restating earlier messages. If the user asks "more detail" on one item, expand only that item.
