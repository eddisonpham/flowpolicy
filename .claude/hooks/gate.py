#!/usr/bin/env python3
"""Hard green-flag gate. Blocks mutating tool calls unless .claude/GREENLIGHT exists.
The user creates it with:  ! echo <step-id> > .claude/GREENLIGHT   and removes it with:  ! rm .claude/GREENLIGHT
Exit code 2 = block (stderr is shown to the agent)."""
import json, os, re, sys

root = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
flag = os.path.join(root, ".claude", "GREENLIGHT")
data = json.load(sys.stdin)
tool, inp = data.get("tool_name", ""), data.get("tool_input", {}) or {}

def block(msg):
    sys.stderr.write("BLOCKED: " + msg + "\nPropose a Step Plan and wait for the user to type GO <step-id> and enable the green flag.\n")
    sys.exit(2)

has_flag = os.path.exists(flag)

if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
    path = os.path.abspath(inp.get("file_path", inp.get("notebook_path", "")))
    if os.sep + ".claude" + os.sep in path:
        block("agent may never modify .claude/")
    if path.endswith(os.path.join("docs", "STATE.md")):
        sys.exit(0)
    if not has_flag:
        block("no green flag for file edits")
    sys.exit(0)

if tool == "Bash":
    cmd = inp.get("command", "")
    READONLY = {"ls","cat","head","tail","pwd","wc","grep","rg","find","tree","stat","file","du","df","which","echo",
                "nvidia-smi","uname","lsb_release","date","diff"}
    GIT_OK = {"status","diff","log","show","rev-parse","branch","ls-files"}
    MUT = r"(>|\btee\b|-delete|-exec|\bsed\s+-i|\brm\b|\bmv\b|\bcp\b|\bmkdir\b|\btouch\b|\bchmod\b|\bln\b)"
    def ro(seg):
        t = seg.split()
        if not t:
            return True
        if re.search(MUT, seg):
            return False
        if t[0] == "git":
            return len(t) > 1 and t[1] in GIT_OK
        if t[0] in ("python3", "python", "uv", "docker") and len(t) == 2 and t[1] == "--version":
            return True
        return t[0] in READONLY
    segs = [s.strip() for s in re.split(r"&&|\|\||;|\|", cmd)]
    readonly = all(ro(s) for s in segs)
    # The gate itself is never modifiable by the agent, flag or not.
    if ".claude" in cmd and not readonly:
        block("agent may never modify .claude/")
    if readonly:
        sys.exit(0)
    if not has_flag:
        block("mutating or unknown shell command without green flag: " + cmd[:120])
    sys.exit(0)

sys.exit(0)
