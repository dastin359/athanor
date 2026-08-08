# Working conventions

## Report times in Pacific

Every timestamp in a report, a status line, or a commit message is
**America/Los_Angeles**. The box runs UTC, so this needs converting rather than
copying — and that is exactly where it fails.

Recorded here, in the repo, because it was already recorded twice in the
session scratchpad (`MEMORY.md`, `AUTOPILOT.md`) and violated anyway on
2026-08-08: the scratchpad reverts to an image snapshot when the container is
replaced, so a rule that lives only there is a rule that expires.

**The rule is the weaker half of the fix.** A convention that has to be applied
by hand at report time gets skipped whenever a raw `date` or a log line is
pasted through. Anything that writes a timestamp should write it in Pacific at
the source:

```bash
TZ=America/Los_Angeles date '+%H:%M %Z'
```
```python
os.environ["TZ"] = "America/Los_Angeles"; time.tzset()
```

Long-running scripts must do this at startup: patching the file afterwards does
not touch a process already running, so an in-flight probe keeps emitting UTC
until it exits.
