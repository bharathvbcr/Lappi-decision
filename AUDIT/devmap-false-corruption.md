# DevMap reported corruption that is not corruption

**2026-09-19, during the build lanes.** `devmap_status` and every `devmap_*` query began returning:

```
database disk image is malformed
```

That message says "rebuild your index". On a 2.05 GB store over 180K+ symbols that is an expensive
instruction to follow, and it would have been the wrong one.

## What is actually true

| Check | Result |
| --- | --- |
| `PRAGMA integrity_check` | **ok** |
| `PRAGMA quick_check` | **ok** |
| `SELECT count(*) FROM sqlite_master` | 38 tables, all readable |
| Latest generation | **167** |
| Nodes in generation 167 | **182,488** |
| WAL file | 0 bytes (nothing uncommitted) |
| `devmap.sqlite.writer.lock` | present, naming **PID 55819** |
| Is PID 55819 alive? | **No** — `kill -0` reports no such process |

**The database is intact and fully queryable.** The store holds a stale writer lock from a process
that has since died, and the MCP server surfaces that as disk corruption.

## Why it matters beyond this session

This is the same failure shape the repo's own rules are about, seen from the other side. The tooling
policy says to record what DevMap could not answer rather than silently substituting grep — and two
lanes did exactly that, correctly, reporting "nothing here is graph-confirmed". But they reported it
against a **false cause**: they believed the index was corrupt. Had anyone acted on that belief, the
response would have been a multi-hour rebuild of a healthy 2 GB store.

**An error message that names the wrong cause is worse than a generic one**, because it directs the
remedy. `malformed` should be reserved for what `integrity_check` actually reports; a lock that
cannot be acquired, or a handle that cannot be reopened, is a different condition with a different
fix.

## Why the lock probably went stale

Generation **167**, from 27 earlier in the same session. The index was being rebuilt continuously
while four agents wrote files into the tree, which is a plausible way to leave a writer lock behind
when a rebuild process is killed mid-write.

## The remedy

**Restart the DevMap MCP server.** The database needs nothing. The stale lock file
(`.devcouncil/codeintel/devmap.sqlite.writer.lock`, naming a dead PID) was left in place rather than
deleted here: it belongs to DevCouncil's own state, the MCP server may hold its own handle on it, and
clearing another tool's lock from outside is how two writers end up in one store.

**Do not rebuild the index.** Nothing about it is broken.

## Consequences recorded elsewhere

Three lanes ran without graph confirmation and said so. Their findings stand on direct reads and
ripgrep, which is the correct fallback with the correct caveat — see `GAP-DEVMAP-CORPUS-PARTIAL`
and the per-lane handoffs. What changes is only the reason: the graph was available and the client
could not reach it, rather than the graph being damaged.
