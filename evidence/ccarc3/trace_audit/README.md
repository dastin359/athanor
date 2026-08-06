# Trace-audit store

`spans.json.gz` and `runs.json.gz` are the data behind the published trace-audit
page (artifact `7447856a-b587-4d52-9c5c-a839de3eb6ee`). They are kept here
because **they can no longer be regenerated from disk.**

The page is built from the `claude -p` NDJSON streams each solver was recorded
with. Those streams lived only in the scratchpad. Of the 25 game directories the
page covers, **13 survive and 12 are gone**, and at least one survivor
(`cd82-fb555c5d`) has since been overwritten by a different, later run — its
on-disk stream is 1,973 rows at $16.52, against the 1,366 rows at $13.01 the page
was built from. A regeneration that scanned the scratchpad would therefore emit a
*degraded* page: fewer games, and one of them a different run wearing the same
name. That is worse than not refreshing at all, because it would look like a
refresh.

So the store is the seed, not a cache. `tools/build_trace_audit.py` loads it,
merges in whatever new games have finished, and writes the page. Games whose
sources are gone keep the spans recorded here; new games are ingested from their
run directories. Nothing is re-derived from sources that no longer exist.

Recovered on 2026-08-06 by fetching the published artifact back, after
`refresh_audit.sh`, `build_spans.py` and `trace_audit.html` were all found
missing from the filesystem — they had only ever existed in the scratchpad.

| file | contents |
|---|---|
| `spans.json.gz` | 25 games -> `{result, attempts[{file, spans[], lines, unparseable}]}` |
| `runs.json.gz` | 25 scored-run summary rows (E, raw, cap, levels, actions, wall, cost, tier) |

Scanned for the ARC API key before commit: zero literal occurrences, zero
key-shaped tokens, zero `ARC_API_KEY=` assignments.
