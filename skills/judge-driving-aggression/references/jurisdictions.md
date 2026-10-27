# Jurisdiction profiles

A profile is a JSON file in `jurisdictions/` that says which rules settle who goes first at an
intersection without a working signal. `scripts/jurisdiction.py` applies it; `python
scripts/jurisdiction.py` self-tests the engine, including that the left-hand profile is the exact
mirror of the California one.

| id | status |
|---|---|
| `us-california` | validated on CAT's 500 WOMD scenes (see `validation-us-california.md`) |
| `left-hand-draft` | the California structure reflected for left-hand traffic; **not any country's law** |
| `china-draft` | regulation 51 as recalled; **unchecked, unvalidated** |
| `germany-draft` | StVO 8 and 9 as recalled; **unchecked, unvalidated** |

Draft profiles print a warning and their results must not be reported as findings until someone
who knows the local traffic code has read the profile and a few labelled scenes agree with it.

## Schema

```json
{
  "id": "us-california",
  "driving_side": "right",
  "all_way_stop": {
    "first_arrival": true,
    "unknown_order": "tie",
    "tie_rules": ["oncoming_left_yields", "yield_right"]
  },
  "uncontrolled_rules": ["oncoming_left_yields"],
  "t_junction": "ending_road_yields",
  "status": "...",
  "notes": ["..."]
}
```

- `first_arrival`: the first vehicle to arrive and stop goes first. When false, the tie rules
  decide every all-way-stop pair.
- `unknown_order`: when both vehicles were already waiting at the start of the clip, `"tie"`
  applies `tie_rules`; anything else gives no verdict.
- `uncontrolled_rules`: no sign and no light; rules applied in order.
- `t_junction`: `"ending_road_yields"` or `"none"`.
- A two-way stop always makes the vehicle at the sign yield to cross traffic. Lights are handled
  by the CAT code, not by the profile.

## Rules (ordered lists; the first that names a holder wins)

| rule | meaning |
|---|---|
| `oncoming_left_yields` | a left turn yields to oncoming traffic (CVC 21801(a); StVO 9(3)) |
| `oncoming_right_yields` | right-turn counterpart for left-hand traffic |
| `oncoming_right_yields_left` | an oncoming right turner yields to a left turner (China) |
| `turner_yields_straight` | a vehicle that turns yields to one going straight on (China) |
| `yield_right` | the vehicle on the left yields to the one on the right (CVC 21800(b); StVO 8(1)) |
| `yield_left` | the mirror of `yield_right` |
| `first_entered` | whoever entered the intersection first (only useful in `uncontrolled_rules`) |

"Left", "right" and "oncoming" are all relative to each vehicle's own heading, so a left-hand
profile needs no coordinate flip.

## Adding a jurisdiction

1. Copy the closest profile, give it a new `id` and set `status` to a string starting with
   `draft`.
2. Write the statute's rules into `all_way_stop`, `uncontrolled_rules` and `t_junction`, citing the
   article in `notes`. If you need a rule that is not in the table, add a function to
   `scripts/jurisdiction.py` (signature `(a, b, rel) -> "a" | "b" | None`) and register it in
   `RULES`.
3. Run `python scripts/jurisdiction.py`. Then screen a few dozen scenes from that country, look at
   every flagged scene and a sample of unflagged ones with a person who knows the local rules, and
   only then change `status` from draft and record the check in a `validation-<id>.md`.

Datasets differ in region: every WOMD scene is from US cities (right-hand traffic; California-like
rules are a reasonable default), nuPlan's Singapore scenes are left-hand, and datasets from China
or Germany need the matching profile.
