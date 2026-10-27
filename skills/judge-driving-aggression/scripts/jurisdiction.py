"""Jurisdiction profiles and the right-of-way decision they drive.

A profile (``jurisdictions/<id>.json``) states which traffic rules decide who goes first at an
intersection without a working signal. ``decide`` applies one to two approaches (anything with
``.control``, ``.turn``, ``.heading``, ``.arrived``, ``.entered``, ``.through``, as
``responsibility.right_of_way.Approach`` has). It has no dependency on the scene code, so it can
be tested on its own (``python jurisdiction.py`` runs the self-test).

Conventions, all relative to each vehicle's own driving direction so that left- and right-hand
traffic need no special case:

* ``rel`` is the heading of b minus the heading of a, in (-180, 180] degrees. rel in (45, 135)
  means b comes from a's right (it heads towards a's left); rel in (-135, -45) from a's left.
* a rule returns "a", "b" or None for who has the right of way; None means it does not apply.
"""

import json
import math
import sys
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent.parent / "jurisdictions"
SAME_DIRECTION = math.radians(45.0)  # the same value as right_of_way.SAME_DIRECTION
ONCOMING = math.radians(135.0)  # and right_of_way.ONCOMING
STOP, NONE = "stop", "none"


def load_profile(name_or_path):
    path = Path(name_or_path)
    if not path.suffix:
        path = PROFILE_DIR / (str(name_or_path) + ".json")
    if not path.exists():
        sys.exit("no profile %r; available: %s" % (str(name_or_path), ", ".join(list_profiles())))
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    unknown = [r for r in profile["all_way_stop"]["tie_rules"] + profile["uncontrolled_rules"] if r not in RULES]
    if unknown:
        raise ValueError("profile %s names unknown rules %s; known: %s" % (path.name, unknown, sorted(RULES)))
    return profile


def list_profiles():
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _rel(a, b):
    return _wrap(b.heading - a.heading)


def _left(turn):
    return turn in ("left", "u-turn")


# Each rule: (a, b, rel) -> "a" | "b" | None, who holds the right of way.

def oncoming_left_yields(a, b, rel):
    """A left turn yields to oncoming traffic (CVC 21801(a); StVO 9(3))."""
    if abs(rel) > ONCOMING and _left(a.turn) != _left(b.turn):
        return "b" if _left(a.turn) else "a"


def oncoming_right_yields(a, b, rel):
    """Left-hand traffic: a right turn crosses the oncoming lane and yields to it."""
    if abs(rel) > ONCOMING and (a.turn == "right") != (b.turn == "right"):
        return "b" if a.turn == "right" else "a"


def oncoming_right_yields_left(a, b, rel):
    """A right turner yields to an oncoming left turner (China, regulation 51)."""
    if abs(rel) > ONCOMING and {a.turn, b.turn} == {"right", "left"}:
        return "b" if a.turn == "right" else "a"


def turner_yields_straight(a, b, rel):
    """A vehicle that turns yields to one that goes straight on (China, regulation 51)."""
    if (a.turn == "straight") != (b.turn == "straight"):
        return "a" if a.turn == "straight" else "b"


def yield_right(a, b, rel):
    """The vehicle on the left yields to the one on the right (CVC 21800(b); StVO 8(1))."""
    if SAME_DIRECTION < rel < ONCOMING:
        return "b"
    if -ONCOMING < rel < -SAME_DIRECTION:
        return "a"


def yield_left(a, b, rel):
    """Left-hand traffic: the vehicle on the right yields to the one on the left."""
    if SAME_DIRECTION < rel < ONCOMING:
        return "a"
    if -ONCOMING < rel < -SAME_DIRECTION:
        return "b"


RULES = {f.__name__: f for f in (oncoming_left_yields, oncoming_right_yields, oncoming_right_yields_left,
                                 turner_yields_straight, yield_right, yield_left)}
RULES["first_entered"] = None  # handled in decide: whoever entered the intersection first


def _first(ta, tb, tie_steps):
    """"a" / "b" for the earlier of two step indices (None: unknown or within the tie window)."""
    if ta is None and tb is None:
        return None
    if ta is None or tb is None:
        return "b" if ta is None else "a"
    return None if abs(ta - tb) <= tie_steps else "a" if ta < tb else "b"


def _apply(names, a, b, rel, tie_steps):
    for name in names:
        if name == "first_entered":
            holder = _first(a.entered, b.entered, tie_steps)
        else:
            holder = RULES[name](a, b, rel)
        if holder is not None:
            return holder, name
    return None, None


def handles(a, b):
    """Whether the profile decides this pair: neither faces a light, both face a stop sign or nothing."""
    return a.control in (STOP, NONE) and b.control in (STOP, NONE)


def decide(profile, a, b, tie_steps=10):
    """(holder, case) for approaches ``a`` and ``b`` that ``handles`` accepts; holder is "a", "b" or None.

    ``tie_steps``: arrivals (or entries) this many 0.1 s steps apart count as simultaneous.
    """
    rel = _rel(a, b)
    stop_a, stop_b = a.control == STOP, b.control == STOP
    if stop_a and stop_b:
        awy = profile["all_way_stop"]
        if awy["first_arrival"]:
            if a.arrived == b.arrived == -1:  # both already waiting when the clip began
                if awy["unknown_order"] != "tie":
                    return None, "all-way-stop/order-unknown"
                holder, name = _apply(awy["tie_rules"], a, b, rel, tie_steps)
                return holder, "all-way-stop/order-unknown-" + name if name else "all-way-stop/order-unknown"
            holder = _first(a.arrived, b.arrived, tie_steps)
            if holder is not None:
                return holder, "all-way-stop/first-in"
        holder, name = _apply(awy["tie_rules"], a, b, rel, tie_steps)
        return holder, "all-way-stop/" + name if name else "all-way-stop/simultaneous"
    if stop_a != stop_b:  # two-way stop: the vehicle at the sign yields to cross traffic
        return ("b" if stop_a else "a"), "stop-sign"
    if profile["t_junction"] == "ending_road_yields" and {a.through, b.through} == {True, False}:
        return ("a" if a.through else "b"), "through-road"
    holder, name = _apply(profile["uncontrolled_rules"], a, b, rel, tie_steps)
    return holder, "uncontrolled/" + name if name else "uncontrolled"


def _self_test():
    from types import SimpleNamespace as N

    def ap(heading_deg, turn, control=STOP, arrived=-1, entered=0, through=True):
        return N(heading=math.radians(heading_deg), turn=turn, control=control, arrived=arrived, entered=entered,
                 through=through)

    def mirror(x):  # reflect the road: headings negate, left and right swap
        swap = {"left": "right", "right": "left"}
        return N(heading=-x.heading, turn=swap.get(x.turn, x.turn), control=x.control, arrived=x.arrived,
                 entered=x.entered, through=x.through)

    us, lh = load_profile("us-california"), load_profile("left-hand-draft")
    # both waiting at an all-way stop; b comes from a's right (rel +90): a yields to b under US rules
    a, b = ap(0, "straight"), ap(90, "straight")
    assert decide(us, a, b) == ("b", "all-way-stop/order-unknown-yield_right"), decide(us, a, b)
    # a turns left across oncoming b: a yields
    a, b = ap(0, "left"), ap(180, "straight")
    assert decide(us, a, b)[0] == "b"
    # the first to arrive goes first, whatever the turn
    a, b = ap(0, "left", arrived=-1), ap(180, "straight", arrived=30)
    assert decide(us, a, b) == ("a", "all-way-stop/first-in")
    # arrivals 0.5 s apart are a tie: left turn yields
    a, b = ap(0, "left", arrived=10), ap(180, "straight", arrived=15)
    assert decide(us, a, b)[0] == "b"
    # two-way stop: the sign yields
    a, b = ap(0, "straight", control=STOP), ap(90, "straight", control=NONE)
    assert decide(us, a, b) == ("b", "stop-sign")
    # a T junction: the road that ends yields
    a, b = ap(0, "left", control=NONE, through=False), ap(90, "straight", control=NONE, through=True)
    assert decide(us, a, b) == ("b", "through-road")
    # mirror symmetry: the left-hand profile is the US one reflected, so on the reflected road it
    # must name the same vehicle (a reflection swaps left and right and negates the headings)
    for a, b in [(ap(0, "left"), ap(180, "straight")), (ap(0, "right"), ap(180, "straight")),
                 (ap(0, "straight"), ap(90, "left")), (ap(0, "straight", arrived=5), ap(180, "left", arrived=50)),
                 (ap(0, "straight", control=NONE), ap(90, "left", control=NONE)),
                 (ap(0, "left", control=NONE, through=False), ap(90, "straight", control=NONE))]:
        assert decide(us, a, b)[0] == decide(lh, mirror(a), mirror(b))[0], (a.turn, b.turn, decide(us, a, b), decide(lh, mirror(a), mirror(b)))
    print("jurisdiction self-test passed; profiles:", ", ".join(list_profiles()))


if __name__ == "__main__":
    sys.exit(_self_test())
