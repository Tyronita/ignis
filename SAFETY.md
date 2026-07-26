# Ignis — fire-safety design, California standards & materials grading

Ignis's higher goal is **safe building design**: use the fire + evacuation simulation to score a building
against code, then optimise it *beyond* the minimum. This file defines the standards we target, how we grade
materials, and the **full list of safety variables** the optimiser can reason about. Computed subset lives in
`ignis/indoor/safety.py`.

## The core safety criterion (performance-based)
```
ASET  = Available Safe Egress Time   (time until conditions become untenable)
RSET  = Required Safe Egress Time    (detection + pre-movement + travel to exit)
SAFE  ⟺  ASET − RSET > margin        (SFPE / performance-based design)
```
Ignis estimates **ASET** from the fire sim (onset of untenability: flashover / smoke filling / head-height
hazard) and **RSET** from the evacuation model (detection delay + walking time to an exit).

## California standards we target
| Code | Scope | What Ignis checks |
|---|---|---|
| **CRC R313** (Title 24) | 1–2 family dwellings | automatic **fire sprinklers required** in new homes (since 2011) |
| **CRC R314 / R315** | dwellings | smoke alarms (interconnected) / CO alarms — detection time |
| **CRC R302 / CBC Ch.7** | fire-resistance | rated wall/door assemblies (hourly, **ASTM E119 / UL 263**) |
| **CBC Ch.9** (NFPA 13/13R, 72) | fire protection | sprinkler coverage & activation, alarm coverage |
| **CBC Ch.10 / NFPA 101** | means of egress | ≥2 exits, **exit-access travel distance**, common path, exit width, illumination, signage |
| **CBC Ch.8 (ASTM E84)** | interior finishes | **flame-spread class** A/B/C, smoke-developed index |
| **ISO 834 / ASTM E119** | fire exposure | standard time-temperature curve (calibration target) |

## Materials grading (our voxel materials → fire behaviour)
| Material | Combustible? | Flame-spread class (ASTM E84) | Fire role |
|---|---|---|---|
| concrete | No (E136 non-comb.) | — | rated structure / compartmentation |
| steel | No | — | structure (conducts heat — future) |
| glass | No (unless rated) | — | fails early under heat (future) |
| drywall (gypsum) | No (Type X = 1-hr) | **Class A** (0–25) | compartmentation |
| wood | Yes | **Class B/C** (26–200) | structure/furniture fuel |
| fabric / foam | Yes | **Class C**, high SDI | **upholstered fuel** — fast HRR, heavy smoke |
| paper | Yes | Class C, very high FSI | light fuel, fast ignition |

## Full safety-variable inventory (the optimiser's objective/constraint space)
**Egress:** # exits · exit width · max exit-access travel distance · common path of travel · dead-end corridors ·
door swing direction · exit illumination · exit signage / photoluminescent markings · **RSET** · pre-movement time.
**Detection & warning:** smoke-alarm coverage & activation time · interconnection · CO alarms.
**Suppression:** sprinkler coverage % · activation time · water supply/budget · hose/agent count & reach.
**Compartmentation:** fire-rated wall/door ratings (hrs) · fraction of rated boundaries · opening protectives.
**Materials / fuel:** flame-spread class per finish · smoke-developed index · fuel-load density (MJ/m²) ·
non-combustible fraction · upholstered-furniture load.
**Fire dynamics:** time-to-flashover · **time-to-untenability (ASET)** · HRR · head-height (1.5–1.8 m)
temperature · smoke-layer height · visibility / CO.
**Structural:** time-to-failure / collapse.
**Outcomes (scored):** **ASET − RSET margin** · % escaped · casualties · $ damage · rooms lost.

## Safe-building-design meta-task
Treat the two agent tasks (suppression, evacuation) as the **inner loop**; the outer loop edits the building —
add/relocate exits, upgrade finishes to Class A, add compartmentation, place sprinklers/detectors, add smoke
vents — and maximises a **safety score** (weighted ASET−RSET, casualties, damage) subject to code minimums.
`safety.py` computes the score and demonstrates a code-minimum vs. optimised comparison.
