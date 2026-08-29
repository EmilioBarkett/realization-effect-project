"""Deterministic design checks for repaired downstream-task contracts.

The repair is deliberately represented as metadata and schedules rather than
as a post-hoc prompt rewrite.  This module validates the machine-readable
parts before generation and validates every generated record before it can be
checkpointed or released.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from typing import Any, Iterable, Mapping

from .prompts import PromptRecord
from .schemas import ConstructSpec


REPAIRED_CONSTRUCTS = frozenset(
    {
        "realization_account_closure",
        "evidence_diagnosticity",
        "persistence_continuation",
        "reference_frame",
        "prior_weighting",
        "authority_deference",
        "exploration_exploitation",
        "ambiguity_orientation",
        "causal_interpretation",
        "consensus_conformity",
        "plan_replanning",
        "temporal_orientation",
        "epistemic_uncertainty",
        "reciprocity_obligation",
        "goal_shielding",
    }
)
DOWNSTREAM_SPLITS = frozenset({"behavior_eval", "steering_eval", "calibration", "collateral_eval"})
DOWNSTREAM_ROLES = frozenset({"behavior", "steering", "calibration", "collateral"})
_WORD = re.compile(r"[a-z0-9]+")
_RESPONSE_DIRECTIVE = re.compile(
    r"\b(?:return|report|output|provide|enter|reply|answer)\b[^.!?\n]{0,220}?"
    r"(?:\bexactly\s+(?:one|two)\b|\bone\s+integer\b|\btwo\s+integers?\b|"
    r"\b0\s*(?:to|-|through)\s*100\b|\b1\s+or\s+2\b)",
    re.IGNORECASE,
)


# This bank is the source of truth for collateral gold answers.  Generated
# collateral text must reproduce the registered statement pair for its
# ``fact_id``; the answer is checked against this table rather than trusting a
# model-supplied ``correct_option`` field.
COLLATERAL_FACT_BANK: dict[str, dict[str, Any]] = {
    "biology_01": {
        "fact_domain": "biology",
        "statement_1": "A whale is a mammal.",
        "statement_2": "A whale is an insect.",
        "correct_option": 1,
    },
    "biology_02": {
        "fact_domain": "biology",
        "statement_1": "A whale is an insect.",
        "statement_2": "A whale is a mammal.",
        "correct_option": 2,
    },
    "geography_01": {
        "fact_domain": "geography",
        "statement_1": "Tokyo is the capital of Japan.",
        "statement_2": "Seoul is the capital of Japan.",
        "correct_option": 1,
    },
    "geography_02": {
        "fact_domain": "geography",
        "statement_1": "Seoul is the capital of Japan.",
        "statement_2": "Tokyo is the capital of Japan.",
        "correct_option": 2,
    },
    "history_01": {
        "fact_domain": "history",
        "statement_1": "The first human Moon landing occurred in 1969.",
        "statement_2": "The first human Moon landing occurred in 1899.",
        "correct_option": 1,
    },
    "history_02": {
        "fact_domain": "history",
        "statement_1": "The first human Moon landing occurred in 1899.",
        "statement_2": "The first human Moon landing occurred in 1969.",
        "correct_option": 2,
    },
    "measurement_01": {
        "fact_domain": "measurement",
        "statement_1": "One meter contains 100 centimeters.",
        "statement_2": "One meter contains 10 centimeters.",
        "correct_option": 1,
    },
    "measurement_02": {
        "fact_domain": "measurement",
        "statement_1": "One meter contains 10 centimeters.",
        "statement_2": "One meter contains 100 centimeters.",
        "correct_option": 2,
    },
}


# The repaired-v3 collateral cell originally repeated these eight fact IDs
# four times.  That is a valid schedule for a repeated-measures fixture, but
# it is not a valid independent prompt pool: a model can emit the same
# wrapper for the repeated fact and trip the duplicate/near-overlap audit.
# Keep that historical bank immutable and register a versioned bank with one
# unique objective fact pair per 32-item collateral cell instead.
COLLATERAL_FACT_BANK_V2_VERSION = "unique_32_v2"
COLLATERAL_FACT_BANK_V2: dict[str, dict[str, Any]] = {
    **COLLATERAL_FACT_BANK,
    "biology_03": {
        "fact_domain": "biology",
        "statement_1": "A dolphin is a mammal.",
        "statement_2": "A dolphin is a flowering plant.",
        "correct_option": 1,
    },
    "biology_04": {
        "fact_domain": "biology",
        "statement_1": "A bat is a mammal.",
        "statement_2": "A bat is a fish.",
        "correct_option": 1,
    },
    "botany_01": {
        "fact_domain": "botany",
        "statement_1": "Oak trees produce acorns.",
        "statement_2": "Oak trees produce pine cones.",
        "correct_option": 1,
    },
    "botany_02": {
        "fact_domain": "botany",
        "statement_1": "Mosses are nonvascular plants.",
        "statement_2": "Mosses are flowering animals.",
        "correct_option": 1,
    },
    "astronomy_01": {
        "fact_domain": "astronomy",
        "statement_1": "Earth completes one orbit around the Sun in about one year.",
        "statement_2": "Earth completes one orbit around the Sun in about one day.",
        "correct_option": 1,
    },
    "astronomy_02": {
        "fact_domain": "astronomy",
        "statement_1": "The Moon reflects sunlight.",
        "statement_2": "The Moon produces its own visible sunlight.",
        "correct_option": 1,
    },
    "geography_03": {
        "fact_domain": "geography",
        "statement_1": "The Pacific Ocean is the largest ocean.",
        "statement_2": "The Atlantic Ocean is the largest ocean.",
        "correct_option": 1,
    },
    "geography_04": {
        "fact_domain": "geography",
        "statement_1": "Australia is both a country and a continent.",
        "statement_2": "Australia is located on the continent of Europe.",
        "correct_option": 1,
    },
    "history_03": {
        "fact_domain": "history",
        "statement_1": "The Berlin Wall fell in 1989.",
        "statement_2": "The Berlin Wall fell in 1798.",
        "correct_option": 1,
    },
    "history_04": {
        "fact_domain": "history",
        "statement_1": "The printing press is associated with Johannes Gutenberg.",
        "statement_2": "The printing press is associated with Isaac Newton.",
        "correct_option": 1,
    },
    "measurement_03": {
        "fact_domain": "measurement",
        "statement_1": "An hour contains 60 minutes.",
        "statement_2": "An hour contains 100 minutes.",
        "correct_option": 1,
    },
    "measurement_04": {
        "fact_domain": "measurement",
        "statement_1": "A dozen contains 12 items.",
        "statement_2": "A dozen contains 10 items.",
        "correct_option": 1,
    },
    "biology_05": {
        "fact_domain": "biology",
        "statement_1": "A penguin is a mammal.",
        "statement_2": "A penguin is a bird.",
        "correct_option": 2,
    },
    "biology_06": {
        "fact_domain": "biology",
        "statement_1": "An octopus has six arms.",
        "statement_2": "An octopus has eight arms.",
        "correct_option": 2,
    },
    "botany_03": {
        "fact_domain": "botany",
        "statement_1": "Photosynthesis occurs only in animals.",
        "statement_2": "Photosynthesis is used by plants to make sugars.",
        "correct_option": 2,
    },
    "botany_04": {
        "fact_domain": "botany",
        "statement_1": "A cactus naturally requires no water at all.",
        "statement_2": "A cactus stores water in its tissues.",
        "correct_option": 2,
    },
    "astronomy_03": {
        "fact_domain": "astronomy",
        "statement_1": "The Sun is a planet.",
        "statement_2": "The Sun is a star.",
        "correct_option": 2,
    },
    "astronomy_04": {
        "fact_domain": "astronomy",
        "statement_1": "Mars is closer to the Sun than Mercury.",
        "statement_2": "Mercury is closer to the Sun than Mars.",
        "correct_option": 2,
    },
    "geography_05": {
        "fact_domain": "geography",
        "statement_1": "The Sahara is in South America.",
        "statement_2": "The Sahara is in Africa.",
        "correct_option": 2,
    },
    "geography_06": {
        "fact_domain": "geography",
        "statement_1": "The capital of France is Madrid.",
        "statement_2": "The capital of France is Paris.",
        "correct_option": 2,
    },
    "history_05": {
        "fact_domain": "history",
        "statement_1": "World War II ended in 1935.",
        "statement_2": "World War II ended in 1945.",
        "correct_option": 2,
    },
    "history_06": {
        "fact_domain": "history",
        "statement_1": "The United States Declaration of Independence was adopted in 1876.",
        "statement_2": "The United States Declaration of Independence was adopted in 1776.",
        "correct_option": 2,
    },
    "measurement_05": {
        "fact_domain": "measurement",
        "statement_1": "A kilogram contains 100 grams.",
        "statement_2": "A kilogram contains 1,000 grams.",
        "correct_option": 2,
    },
    "math_01": {
        "fact_domain": "mathematics",
        "statement_1": "A triangle has four sides.",
        "statement_2": "A triangle has three sides.",
        "correct_option": 2,
    },
}


# The common 32-fact bank is sufficient to prevent within-construct repeats,
# but every Wave 2--4 construct using it still shares the same objective cards.
# A shared card is legitimate provenance for a paired control only when that
# pairing is preregistered.  These collateral cells are independent tasks, so
# the release banks below are construct-disjoint across all repaired Waves
# 2--4.  The seeds are ordinary, independently checkable facts; the helper
# only alternates which numbered statement is true so the registered gold
# positions remain exactly balanced (16/16) in every bank.
def _construct_disjoint_fact_bank(
    prefix: str,
    seeds: Iterable[tuple[str, str, str]],
) -> dict[str, dict[str, Any]]:
    rows = tuple(seeds)
    if len(rows) != 32:
        raise ValueError(f"{prefix} must define exactly 32 collateral facts.")
    bank: dict[str, dict[str, Any]] = {}
    for index, (fact_domain, true_statement, false_statement) in enumerate(rows, start=1):
        fact_id = f"{prefix}_{index:02d}"
        correct_option = 1 if index % 2 else 2
        bank[fact_id] = {
            "fact_domain": fact_domain,
            "statement_1": true_statement if correct_option == 1 else false_statement,
            "statement_2": false_statement if correct_option == 1 else true_statement,
            "correct_option": correct_option,
        }
    return bank


COLLATERAL_FACT_BANK_V3_VERSION_BY_CONSTRUCT = {
    "reference_frame": "unique_32_wave2_reference_v3",
    "prior_weighting": "unique_32_wave2_prior_v3",
    "authority_deference": "unique_32_wave2_authority_v3",
    "exploration_exploitation": "unique_32_wave2_exploration_v3",
    "ambiguity_orientation": "unique_32_wave3_ambiguity_v3",
    "causal_interpretation": "unique_32_wave3_causal_v3",
    "consensus_conformity": "unique_32_wave3_consensus_v3",
    "plan_replanning": "unique_32_wave3_plan_v3",
    "temporal_orientation": "unique_32_wave4_temporal_v3",
    "epistemic_uncertainty": "unique_32_wave4_epistemic_v3",
    "reciprocity_obligation": "unique_32_wave4_reciprocity_v3",
    "goal_shielding": "unique_32_wave4_goal_v3",
}

COLLATERAL_FACT_BANK_V3_BY_CONSTRUCT: dict[str, dict[str, dict[str, Any]]] = {
    "reference_frame": _construct_disjoint_fact_bank(
        "reference",
        (
            ("geography", "The capital of Belgium is Brussels.", "The capital of Belgium is Antwerp."),
            ("geography", "The capital of Denmark is Copenhagen.", "The capital of Denmark is Aarhus."),
            ("geography", "The capital of the Netherlands is Amsterdam.", "The capital of the Netherlands is Rotterdam."),
            ("geography", "The capital of Algeria is Algiers.", "The capital of Algeria is Oran."),
            ("geography", "The capital of Tunisia is Tunis.", "The capital of Tunisia is Sfax."),
            ("geography", "The capital of Libya is Tripoli.", "The capital of Libya is Benghazi."),
            ("geography", "The capital of Ethiopia is Addis Ababa.", "The capital of Ethiopia is Gondar."),
            ("geography", "The capital of Tanzania is Dodoma.", "The capital of Tanzania is Dar es Salaam."),
            ("geography", "The capital of Uganda is Kampala.", "The capital of Uganda is Entebbe."),
            ("geography", "The capital of Zimbabwe is Harare.", "The capital of Zimbabwe is Bulawayo."),
            ("geography", "The capital of Zambia is Lusaka.", "The capital of Zambia is Kitwe."),
            ("geography", "The capital of Senegal is Dakar.", "The capital of Senegal is Saint-Louis."),
            ("geography", "The capital of Cameroon is Yaounde.", "The capital of Cameroon is Douala."),
            ("geography", "The capital of Angola is Luanda.", "The capital of Angola is Huambo."),
            ("geography", "The capital of Mozambique is Maputo.", "The capital of Mozambique is Beira."),
            ("geography", "The capital of Madagascar is Antananarivo.", "The capital of Madagascar is Toamasina."),
            ("geography", "The capital of Pakistan is Islamabad.", "The capital of Pakistan is Karachi."),
            ("geography", "The capital of Bangladesh is Dhaka.", "The capital of Bangladesh is Chittagong."),
            ("geography", "The capital of Nepal is Kathmandu.", "The capital of Nepal is Pokhara."),
            ("geography", "The capital of Bhutan is Thimphu.", "The capital of Bhutan is Paro."),
            ("geography", "The capital of Mongolia is Ulaanbaatar.", "The capital of Mongolia is Erdenet."),
            ("geography", "The capital of Kazakhstan is Astana.", "The capital of Kazakhstan is Almaty."),
            ("geography", "The capital of Uzbekistan is Tashkent.", "The capital of Uzbekistan is Samarkand."),
            ("geography", "The capital of Iran is Tehran.", "The capital of Iran is Isfahan."),
            ("geography", "The capital of Iraq is Baghdad.", "The capital of Iraq is Basra."),
            ("geography", "The capital of Saudi Arabia is Riyadh.", "The capital of Saudi Arabia is Jeddah."),
            ("geography", "The capital of the United Arab Emirates is Abu Dhabi.", "The capital of the United Arab Emirates is Dubai."),
            ("geography", "The capital of Qatar is Doha.", "The capital of Qatar is Al Rayyan."),
            ("geography", "The capital of Oman is Muscat.", "The capital of Oman is Salalah."),
            ("geography", "The capital of Jordan is Amman.", "The capital of Jordan is Aqaba."),
            ("geography", "The capital of Azerbaijan is Baku.", "The capital of Azerbaijan is Ganja."),
            ("geography", "The capital of Armenia is Yerevan.", "The capital of Armenia is Gyumri."),
        ),
    ),
    "prior_weighting": _construct_disjoint_fact_bank(
        "prior",
        (
            ("chemistry", "The chemical symbol for hydrogen is H.", "The chemical symbol for hydrogen is He."),
            ("chemistry", "The chemical symbol for carbon is C.", "The chemical symbol for carbon is Ca."),
            ("chemistry", "The chemical symbol for nitrogen is N.", "The chemical symbol for nitrogen is Ne."),
            ("chemistry", "The chemical symbol for sodium is Na.", "The chemical symbol for sodium is K."),
            ("chemistry", "The chemical symbol for iron is Fe.", "The chemical symbol for iron is F."),
            ("chemistry", "The chemical symbol for copper is Cu.", "The chemical symbol for copper is Co."),
            ("chemistry", "The chemical symbol for silver is Ag.", "The chemical symbol for silver is Al."),
            ("chemistry", "The chemical symbol for tin is Sn.", "The chemical symbol for tin is Si."),
            ("chemistry", "The chemical symbol for lead is Pb.", "The chemical symbol for lead is Pt."),
            ("chemistry", "The chemical symbol for potassium is K.", "The chemical symbol for potassium is Ca."),
            ("chemistry", "The chemical symbol for calcium is Ca.", "The chemical symbol for calcium is C."),
            ("chemistry", "The chemical symbol for chlorine is Cl.", "The chemical symbol for chlorine is C."),
            ("chemistry", "The chemical symbol for fluorine is F.", "The chemical symbol for fluorine is Fe."),
            ("chemistry", "The chemical symbol for helium is He.", "The chemical symbol for helium is H."),
            ("chemistry", "The chemical symbol for neon is Ne.", "The chemical symbol for neon is N."),
            ("chemistry", "The chemical symbol for argon is Ar.", "The chemical symbol for argon is Al."),
            ("chemistry", "Lithium has atomic number 3.", "Lithium has atomic number 4."),
            ("chemistry", "Beryllium has atomic number 4.", "Beryllium has atomic number 5."),
            ("chemistry", "Boron has atomic number 5.", "Boron has atomic number 6."),
            ("chemistry", "Carbon has atomic number 6.", "Carbon has atomic number 7."),
            ("chemistry", "Nitrogen has atomic number 7.", "Nitrogen has atomic number 9."),
            ("chemistry", "Fluorine has atomic number 9.", "Fluorine has atomic number 10."),
            ("chemistry", "Neon has atomic number 10.", "Neon has atomic number 11."),
            ("chemistry", "Sodium has atomic number 11.", "Sodium has atomic number 12."),
            ("chemistry", "Magnesium has atomic number 12.", "Magnesium has atomic number 13."),
            ("chemistry", "Aluminum has atomic number 13.", "Aluminum has atomic number 14."),
            ("chemistry", "Silicon has atomic number 14.", "Silicon has atomic number 15."),
            ("chemistry", "Phosphorus has atomic number 15.", "Phosphorus has atomic number 16."),
            ("chemistry", "Sulfur has atomic number 16.", "Sulfur has atomic number 17."),
            ("chemistry", "Chlorine has atomic number 17.", "Chlorine has atomic number 18."),
            ("chemistry", "Argon has atomic number 18.", "Argon has atomic number 20."),
            ("chemistry", "Calcium has atomic number 20.", "Calcium has atomic number 21."),
        ),
    ),
    "authority_deference": _construct_disjoint_fact_bank(
        "authority",
        (
            ("biology", "A cat is a mammal.", "A cat is a reptile."),
            ("biology", "A dog is a mammal.", "A dog is a bird."),
            ("biology", "A horse is a mammal.", "A horse is a fish."),
            ("biology", "An elephant is a mammal.", "An elephant is an insect."),
            ("biology", "A giraffe is a mammal.", "A giraffe is an amphibian."),
            ("biology", "A lion is a mammal.", "A lion is a reptile."),
            ("biology", "A tiger is a mammal.", "A tiger is a bird."),
            ("biology", "A rabbit is a mammal.", "A rabbit is a fish."),
            ("biology", "An eagle is a bird.", "An eagle is a mammal."),
            ("biology", "A robin is a bird.", "A robin is a reptile."),
            ("biology", "An ostrich is a bird.", "An ostrich is a fish."),
            ("biology", "An emu is a bird.", "An emu is an insect."),
            ("botany", "A maple tree is a plant.", "A maple tree is an animal."),
            ("botany", "Bamboo is a plant.", "Bamboo is a fungus."),
            ("botany", "A cactus is a plant.", "A cactus is a mineral."),
            ("botany", "A fern is a plant.", "A fern is an animal."),
            ("biology", "The kidneys filter waste from blood.", "The kidneys filter sunlight from air."),
            ("biology", "The stomach begins the digestion of many proteins.", "The stomach begins the digestion of sunlight."),
            ("biology", "The liver processes nutrients and chemicals.", "The liver processes only sound waves."),
            ("biology", "The lungs exchange gases with the blood.", "The lungs exchange bones with the blood."),
            ("biology", "A snake is a reptile.", "A snake is a mammal."),
            ("biology", "A lizard is a reptile.", "A lizard is a bird."),
            ("biology", "A turtle is a reptile.", "A turtle is a fish."),
            ("biology", "A crocodile is a reptile.", "A crocodile is an insect."),
            ("biology", "A bee is an insect.", "A bee is a mammal."),
            ("biology", "An ant is an insect.", "An ant is a bird."),
            ("biology", "A butterfly is an insect.", "A butterfly is a reptile."),
            ("biology", "A beetle is an insect.", "A beetle is a fish."),
            ("botany", "A rose is a flowering plant.", "A rose is a mineral."),
            ("biology", "A salmon is a fish.", "A salmon is a bird."),
            ("biology", "An earthworm is an invertebrate.", "An earthworm is a vertebrate."),
            ("biology", "Many bats use echolocation to navigate.", "Bats use photosynthesis to navigate."),
        ),
    ),
    "exploration_exploitation": _construct_disjoint_fact_bank(
        "exploration",
        (
            ("measurement", "One kilometer contains 1,000 meters.", "One kilometer contains 100 meters."),
            ("measurement", "One liter contains 1,000 milliliters.", "One liter contains 100 milliliters."),
            ("measurement", "One byte contains 8 bits.", "One byte contains 7 bits."),
            ("measurement", "A standard week contains 7 days.", "A standard week contains 8 days."),
            ("measurement", "A leap year contains 366 days.", "A leap year contains 365 hours."),
            ("mathematics", "A full circle measures 360 degrees.", "A full circle measures 180 degrees."),
            ("mathematics", "A hexagon has 6 sides.", "A hexagon has 8 sides."),
            ("mathematics", "A square has 4 sides.", "A square has 5 sides."),
            ("measurement", "A gross contains 144 items.", "A gross contains 100 items."),
            ("mathematics", "A parallelogram has two pairs of parallel sides.", "A parallelogram has no parallel sides."),
            ("measurement", "One foot contains 12 inches.", "One foot contains 10 inches."),
            ("measurement", "One yard contains 3 feet.", "One yard contains 4 feet."),
            ("measurement", "One meter contains 1,000 millimeters.", "One meter contains 100 millimeters."),
            ("measurement", "A metric tonne contains 1,000 kilograms.", "A metric tonne contains 100 kilograms."),
            ("measurement", "One hectare contains 10,000 square meters.", "One hectare contains 1,000 square meters."),
            ("mathematics", "One percent is one hundredth.", "One percent is one tenth."),
            ("mathematics", "The decimal 0.5 equals one half.", "The decimal 0.5 equals one fifth."),
            ("mathematics", "The decimal 0.25 equals one quarter.", "The decimal 0.25 equals one third."),
            ("mathematics", "A pentagon has 5 sides.", "A pentagon has 6 sides."),
            ("mathematics", "An octagon has 8 sides.", "An octagon has 6 sides."),
            ("mathematics", "A cube has 6 faces.", "A cube has 8 faces."),
            ("mathematics", "The interior angles of a triangle sum to 180 degrees.", "The interior angles of a triangle sum to 360 degrees."),
            ("mathematics", "The number pi is approximately 3.14.", "The number pi is approximately 2.14."),
            ("mathematics", "The product of two negative numbers is positive.", "The product of two negative numbers is negative."),
            ("mathematics", "Two to the fifth power equals 32.", "Two to the fifth power equals 25."),
            ("mathematics", "Three multiplied by four equals 12.", "Three multiplied by four equals 14."),
            ("mathematics", "Twelve squared equals 144.", "Twelve squared equals 124."),
            ("mathematics", "The binary representation of decimal two is 10.", "The binary representation of decimal two is 11."),
            ("mathematics", "An isosceles triangle has at least two equal sides.", "An isosceles triangle has no equal sides."),
            ("mathematics", "A square's diagonals are perpendicular.", "A square's diagonals are parallel."),
            ("measurement", "One gram contains 1,000 milligrams.", "One gram contains 100 milligrams."),
            ("measurement", "One minute contains 60 seconds.", "One minute contains 100 seconds."),
        ),
    ),
    "ambiguity_orientation": _construct_disjoint_fact_bank(
        "ambiguity",
        (
            ("geology", "Granite is an igneous rock.", "Granite is a sedimentary rock."),
            ("geology", "Basalt is an igneous rock.", "Basalt is a metamorphic rock."),
            ("geology", "Sandstone is a sedimentary rock.", "Sandstone is an igneous rock."),
            ("geology", "Limestone is a sedimentary rock.", "Limestone is an igneous rock."),
            ("geology", "Marble is a metamorphic rock.", "Marble is an igneous rock."),
            ("geology", "Slate is a metamorphic rock.", "Slate is a sedimentary rock."),
            ("geology", "Quartz is a mineral.", "Quartz is an animal."),
            ("geology", "Diamond is an allotrope of carbon.", "Diamond is a compound of water."),
            ("geology", "Earth's crust is its outer rocky layer.", "Earth's crust is its innermost metallic layer."),
            ("geology", "A volcano can release lava.", "A volcano can release ice."),
            ("geology", "Tectonic plates move over geologic time.", "Tectonic plates are permanently fixed."),
            ("geology", "Erosion wears down exposed surfaces.", "Erosion creates matter from nothing."),
            ("geology", "Soil commonly contains mineral particles.", "Soil contains only light."),
            ("geology", "A fossil records evidence of past life.", "A fossil records a future event."),
            ("geology", "A glacier is moving ice.", "A glacier is molten rock."),
            ("geology", "Magma is molten rock below Earth's surface.", "Magma is frozen air below Earth's surface."),
            ("geology", "Lava is magma that reaches Earth's surface.", "Lava is groundwater below Earth's surface."),
            ("geology", "Sediment consists of deposited particles.", "Sediment consists of stars."),
            ("geology", "An earthquake is sudden ground shaking.", "An earthquake is daily rainfall."),
            ("geology", "A fault is a fracture where rocks can move.", "A fault is a flowering plant."),
            ("geology", "A mineral has an ordered crystal structure.", "A mineral has no atomic structure."),
            ("geology", "The water table is the upper level of groundwater.", "The water table is the upper level of clouds."),
            ("geology", "Karst landscapes can form in soluble limestone.", "Karst landscapes can form only in solid steel."),
            ("geology", "Coal is an organic sedimentary rock.", "Coal is an igneous metal."),
            ("geology", "Petroleum is a fossil fuel.", "Petroleum is a pure metal."),
            ("geology", "Earth's mantle lies beneath its crust.", "Earth's mantle lies outside the atmosphere."),
            ("geology", "Earth's core contains substantial iron and nickel.", "Earth's core contains only oxygen gas."),
            ("geology", "Weathering breaks down rock at or near the surface.", "Weathering creates a mountain instantly."),
            ("geology", "A geode can contain crystals lining its cavity.", "A geode always contains liquid metal."),
            ("geology", "Pumice can float because it contains many pores.", "Pumice is always denser than lead."),
            ("geology", "Obsidian is volcanic glass.", "Obsidian is a sedimentary clay."),
            ("geology", "Sedimentary layers can preserve fossils.", "Sedimentary layers cannot preserve any fossils."),
        ),
    ),
    "causal_interpretation": _construct_disjoint_fact_bank(
        "causal",
        (
            ("chemistry", "Ordinary candle combustion requires oxygen.", "Ordinary candle combustion requires a vacuum."),
            ("chemistry", "Rusting of iron generally involves oxygen and water.", "Rusting of iron generally requires helium alone."),
            ("chemistry", "Acid-base neutralization can form salt and water.", "Acid-base neutralization turns all matter into gold."),
            ("chemistry", "A catalyst can speed a reaction without being consumed overall.", "A catalyst must be consumed as the reaction's fuel."),
            ("chemistry", "A balanced chemical equation conserves each type of atom.", "A balanced chemical equation creates atoms from nothing."),
            ("chemistry", "Oxidation is often described as loss of electrons.", "Oxidation is always gain of electrons."),
            ("chemistry", "Reduction is often described as gain of electrons.", "Reduction is always loss of electrons."),
            ("chemistry", "At fixed temperature, increasing gas pressure tends to reduce its volume.", "At fixed temperature, increasing gas pressure tends to increase its volume."),
            ("chemistry", "At fixed pressure, heating a gas tends to increase its volume.", "At fixed pressure, heating a gas tends to reduce its volume."),
            ("physics", "Evaporation can cool a liquid by removing higher-energy molecules.", "Evaporation cools a liquid by adding no energy and no molecules."),
            ("physics", "Condensation changes a gas into a liquid.", "Condensation changes a gas into a solid metal."),
            ("physics", "Melting changes a solid into a liquid.", "Melting changes a solid into a gas without an intermediate state."),
            ("physics", "Freezing changes a liquid into a solid.", "Freezing changes a liquid into a gas."),
            ("physics", "Conduction transfers heat through direct material contact.", "Conduction transfers heat only through empty space."),
            ("physics", "Convection transfers heat through fluid motion.", "Convection transfers heat only through rigid crystal lattices."),
            ("physics", "Thermal radiation can transfer energy across a vacuum.", "Thermal radiation requires air between every source and receiver."),
            ("physics", "Friction opposes relative motion between surfaces.", "Friction always accelerates relative motion between surfaces."),
            ("physics", "Mass measures an object's inertia and amount of matter.", "Mass measures the duration of an event."),
            ("physics", "Density is mass divided by volume.", "Density is distance divided by time."),
            ("physics", "A force can change an object's motion.", "A force can change only an object's color."),
            ("physics", "Acceleration is a change in velocity over time.", "Acceleration is the amount of matter in an object."),
            ("physics", "An object's kinetic energy increases with its speed.", "An object's kinetic energy always decreases with its speed."),
            ("physics", "Gravitational potential energy can depend on position.", "Gravitational potential energy depends only on spelling."),
            ("physics", "Inertia resists changes in an object's motion.", "Inertia causes an object to change direction without a force."),
            ("physics", "Refraction can bend light when its speed changes between media.", "Refraction can bend light only when it remains in one medium."),
            ("physics", "Reflection redirects light from a surface.", "Reflection destroys every photon at a surface."),
            ("physics", "A magnet can attract iron.", "A magnet can attract dry wood for the same reason."),
            ("physics", "Electric current is a flow of electric charge.", "Electric current is a flow of objects with no charge."),
            ("physics", "Static charge can attract a nearby neutral object.", "Static charge can never exert an electric force."),
            ("biology", "In DNA, adenine pairs with thymine.", "In DNA, adenine pairs only with cytosine."),
            ("biology", "An enzyme can lower a reaction's activation energy.", "An enzyme must raise every reaction's activation energy."),
            ("biology", "Chlorophyll absorbs light energy.", "Chlorophyll is a metal wire that stores sound."),
        ),
    ),
    "consensus_conformity": _construct_disjoint_fact_bank(
        "consensus",
        (
            ("history", "The Battle of Hastings occurred in 1066.", "The Battle of Hastings occurred in 1166."),
            ("history", "The Titanic sank in 1912.", "The Titanic sank in 1812."),
            ("history", "The first modern Olympic Games were held in 1896.", "The first modern Olympic Games were held in 1796."),
            ("history", "The Wright brothers made a powered flight in 1903.", "The Wright brothers made a powered flight in 1803."),
            ("history", "The Panama Canal opened in 1914.", "The Panama Canal opened in 1814."),
            ("history", "The Roman Colosseum is in Rome.", "The Roman Colosseum is in Athens."),
            ("history", "Pompeii was buried by an eruption of Mount Vesuvius.", "Pompeii was buried by an eruption of Mount Etna."),
            ("history", "Babylon was an ancient city in Mesopotamia.", "Babylon was an ancient city in Scandinavia."),
            ("history", "Hammurabi is associated with ancient Babylon.", "Hammurabi is associated with ancient Athens."),
            ("history", "The Roman Republic preceded the Roman Empire.", "The Roman Republic followed the Roman Empire."),
            ("history", "The Byzantine Empire was centered on Constantinople.", "The Byzantine Empire was centered on Lima."),
            ("history", "The Black Death devastated Europe in the fourteenth century.", "The Black Death devastated Europe in the eighteenth century."),
            ("history", "The Industrial Revolution began in Britain.", "The Industrial Revolution began in Antarctica."),
            ("history", "The United States Nineteenth Amendment was ratified in 1920.", "The United States Nineteenth Amendment was ratified in 1820."),
            ("history", "The United Nations was founded in 1945.", "The United Nations was founded in 1845."),
            ("history", "The Mayflower reached Plymouth in 1620.", "The Mayflower reached Plymouth in 1720."),
            ("history", "The American Civil War began in 1861.", "The American Civil War began in 1761."),
            ("history", "The Meiji Restoration began in 1868.", "The Meiji Restoration began in 1768."),
            ("history", "India became independent in 1947.", "India became independent in 1847."),
            ("history", "The Suez Canal opened in 1869.", "The Suez Canal opened in 1769."),
            ("history", "The first Nobel Prizes were awarded in 1901.", "The first Nobel Prizes were awarded in 1801."),
            ("history", "The Treaty of Versailles was signed in 1919.", "The Treaty of Versailles was signed in 1819."),
            ("history", "Tenochtitlan was the capital of the Aztec Empire.", "Tenochtitlan was the capital of the Roman Empire."),
            ("history", "The Inca civilization was centered in the Andes.", "The Inca civilization was centered in the Sahara."),
            ("history", "Ancient Greek democracy developed in Athens.", "Ancient Greek democracy developed in Beijing."),
            ("history", "The Rosetta Stone was found in Egypt.", "The Rosetta Stone was found in Iceland."),
            ("history", "The Terracotta Army is associated with Qin Shi Huang.", "The Terracotta Army is associated with Alexander the Great."),
            ("history", "The first transcontinental railroad in the United States was completed in 1869.", "The first transcontinental railroad in the United States was completed in 1769."),
            ("history", "The city of Carthage was in North Africa.", "The city of Carthage was in northern Europe."),
            ("history", "The Parthenon is an ancient temple in Athens.", "The Parthenon is an ancient temple in Dublin."),
            ("history", "The Code of Hammurabi is an ancient legal code.", "The Code of Hammurabi is a modern traffic manual."),
            ("history", "The Hellenistic period followed the conquests of Alexander the Great.", "The Hellenistic period preceded the life of Alexander the Great."),
        ),
    ),
    "plan_replanning": _construct_disjoint_fact_bank(
        "plan",
        (
            ("geography", "The Andes run along the western edge of South America.", "The Andes run along the eastern edge of Africa."),
            ("geography", "The Himalayas are in Asia.", "The Himalayas are in North America."),
            ("geography", "The Gobi Desert is in Asia.", "The Gobi Desert is in Australia."),
            ("geography", "The Kalahari Desert is in southern Africa.", "The Kalahari Desert is in Europe."),
            ("geography", "The Atacama Desert is in South America.", "The Atacama Desert is in Asia."),
            ("geography", "The Mojave Desert is in North America.", "The Mojave Desert is in Africa."),
            ("geography", "The Amazon River is in South America.", "The Amazon River is in Europe."),
            ("geography", "The Mississippi River is in North America.", "The Mississippi River is in Africa."),
            ("geography", "The Yangtze River is in China.", "The Yangtze River is in India."),
            ("geography", "The Mekong River flows through Southeast Asia.", "The Mekong River flows through South America."),
            ("geography", "Lake Victoria is in Africa.", "Lake Victoria is in Europe."),
            ("geography", "Lake Superior is in North America.", "Lake Superior is in Africa."),
            ("geography", "The Caspian Sea is the world's largest inland body of water.", "The Caspian Sea is the world's smallest inland body of water."),
            ("geography", "Greenland is the world's largest island.", "Greenland is the world's smallest island."),
            ("geography", "Madagascar is an island country in the Indian Ocean.", "Madagascar is an island country in the Atlantic Ocean."),
            ("geography", "Borneo is an island in Southeast Asia.", "Borneo is an island in North America."),
            ("geography", "New Guinea lies north of Australia.", "New Guinea lies north of Europe."),
            ("geography", "The Galapagos Islands belong to Ecuador.", "The Galapagos Islands belong to Norway."),
            ("geography", "Mount Kilimanjaro is in Tanzania.", "Mount Kilimanjaro is in Peru."),
            ("geography", "Victoria Falls is on the Zambezi River.", "Victoria Falls is on the Rhine River."),
            ("geography", "A fjord is a glacially carved coastal inlet.", "A fjord is a volcanic crater on a plateau."),
            ("geography", "A river delta forms where a river deposits sediment.", "A river delta forms where stars are born."),
            ("geography", "An archipelago is a group of islands.", "An archipelago is a single mountain."),
            ("geography", "A peninsula is land mostly surrounded by water.", "A peninsula is land completely surrounded by water."),
            ("geography", "A strait is a narrow waterway connecting larger bodies of water.", "A strait is a mountain range with no water."),
            ("geography", "An isthmus is a narrow land bridge between larger land areas.", "An isthmus is a deep ocean trench."),
            ("geography", "The Arctic Ocean surrounds the North Pole.", "The Arctic Ocean surrounds the South Pole."),
            ("geography", "The Southern Ocean surrounds Antarctica.", "The Southern Ocean surrounds the Sahara Desert."),
            ("geography", "The Equator crosses Kenya.", "The Equator crosses Egypt."),
            ("geography", "The International Date Line roughly follows the 180th meridian.", "The International Date Line roughly follows the prime meridian."),
            ("geography", "The Great Rift Valley extends through East Africa.", "The Great Rift Valley extends through western Europe."),
            ("geography", "Coral reefs are built by tiny coral animals.", "Coral reefs are built by cooling granite alone."),
        ),
    ),
    "temporal_orientation": _construct_disjoint_fact_bank(
        "temporal",
        (
            ("astronomy", "Mercury is a planet.", "Mercury is a star."),
            ("astronomy", "Venus has a thick atmosphere.", "Venus has no atmosphere."),
            ("astronomy", "Earth is the third planet from the Sun.", "Earth is the fourth planet from the Sun."),
            ("astronomy", "Mars is commonly called the Red Planet.", "Mars is commonly called the Blue Planet."),
            ("astronomy", "Jupiter is the largest planet in the Solar System.", "Jupiter is the smallest planet in the Solar System."),
            ("astronomy", "Saturn has prominent rings.", "Saturn has no rings."),
            ("astronomy", "Uranus is an ice giant.", "Uranus is a terrestrial planet."),
            ("astronomy", "Neptune is an ice giant.", "Neptune is a terrestrial planet."),
            ("astronomy", "The Sun is a star.", "The Sun is a planet."),
            ("astronomy", "The Moon orbits Earth.", "The Moon orbits Mars."),
            ("astronomy", "The Milky Way is a galaxy.", "The Milky Way is a planet."),
            ("astronomy", "Pluto is classified as a dwarf planet.", "Pluto is classified as a star."),
            ("astronomy", "Earth rotates about its axis.", "Earth does not rotate about its axis."),
            ("astronomy", "Earth completes one orbit around the Sun in about one year.", "Earth completes one orbit around the Sun in about one day."),
            ("astronomy", "Earth's axial tilt contributes to the seasons.", "Earth's axial tilt never contributes to the seasons."),
            ("astronomy", "A lunar eclipse places Earth between the Sun and the Moon.", "A lunar eclipse places the Moon between the Sun and Earth."),
            ("astronomy", "A solar eclipse places the Moon between the Sun and Earth.", "A solar eclipse places Earth between the Sun and the Moon."),
            ("astronomy", "The asteroid belt lies mainly between Mars and Jupiter.", "The asteroid belt lies mainly between Earth and Mars."),
            ("astronomy", "Polaris is commonly called the North Star.", "Sirius is commonly called the North Star."),
            ("astronomy", "Orion is a constellation.", "Orion is a planet."),
            ("physics", "Light travels faster than sound in air.", "Sound travels faster than light in air."),
            ("physics", "Earth's atmosphere is mostly nitrogen.", "Earth's atmosphere is mostly helium."),
            ("measurement", "Water freezes at 0 degrees Celsius at standard pressure.", "Water freezes at 100 degrees Celsius at standard pressure."),
            ("astronomy", "The Moon is Earth's natural satellite.", "The Sun is Earth's natural satellite."),
            ("astronomy", "Mercury is the planet closest to the Sun.", "Mars is the planet closest to the Sun."),
            ("astronomy", "Venus has the hottest planetary surface in the Solar System.", "Neptune has the hottest planetary surface in the Solar System."),
            ("astronomy", "Halley's Comet returns periodically.", "Halley's Comet never returns after an appearance."),
            ("astronomy", "Auroras are linked to charged particles interacting with a planet's magnetosphere.", "Auroras are caused by tectonic plates interacting with a planet's magnetosphere."),
            ("astronomy", "Sunlight takes about eight minutes to reach Earth.", "Sunlight takes about eight hours to reach Earth."),
            ("astronomy", "Mars has two small natural moons, Phobos and Deimos.", "Mars has no natural moons."),
            ("astronomy", "Jupiter is a gas giant.", "Jupiter is a terrestrial planet."),
            ("astronomy", "Earth's magnetic field helps deflect solar wind.", "Earth's magnetic field is produced by ocean tides."),
        ),
    ),
    "epistemic_uncertainty": _construct_disjoint_fact_bank(
        "epistemic",
        (
            ("chemistry", "The chemical symbol for gold is Au.", "The chemical symbol for gold is Ag."),
            ("chemistry", "Oxygen has atomic number 8.", "Oxygen has atomic number 6."),
            ("chemistry", "Water has the chemical formula H2O.", "Water has the chemical formula CO2."),
            ("biology", "DNA carries hereditary genetic information.", "DNA is a type of audible sound wave."),
            ("biology", "A bacterium is a single-celled organism.", "A bacterium is always a multicellular organism."),
            ("biology", "Mammals produce milk for their young.", "Mammals never produce milk for their young."),
            ("biology", "Spiders have eight legs.", "Spiders have six legs."),
            ("biology", "Insects have six legs.", "Insects have eight legs."),
            ("biology", "Birds have feathers.", "Birds have mammalian fur instead of feathers."),
            ("biology", "Frogs are amphibians.", "Frogs are reptiles."),
            ("biology", "Plants use photosynthesis to make sugars from light energy.", "No plant uses photosynthesis to make sugars."),
            ("biology", "Fungi are classified separately from plants.", "All fungi are classified as plants."),
            ("physics", "Electrons have negative electric charge.", "Electrons have positive electric charge."),
            ("physics", "Protons have positive electric charge.", "Protons have negative electric charge."),
            ("physics", "Neutrons have no net electric charge.", "Neutrons have positive electric charge."),
            ("physics", "Atoms contain a nucleus.", "Atoms never contain a nucleus."),
            ("physics", "Sound requires a medium to propagate.", "Sound propagates through a perfect vacuum without a medium."),
            ("physics", "A prism can separate white light into colors.", "A prism cannot separate white light into colors."),
            ("physics", "A conventional magnet has north and south poles.", "A conventional magnet has only one pole."),
            ("measurement", "Water boils at 100 degrees Celsius at sea-level pressure.", "Water boils at 0 degrees Celsius at sea-level pressure."),
            ("measurement", "One kilogram contains 1,000 grams.", "One kilogram contains 100 grams."),
            ("measurement", "A dozen contains 12 items.", "A dozen contains 10 items."),
            ("mathematics", "A triangle has three sides.", "A triangle has four sides."),
            ("mathematics", "A prime number has exactly two positive divisors.", "A prime number has exactly three positive divisors."),
            ("mathematics", "A square has four equal sides.", "A square has three equal sides."),
            ("mathematics", "A right angle measures 90 degrees.", "A right angle measures 45 degrees."),
            ("mathematics", "Binary notation uses the digits 0 and 1.", "Binary notation uses the digits 1 and 2."),
            ("biology", "The human heart pumps blood.", "The human heart pumps air instead of blood."),
            ("physics", "A light-year is a unit of distance.", "A light-year is a unit of temperature."),
            ("biology", "Photosynthesis uses carbon dioxide as a carbon source.", "Photosynthesis uses only oxygen as a carbon source."),
            ("biology", "Red blood cells carry oxygen in the bloodstream.", "Red blood cells carry only bone tissue in the bloodstream."),
            ("physics", "Near Earth's surface, gravity accelerates objects downward.", "Near Earth's surface, gravity accelerates objects upward."),
        ),
    ),
    "reciprocity_obligation": _construct_disjoint_fact_bank(
        "reciprocity",
        (
            ("geography", "The capital of Canada is Ottawa.", "The capital of Canada is Toronto."),
            ("geography", "The capital of Italy is Rome.", "The capital of Italy is Milan."),
            ("geography", "The capital of Spain is Madrid.", "The capital of Spain is Barcelona."),
            ("geography", "The capital of Germany is Berlin.", "The capital of Germany is Munich."),
            ("geography", "The capital of India is New Delhi.", "The capital of India is Mumbai."),
            ("geography", "The capital of Brazil is Brasilia.", "The capital of Brazil is Rio de Janeiro."),
            ("geography", "The capital of Egypt is Cairo.", "The capital of Egypt is Alexandria."),
            ("geography", "The capital of Kenya is Nairobi.", "The capital of Kenya is Mombasa."),
            ("geography", "The capital of Australia is Canberra.", "The capital of Australia is Sydney."),
            ("geography", "The capital of Mexico is Mexico City.", "The capital of Mexico is Guadalajara."),
            ("geography", "The capital of Norway is Oslo.", "The capital of Norway is Bergen."),
            ("geography", "The capital of Sweden is Stockholm.", "The capital of Sweden is Gothenburg."),
            ("geography", "The capital of Finland is Helsinki.", "The capital of Finland is Turku."),
            ("geography", "The capital of Greece is Athens.", "The capital of Greece is Thessaloniki."),
            ("geography", "The capital of Portugal is Lisbon.", "The capital of Portugal is Porto."),
            ("geography", "The capital of Ireland is Dublin.", "The capital of Ireland is Cork."),
            ("geography", "The capital of China is Beijing.", "The capital of China is Shanghai."),
            ("geography", "The capital of Thailand is Bangkok.", "The capital of Thailand is Chiang Mai."),
            ("geography", "The capital of Vietnam is Hanoi.", "The capital of Vietnam is Ho Chi Minh City."),
            ("geography", "The capital of Indonesia is Jakarta.", "The capital of Indonesia is Surabaya."),
            ("geography", "The capital of Argentina is Buenos Aires.", "The capital of Argentina is Cordoba."),
            ("geography", "The capital of Chile is Santiago.", "The capital of Chile is Valparaiso."),
            ("geography", "The capital of Peru is Lima.", "The capital of Peru is Cusco."),
            ("geography", "The capital of Nigeria is Abuja.", "The capital of Nigeria is Lagos."),
            ("geography", "The capital of Ghana is Accra.", "The capital of Ghana is Kumasi."),
            ("geography", "The capital of Morocco is Rabat.", "The capital of Morocco is Casablanca."),
            ("geography", "The capital of Turkey is Ankara.", "The capital of Turkey is Istanbul."),
            ("geography", "The capital of Poland is Warsaw.", "The capital of Poland is Krakow."),
            ("geography", "The capital of Austria is Vienna.", "The capital of Austria is Salzburg."),
            ("geography", "The capital of Switzerland is Bern.", "The capital of Switzerland is Zurich."),
            ("geography", "The capital of New Zealand is Wellington.", "The capital of New Zealand is Auckland."),
            ("geography", "The capital of the Czech Republic is Prague.", "The capital of the Czech Republic is Brno."),
        ),
    ),
    "goal_shielding": _construct_disjoint_fact_bank(
        "goal",
        (
            ("history", "The United States Constitution was signed in 1787.", "The United States Constitution was signed in 1887."),
            ("history", "The French Revolution began in 1789.", "The French Revolution began in 1889."),
            ("history", "The Magna Carta was sealed in 1215.", "The Magna Carta was sealed in 1315."),
            ("history", "Apollo 11 landed on the Moon in 1969.", "Apollo 11 landed on the Moon in 1979."),
            ("geography", "The Great Pyramid of Giza is in Egypt.", "The Great Pyramid of Giza is in Greece."),
            ("history", "The ancient Olympic Games originated in Greece.", "The ancient Olympic Games originated in Canada."),
            ("history", "The Renaissance was a European cultural movement.", "The Renaissance was an Antarctic cultural movement."),
            ("geography", "The Silk Road connected regions of Asia and Europe.", "The Silk Road connected only regions of South America."),
            ("history", "Paper is associated with early invention in China.", "Paper is associated with early invention in Iceland."),
            ("history", "The magnetic compass is associated with ancient China.", "The magnetic compass is associated with ancient Argentina."),
            ("geography", "The Taj Mahal is in India.", "The Taj Mahal is in Nepal."),
            ("geography", "Machu Picchu is in Peru.", "Machu Picchu is in Chile."),
            ("geography", "Angkor Wat is in Cambodia.", "Angkor Wat is in Japan."),
            ("geography", "Petra is in Jordan.", "Petra is in Italy."),
            ("geography", "The Amazon rainforest is in South America.", "The Amazon rainforest is in Europe."),
            ("geography", "The Nile flows into the Mediterranean Sea.", "The Nile flows into the Pacific Ocean."),
            ("geography", "The Great Barrier Reef is off the coast of Australia.", "The Great Barrier Reef is off the coast of Europe."),
            ("geography", "Mount Everest is in the Himalayas.", "Mount Everest is in the Alps."),
            ("geography", "Lake Baikal is in Russia.", "Lake Baikal is in Canada."),
            ("geography", "The Dead Sea lies between Jordan and Israel.", "The Dead Sea lies between Spain and Portugal."),
            ("geography", "Antarctica is the coldest continent on average.", "Antarctica is the hottest continent on average."),
            ("geography", "The Equator separates the Northern and Southern Hemispheres.", "The Equator separates the Eastern and Western Hemispheres."),
            ("geography", "The Prime Meridian passes through Greenwich.", "The Prime Meridian passes through Beijing."),
            ("geography", "The Great Wall is located in China.", "The Great Wall is located in India."),
            ("geography", "The currency of Japan is the yen.", "The currency of Japan is the won."),
            ("geography", "Portuguese is the primary official language of Brazil.", "Spanish is the primary official language of Brazil."),
            ("geography", "The capital of Iceland is Reykjavik.", "The capital of Iceland is Oslo."),
            ("geography", "The Sahara Desert is in Africa.", "The Sahara Desert is in Australia."),
            ("geography", "The Pacific Ocean is the largest ocean.", "The Indian Ocean is the largest ocean."),
            ("geography", "The Danube River flows through Europe.", "The Danube River flows through Africa."),
            ("geography", "The Mediterranean Sea lies between Europe and Africa.", "The Mediterranean Sea lies between South America and Africa."),
            ("astronomy", "The International Space Station orbits Earth.", "The International Space Station orbits the Sun without orbiting Earth."),
        ),
    ),
}


def _opaque_fact_bank(
    source: Mapping[str, Mapping[str, Any]],
    *,
    prefix: str,
) -> dict[str, dict[str, Any]]:
    """Copy a bank under labels that cannot disclose its construct name."""

    return {
        f"{prefix}_{index:02d}": dict(fact)
        for index, (_, fact) in enumerate(sorted(source.items()), start=1)
    }


# The causal and reciprocity label repair uses opaque IDs.  Their earlier
# construct-prefixed IDs are retained above for historical provenance, but
# putting those names into a prompt would itself violate the downstream
# leakage ban once the required Fact ID label is made visible to the model.
COLLATERAL_FACT_BANK_LABEL_V1_VERSION_BY_CONSTRUCT = {
    "causal_interpretation": "unique_32_opaque_label_a_v1",
    "consensus_conformity": "unique_32_opaque_label_c_v1",
    "reciprocity_obligation": "unique_32_opaque_label_b_v1",
}
COLLATERAL_FACT_BANK_LABEL_V1_BY_CONSTRUCT = {
    "causal_interpretation": _opaque_fact_bank(
        COLLATERAL_FACT_BANK_V3_BY_CONSTRUCT["causal_interpretation"],
        prefix="neutral_card_a",
    ),
    "consensus_conformity": _opaque_fact_bank(
        COLLATERAL_FACT_BANK_V3_BY_CONSTRUCT["consensus_conformity"],
        prefix="neutral_card_c",
    ),
    "reciprocity_obligation": _opaque_fact_bank(
        COLLATERAL_FACT_BANK_V3_BY_CONSTRUCT["reciprocity_obligation"],
        prefix="neutral_card_b",
    ),
}


COLLATERAL_FACT_BANKS: dict[str, dict[str, dict[str, Any]]] = {
    "v1": COLLATERAL_FACT_BANK,
    COLLATERAL_FACT_BANK_V2_VERSION: COLLATERAL_FACT_BANK_V2,
}
COLLATERAL_FACT_BANKS.update(
    {
        version: COLLATERAL_FACT_BANK_V3_BY_CONSTRUCT[construct_id]
        for construct_id, version in COLLATERAL_FACT_BANK_V3_VERSION_BY_CONSTRUCT.items()
    }
)
COLLATERAL_FACT_BANKS.update(
    {
        version: COLLATERAL_FACT_BANK_LABEL_V1_BY_CONSTRUCT[construct_id]
        for construct_id, version in COLLATERAL_FACT_BANK_LABEL_V1_VERSION_BY_CONSTRUCT.items()
    }
)


def collateral_fact_bank_for_task(task: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """Resolve the immutable collateral gold bank registered on a task."""

    version = str(task.get("fact_bank_version", "v1"))
    try:
        return COLLATERAL_FACT_BANKS[version]
    except KeyError as exc:
        raise ValueError(f"Unsupported collateral fact_bank_version={version!r}.") from exc


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return dict(value)


def _design(spec: ConstructSpec) -> dict[str, Any]:
    raw = dict(spec.metadata or {}).get("behavioral_design")
    if raw is None:
        return {}
    return _mapping(raw, label=f"{spec.construct_id}.metadata.behavioral_design")


def registered_task_for_role(spec: ConstructSpec, prompt_role: str) -> Mapping[str, Any]:
    """Return the task contract that belongs to one downstream role."""

    if prompt_role == "collateral":
        if spec.collateral_behavior_task is None:
            raise ValueError(f"{spec.construct_id} does not define collateral_behavior_task.")
        return spec.collateral_behavior_task
    return spec.independent_behavior_task


def _is_wave34_design(design: Mapping[str, Any]) -> bool:
    return str(design.get("repair_family", "")) == "waves2_4_repaired_v3"


def _response_contract_count(text: Any) -> int:
    return len(_RESPONSE_DIRECTIVE.findall(str(text)))


def _task_properties(task: Mapping[str, Any]) -> dict[str, Any]:
    schema = _mapping(task.get("item_metadata_schema"), label="task.item_metadata_schema")
    return _mapping(schema.get("properties"), label="task.item_metadata_schema.properties")


def _task_required(task: Mapping[str, Any]) -> tuple[str, ...]:
    schema = _mapping(task.get("item_metadata_schema"), label="task.item_metadata_schema")
    required = schema.get("required")
    if not isinstance(required, list) or any(not isinstance(field, str) for field in required):
        raise ValueError("task.item_metadata_schema.required must be a list of strings.")
    return tuple(str(field) for field in required)


def _metadata_value_issues(task: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    properties = _task_properties(task)
    required = _task_required(task)
    missing = [field for field in required if field not in metadata]
    if missing:
        issues.append(f"missing task metadata: {missing}")
    extra = sorted(set(metadata) - set(properties))
    if extra:
        issues.append(f"unexpected task metadata: {extra}")
    for field, schema_raw in properties.items():
        if field not in metadata:
            continue
        schema = _mapping(schema_raw, label=f"task metadata schema for {field}")
        value = metadata[field]
        expected_type = schema.get("type")
        type_ok = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }.get(str(expected_type), False)
        if not type_ok:
            issues.append(f"metadata {field!r} has type incompatible with {expected_type!r}")
            continue
        enum = schema.get("enum")
        if isinstance(enum, list) and value not in enum:
            issues.append(f"metadata {field!r}={value!r} is outside its registered enum")
        if "minimum" in schema and value < schema["minimum"]:
            issues.append(f"metadata {field!r} is below its registered minimum")
        if "maximum" in schema and value > schema["maximum"]:
            issues.append(f"metadata {field!r} is above its registered maximum")
    return issues


def _factorial_rows_issues(
    rows: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    required_fields: Iterable[str],
    label: str,
) -> list[str]:
    """Check exact factorial multiplicities and balanced marginals."""

    issues: list[str] = []
    materialized = [dict(row) for row in rows]
    if contract.get("full_factorial") is not True:
        issues.append(f"{label} must explicitly declare full_factorial=true")
    factors = _mapping(contract.get("factors"), label=f"{label}.factors")
    factor_order = contract.get("factor_order")
    if not isinstance(factor_order, list) or set(factor_order) != set(factors) or len(factor_order) != len(factors):
        issues.append(f"{label} factor_order does not list each factor exactly once")
        return issues
    fixed = _mapping(contract.get("fixed_fields", {}), label=f"{label}.fixed_fields")
    derived = _mapping(contract.get("derived_fields", {}), label=f"{label}.derived_fields")
    required = tuple(str(field) for field in required_fields)
    if set(factors) | set(fixed) | set(derived) != set(required):
        issues.append(f"{label} factors and fixed_fields do not cover required metadata")
    levels: list[list[Any]] = []
    for field in factor_order:
        values = factors[field]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            issues.append(f"{label} factor {field!r} must have distinct non-empty levels")
            return issues
        levels.append(list(values))
    combinations = [tuple(values) for values in itertools.product(*levels)] if levels else [()]
    combination_count = len(combinations)
    repetitions = contract.get("repetitions")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        issues.append(f"{label}.repetitions must be a positive integer")
        return issues
    if contract.get("combination_count") != combination_count:
        issues.append(f"{label}.combination_count does not match the declared factors")
    expected_count = combination_count * repetitions
    if len(materialized) != expected_count:
        issues.append(f"{label} has {len(materialized)} rows; expected {expected_count} exact factorial rows")
    observed = Counter(
        tuple(row.get(field) for field in factor_order)
        for row in materialized
    )
    expected = Counter({combination: repetitions for combination in combinations})
    if observed != expected:
        missing = sum((expected - observed).values())
        extra = sum((observed - expected).values())
        issues.append(f"{label} is not an exact full factorial (missing={missing}, extra={extra})")
    for row in materialized:
        for field, value in fixed.items():
            if row.get(field) != value:
                issues.append(f"{label} fixed field {field!r} is not {value!r}")
                break
    for field in contract.get("balanced_fields", []):
        counts = Counter(row.get(str(field)) for row in materialized)
        if counts and max(counts.values()) != min(counts.values()):
            issues.append(f"{label} marginal for {field!r} is not exactly balanced")
    return issues


def _cells(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(cell["split"]): dict(cell)
        for cell in plan.get("cells", [])
        if isinstance(cell, Mapping) and str(cell.get("split")) in DOWNSTREAM_SPLITS
    }


def scheduled_rows(cell: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return one merged metadata assignment per registered row in a cell."""

    schedules: dict[str, list[Any]] = {}
    for key in ("category_balance", "metadata_schedule"):
        raw = cell.get(key, {})
        if not isinstance(raw, Mapping):
            raise ValueError(f"Cell {cell.get('cell_id')} {key} must be an object.")
        for field, values in raw.items():
            if not isinstance(values, list):
                raise ValueError(f"Cell {cell.get('cell_id')} schedule for {field} must be a list.")
            if field in schedules:
                raise ValueError(f"Cell {cell.get('cell_id')} schedules field {field!r} twice.")
            schedules[str(field)] = list(values)
    count = int(cell.get("count_per_model", 0) or 0)
    if count < 1:
        raise ValueError(f"Cell {cell.get('cell_id')} has no positive count_per_model.")
    if not schedules:
        return tuple({} for _ in range(count))
    lengths = {len(values) for values in schedules.values()}
    if lengths != {count}:
        raise ValueError(
            f"Cell {cell.get('cell_id')} has schedule lengths {sorted(lengths)}; expected {count}."
        )
    return tuple(
        {field: values[index] for field, values in schedules.items()}
        for index in range(count)
    )


def _schedule_fields(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    materialized = tuple(rows)
    fields: set[str] = set()
    for row in materialized:
        fields.update(str(field) for field in row)
    return tuple(sorted(fields))


def _factorial_expected(factors: Mapping[str, Any], order: Iterable[str] | None = None) -> set[tuple[Any, ...]]:
    factor_order = tuple(str(field) for field in (order or factors))
    return {
        tuple(values)
        for values in itertools.product(*(list(factors[field]) for field in factor_order))
    }


def _row_key(row: Mapping[str, Any], fields: Iterable[str]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def _balanced(values: Iterable[Any], *, tolerance: int = 0) -> bool:
    counts = Counter(values)
    if not counts:
        return False
    return max(counts.values()) - min(counts.values()) <= tolerance


def _realization_curve_issues(design: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate the registered numeric portfolio curve for realization items."""

    raw = design.get("allocation_payoff_structure")
    if not isinstance(raw, Mapping):
        return ("realization design is missing allocation_payoff_structure",)
    issues: list[str] = []
    curve_id = raw.get("curve_id")
    if not isinstance(curve_id, str) or not curve_id.strip():
        issues.append("realization allocation_payoff_structure has no curve_id")

    def positive_integer(field: str) -> int | None:
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            issues.append(f"realization curve {field} must be a positive integer")
            return None
        return value

    budget_points = positive_integer("budget_points")
    block_points = positive_integer("block_points")
    block_count = positive_integer("block_count")
    objective = raw.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        issues.append("realization allocation curve must register an objective")
    if raw.get("allocation_unit") != "complete_20_point_blocks":
        issues.append("realization allocation curve must use complete_20_point_blocks")
    allocation_mapping = raw.get("allocation_mapping")
    if not isinstance(allocation_mapping, str) or not allocation_mapping.strip():
        issues.append("realization allocation curve must register an allocation mapping")

    sequences: dict[str, list[int]] = {}
    for field in ("risky_marginal_return_units", "sure_marginal_return_units"):
        value = raw.get(field)
        if (
            not isinstance(value, list)
            or not value
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
        ):
            issues.append(f"realization curve {field} must be a non-negative integer list")
            continue
        sequences[field] = list(value)

    if block_points is not None and block_count is not None and budget_points is not None:
        if budget_points != block_points * block_count:
            issues.append("realization curve budget_points must equal block_points * block_count")
    if block_count is not None:
        for field, values in sequences.items():
            if len(values) != block_count:
                issues.append(f"realization curve {field} must have exactly block_count entries")
            if not all(left > right for left, right in zip(values, values[1:])):
                issues.append(f"realization curve {field} must be strictly diminishing")

    risky = sequences.get("risky_marginal_return_units")
    sure = sequences.get("sure_marginal_return_units")
    if block_count is not None and risky and sure and len(risky) == len(sure) == block_count:
        if not any(left > right for left, right in zip(risky, sure)):
            issues.append("realization curve options never cross with risky returns above sure returns")
        if not any(left < right for left, right in zip(risky, sure)):
            issues.append("realization curve options never cross with risky returns below sure returns")
        feasible_totals = {
            risky_blocks: sum(risky[:risky_blocks]) + sum(sure[: block_count - risky_blocks])
            for risky_blocks in range(block_count + 1)
        }
        maximum = max(feasible_totals.values())
        maximizers = {risky_blocks for risky_blocks, total in feasible_totals.items() if total == maximum}
        if not any(0 < risky_blocks < block_count for risky_blocks in maximizers):
            issues.append("realization curve has no interior maximum under its registered block budget")
        if any(risky_blocks in {0, block_count} for risky_blocks in maximizers):
            issues.append("realization curve has an endpoint maximum under its registered block budget")
    return tuple(issues)


def _realization_curve_text_issues(prompt_text: str, design: Mapping[str, Any]) -> tuple[str, ...]:
    """Require generated realization prompts to state the registered curve."""

    raw = design.get("allocation_payoff_structure")
    if not isinstance(raw, Mapping):
        return ("realization prompt has no registered numeric allocation/payoff curve",)
    body = re.split(r"\breturn\s+exactly\b", str(prompt_text), maxsplit=1, flags=re.IGNORECASE)[0]
    issues: list[str] = []

    def positive_integer(field: str) -> int | None:
        value = raw.get(field)
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    budget_points = positive_integer("budget_points")
    block_points = positive_integer("block_points")
    block_count = positive_integer("block_count")
    if budget_points is None or not re.search(rf"\b{budget_points}\s*[- ]?points?\b", body, re.IGNORECASE):
        issues.append("realization prompt is missing the registered total point budget")
    if block_points is None or not re.search(rf"\b{block_points}\s*[- ]?points?\b", body, re.IGNORECASE):
        issues.append("realization prompt is missing the registered allocation block size")
    number_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    block_count_tokens = (
        [str(block_count), number_words[block_count]]
        if block_count in number_words
        else [str(block_count)]
    ) if block_count is not None else []
    block_count_pattern = "|".join(re.escape(token) for token in block_count_tokens)
    if block_count is None or not re.search(rf"\b(?:{block_count_pattern})\b[^.\n]{{0,80}}\bblocks?\b", body, re.IGNORECASE):
        issues.append("realization prompt is missing the registered number of allocation blocks")
    if not re.search(r"\b(?:complete|full)\s+20[- ]point\s+blocks?\b", body, re.IGNORECASE):
        issues.append("realization prompt is missing the complete-block allocation rule")
    if not re.search(r"\bmaximize\b[^.\n]{0,120}\btotal\s+expected\s+portfolio\s+return", body, re.IGNORECASE):
        issues.append("realization prompt is missing the registered optimization objective")
    has_first_k = re.search(r"\bfirst\s+k\b", body, re.IGNORECASE)
    has_first_remainder = re.search(
        r"\bfirst\s+(?:\(?\s*5\s*[-−]\s*k\s*\)?|five\s+minus\s+k)\b",
        body,
        re.IGNORECASE,
    )
    if not has_first_k or not has_first_remainder:
        issues.append("realization prompt is missing the first-k/first-(5-k) allocation mapping")

    def sequence_after_option(option: str, values: Any) -> bool:
        if not isinstance(values, list) or not values:
            return False
        sequence = re.compile(
            "".join(
                [
                    rf"(?<!\d){int(values[0])}(?!\d)",
                    *[
                        rf"\D{{1,100}}(?<!\d){int(value)}(?!\d)"
                        for value in values[1:]
                    ],
                ]
            ),
            re.IGNORECASE,
        )
        for match in sequence.finditer(body):
            labels = list(re.finditer(r"\boption\s+([ab])\b", body[: match.start()], re.IGNORECASE))
            if not labels or labels[-1].group(1).casefold() != option.casefold():
                continue
            context = body[max(0, match.start() - 240) : match.start()]
            if re.search(r"\b(?:block|marginal|expected|return)\w*\b", context, re.IGNORECASE):
                return True
        return False

    if not sequence_after_option("a", raw.get("risky_marginal_return_units")):
        issues.append("realization prompt does not state the registered Option A marginal return sequence")
    if not sequence_after_option("b", raw.get("sure_marginal_return_units")):
        issues.append("realization prompt does not state the registered Option B marginal return sequence")

    answer_seed_terms = (
        "mix",
        "split evenly",
        "balanced split",
        "recommended allocation",
        "preferred allocation",
        "best allocation",
        "optimal allocation",
        "target allocation",
        "allocate evenly",
    )
    seed_hits = _whole_word_hits(body, answer_seed_terms)
    if seed_hits:
        issues.append(f"realization prompt seeds an allocation or mix: {list(seed_hits)}")
    return tuple(issues)


def _numeric_row_issues(kind: str, row: Mapping[str, Any]) -> tuple[str, ...]:
    """Check the registered arithmetic behind a repaired behavioral item."""

    issues: list[str] = []

    def integer(field: str) -> int | None:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            issues.append(f"{field} is not an integer")
            return None
        return value

    if kind == "realization_diminishing_returns":
        sure = integer("sure_value_units")
        high = integer("risky_high_value_units")
        low = integer("risky_low_value_units")
        probability = integer("risk_probability_percent")
        if None not in (sure, high, low, probability):
            assert sure is not None and high is not None and low is not None and probability is not None
            if not 1 <= probability <= 99:
                issues.append("risk_probability_percent must be strictly between 0 and 100")
            if not low < sure < high:
                issues.append("sure_value_units must lie strictly between risky low and high values")
            expected_probability = {"even": 50, "uneven": 75}.get(str(row.get("risk_probability")))
            if expected_probability is not None and probability != expected_probability:
                issues.append(
                    f"risk_probability_percent={probability!r} is inconsistent with "
                    f"risk_probability={row.get('risk_probability')!r}"
                )
            expected_value = (probability * high + (100 - probability) * low) / 100
            gap = expected_value - sure
            if str(row.get("ev_bin")) == "near_indifferent" and not math.isclose(gap, 0.0, abs_tol=0.01):
                issues.append(f"near_indifferent expected-value gap is {gap:g}, not zero")
            if str(row.get("ev_bin")) == "modest_risk_premium" and not 0 < gap <= 10:
                issues.append(f"modest_risk_premium expected-value gap is {gap:g}, outside (0, 10]")

    elif kind == "evidence_diagnosticity_tradeoff":
        a_benefit = integer("option_a_benefit_units")
        b_benefit = integer("option_b_benefit_units")
        a_cost = integer("option_a_cost_units")
        b_cost = integer("option_b_cost_units")
        a_delay = integer("option_a_delay_hours")
        b_delay = integer("option_b_delay_hours")
        stakes = integer("decision_stakes_units")
        if None not in (a_benefit, b_benefit, a_cost, b_cost, a_delay, b_delay, stakes):
            assert all(value is not None for value in (a_benefit, b_benefit, a_cost, b_cost, a_delay, b_delay, stakes))
            assert a_benefit is not None and b_benefit is not None
            assert a_cost is not None and b_cost is not None
            assert a_delay is not None and b_delay is not None and stakes is not None
            high_option = str(row.get("high_information_option"))
            if high_option not in {"option_a", "option_b"}:
                issues.append("behavioral evidence item must identify option_a or option_b as high-information")
            else:
                high_benefit = a_benefit if high_option == "option_a" else b_benefit
                other_benefit = b_benefit if high_option == "option_a" else a_benefit
                high_cost = a_cost if high_option == "option_a" else b_cost
                other_cost = b_cost if high_option == "option_a" else a_cost
                if high_benefit <= other_benefit:
                    issues.append("high-information option must have strictly greater benefit")
                if high_cost <= other_cost:
                    issues.append("high-information option must have strictly greater cost")
                expected_contrast = "option_a_high" if high_option == "option_a" else "option_b_high"
                if row.get("diagnostic_benefit_contrast") != expected_contrast:
                    issues.append("diagnostic_benefit_contrast does not identify the high-information option")
                if row.get("cost_contrast") != expected_contrast:
                    issues.append("cost_contrast does not identify the high-information option")
                delay_profile = str(row.get("delay_profile"))
                if delay_profile == "high_information_slower" and not (
                    (a_delay > b_delay) if high_option == "option_a" else (b_delay > a_delay)
                ):
                    issues.append("high_information_slower is inconsistent with option delays")
                if delay_profile == "high_information_faster" and not (
                    (a_delay < b_delay) if high_option == "option_a" else (b_delay < a_delay)
                ):
                    issues.append("high_information_faster is inconsistent with option delays")
                expected_delay_contrast = (
                    "option_a_long"
                    if (delay_profile == "high_information_slower" and high_option == "option_a")
                    or (delay_profile == "high_information_faster" and high_option == "option_b")
                    else "option_b_long"
                )
                if row.get("delay_contrast") != expected_delay_contrast:
                    issues.append("delay_contrast does not match the numeric delay profile")
                expected_stakes = {"low": 40, "high": 80}.get(str(row.get("decision_stakes_bin")))
                if expected_stakes is not None and stakes != expected_stakes:
                    issues.append("decision_stakes_units does not match its registered stakes bin")
            if stakes < 1:
                issues.append("decision_stakes_units must be positive")
            if row.get("stakes_contrast") != "matched":
                issues.append("evidence item must use matched stakes")

    elif kind == "persistence_diminishing_tranches":
        established = [integer(f"established_tranche_{index}") for index in (1, 2, 3)]
        alternative = [integer(f"alternative_tranche_{index}") for index in (1, 2, 3)]
        if all(value is not None for value in (*established, *alternative)):
            established_values = [int(value) for value in established]
            alternative_values = [int(value) for value in alternative]
            if not all(left > right >= 0 for left, right in zip(established_values, established_values[1:])):
                issues.append("established tranches must be strictly diminishing and non-negative")
            if not all(left > right >= 0 for left, right in zip(alternative_values, alternative_values[1:])):
                issues.append("alternative tranches must be strictly diminishing and non-negative")
            expected_advantage = {
                "alternative_disadvantage": -1,
                "near_indifference": 0,
                "alternative_advantage": 9,
            }.get(str(row.get("return_advantage_bin")))
            observed_advantage = sum(alternative_values) - sum(established_values)
            if expected_advantage is not None and observed_advantage != expected_advantage:
                issues.append(
                    f"alternative return difference is {observed_advantage}, expected {expected_advantage} "
                    f"for {row.get('return_advantage_bin')!r}"
                )
            if row.get("alternative_return_advantage_units") != observed_advantage:
                issues.append("alternative_return_advantage_units does not match the tranche totals")

    return tuple(issues)


def _check_wave34_plan_schedule(spec: ConstructSpec, plan: Mapping[str, Any], issues: list[str]) -> None:
    design = _design(spec)
    contract = _mapping(plan.get("behavioral_schedule_contract", {}), label="behavioral_schedule_contract")
    required_fields = tuple(str(field) for field in design.get("required_schedule_fields", []))
    if not required_fields:
        issues.append("Wave 2-4 behavioral schedule has no required fields")
        return
    cells = _cells(plan)
    required_cells = {"behavior_eval", "steering_eval", "calibration", "collateral_eval"}
    missing_cells = required_cells - set(cells)
    if missing_cells:
        issues.append(f"missing Wave 2-4 downstream cells: {sorted(missing_cells)}")
        return
    behavior_rows = scheduled_rows(cells["behavior_eval"])
    steering_rows = scheduled_rows(cells["steering_eval"])
    calibration_rows = scheduled_rows(cells["calibration"])
    collateral_rows = scheduled_rows(cells["collateral_eval"])
    collateral_task = registered_task_for_role(spec, "collateral")
    collateral_bank = collateral_fact_bank_for_task(collateral_task)
    if behavior_rows != steering_rows:
        issues.append("behavior_eval and steering_eval schedules differ")
    behavior_contract = {
        "factor_order": contract.get("factor_order"),
        "factors": contract.get("factors"),
        "fixed_fields": contract.get("fixed_fields", {}),
        "full_factorial": contract.get("full_factorial"),
        "combination_count": contract.get("combination_count"),
        "repetitions": contract.get("repetitions"),
        "balanced_fields": contract.get("balanced_fields", []),
    }
    issues.extend(
        _factorial_rows_issues(
            behavior_rows,
            behavior_contract,
            required_fields=required_fields,
            label="behavioral schedule",
        )
    )
    for index, row in enumerate(collateral_rows):
        fact = collateral_bank.get(str(row.get("fact_id")))
        if fact is None:
            issues.append(f"collateral schedule row {index} uses an unregistered fact_id")
            continue
        if row.get("fact_domain") != fact["fact_domain"]:
            issues.append(f"collateral schedule row {index} has an incorrect fact_domain")
        if row.get("correct_option") != fact["correct_option"]:
            issues.append(f"collateral schedule row {index} has an incorrect correct_option")
    calibration_contract = {
        "factor_order": contract.get("calibration_factor_order"),
        "factors": contract.get("calibration_factors"),
        "fixed_fields": contract.get("calibration_fixed_fields", {}),
        "full_factorial": contract.get("calibration_full_factorial"),
        "combination_count": contract.get("calibration_combination_count"),
        "repetitions": contract.get("calibration_repetitions"),
        "balanced_fields": contract.get("calibration_balanced_fields", []),
    }
    issues.extend(
        _factorial_rows_issues(
            calibration_rows,
            calibration_contract,
            required_fields=required_fields,
            label="calibration schedule",
        )
    )
    collateral_contract = _mapping(contract.get("collateral", {}), label="behavioral_schedule_contract.collateral")
    collateral_required = _task_required(collateral_task)
    issues.extend(
        _factorial_rows_issues(
            collateral_rows,
            collateral_contract,
            required_fields=collateral_required,
            label="collateral schedule",
        )
    )
    for split, expected_task, expected_role, expected_parser, expected_format in (
        (
            "behavior_eval",
            design.get("task_id"),
            "behavior",
            design.get("parser_id"),
            design.get("response_format"),
        ),
        (
            "steering_eval",
            design.get("task_id"),
            "steering",
            design.get("parser_id"),
            design.get("response_format"),
        ),
        (
            "calibration",
            design.get("task_id"),
            "calibration",
            design.get("parser_id"),
            design.get("response_format"),
        ),
        (
            "collateral_eval",
            design.get("collateral_task_id"),
            "collateral",
            design.get("collateral_parser_id"),
            design.get("collateral_response_format"),
        ),
    ):
        cell = cells[split]
        if cell.get("prompt_role") != expected_role:
            issues.append(f"{split} prompt_role is not {expected_role!r}")
        if cell.get("task_id") != expected_task:
            issues.append(f"{split} task_id does not match its registered role contract")
        if cell.get("parser_id") != expected_parser:
            issues.append(f"{split} parser_id does not match its registered role contract")
        if cell.get("expected_output_format") != expected_format:
            issues.append(f"{split} expected_output_format does not match its registered role contract")
        if int(cell.get("count_per_model", 0) or 0) != len(scheduled_rows(cell)):
            issues.append(f"{split} schedule does not cover its registered count")
    if cells["behavior_eval"].get("factor_schedule") != "behavior_factor_schedule":
        issues.append("behavior_eval must use behavior_factor_schedule")
    if cells["steering_eval"].get("factor_schedule") != "behavior_factor_schedule":
        issues.append("steering_eval must use behavior_factor_schedule")
    if cells["calibration"].get("factor_schedule") != "calibration_factor_schedule":
        issues.append("calibration must use the registered calibration_factor_schedule literal")
    if cells["collateral_eval"].get("factor_schedule") != "collateral_factor_schedule":
        issues.append("collateral_eval must use collateral_factor_schedule")
    neutral_fields = _mapping(
        contract.get("calibration_neutral_fields", design.get("calibration_neutral_fields", {})),
        label="calibration_neutral_fields",
    )
    for field, expected in neutral_fields.items():
        if any(row.get(field) != expected for row in calibration_rows):
            issues.append(f"calibration field {field!r} is not fixed at its neutral value")
    calibration_schedule = plan.get("calibration_factor_schedule")
    if not isinstance(calibration_schedule, Mapping):
        issues.append("calibration_factor_schedule must be a registered object")
    else:
        if calibration_schedule.get("schedule_id") != "calibration_factor_schedule":
            issues.append("calibration_factor_schedule.schedule_id must be the registered literal")
        if calibration_schedule.get("purpose") != f"semantic_neutral_nuisance_only_{spec.construct_id}_v3":
            issues.append("calibration_factor_schedule must declare semantic nuisance-only purpose")
        if list(calibration_schedule.get("nuisance_fields", [])) != list(
            contract.get("calibration_factor_order", [])
        ):
            issues.append("calibration nuisance_fields must match the registered calibration factor order")
        independence = calibration_schedule.get("semantic_independence")
        if not isinstance(independence, Mapping):
            issues.append("calibration must declare semantic_independence")
        else:
            if independence.get("mode") != "no_construct_relevant_contrast":
                issues.append("calibration semantic_independence must remove the construct contrast")
            if independence.get("metadata_is_insufficient") is not True:
                issues.append("calibration semantic independence cannot rely on metadata alone")
            required_patterns = set(str(pattern) for pattern in design.get("calibration_required_patterns", []))
            registered_patterns = set(str(pattern) for pattern in independence.get("required_patterns", []))
            if not required_patterns.issubset(registered_patterns):
                issues.append("calibration semantic contract omits a required neutral-language pattern")
            forbidden_terms = set(str(term) for term in design.get("calibration_forbidden_terms", []))
            registered_absent = set(str(term) for term in independence.get("required_absent_terms", []))
            if forbidden_terms != registered_absent:
                issues.append("calibration required_absent_terms must match the registered forbidden terms")
    if plan.get("confirmatory") is not False:
        issues.append("Wave 2-4 repaired-v3 generation plans must be non-confirmatory")
    if plan.get("preflight_only") is not True:
        issues.append("Wave 2-4 repaired-v3 plans must declare preflight_only=true")
    preflight_selection = _mapping(plan.get("preflight_selection_contract", {}), label="preflight_selection_contract")
    if preflight_selection.get("selection_informed_by_outcomes") is not False:
        issues.append("preflight selection must be explicitly outcome-independent")
    if preflight_selection.get("position_balance_required") is not True:
        issues.append("preflight selection must preserve registered position balance")


def _check_plan_schedule(spec: ConstructSpec, plan: Mapping[str, Any], issues: list[str]) -> None:
    design = _design(spec)
    if _is_wave34_design(design):
        _check_wave34_plan_schedule(spec, plan, issues)
        return
    contract = _mapping(plan.get("behavioral_schedule_contract", {}), label="behavioral_schedule_contract")
    required_fields = tuple(str(field) for field in contract.get("required_fields", design.get("required_schedule_fields", [])))
    if not required_fields:
        issues.append("behavioral schedule contract has no required fields")
        return
    cells = _cells(plan)
    missing_cells = DOWNSTREAM_SPLITS - set(cells)
    if missing_cells:
        issues.append(f"missing downstream cells: {sorted(missing_cells)}")
        return
    behavior_rows = scheduled_rows(cells["behavior_eval"])
    steering_rows = scheduled_rows(cells["steering_eval"])
    if behavior_rows != steering_rows:
        issues.append("behavior_eval and steering_eval schedules differ")
    for split in ("behavior_eval", "steering_eval"):
        rows = scheduled_rows(cells[split])
        missing = [field for field in required_fields if any(field not in row for row in rows)]
        if missing:
            issues.append(f"{split} schedule is missing required metadata fields: {missing}")
        if len(rows) != int(cells[split].get("count_per_model", 0) or 0):
            issues.append(f"{split} schedule does not cover its registered count")

    raw_factors = contract.get("factors")
    if raw_factors is not None:
        factors = _mapping(raw_factors, label="behavioral_schedule_contract.factors")
        factor_order = contract.get("factor_order", list(factors))
        if not isinstance(factor_order, list) or set(factor_order) != set(factors):
            issues.append("factor_order does not match behavioral schedule factors")
        else:
            expected = _factorial_expected(factors, factor_order)
            observed = {
                _row_key(row, factor_order)
                for row in behavior_rows
                if all(field in row for field in factor_order)
            }
            if contract.get("full_factorial") is True and observed != expected:
                issues.append(
                    "behavior schedule is not the registered full factorial "
                    f"(missing={len(expected - observed)}, extra={len(observed - expected)})"
                )
            required_values = _mapping(contract.get("required_values", {}), label="required_values")
            for field, values in required_values.items():
                seen = {row.get(field) for row in behavior_rows}
                if not set(values).issubset(seen):
                    issues.append(f"behavior schedule is missing values for {field}: {sorted(set(values) - seen)}")
            for field in contract.get("balanced_fields", []):
                values = [row.get(str(field)) for row in behavior_rows]
                if not _balanced(values):
                    issues.append(f"behavior schedule is materially unbalanced for {field}")

    kind = str(design.get("design_kind", ""))
    if kind == "realization_diminishing_returns":
        if plan.get("allocation_payoff_structure") != design.get("allocation_payoff_structure"):
            issues.append("realization plan allocation_payoff_structure does not match the construct spec")
    for index, row in enumerate(behavior_rows):
        for issue in _numeric_row_issues(kind, row):
            issues.append(f"behavior schedule row {index}: {issue}")

    calibration_rows = scheduled_rows(cells["calibration"])
    neutral_fields = _mapping(contract.get("calibration_neutral_fields", {}), label="calibration_neutral_fields")
    for field, expected in neutral_fields.items():
        if any(row.get(field) != expected for row in calibration_rows):
            issues.append(f"calibration field {field!r} is not fixed at its neutral value")


def _check_spec_contract(spec: ConstructSpec, issues: list[str]) -> None:
    design = _design(spec)
    if not design:
        return
    if _is_wave34_design(design):
        task = spec.independent_behavior_task
        required_metadata = tuple(str(field) for field in design.get("required_task_metadata", []))
        properties = _task_properties(task)
        required = set(_task_required(task))
        missing = [field for field in required_metadata if field not in properties or field not in required]
        if missing:
            issues.append(f"spec is missing required Wave 2-4 behavioral metadata: {missing}")
        for field, expected in (
            ("task_id", task.get("task_id")),
            ("parser_id", spec.parsing_rules.get("parser_id")),
            ("response_format", task.get("response_format")),
        ):
            if design.get(field) != expected:
                issues.append(f"Wave 2-4 design {field} does not match the registered task contract")
        if not design.get("forbidden_downstream_terms"):
            issues.append("Wave 2-4 spec has no downstream probe/construct leakage ban")
        if _response_contract_count(task.get("prompt_template")) != 1:
            issues.append("Wave 2-4 independent task template must contain exactly one response contract")
        valid_ranges = spec.parsing_rules.get("valid_ranges")
        if not isinstance(valid_ranges, Mapping):
            issues.append("Wave 2-4 parsing_rules.valid_ranges is missing")
        else:
            for outcome in (design.get("primary_outcome"), *list(design.get("secondary_outcomes", []))[:1]):
                if outcome not in valid_ranges:
                    issues.append(f"valid_ranges is missing parser outcome alias {outcome!r}")
        position_field = design.get("position_field")
        if position_field is not None:
            schema = properties.get(position_field)
            if not isinstance(schema, Mapping) or schema.get("type") != "integer" or schema.get("enum") != [1, 2]:
                issues.append(f"position field {position_field!r} must be an integer enum [1, 2]")
        collateral = registered_task_for_role(spec, "collateral")
        collateral_bank = collateral_fact_bank_for_task(collateral)
        collateral_properties = _task_properties(collateral)
        collateral_required = set(_task_required(collateral))
        for field in ("fact_id", "fact_domain", "correct_option"):
            if field not in collateral_properties or field not in collateral_required:
                issues.append(f"collateral task is missing required field {field!r}")
        if collateral.get("task_id") != design.get("collateral_task_id"):
            issues.append("collateral task_id does not match its behavioral design")
        if collateral.get("response_format") != design.get("collateral_response_format"):
            issues.append("collateral response format does not match its behavioral design")
        if collateral.get("fact_bank") is None:
            issues.append("collateral task has no objective fact bank")
        else:
            fact_ids = {
                str(item.get("fact_id"))
                for item in collateral.get("fact_bank", [])
                if isinstance(item, Mapping)
            }
            if fact_ids != set(collateral_bank):
                issues.append("collateral task fact bank does not match the objective registry")
        if collateral.get("fact_bank_version", "v1") != design.get("collateral_fact_bank_version", "v1"):
            issues.append("collateral task fact_bank_version does not match its behavioral design")
        calibration_neutral = _mapping(
            design.get("calibration_neutral_fields", {}),
            label="behavioral_design.calibration_neutral_fields",
        )
        for field, value in calibration_neutral.items():
            schema = properties.get(field)
            if not isinstance(schema, Mapping) or value not in list(schema.get("enum", [])):
                issues.append(f"calibration neutral value {field}={value!r} is not in the task enum")
        required_patterns = design.get("calibration_required_patterns")
        if not isinstance(required_patterns, list) or not required_patterns:
            issues.append("calibration must register semantic required patterns")
        if spec.construct_id == "reference_frame" and not isinstance(design.get("pair_contract"), Mapping):
            issues.append("reference_frame must register its minimal pair contract")
        return
    required_metadata = tuple(str(field) for field in design.get("required_task_metadata", []))
    properties = dict(spec.independent_behavior_task["item_metadata_schema"]["properties"])
    required = set(spec.independent_behavior_task["item_metadata_schema"]["required"])
    missing = [field for field in required_metadata if field not in properties or field not in required]
    if missing:
        issues.append(f"spec is missing required behavioral metadata: {missing}")
    leakage_terms = design.get("forbidden_downstream_terms", [])
    if not isinstance(leakage_terms, list) or not leakage_terms:
        issues.append("spec has no downstream probe/construct leakage ban")
    kind = str(design.get("design_kind", ""))
    if kind == "realization_diminishing_returns":
        for field in ("ev_bin", "dominance_status", "allocation_region", "curve_profile"):
            if field not in properties:
                issues.append(f"realization spec lacks {field} metadata")
        if design.get("curve_id") != "crossing_concave_portfolio_5x20_v1":
            issues.append("realization spec does not register the diminishing-return curve")
        issues.extend(_realization_curve_issues(design))
    elif kind == "evidence_diagnosticity_tradeoff":
        for field in ("high_information_option", "diagnosticity_strength", "cost_profile", "delay_profile", "decision_stakes_bin", "stakes_structure"):
            if field not in properties:
                issues.append(f"evidence spec lacks {field} metadata")
        if design.get("shared_stakes_field") != "decision_stakes_bin":
            issues.append("evidence spec does not identify one shared decision-stakes field")
    elif kind == "persistence_diminishing_tranches":
        for field in ("return_advantage_bin", "tranche_profile", "option_a_semantics", "dominance_status", "allocation_region"):
            if field not in properties:
                issues.append(f"persistence spec lacks {field} metadata")
        if design.get("curve_id") != "diminishing_marginal_returns_tranches_v1":
            issues.append("persistence spec does not register a diminishing-marginal-return curve")
    else:
        issues.append(f"unknown behavioral repair design_kind={kind!r}")


def behavioral_design_issues(spec: ConstructSpec, plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic spec/plan design failures without model output."""

    issues: list[str] = []
    if spec.construct_id not in REPAIRED_CONSTRUCTS or not _design(spec):
        return ()
    try:
        _check_spec_contract(spec, issues)
        _check_plan_schedule(spec, plan, issues)
    except ValueError as exc:
        issues.append(str(exc))
    return tuple(issues)


def validate_behavioral_design(spec: ConstructSpec, plan: Mapping[str, Any]) -> dict[str, Any]:
    issues = behavioral_design_issues(spec, plan)
    if issues:
        raise ValueError(f"{spec.construct_id} behavioral design validation failed: {'; '.join(issues)}")
    return {
        "construct_id": spec.construct_id,
        "design_kind": _design(spec).get("design_kind"),
        "required_task_metadata": list(_design(spec).get("required_task_metadata", [])),
        "status": "pass",
    }


def _whole_word_hits(text: str, terms: Iterable[Any]) -> tuple[str, ...]:
    folded = str(text).casefold()
    hits: list[str] = []
    for raw_term in terms:
        term = str(raw_term).strip().casefold()
        if not term:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
        if re.search(pattern, folded):
            hits.append(term)
    return tuple(sorted(set(hits)))


def _wave34_record_issues(
    spec: ConstructSpec,
    plan: Mapping[str, Any],
    record: PromptRecord,
) -> tuple[str, ...]:
    design = _design(spec)
    cells = _cells(plan)
    cell = cells.get(record.split)
    if cell is None:
        return (f"record split {record.split!r} is not a registered downstream cell",)
    task = registered_task_for_role(spec, record.prompt_role)
    metadata_raw = record.metadata.get("task_metadata")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else dict(record.metadata)
    issues: list[str] = []
    issues.extend(_metadata_value_issues(task, metadata))
    expected_task_id = str(task.get("task_id"))
    expected_parser = (
        str(design.get("collateral_parser_id"))
        if record.prompt_role == "collateral"
        else str(design.get("parser_id"))
    )
    expected_format = (
        str(design.get("collateral_response_format"))
        if record.prompt_role == "collateral"
        else str(design.get("response_format"))
    )
    if record.task_id != expected_task_id:
        issues.append(f"task_id={record.task_id!r}; expected {expected_task_id!r}")
    if record.parser_id != expected_parser:
        issues.append(f"parser_id={record.parser_id!r}; expected {expected_parser!r}")
    if record.expected_output_format != expected_format:
        issues.append(
            f"expected_output_format={record.expected_output_format!r}; expected {expected_format!r}"
        )
    if record.prompt_role != str(cell.get("prompt_role")):
        issues.append(f"prompt role {record.prompt_role!r} does not match its cell")
    if record.task_id != str(cell.get("task_id")):
        issues.append("record task_id differs from its cell task_id")
    if record.parser_id != str(cell.get("parser_id")):
        issues.append("record parser_id differs from its cell parser_id")
    if record.expected_output_format != str(cell.get("expected_output_format")):
        issues.append("record response format differs from its cell response format")

    if record.prompt_role == "calibration":
        fixed_fields = _mapping(
            design.get("calibration_fixed_fields", {}),
            label="calibration_fixed_fields",
        )
    elif record.prompt_role in {"behavior", "steering"}:
        fixed_fields = _mapping(
            design.get("behavior_fixed_fields", {}),
            label="behavior_fixed_fields",
        )
    else:
        # Collateral is a standalone objective fact task.  It uses the
        # collateral task schema below and must not inherit construct-specific
        # fixed fields (for example reciprocity's prior-contact controls).
        fixed_fields = {}
    for field, expected in fixed_fields.items():
        if metadata.get(field) != expected:
            issues.append(f"{field}={metadata.get(field)!r}; expected fixed value {expected!r}")

    forbidden = list(design.get("forbidden_downstream_terms", []))
    forbidden.extend(str(term) for term in plan.get("forbidden_terms", []))
    if record.prompt_role == "calibration":
        forbidden.extend(str(term) for term in design.get("calibration_forbidden_terms", []))
    hits = _whole_word_hits(record.prompt_text, forbidden)
    if hits:
        issues.append(f"probe/construct leakage terms: {list(hits)}")
    if "Continue processing the scenario.".casefold() in record.prompt_text.casefold():
        issues.append("downstream prompt contains the probe-only continuation suffix")
    response_count = _response_contract_count(record.prompt_text)
    if response_count != 1:
        issues.append(f"response request count is {response_count}; expected exactly one")
    if record.prompt_text.rstrip()[-1:] not in ".!?":
        issues.append("prompt lacks terminal punctuation")
    if expected_format not in {
        "single_integer_1_or_2",
        "single_integer_0_to_100",
        "two_integers_sum_100",
        "single_integer_allocation_0_to_100",
        "two_integers_on_separate_lines",
    }:
        issues.append(f"unsupported response format {expected_format!r}")

    if record.prompt_role == "calibration":
        for pattern in design.get("calibration_required_patterns", []):
            try:
                matched = re.search(str(pattern), record.prompt_text, re.IGNORECASE)
            except re.error as exc:
                issues.append(f"invalid calibration semantic pattern {pattern!r}: {exc}")
                continue
            if not matched:
                issues.append(f"calibration prompt is missing required neutral-language pattern {pattern!r}")

    if record.prompt_role == "collateral":
        fact_id = metadata.get("fact_id")
        collateral_task = registered_task_for_role(spec, "collateral")
        fact = collateral_fact_bank_for_task(collateral_task).get(str(fact_id))
        if fact is None:
            issues.append(f"collateral fact_id={fact_id!r} is not in the objective fact bank")
        else:
            if metadata.get("fact_domain") != fact["fact_domain"]:
                issues.append("collateral fact_domain does not match the objective fact bank")
            if metadata.get("correct_option") != fact["correct_option"]:
                issues.append("collateral correct_option does not match the objective fact bank")
            prompt_folded = record.prompt_text.casefold()
            for field in ("statement_1", "statement_2"):
                if str(fact[field]).casefold() not in prompt_folded:
                    issues.append(f"collateral prompt does not contain the registered {field}")
            if str(collateral_task.get("fact_bank_version", "v1")) != "v1":
                # Every versioned repair deliberately exposes the registered
                # fact ID as a neutral card label.  It is a bounded content
                # variation that makes uniqueness auditable even when two
                # models choose the same wrapper around distinct statements.
                if not re.search(r"\bfact\s+id\s*:\s*" + re.escape(str(fact_id)) + r"\b", prompt_folded):
                    issues.append("versioned collateral prompt is missing its registered fact ID label")
    return tuple(dict.fromkeys(issues))


def behavioral_record_issues(
    spec: ConstructSpec,
    plan: Mapping[str, Any],
    record: PromptRecord,
) -> tuple[str, ...]:
    """Validate model-produced downstream metadata and lexical contracts."""

    design = _design(spec)
    if not design or record.prompt_role not in DOWNSTREAM_ROLES:
        return ()
    if _is_wave34_design(design):
        return _wave34_record_issues(spec, plan, record)
    issues: list[str] = []
    task_metadata = record.metadata.get("task_metadata")
    metadata = dict(task_metadata) if isinstance(task_metadata, Mapping) else record.metadata
    required = list(spec.independent_behavior_task["item_metadata_schema"]["required"])
    missing = [field for field in required if field not in metadata]
    if missing:
        issues.append(f"missing task metadata: {missing}")
    if record.prompt_role in {"behavior", "steering"}:
        expected_values = _mapping(design.get("fixed_metadata", {}), label="behavioral_design.fixed_metadata")
        for field, expected in expected_values.items():
            if metadata.get(field) != expected:
                issues.append(f"{field}={metadata.get(field)!r}; expected fixed value {expected!r}")
    forbidden = list(design.get("forbidden_downstream_terms", []))
    forbidden.extend(str(term) for term in plan.get("forbidden_terms", []))
    hits = _whole_word_hits(record.prompt_text, forbidden)
    if hits:
        issues.append(f"probe/construct leakage terms: {list(hits)}")
    if "Continue processing the scenario.".casefold() in record.prompt_text.casefold():
        issues.append("downstream prompt contains the probe-only continuation suffix")
    directives = _RESPONSE_DIRECTIVE.findall(record.prompt_text)
    if len(directives) != 1:
        issues.append(f"response request count is {len(directives)}; expected exactly one")
    if record.prompt_text.rstrip()[-1:] not in ".!?":
        issues.append("prompt lacks terminal punctuation")
    if str(record.expected_output_format) not in {"single_integer_allocation_0_to_100", "single_integer_0_to_100", "single_integer_1_or_2"}:
        issues.append(f"unsupported response format {record.expected_output_format!r}")

    # Calibration rows are neutral variance controls.  Their metadata is
    # checked against the calibration schedule, but they must not be forced to
    # carry the behavior/steering repair labels used to describe target items.
    if record.prompt_role not in {"behavior", "steering"}:
        return tuple(issues)

    kind = str(design.get("design_kind"))
    for issue in _numeric_row_issues(kind, metadata):
        issues.append(issue)
    if kind == "realization_diminishing_returns":
        issues.extend(_realization_curve_text_issues(record.prompt_text, design))
        if metadata.get("dominance_status") != "no_strict_dominance":
            issues.append("realization item is not marked no_strict_dominance")
        if metadata.get("allocation_region") != "interior":
            issues.append("realization item is not marked interior")
        if metadata.get("curve_profile") != "diminishing_returns_portfolio":
            issues.append("realization item lacks the registered concave portfolio profile")
    elif kind == "evidence_diagnosticity_tradeoff":
        if metadata.get("stakes_structure") != "shared_single_decision_value":
            issues.append("evidence item does not use one shared decision-stakes value")
        if metadata.get("cost_profile") != "high_information_costlier":
            issues.append("evidence item does not guarantee a real cost disadvantage for the high-information option")
        if metadata.get("dominance_status") != "no_strict_dominance":
            issues.append("evidence item is not marked no_strict_dominance")
    elif kind == "persistence_diminishing_tranches":
        if metadata.get("tranche_profile") != "diminishing_marginal_returns":
            issues.append("persistence item uses a missing or linear tranche profile")
        if metadata.get("dominance_status") != "no_strict_dominance":
            issues.append("persistence item is not marked no_strict_dominance")
        if metadata.get("allocation_region") != "interior":
            issues.append("persistence item is not marked interior")
        if metadata.get("option_a_semantics") not in {"established_goal", "abstract_option"}:
            issues.append("persistence item lacks explicit Option A semantics")
    return tuple(issues)


def validate_behavioral_records(
    spec: ConstructSpec,
    plan: Mapping[str, Any],
    records: Iterable[PromptRecord],
) -> dict[str, Any]:
    records = tuple(records)
    issues: list[dict[str, Any]] = []
    for record in records:
        for issue in behavioral_record_issues(spec, plan, record):
            issues.append({"prompt_id": record.prompt_id, "issue": issue})
    if issues:
        raise ValueError(
            f"{spec.construct_id} behavioral record validation failed for {len(issues)} issue(s): "
            + "; ".join(f"{item['prompt_id']}: {item['issue']}" for item in issues[:5])
        )
    return {"construct_id": spec.construct_id, "record_count": len(records), "status": "pass"}


__all__ = [
    "COLLATERAL_FACT_BANK",
    "COLLATERAL_FACT_BANK_V2",
    "COLLATERAL_FACT_BANK_V2_VERSION",
    "COLLATERAL_FACT_BANK_V3_BY_CONSTRUCT",
    "COLLATERAL_FACT_BANK_V3_VERSION_BY_CONSTRUCT",
    "COLLATERAL_FACT_BANK_LABEL_V1_BY_CONSTRUCT",
    "COLLATERAL_FACT_BANK_LABEL_V1_VERSION_BY_CONSTRUCT",
    "REPAIRED_CONSTRUCTS",
    "behavioral_design_issues",
    "behavioral_record_issues",
    "collateral_fact_bank_for_task",
    "registered_task_for_role",
    "scheduled_rows",
    "validate_behavioral_design",
    "validate_behavioral_records",
]
