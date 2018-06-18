"""
This module reads the dictionaries from the resource path and loads their values
into a set of variables for import by other modules.
"""

from mercury.common.schema import Dictionary

REASON_VALUES = Dictionary.getReasons()
POPULATION_VALUES = Dictionary.getPopulations()
TARGET_VALUES = Dictionary.getTargets()
STATE_ACTOR_VALUES = Dictionary.getStateActors()
NON_STATE_ACTOR_VALUES = Dictionary.getNonStateActors()
UNSPECIFIED_TARGETS = Dictionary.getUnspecifiedTargets()
UNSPECIFIED_ACTORS = Dictionary.getUnspecifiedActors()