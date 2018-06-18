'''
Module that provides abstraction of the attribute names and structure for
Mercury data
'''

__all__ = '''
  Dictionary
  JSONField
  EventType
  DiseaseType
  CountryName
  Subtype
  '''.split()

from .dictionaries import Dictionary


class JSONField(object):
  EVENT_ID = 'Event_ID'
  EVENT_TYPE = 'Event_Type'
  SUBTYPE = 'Event_Subtype'
  DISEASE = 'Disease'
  COUNTRY = 'Country'
  STATE = 'State'
  COUNTY = 'County'
  CITY = 'City'
  LATITUDE = 'Latitude'
  LONGITUDE = 'Longitude'
  POPULATION = 'Population'
  REASON = 'Reason'
  VIOLENT = 'Violent'
  PROBABILITY = 'Probability'
  ACTOR = 'Actor'
  TARGET = 'Target' # Kept for gradual deprecation
  TARGETS = 'Targets'
  GSR_TARGET = 'Target'
  GSR_TARGET_STATUS = 'Target_Status'
  WARNING_TARGET = 'Target'
  WARNING_TARGET_STATUS = 'Target_Status'
  EVENT_DATE = 'Event_Date'
  CASE_COUNT = 'Case_Count'
  REVISION_DATE = 'Revision_Date'
  WARNING_ID = 'Warning_ID'
  TIMESTAMP = 'timestamp'
  EARLIEST_REPORTED_DATE = 'Earliest_Reported_Date'
  APPROXIMATE_LOCATION = 'Approximate_Location'
  SEQUENCE = 'sequence'
  LATEST = 'latest'

class EventType(object):
  CIVIL_UNREST = 'Civil Unrest'
  WIDESPREAD_CIVIL_UNREST= 'Widespread Civil Unrest'
  MILITARY_ACTION = 'Military Action'
  NONSTATE_ACTOR = 'Non-State Actor'
  ICEWS_PROTEST = 'ICEWS Protest'
  DISEASE = 'Disease'


class Subtype(object):
  ARMED_CONFLICT = 'Armed Conflict'
  FORCE_POSTURE = 'Force Posture'
  ARMED_ASSAULT = 'Armed Assault'
  BOMBING = 'Bombing'
  HOSTAGE_TAKING = 'Hostage Taking'


class DiseaseType(object):
  RARE = 'rare'
  DENGUE = 'Dengue'
  MERS = 'MERS'
  AVIAN_INFLUENZA = 'Avian Influenza'


class CountryName(object):
  SAUDI_ARABIA = 'Saudi Arabia'
  EGYPT = 'Egypt'

class Wildcards(object):
  TARGET_STATUS = "Unknown"
  TARGET_TARGET = "Unspecified"
  STATE_ACTOR = "Unspecified"
  NONSTATE_ACTOR = "Unspecified"
  ALL_ACTORS = [STATE_ACTOR, NONSTATE_ACTOR]

class ScoreComponents(object):
  QS = "Quality Score"
  LS = "Location Score"
  LT = "Lead Time"
  DS = "Date Score"
  PS = "Population Score"
  RS = "Reason Score"
  VS = "Violence Score"
  AS = "Actor Score"
  ESS = "Event Subtype Score"
  TSS = "Target Status Score"
  TTS = "Target Target Score"
  DIS = "Disease Score"
