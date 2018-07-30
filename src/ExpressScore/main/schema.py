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

#from .dictionaries import Dictionary


class db(object):
  GSR_TYPE = "GSR"
  WARNING_TYPE = 'warnings'


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
  ACTOR = 'Actor'
  EVENT_DATE = 'Event_Date'
  CASE_COUNT = 'Case_Count'
  REVISION_DATE = 'Revision_Date'
  WARNING_ID = 'Warning_ID'
  TIMESTAMP = 'timestamp'
  EARLIEST_REPORTED_DATE = 'Earliest_Reported_Date'
  APPROXIMATE_LOCATION = 'Approximate_Location'
  SEQUENCE = 'sequence'
  LATEST = 'latest'
  WARN_VALUE = 'Warning Case Count'
  EVENT_VALUE = "GSR Case Count"
  PARTICIPANT_ID = "participant_id"

class EventType(object):
  CU = 'Civil Unrest'
  MA = 'Military Activity'
  DIS = 'Disease'


class Subtype(object):
  CONFLICT = 'Conflict'
  FORCE_POSTURE = 'Force Posture'

class DiseaseType(object):
  MERS = 'MERS'

class LocationName(object):
  SAUDI_ARABIA = 'Saudi Arabia'
  EGYPT = 'Egypt'
  TAHRIR = "Tahrir Square"
  JORDAN = "Jordan"
  AMMAN = "Amman"
  IRBID = "Irbid"
  MADABA = "Madaba"

class Wildcards(object):
  STATE_ACTOR = "Unspecified"
  NONSTATE_ACTOR = "Unspecified"
  ALL_ACTORS = [STATE_ACTOR, NONSTATE_ACTOR]

class ScoreComponents(object):
  QS = "Quality Score"
  LS = "Location Score"
  LT = "Lead Time"
  DS = "Date Score"
  AS = "Actor Score"
  ESS = "Event Subtype Score"
