'''
Module providing access to the dictionaries of allowable values for various
attributes
'''

__all__ = '''
  Dictionary
  '''.split()

import os
from os.path import join as pathJoin
import json

class Dictionary(object):
  '''
Class encapsulation of access to the Mercury dictionaries
  '''
  RESOURCE_PATH = pathJoin(os.getcwd(), '..', 'resources', 'common')

  @classmethod
  def getReasons(cls):
    '''
Return a list of allowable values for the Reason attribute of Civil Unrest
events
    '''
    return cls.getValues('cu_reason_dictionary.json')

  @classmethod
  def getPopulations(cls):
    '''
Return a list of allowable values for the Population attribute of MANSA events
    '''
    return cls.getValues('population_dictionary.json')

  @classmethod
  def getTargets(cls):
    '''
Return a list of allowable values for the Target attribute of MANSA events
    '''
    return cls.getValues('target_dictionary.json', flat=True)

  @classmethod
  def getStateActors(cls):
    '''
Return a list of allowable values for the Actor attribute of Military Action
events
    '''
    return cls.getValues('state_actor_dictionary.json')

  @classmethod
  def getNonStateActors(cls):
    '''
Return a list of allowable values for the Actor attribute of Non-State Actor 
events
    '''
    return cls.getValues('non_state_actor_dictionary.json')

  @classmethod
  def getUnspecifiedTargets(cls):
    '''
Return a list of targets to be treated as unspecified in scoring code.
    :return: list of values
    '''
    targets = cls.getTargets()
    return [i for i in targets if i['Target'].startswith('Unspecified')]

  @classmethod
  def getUnspecifiedActors(cls):
    '''
    Return a list of actors to be treated as unspecified in scoring code.
    :return:
    '''
    return cls.getValues('unspecified_actor_dictionary.json')

  @classmethod
  def getValues(cls, filename, flat=False):
    '''
Access the specified filename and return a list of the allowable values defined
within.  If the boolean evaluation of the optional keyword parameter "flat" is
false or if the parameter is omitted, then The specified file is expected to
contain a serialized JSON object with a top level attributed called "Values"
whose value is an array.  Otherwise, the file's contents is expected to be a
serialized JSON array.
    '''
    with open(pathJoin(cls.RESOURCE_PATH, filename), 'rb') as source:
      decoded = json.loads(source.read().decode('utf-8-sig'))
    return decoded if flat else decoded['values']

