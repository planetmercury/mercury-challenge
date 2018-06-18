'''
Package implementing Mercury T&E scoring
'''

class ScoringError(Exception):
  '''
Trivial extension of the base Exception class representing errors encountered
while performing scoring
  '''
  pass

from .scoring import *

