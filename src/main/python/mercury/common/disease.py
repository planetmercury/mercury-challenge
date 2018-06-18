'''
Module providing common disease-related functionality across the T&E services
'''

__all__ = '''
  isDiseaseRare
  '''.split()

from .schema import JSONField, DiseaseType, CountryName

_COMMON_DISEASES = (
  (DiseaseType.MERS, CountryName.SAUDI_ARABIA),
  (DiseaseType.AVIAN_INFLUENZA, CountryName.EGYPT),
  )

def isDiseaseRare(warning):
  '''
Return a boolean representation of whether the specified warning (represented
as a dictionary) represents a rare disease.
  '''
  if JSONField.DISEASE not in warning: return False
  combo = (warning[JSONField.DISEASE], warning[JSONField.COUNTRY])
  return combo not in _COMMON_DISEASES

