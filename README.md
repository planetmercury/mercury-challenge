# mercury-challenge
Public Repository for the IARPA Mercury Challenge.  Challenge participants should follow this repository to receive notices of code updates and other release information.

## Software
### Official Scoring
The official scoring package requires a Docker installation.  This package provides the requisite REST endpoints for warning submission, GSR intake, and scoring requests.
### ExpressScore 
The **ExpressScorer** package provides Challenge participants with a lightweight testing engine that does not require a Docker installation.  The code is found at https://github.com/planetmercury/mercury-challenge/tree/master/src/ExpressScore and tutorial Jupyter notebooks are found at https://github.com/planetmercury/mercury-challenge/blob/master/src/ExpressScore/notebooks
### Baserate Models
The Baserate models make predictions using only ground truth history.  The scores for these predictions provide a minimum threshold that Challenge participants must exceed in order to be ranked.

## Documentation
### Challenge Handbook
The Mercury Challenge Handbook and Appendix provide complete documentation for the challenge, including definitions of the several event types and scoring rules.  They can be found in https://github.com/planetmercury/mercury-challenge/blob/master/doc
### Jupyter Notebooks for Code Tutorials
- *CaseCountScorer* Tutorial: https://github.com/planetmercury/mercury-challenge/blob/master/src/ExpressScore/notebooks/Case%20Count%20Scorer.ipynb
