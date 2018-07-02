# mercury-challenge
Public Repository for the IARPA Mercury Challenge.  Challenge participants should follow this repository to receive notices of code updates and other release information.  The files in this directory are provided for IARPA’s Mercury Challenge.  You can learn about the challenge and register to compete at https://iarpa.gov/challenges/mercury.html.  The discussion forum for the Challenge can be found at <http://apps.topcoder.com/forums//?module=ThreadList&forumID=630604> (login required.)

## Software
### Official Scoring
The official scoring package requires a Docker installation.  This package provides the requisite REST endpoints for warning submission, GSR intake, and scoring requests.
The official scoring system has been tested on MacOS Sierra and Ubuntu Linux (version).  In our testing it was incompatible with some installations of Windows 7.
### ExpressScore 
The **ExpressScorer** package provides Challenge participants with a lightweight testing engine that does not require a Docker installation.  The code is found at https://github.com/planetmercury/mercury-challenge/tree/master/src/ExpressScore and tutorial Jupyter notebooks are found at https://github.com/planetmercury/mercury-challenge/blob/master/src/ExpressScore/notebooks
**ExpressScore** has been tested on MacOS Sierra and requires Python 3.6 or greater.  It should work with other operating systems and Python 3.x versions but has not been tested with them.
### Baserate Models
The Baserate models make predictions using only ground truth history.  The scores for these predictions provide a minimum threshold that Challenge participants must exceed in order to be ranked.

## Documentation
### Challenge Handbook
The Mercury Challenge Handbook and Appendix provide complete documentation for the challenge, including definitions of the several event types and scoring rules.  They can be found in https://github.com/planetmercury/mercury-challenge/blob/master/doc
### Jupyter Notebooks for Code Tutorials
- *Case Count Scorer* Tutorial: https://github.com/planetmercury/mercury-challenge/blob/master/src/ExpressScore/notebooks/Case%20Count%20Scorer.ipynb
- *Military Action Scorer* Tutorial:  https://github.com/planetmercury/mercury-challenge/blob/master/src/ExpressScore/notebooks/Military%20Action%20Scorer.ipynb
