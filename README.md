# mercury-challenge
Public Repository for the IARPA Mercury Challenge.  Challenge participants should follow this repository to receive notices of code updates and other release information.  The files in this directory are provided for IARPA’s Mercury Challenge.  You can learn about the challenge and register to compete at https://iarpa.gov/challenges/mercury.html.  The discussion forum for the Challenge can be found at <http://apps.topcoder.com/forums/?module=ThreadList&forumID=630604> (login required.)  Addtionally, there will be a live Q&A web session on 12 July.  To register go to <https://eventmanagement.cvent.com/MercuryChallengeQASession> and use the password "Mercury317CQA".

## Software
### Official Scoring
The official scoring package requires a Docker installation.  This package provides the requisite REST endpoints for warning submission, GSR intake, and scoring requests.  This package has not yet been released.
### ExpressScore 
The **ExpressScorer** package provides Challenge participants with a lightweight testing engine that does not require a Docker installation.  The code is found at https://github.com/planetmercury/mercury-challenge/tree/master/src/ExpressScore and tutorial Jupyter notebooks are found at https://github.com/planetmercury/mercury-challenge/blob/master/src/ExpressScore/notebooks
**ExpressScore** has been tested on MacOS Sierra, Windows 7, and Windows 10 and requires Python 3.4 or greater.
### Installation

Before beginning, we recommend installing Anaconda (https://www.anaconda.com/download/)
- Anaconda is a free and open source distribution of the Python and R programming languages for data science and machine learning related applications (large-scale data processing, predictive analytics, scientific computing), that aims to simplify package management and deployment.

#### Windows 7/8/10
Install Visual Studio Build Tools 2017
- Installation file found here: https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15#
- After downloading, open the installer and ensure **Visual C++ build tools**, **Testing tools core features - Build Tools** are both checked.
- Depending on which Windows version you're on, ensure that the proper SDK is checked.
  - **Windows 10 SDK** for Windows 10
  - **Windows 8.1 SDK and UCRT SDK** for Windows 7 & 8
#### All Platforms
In the terminal of your choice, navigate to src/ExpressScore. </br></br>
Run `conda env create -f expressscore.yml -v` to install the required dependencies for the ExpressScorer.</br></br>
Once complete, activate the environment.
- Run `activate mc_minimal` if on Windows
- Run `source activate mc_minimal` if on Mac or Linux

To ensure successful installation, open the **test** directory inside src/ExpressScore and run **test_ma_scoring.py** and **test_case_count_scoring.py**. If all tests pass, congrats! You now have a working ExpressScorer environment.

### Baserate Models
The Baserate models make predictions using only ground truth history.  The scores for these predictions provide a minimum threshold that Challenge participants must exceed in order to be ranked.

## Documentation
### Challenge Handbook
The Mercury Challenge Handbook and Appendix provide complete documentation for the challenge, including definitions of the several event types and scoring rules.  They can be found in https://github.com/planetmercury/mercury-challenge/blob/master/doc
### Jupyter Notebooks for Code Tutorials
- *Case Count Scorer* Tutorial: https://github.com/planetmercury/mercury-challenge/blob/master/src/ExpressScore/notebooks/Case%20Count%20Scorer.ipynb
- *Military Action Scorer* Tutorial:  https://github.com/planetmercury/mercury-challenge/blob/master/src/ExpressScore/notebooks/Military%20Action%20Scorer.ipynb
