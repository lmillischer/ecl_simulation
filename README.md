# ECL Simulation

A package to generate bank and portfolio specific expected credit loss estimates.

## Installation

* Download and install Anaconda
* Download and install PyCharm
* Check that git is installed and recognized by PyCharm
* Connect PyCharm to your GitHub account
  * This step and the previous step might not be easily allowed by the firewall of your institution - get the rigth authorizations from your IT department.
* Go to "new project from VCS" and select this repository (you should have been invited)
* PyCharm should then propose setting up a new conda environment using environment.yml (this will automatically use the correct python version and install all required packages)
  * If this does not happen, you can use the Anaconda Prompt, navigate to the project folder and run `conda env create -f environment.yml`
* The correct interpreter (associated with the conda environment) should be automatically selected.
* You're good to go!

## Usage

* Open the project in PyCharm
* In the `controls.py` file you can select which parts of the model to run.
* Open and run the `main.py` file
* Please at this stage do not PUSH code to the repository.

## Author and contact

Laurent Millischer - lmillischer@worldbank.org

## Update the environment

`conda env export --no-builds > environment.yml`
