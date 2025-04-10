@echo off
echo Setting up CommitHunter development environment...

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call .\venv\Scripts\activate

:: Update pip
python -m pip install --upgrade pip

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt


echo Setup complete!
echo To activate the environment, run: .\venv\Scripts\activate