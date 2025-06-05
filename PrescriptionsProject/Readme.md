# DME Prescription Parser

## Requirements

unzip the file and cd into the directory that is created.

requirements.txt contains the requirements to be installed. You will also likely need a virtual environment in which to install the requirements.

This project runs with python version 3.12.

# Modify the .env file

The project has a .env file in /synapse/PrescriptionsProject/src/ to hold it's configuration parameters. Modify this line with your own GEMINI_API_KEY.
```
GEMINI_API_KEY = "Put your key here"
```

If you don't have one you can create one at https://aistudio.google.com/apikey

# How to Install a virtual environment

/synapse/PrescriptionsProject/ directory needs to have the virtual environment
```
> python -m venv .venv
```
This will create a virtual environment for you to use and install dependencies.

# How to activate the virtual environment
```
> .\.venv\Scripts\activate.bat (activate.sh on linux or mac)
> (.venv)>
```

# How to deactivate the virtual environment
```
> .venv/Scripts/deactivate.bat (activate.sh on linux or mac)
```

# Installing the Requirements
```
> pip install -r requirements.txt
```

# Running the PrescriptionParser
```
> cd ./src

> python PrescriptionParser.py "CPAP supplies
> requested. Full face mask with headgear and filters. Patient has been
> compliant. Ordered by Dr. House."
```
Response:
```
{
"device": "CPAP",
"ordering_provider": "Dr. House",
"mask_type": "Full face mask",
"add_ons": [
"headgear",
"filters"
],
"compliance_status": "compliant"
}
```


# Running the tests for Prescription Parser
```
> python .\TestPrescriptionParser.py
> See TestOutput.txt for the output
```

## Approach

I saw that the set of items requested for a prescription was limited. pydantic has a nice way to create
a lightweight class of related fields. In my reading I found that one could populate a pydantic object as
I've done in the code. Tests are segmented into their own file.

## Status

I attempted to complete the challenge in 1 hr. I was unsuccessful. It took me about 1.5 hrs just to get it to run at
all. After about 2 hrs I was able to get down to the real problem of parsing very fuzzy input. I tried to adjust my
prompt instructions, but was unable to get it to run successfully. The main issue is that the AI is unable to resolve
the differences between add_ons, accessories, components and features despite my enhanced prompt instructions.

So the solution is as you see it. I know I blew up the 1 hr limit, however I find it difficult to believe that it is
a fair test of a potential employee's skills if one is unable to get anything at all running.

If I had additional time, I would have put it into a FastAPI solution so that another solution could call it via a web
interface with swagger - like page.
