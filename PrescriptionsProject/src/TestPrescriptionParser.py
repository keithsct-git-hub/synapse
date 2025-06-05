import os
import json
from typing import Optional, List
from pydantic import BaseModel, Field, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from PrescriptionParser import Prescription

from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")  # Ensure GOOGLE_API_KEY is set


def parse_prescription(prescription_text: str, llm_model_name: str) -> Optional[Prescription]:
    """
    Uses Gemini to parse a prescription text into a Prescription Pydantic model.

    Args:
        prescription_text (str): The content of the prescription to parse.
        llm_model_name (str): The name of the Gemini model to use (e.g., "models/gemini-1.5-pro-latest").

    Returns:
        Optional[Prescription]: The parsed Prescription object, or None if parsing fails.
    """
    parser = PydanticOutputParser(pydantic_object=Prescription)

    try:
        llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=1.0)

        # Updated prompt for better JSON adherence and field omission
        prompt_template = f"""
            You are an expert at reading prescriptions for durable medical equipment.
            Your task is to accurately extract prescription information from the provided text.

            Extract the following fields. If a field is not present in the text, you MUST omit it from the output JSON.
            For list fields (like 'add_ons', 'accessories', 'features', 'usage', 'components'), extract all relevant items into a JSON array of strings.

            {parser.get_format_instructions()}

            Text: "{prescription_text}"
        """

        response = llm.invoke(prompt_template)
        output_text = response.content

        try:
            # Pydantic will validate and parse the JSON string into the Prescription object
            prescription = parser.parse(output_text)
            return prescription
        except ValidationError as e:
            print(f"Pydantic Validation Error for text: '{prescription_text}'\n{e}")
            print(f"LLM Raw Output:\n{output_text}\n{'=' * 50}\n")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON Decoding Error for text: '{prescription_text}'\n{e}")
            print(f"LLM Raw Output:\n{output_text}\n{'=' * 50}\n")
            return None

    except Exception as e:
        print(f"An error occurred during LLM invocation for text: '{prescription_text}'\n{e}")
        return None


# --- Test Cases ---
test_cases = [
    {
        "input_text": """
Patient diagnosed with COPD, SpO2 measured at
87% on room air. Needs portable oxygen concentrator for use during
exertion and sleep. Dr. Chase signed the order.
        """,
        "expected_output": {
            "device": "portable oxygen concentrator",
            "diagnosis": "COPD",
            "SpO2": "87%",
            "usage": ["exertion", "sleep"],
            "ordering_provider": "Dr. Chase"
        }
    },
    {
        "input_text": """
Patient has MS with significant mobility issues. Recommended a lightweight manual wheelchair with elevating leg rests. Ordered by Dr. Taub.""",
        "expected_output": {
            "device": "manual wheelchair",
            "type": "lightweight",
            "features": ["elevating leg rests"],
            "diagnosis": "MS",
            "ordering_provider": "Dr. Taub"
        }
    },
    {
        "input_text": """
Asthma
diagnosis confirmed. Prescribing nebulizer with mouthpiece and
tubing. Dr. Foreman completed the documentation.
        """,
        "expected_output": {
            "device": "nebulizer",
            "accessories": ["mouthpiece", "tubing"],
            "diagnosis": "Asthma",
            "ordering_provider": "Dr. Foreman"
        }
    },
    {
        "input_text": """
Patient is
non-ambulatory and requires hospital bed with trapeze bar and side
rails. Diagnosis: late-stage ALS. Order submitted by Dr.
Cuddy.
        """,
        "expected_output": {
            "device": "hospital bed",
            "features": ["trapeze bar", "side rails"],
            "diagnosis": "ALS",
            "mobility_status": "non-ambulatory",
            "ordering_provider": "Dr. Cuddy"
        }
    },
    {
        "input_text": """
CPAP supplies
requested. Full face mask with headgear and filters. Patient has been
compliant. Ordered by Dr. House.
        """,
        "expected_output": {
            "device": "CPAP supplies",  # Changed from 'product' to 'device' to match Pydantic model
            "components": ["full face mask", "headgear", "filters"],
            "compliance_status": "compliant",
            "ordering_provider": "Dr. House"
        }
    }
]

# --- Main Test Execution ---
if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY is set in your environment
    if os.getenv("GOOGLE_API_KEY") is None:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it to your Gemini API key before running.")
        exit(1)

    # Choose your preferred model from the list of available models
    # Based on your previous output, these are good choices:
    # llm_model = "models/gemini-1.5-pro-latest"
    llm_model = "models/gemini-1.5-flash-latest"  # Often faster and cheaper for this type of task

    print(f"--- Running tests with model: {llm_model} ---")
    print("Note: LLM output can be non-deterministic, so perfect matches are not guaranteed.")
    print("      Focus on whether the key information is correctly extracted.\n")

    for i, test_case in enumerate(test_cases):
        print(f"### Test Case {i + 1} ###")
        print("Input Text:")
        print(test_case["input_text"].strip())

        # Parse the input text using the LLM
        actual_prescription_obj = parse_prescription(test_case["input_text"], llm_model)

        if actual_prescription_obj:
            # Convert Pydantic object to dictionary and then to pretty-printed JSON string
            actual_output_dict = actual_prescription_obj.model_dump(
                exclude_none=True)  # Exclude None values for cleaner comparison
            actual_output_json = json.dumps(actual_output_dict, indent=2)
        else:
            actual_output_json = "Error: Could not parse output."

        expected_output_json = json.dumps(test_case["expected_output"], indent=2)

        print("\nExpected Output:")
        print(expected_output_json)
        print("\nActual Output:")
        print(actual_output_json)

        # Basic comparison (you might need more sophisticated comparison for real tests)
        # Sort keys to ensure consistent order for comparison
        if actual_prescription_obj:
            sorted_actual = json.dumps(dict(sorted(actual_output_dict.items())), indent=2)
            sorted_expected = json.dumps(dict(sorted(test_case["expected_output"].items())), indent=2)

            if sorted_actual == sorted_expected:
                print("\nResult: PASSED (Exact Match)")
            else:
                print("\nResult: FAILED (Mismatch)")
                # You could add a diff tool here for better analysis
        else:
            print("\nResult: FAILED (Parsing Error)")

        print("\n" + "=" * 80 + "\n")

    print("--- Test run complete ---")
