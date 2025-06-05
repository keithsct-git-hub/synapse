import os
import sys
import time
from langchain_google_genai import ChatGoogleGenerativeAI  # For Gemini models
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


class Prescription(BaseModel):
    """
    This object represents all the information of interest that we want to parse from a prescription.
    """
    # Mandatory fields - no Optional, no default
    device: str = Field(description="Name of the device required for the patient.")
    ordering_provider: str = Field(description="The doctor's name that ordered the prescription.")

    # Optional fields - use Optional[type] and default to None
    mask_type: Optional[str] | None = Field(None, description="Type of mask for the required equipment.")
    add_ons: list[str] | None = Field(None, description="Any add ons that are required by the prescription.")
    qualifier: Optional[str] | None = Field(None, description="Requirements for the device.")
    diagnosis: Optional[str] | None = Field(None, description="The diagnosis for the patient.")
    SpO2: Optional[str] | None = Field(None, description="The SpO2 for the patient.")
    usage: list[str] | None = Field(None, description="When the patient should use the device.")
    type: Optional[str] | None = Field(None, description="The type of the device.")
    features: Optional[str] | None = Field(None, description="Features that the device includes.")
    accessories: list[str] | None = Field(None, description="Any accessories that are required by the prescription.")
    mobility_status: Optional[str] | None = Field(None, description="Description of the patient's mobility status.")
    components: Optional[str] | None = Field(None, description="The components to be included with the device.")
    compliance_status: Optional[str] | None = (
        Field(None, description="Whether the patient is noted as compliant or non-compliant."))


def parse_prescription(prescription_text: str) -> (Prescription, dict[str, any]):
    """
    Parse some text and return the fields of a Prescription object
    :param prescription_text: a str with the text to parse
    :return: a tuple with the prescription object (json) and a metrics object with time and tokens.
    """
    parser = PydanticOutputParser(pydantic_object=Prescription)
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.9)

    # Define the prompt more clearly, making sure the model understands to output JSON
    prompt_template = f"""
        You are an expert at reading medical prescriptions for durable medical equipment and extracting structured information.
        Your task is to accurately extract the following fields from the provided text and output them as a JSON object.
        
        **Crucial Extraction Guidelines:**
        
        1.  **Strict Field Inclusion:** Only extract the exact fields listed below. If a field's value is not explicitly present in the text, you MUST omit that field from the output JSON. Do not provide any values that are null, empty strings, or placeholders if the information is missing.
        2.  **Device and Type Distinction:**
            * **`device`**: This is the core medical equipment being prescribed (e.g., "manual wheelchair", "CPAP machine", "nebulizer", "hospital bed").
            * **`type`**: If a modifier directly describes the *category* or *kind* of the device (e.g., "lightweight" for a "wheelchair", "bilevel" for a "CPAP"), extract that modifier as the `type`. Do NOT include the type as a `feature`.
        3.  **Features vs. Add-ons Distinction:** This is critical.
            * **`features`**: These are *integral parts, inherent capabilities, or built-in components* of the `device` itself. They are not typically ordered or used separately.
                * **Examples of Features:** "elevating leg rests" (built into a wheelchair), "heated humidification" (built into a CPAP machine), "trapeze bar" (often permanently attached or part of a hospital bed frame), "side rails" (built into a hospital bed), "cellular modem" (built into a device).
            * **`add_ons`**: These are *separate, additional items or accessories* that are *not inherently part of the main device* but are prescribed to be used *alongside* it. They can often be ordered or replaced independently.
                * **Examples of Add-ons:** "humidifier" (a separate unit attached to a CPAP), "mouthpiece" and "tubing" (separate for a nebulizer), "bacterial filter" (replaceable), "travel bag" (separate item).
        4.  **List Fields:** For fields that are lists (`add_ons`, `accessories`, `features`, `usage`, `components`), extract all relevant items into a JSON array of strings.
        5.  **`usage` Modifier Stripping:** If a `usage` item includes a modifier (e.g., "during exertion"), extract *only the core term* (e.g., "exertion").
        6.  **`mobility_status` Strictness:**
            * **`mobility_status`**: This field MUST only be one of the following exact terms: "non-ambulatory" or "ambulatory". If the text contains other descriptors (e.g., "significant mobility issues"), you MUST omit the `mobility_status` field entirely from the output.
        7.  **Do not provide any values that are null or empty string.**
        
        **Fields to Extract:**
        
        * `device` (MANDATORY)
        * `mask_type`
        * `add_ons`
        * `qualifier`
        * `ordering_provider` (MANDATORY)
        * `diagnosis`
        * `SpO2`
        * `usage`
        * `type`
        * `features`
        * `accessories`
        * `mobility_status`
        * `components`
        * `compliance_status`
        
        Text: "{prescription_text}"
        
        {parser.get_format_instructions()}
    """, prescription_text

    start_time = time.time()
    response = llm.invoke(prompt_template)
    output_text = response.content
    metadata = response.usage_metadata
    end_time = time.time()
    elapsed_time = end_time - start_time
    metrics = {"input_tokens": metadata['input_tokens'], "output_tokens": metadata['output_tokens'],
               "total_tokens": metadata['total_tokens']}
    metrics.update({"response_time": elapsed_time})
    try:
        prescription = parser.parse(output_text)
        prescription.model_dump_json(exclude_none=True, indent=2)
        return prescription, metrics
    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        print(f"LLM output:\n{output_text}")


if __name__ == "__main__":
    if os.getenv("GOOGLE_API_KEY") is None or len(os.getenv("GOOGLE_API_KEY")) == 0:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it to your Gemini API key in the .env file before running.")
        exit(1)
    prescript, metrics = parse_prescription(sys.argv[1])
    print(prescript.model_dump_json(exclude_none=True, indent=2))
    print(metrics)
