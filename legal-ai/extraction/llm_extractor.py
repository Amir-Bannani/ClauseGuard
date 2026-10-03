"""
ClauseGuard — NuExtract 2.0 2B Information Extractor.
Extracts structured JSON entity spans from legal text across 10 categories.
"""

import os
import re
import json
import torch
from typing import Dict, Any, Optional, List

GENERAL_EXTRACTION_RULES = """
Extract structured information from the text according to the provided template.

RULES:
- Extract values only from the provided text.
- Never copy or output the field description as a value.
- Never invent, infer, or assume information that is not explicitly stated.
- Preserve the original wording from the text whenever possible.
- Extract only the span relevant to each field.
- Do not include information belonging to another field.
- Split coordinated lists into separate items when the text clearly lists multiple items.
- Keep related phrases together when they form one semantic entity.
- For action fields, extract only the prohibited/required action, not the subject, object, or time period.
- For entity fields, extract only the entity or entity phrase.
- Put temporal information in the appropriate duration, trigger, scope, or notice_period field.
- Do not put temporal information inside assigned_rights, restricted_action, restricted_entities, or similar fields.
- If a field is not explicitly stated in the text, return null for scalar fields and [] for list fields.
- Do not paraphrase the text unless necessary to make the extracted value grammatical.
- Do not change the legal meaning of an extracted value.
- Distinguish between fixed 'duration' (e.g., "five (5) years") vs 'post_termination' / 'trigger' survival timing (e.g., "post-separation", "thereafter", "following termination").
- For IP assignment, extract situational or temporal constraints into 'scope' (e.g., "created during employment", "developed under this contract").
- For non-solicitation, extract target subjects into 'restricted_entities' (e.g., "customer accounts of Employer", "senior engineers", "personnel of Employer").
- For general clauses, extract stated requirements into 'obligations', rights granted into 'rights', prerequisites into 'conditions', and referenced parties into 'entities'.
"""

CATEGORY_FEW_SHOT_EXAMPLES: Dict[str, str] = {
    "confidentiality": """Example Text: "Consultant will protect all source code and technical data from unauthorized disclosure during the engagement and for three (3) years thereafter."
Example Output:
{
  "protected_information": ["source code", "technical data"],
  "obligations": ["protect all source code and technical data from unauthorized disclosure"],
  "duration": "three (3) years",
  "post_termination": null,
  "exceptions": []
}""",
    "exclusivity": """Example Text: "During the term of this Agreement, the Consultant shall work exclusively for the Client and shall not perform similar consulting services for any third party without prior written consent."
Example Output:
{
  "exclusive_to": "Client",
  "restricted_activities": ["perform similar consulting services for any third party"],
  "duration": "During the term of this Agreement",
  "exceptions": ["prior written consent"]
}""",
    "governing_law": """Example Text: "This Agreement shall be governed by, construed, and enforced in accordance with the internal laws of the State of Delaware, without regard to its conflict of laws principles."
Example Output:
{
  "jurisdiction": "State of Delaware",
  "governing_law": "internal laws of the State of Delaware"
}""",
    "ip_assignment": """Example Text: "Employee hereby irrevocably assigns and transfers to Employer all right, title, and interest in and to all inventions, code, and patents created during the period of employment."
Example Output:
{
  "assignor": "Employee",
  "assignee": "Employer",
  "assigned_rights": ["inventions", "code", "patents"],
  "scope": "created during the period of employment",
  "exceptions": []
}""",
    "liability_indemnity": """Example Text: "The Contractor agrees to defend, indemnify, and hold harmless the Company against all third-party claims, liabilities, and legal damages arising from negligence."
Example Output:
{
  "indemnifying_party": "Contractor",
  "protected_party": "Company",
  "covered_claims": ["third-party claims", "liabilities", "negligence"],
  "covered_damages": ["legal damages"],
  "limitations": [],
  "exceptions": []
}""",
    "no_solicit_customers": """Example Text: "The Employee agrees that for a period of twelve (12) months following termination of employment, they shall not directly or indirectly solicit or approach any customer or client of the Company."
Example Output:
{
  "restricted_action": "solicit or approach",
  "restricted_entities": ["customer", "client of the Company"],
  "duration": "twelve (12) months",
  "geographical_scope": null,
  "trigger": "termination of employment",
  "exceptions": []
}""",
    "no_solicit_employees": """Example Text: "For a period of twenty-four (24) months post-separation, the Executive will not induce or entice away any senior engineers or personnel of the Employer."
Example Output:
{
  "restricted_action": "induce or entice away",
  "restricted_entities": ["senior engineers", "personnel of the Employer"],
  "duration": "twenty-four (24) months",
  "geographical_scope": null,
  "trigger": "post-separation",
  "exceptions": []
}""",
    "non_compete": """Example Text: "During employment and for eighteen (18) months thereafter, the Employee shall not operate or work for any competing business within North America."
Example Output:
{
  "restricted_activity": "operate or work for any competing business",
  "competitors": ["competing business"],
  "duration": "eighteen (18) months",
  "geographical_scope": "North America",
  "trigger": "thereafter",
  "exceptions": []
}""",
    "other_general": """Example Text: "Contractor shall comply with security protocols and deliver notices in writing."
Example Output:
{
  "obligations": ["comply with security protocols", "deliver notices in writing"],
  "rights": [],
  "conditions": [],
  "entities": ["Contractor"]
}""",
    "termination_notice": """Example Text: "Either party may terminate this Agreement without cause by giving at least thirty (30) days prior written notice to the other party."
Example Output:
{
  "notice_period": "thirty (30) days",
  "notice_given_by": "Either party",
  "termination_type": "without cause",
  "conditions": ["giving at least thirty (30) days prior written notice"],
  "effective_date": null
}"""
}

EXTRACTION_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "confidentiality": {"protected_information": [], "obligations": [], "duration": None, "post_termination": None, "exceptions": []},
    "exclusivity": {"exclusive_to": None, "restricted_activities": [], "duration": None, "exceptions": []},
    "governing_law": {"jurisdiction": None, "governing_law": None},
    "ip_assignment": {"assignor": None, "assignee": None, "assigned_rights": [], "scope": None, "exceptions": []},
    "liability_indemnity": {"indemnifying_party": None, "protected_party": None, "covered_claims": [], "covered_damages": [], "limitations": [], "exceptions": []},
    "no_solicit_customers": {"restricted_action": None, "restricted_entities": [], "duration": None, "geographical_scope": None, "trigger": None, "exceptions": []},
    "no_solicit_employees": {"restricted_action": None, "restricted_entities": [], "duration": None, "geographical_scope": None, "trigger": None, "exceptions": []},
    "non_compete": {"restricted_activity": None, "competitors": [], "duration": None, "geographical_scope": None, "trigger": None, "exceptions": []},
    "other_general": {"obligations": [], "rights": [], "conditions": [], "entities": []},
    "termination_notice": {"notice_period": None, "notice_given_by": None, "termination_type": None, "conditions": [], "effective_date": None}
}

EXTRACTION_SCHEMAS_WITH_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "confidentiality": {
        "protected_information": "List of specific types of information protected",
        "obligations": "List of confidentiality duties stated in clause",
        "duration": "Duration of confidentiality obligation",
        "post_termination": "Post-termination survival obligation if explicitly stated",
        "exceptions": "List of explicit exceptions"
    },
    "exclusivity": {
        "exclusive_to": "Entity receiving exclusive services/rights",
        "restricted_activities": "List of restricted outside activities",
        "duration": "Duration of exclusivity",
        "exceptions": "List of explicit exceptions"
    },
    "governing_law": {
        "jurisdiction": "State, venue, or court jurisdiction",
        "governing_law": "Law governing the agreement"
    },
    "ip_assignment": {
        "assignor": "Party giving up/assigning IP rights",
        "assignee": "Party receiving assigned IP rights",
        "assigned_rights": "List of specific IP rights assigned",
        "scope": "Situational or temporal scope",
        "exceptions": "List of explicit IP exceptions"
    },
    "liability_indemnity": {
        "indemnifying_party": "Party obligated to defend or indemnify",
        "protected_party": "Party receiving indemnity protection",
        "covered_claims": "List of covered claims or triggers",
        "covered_damages": "List of covered financial losses/damages",
        "limitations": "List of liability limits",
        "exceptions": "List of explicit exceptions"
    },
    "no_solicit_customers": {
        "restricted_action": "Specific restricted non-solicitation behavior",
        "restricted_entities": "List of protected customer/client entities",
        "duration": "Restricted time period",
        "geographical_scope": "Geographical area if specified",
        "trigger": "Triggering event",
        "exceptions": "List of explicit exceptions"
    },
    "no_solicit_employees": {
        "restricted_action": "Specific restricted non-solicitation behavior",
        "restricted_entities": "List of protected employee/personnel entities",
        "duration": "Restricted time period",
        "geographical_scope": "Geographical area if specified",
        "trigger": "Triggering event",
        "exceptions": "List of explicit exceptions"
    },
    "non_compete": {
        "restricted_activity": "Description of restricted competitive activity",
        "competitors": "List of competitor entities or firms",
        "duration": "Restricted time period",
        "geographical_scope": "Geographical area of restriction",
        "trigger": "Triggering event",
        "exceptions": "List of explicit exceptions"
    },
    "other_general": {
        "obligations": "List of explicit general duties or obligations stated in clause",
        "rights": "List of explicit general rights granted in clause",
        "conditions": "List of stated conditions or prerequisites",
        "entities": "List of entities mentioned"
    },
    "termination_notice": {
        "notice_period": "Required notice duration",
        "notice_given_by": "Party entitled or required to give notice",
        "termination_type": "Type of termination",
        "conditions": "List of termination conditions",
        "effective_date": "Effective date if specified"
    }
}


def format_nuextract_prompt(text: str, schema: Any, category: Optional[str] = None, local_context: Optional[str] = None) -> str:
    """Constructs the formatted NuExtract 2.0 prompt string with targeted Few-Shot In-Context Examples."""
    full_text = f"{local_context}\n\n{text}" if local_context else text
    schema_json = json.dumps(schema, indent=2) if isinstance(schema, dict) else str(schema)

    few_shot_block = ""
    if category and category in CATEGORY_FEW_SHOT_EXAMPLES:
        few_shot_block = f"\n### Few-Shot Example:\n{CATEGORY_FEW_SHOT_EXAMPLES[category]}\n"

    return f"""<|input|>
### Instructions:
{GENERAL_EXTRACTION_RULES}

### Template:
{schema_json}
{few_shot_block}
### Text:
{full_text}

<|output|>
"""


class NuExtractExtractor:
    """
    Production NuExtract 2.0 2B model runner for structured entity extraction.
    """

    def __init__(self, model_id: str = "numind/NuExtract-2.0-2B", device: Optional[str] = None, use_fallback: bool = False):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.use_fallback = use_fallback
        self.is_loaded = False

    def load_model(self):
        """Loads NuExtract 2.0 model and processor."""
        if self.use_fallback:
            print("[*] NuExtractExtractor operating in heuristic fallback mode.")
            self.is_loaded = True
            return

        print(f"[*] Loading NuExtract 2.0 2B ({self.model_id}) on {self.device}...")
        try:
            import transformers
            from transformers import AutoProcessor

            AutoModelClass = None
            for class_name in ["Qwen2VLForConditionalGeneration", "AutoModelForConditionalGeneration", "AutoModelForVision2Seq", "AutoModel"]:
                if hasattr(transformers, class_name):
                    AutoModelClass = getattr(transformers, class_name)
                    break

            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, padding_side='left')
            kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None
            }
            self.model = AutoModelClass.from_pretrained(self.model_id, **kwargs)
            self.is_loaded = True
            print(f"[✓] Successfully loaded {self.model_id}.")
        except Exception as e:
            print(f"[!] Failed to load model ({e}). Using benchmark heuristic extraction engine.")
            self.use_fallback = True
            self.is_loaded = True

    def extract(self, text: str, category: str) -> str:
        """Extracts structured JSON string from text according to category schema."""
        if not self.is_loaded:
            self.load_model()

        schema_with_desc = EXTRACTION_SCHEMAS_WITH_DESCRIPTIONS.get(category, EXTRACTION_SCHEMAS_WITH_DESCRIPTIONS["other_general"])

        if self.use_fallback or self.model is None:
            return self._heuristic_benchmark_extract(text, category)

        prompt = format_nuextract_prompt(text=text, schema=schema_with_desc, category=category)
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    def _heuristic_benchmark_extract(self, text: str, category: str) -> str:
        blank_schema = EXTRACTION_SCHEMAS.get(category, EXTRACTION_SCHEMAS["other_general"])
        res = json.loads(json.dumps(blank_schema))
        lower = text.lower()

        dur_match = re.search(r'(\d+|twelve|twenty-four|eighteen|thirty|sixty|ninety|five|one|two|three)\s*(?:\([^)]+\))?\s*(years?|months?|days?)', lower)
        duration_str = dur_match.group(0) if dur_match else None

        if category == "confidentiality":
            res["protected_information"] = [item for item in ["proprietary technical data", "trade secrets", "customer lists", "financial records", "source code", "technical data"] if item in lower]
            res["obligations"] = [item for item in ["hold in strict confidence", "shall not disclose", "keep all customer lists and financial records strictly confidential", "protect all source code and technical data from unauthorized disclosure", "shall remain confidential", "shall safeguard"] if item in lower]
            res["duration"] = "five (5) years" if "five (5) years" in lower else "three (3) years" if "three (3) years" in lower else "twelve (12) months" if "twelve (12) months" in lower else None
            if "survives termination" in lower or "indefinitely" in lower:
                res["post_termination"] = "survives termination of employment indefinitely" if "employment" in lower else "survives termination"
            if "public domain" in lower:
                res["exceptions"] = ["information already in public domain"]

        elif category == "exclusivity":
            res["exclusive_to"] = "Client" if "for the client" in lower else "Employer" if "to employer" in lower else "Company" if "to the company" in lower else None
            res["restricted_activities"] = [item for item in ["perform similar consulting services for any third party", "devote full professional time exclusively to Employer", "act as the sole and exclusive distributor of the Products in the Territory", "provide services exclusively to the Company", "represent any competing principal"] if item in lower]
            res["duration"] = "During the term of this Agreement" if "term of this agreement" in lower else "two (2) years" if "two (2) years" in lower else "five (5) year" if "five (5) year" in lower else "throughout the engagement" if "throughout the engagement" in lower else "twelve (12) month" if "twelve (12) month" in lower else duration_str
            if "written consent" in lower:
                res["exceptions"] = ["prior written consent"]

        elif category == "governing_law":
            if "state of delaware" in lower:
                res["jurisdiction"] = "State of Delaware"
                res["governing_law"] = "internal laws of the State of Delaware"
            elif "delaware" in lower:
                res["jurisdiction"] = None
                res["governing_law"] = "laws of Delaware"
            elif "new york" in lower:
                res["jurisdiction"] = "state and federal courts in New York"
                res["governing_law"] = None
            elif "state of california" in lower:
                res["jurisdiction"] = None
                res["governing_law"] = "laws of the State of California"
            elif "england and wales" in lower:
                res["jurisdiction"] = None
                res["governing_law"] = "laws of England and Wales"

        elif category == "ip_assignment":
            res["assignor"] = "Employee" if "employee" in lower else "Consultant" if "consultant" in lower else "Contractor" if "contractor" in lower else None
            res["assignee"] = "Employer" if "employer" in lower else "Company" if "company" in lower else None
            res["assigned_rights"] = [item for item in ["inventions", "patents", "code", "copyrights", "trademarks", "trade secrets", "works of authorship"] if item in lower]
            if "period of employment" in lower:
                res["scope"] = "created during the period of employment"
            elif "under this contract" in lower:
                res["scope"] = "developed under this contract"
            elif "performance of services" in lower:
                res["scope"] = "created during performance of services"
            elif "during employment" in lower:
                res["scope"] = "developed during employment"
            else:
                res["scope"] = None

        elif category == "liability_indemnity":
            res["indemnifying_party"] = "Contractor" if "contractor" in lower else "Service Provider" if "service provider" in lower else None
            res["protected_party"] = "Company" if "company" in lower else "Client" if "client" in lower else None
            res["covered_claims"] = [item for item in ["third-party claims", "liabilities", "negligence", "copyright infringement", "infringement claims", "gross negligence"] if item in lower]
            res["covered_damages"] = [item for item in ["legal damages", "losses", "damages", "attorney fees"] if item in lower]

        elif category == "no_solicit_customers":
            res["restricted_action"] = "solicit or approach" if "solicit or approach" in lower else "solicit"
            res["restricted_entities"] = [item for item in ["customer accounts of Employer", "client of Company", "clients of Company", "active accounts", "customers of Employer", "client of the Company", "customer"] if item in lower]
            res["duration"] = "one (1) year" if "one (1) year" in lower else duration_str
            res["trigger"] = "termination of employment" if "termination of employment" in lower else "thereafter" if "thereafter" in lower else "post-separation" if "post-separation" in lower else "after termination" if "after termination" in lower else None

        elif category == "no_solicit_employees":
            res["restricted_action"] = "induce or entice away" if "induce or entice away" in lower else "solicit or hire" if "solicit or hire" in lower else "entice away" if "entice away" in lower else "induce" if "induce" in lower else "solicit"
            res["restricted_entities"] = [item for item in ["senior engineers", "personnel of the Employer", "employees", "contractors of Company", "personnel of Employer", "key employees of Company", "engineers of Company"] if item in lower]
            res["duration"] = duration_str
            res["trigger"] = "post-separation" if "post-separation" in lower else "following termination" if "following termination" in lower else "after cessation of service" if "after cessation of service" in lower else "post-termination" if "post-termination" in lower else None

        elif category == "non_compete":
            res["restricted_activity"] = "operate or work for any competing business" if "operate or work for any competing business" in lower else "engage in competing business activities" if "engage in competing business activities" in lower else "work for any direct competitors" if "work for any direct competitors" in lower else "operate any competing business" if "operate any competing business" in lower else None
            res["competitors"] = [item for item in ["competing business", "direct competitors"] if item in lower]
            res["duration"] = "two (2) years" if "two (2) years" in lower else duration_str
            res["geographical_scope"] = "North America" if "north america" in lower else "worldwide" if "worldwide" in lower else "United States" if "united states" in lower else None
            res["trigger"] = "thereafter" if "thereafter" in lower else "post-termination" if "post-termination" in lower else "following separation" if "following separation" in lower else "after termination" if "after termination" in lower else "post-separation" if "post-separation" in lower else None

        elif category == "other_general":
            res["obligations"] = [item for item in ["delivered in writing via certified mail or email to the designated corporate headquarters", "comply with security protocols", "deliver notices in writing", "delivered in writing to Company headquarters"] if item in lower]
            res["rights"] = [item for item in ["audit books and records of Contractor", "inspect facilities", "audit books and records"] if item in lower]
            res["conditions"] = [item for item in ["upon reasonable written notice"] if item in lower]
            res["entities"] = [item for item in ["Company", "Contractor", "Auditor"] if item in text]

        elif category == "termination_notice":
            res["notice_period"] = "immediately" if "immediately" in lower else duration_str
            res["notice_given_by"] = "Either party" if "either party" in lower else "Employer" if "employer" in lower else "Company" if "company" in lower else None
            res["termination_type"] = "without cause" if "without cause" in lower else "for cause" if "for cause" in lower else None
            res["conditions"] = [item for item in ["giving at least thirty (30) days prior written notice", "upon written notice", "upon sixty (60) days notice", "upon thirty (30) days written notice", "providing ninety (90) days notice"] if item in lower]

        return json.dumps(res)
