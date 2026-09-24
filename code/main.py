import io
import os
import json
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import ollama
from PIL import Image
import pillow_avif  # Registers AVIF support with Pillow

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset')

INPUT_FILE_PATH = os.path.join(DATA_DIR, os.getenv('INPUT_FILE_NAME'))
OUTPUT_FILE_PATH = os.path.join(DATA_DIR, os.getenv('OUTPUT_FILE_NAME'))

IMAGE_DIR = os.path.join(DATA_DIR, "images", os.getenv('IMAGE_DIRECTORY'))

LOG_FILE_PATH =  os.path.join(Path(__file__).resolve().parent, "logs")

execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(
    LOG_FILE_PATH,
    f"claim_classifier_{execution_timestamp}.log"
)

# ============================================================
# LOGGING SETUP
# ============================================================

logger = logging.getLogger("claim_classifier")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if the module is reloaded
if not logger.handlers:

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

# Reading the necessary prerequisites from CSV files
df_evidence_requirements = pd.read_csv(os.path.join(DATA_DIR, 'evidence_requirements.csv'))
df_user_history = pd.read_csv(os.path.join(DATA_DIR, 'user_history.csv'))

# Reading the actual input file
df_claims = pd.read_csv(INPUT_FILE_PATH)
df_claims = df_claims[["user_id","image_paths","user_claim","claim_object"]]

# Importing necessary Ollama models
OLLAMA_TEXT_MODEL = os.getenv('OLLAMA_TEXT_MODEL')
OLLAMA_IMAGE_MODEL = os.getenv('OLLAMA_IMAGE_MODEL')

def image_to_bytes(image_path, max_size=1024):
    """
        Reads an image from disk and converts it to JPEG bytes
        in memory.
    
        This handles your case where files have a .jpg extension
        but are actually AVIF images.
        """
    
    with Image.open(image_path) as img:

        img = img.convert("RGB")

        # Resize while maintaining aspect ratio
        img.thumbnail((max_size, max_size))

        buffer = io.BytesIO()

        img.save(
            buffer,
            format="JPEG",
            quality=50,
            optimize=True
        )

        return buffer.getvalue()

def image_classifier(claim):

    item = claim['claim_object']
    df_temp = df_evidence_requirements[(df_evidence_requirements['claim_object']==item) | (df_evidence_requirements['claim_object']=='all')]

    prompt = f"""
    This is a claim for: {claim['claim_object']}.
    Your task is to verify damage claims using images, a short claim conversation, user history, and minimum evidence requirements.
    
    The fields you have to generate are:
    - `evidence_standard_met`: `true` if the image set is sufficient to evaluate the claim; otherwise `false`
    - `evidence_standard_met_reason`: short reason for the evidence decision
    - `risk_flags`: semicolon-separated risk flags, or `none`
    - `issue_type`: visible issue type
    - `object_part`: relevant object part
    - `claim_status`: final decision: `supported`, `contradicted`, or `not_enough_information`
    - `claim_status_justification`: concise image-grounded explanation; mention relevant image IDs when helpful
    - `supporting_image_ids`: image IDs supporting the decision, separated by semicolons; use `none` if no image is sufficient
    - `valid_image`: `true` if the image set is usable for automated review; otherwise `false`
    - `severity`: `none`, `low`, `medium`, `high`, or `unknown`
    
    Refer to the {df_temp[["applies_to","minimum_image_evidence"]].to_string(index=False)} to determine the `evidence_standard_met` and `evidence_standard_met_reason` fields by analyzing the images and the claim conversation.
    Decide wisely whether the image evidence is necessary or not. Resturn false if you feel so.
    Refer to the user history for this user: {df_user_history[df_user_history['user_id']==claim['user_id']].to_string(index=False)}.
    Use this to determine `risk_flags` and `claim_status`.
    Use the image to identify the `issue_type`, `object_part`.
    Also use the same image to determine the `valid_image` and `supporting_image_ids` fields.
    Finally, provide a concise justification for the `claim_status` and `claim_status_justification`.
    Decide the `severity` wisely at the end based on the images and the claim conversation.
    
    Return ONLY JSON:
    
    {{
        "evidence_standard_met": ... ,
        "evidence_standard_met_reason": ... ,
        "risk_flags": ... ,
        "issue_type": ... ,
        "object_part": ... ,
        "claim_status": ... ,
        "claim_status_justification": ... ,
        "supporting_image_ids": ... ,
        "valid_image": ... ,
        "severity": ... 
    }}
    """

#     prompt = prompt = f"""
# Verify the damage claim using the images and information below.

# CLAIM:
# {claim['claim_object']}

# EVIDENCE REQUIREMENTS:
# {df_temp[["applies_to", "minimum_image_evidence"]].to_string(index=False)}

# USER HISTORY:
# {df_user_history[df_user_history['user_id'] == claim['user_id']].to_string(index=False)}

# Evaluate:
# - evidence sufficiency
# - visible damage and affected part
# - consistency with the claim
# - risk indicators
# - image usability
# - damage severity

# claim_status must be one of:
# supported | contradicted | not_enough_information

# severity must be one of:
# none | low | medium | high | unknown

# If evidence is insufficient, use not_enough_information.
# If an image is unusable, set valid_image=false.
# Use "none" for no risk flags.
# Use "none" for supporting_image_ids when no image supports the decision.

# Keep reasons concise.
# Return ONLY JSON.

# {{
#   "evidence_standard_met": true,
#   "evidence_standard_met_reason": "",
#   "risk_flags": "none",
#   "issue_type": "",
#   "object_part": "",
#   "claim_status": "not_enough_information",
#   "claim_status_justification": "",
#   "supporting_image_ids": "none",
#   "valid_image": true,
#   "severity": "unknown"
# }}
# """

    start_time = pd.Timestamp.now()
    
    # --------------------------------------------------
    # Build image paths
    # --------------------------------------------------

    image_paths = [
        os.path.join(
            IMAGE_DIR,
            *Path(x.strip()).parts[-2:]
        )
        for x in claim["image_paths"].split(";")
    ]

    print("\nClaim:", claim["claim_object"])

    # --------------------------------------------------
    # Convert images to JPEG bytes in memory
    # --------------------------------------------------

    image_data = []

    for image_path in image_paths:

        # print("Reading:", image_path)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image_bytes = image_to_bytes(image_path)

        image_data.append(image_bytes)

        # print(
        #     f"Loaded successfully: "
        #     f"{len(image_bytes):,} bytes"
        # )

    # --------------------------------------------------
    # Send images to Ollama
    # --------------------------------------------------

    response = ollama.chat(
        model=OLLAMA_IMAGE_MODEL,
        # keep_alive="10m",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": image_data
            }
        ],
        format="json",
        options={
            "temperature": 0,
            "num_predict": 250
        }
    )

    emd_time = pd.Timestamp.now()
    logger.info(f"Time taken for image inference: {emd_time - start_time}")

    # ----------------------------------------------------
    # Ollama token metrics
    # ----------------------------------------------------

    prompt_eval_count = getattr(
        response,
        "prompt_eval_count",
        None
    )

    prompt_eval_duration = getattr(
        response,
        "prompt_eval_duration",
        None
    )

    eval_count = getattr(
        response,
        "eval_count",
        None
    )

    eval_duration = getattr(
        response,
        "eval_duration",
        None
    )

    total_duration = getattr(
        response,
        "total_duration",
        None
    )

    load_duration = getattr(
        response,
        "load_duration",
        None
    )

    # Ollama durations are generally nanoseconds
    def ns_to_sec(value):
        if value is None:
            return None

        return round(
            value / 1_000_000_000,
            4
        )

    prompt_eval_sec = ns_to_sec(
        prompt_eval_duration
    )

    eval_sec = ns_to_sec(
        eval_duration
    )

    total_ollama_sec = ns_to_sec(
        total_duration
    )

    load_sec = ns_to_sec(
        load_duration
    )

    # Output tokens / generation time
    tokens_per_sec = None

    if eval_count and eval_duration:
        tokens_per_sec = round(
            eval_count /
            (eval_duration / 1_000_000_000),
            2
        )

    logger.info(
        f"ollama_total_sec={total_ollama_sec} | "
        f"load_sec={load_sec} | "
        f"prompt_eval_sec={prompt_eval_sec} | "
        f"eval_sec={eval_sec} | "
        f"prompt_tokens={prompt_eval_count} | "
        f"output_tokens={eval_count} | "
        f"tokens_per_sec={tokens_per_sec}"
    )

    response_json = json.loads(response.message["content"])

    # return pd.Series(
    #     response_json["evidence_standard_met"],
    #     response_json["evidence_standard_met_reason"],
    #     response_json["risk_flags"],
    #     response_json["issue_type"],
    #     response_json["object_part"],
    #     response_json["claim_status"],
    #     response_json["claim_status_justification"],
    #     response_json["supporting_image_ids"],
    #     response_json["valid_image"],
    #     response_json["severity"]
    # )

    return response_json
    # print(json.loads(response.message["content"])["risk_flags"])
    # return response.message["content"]
    # return json.loads(response.message["content"])

# df_claims = df_claims.iloc[0:2]

df_claims["response"] = df_claims.apply(image_classifier, axis=1)
df_claims = df_claims.join(df_claims.pop("response").apply(pd.Series))

print(f"Claim report generated successfully for {len(df_claims)} claims")

# Save output
df_claims.to_csv(
    OUTPUT_FILE_PATH,
    index=False
)

print(f"Output saved to: {OUTPUT_FILE_PATH}")

# Process all tickets
# print(df_claims_subset.iloc[0])
# print(image_classifier(df_claims.iloc[0]))
