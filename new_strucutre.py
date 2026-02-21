"""
Reformat calibration dataset for MedGemma-4B-IT instruction-tuned model.
Converts plain narrative format to chat template format with proper turn structure.
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def reformat_for_medgemma_it(input_file, output_file, system_prompt_type="night_sentinel"):
    """
    Reformat calibration dataset to MedGemma-4B-IT chat template format.
    
    Args:
        input_file: Path to original calibration file with narratives
        output_file: Path to reformatted output file
        system_prompt_type: Type of system prompt ("night_sentinel" or "clinical_analysis")
    """
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False
    
    # System prompts for different contexts
    system_prompts = {
        "night_sentinel": (
            "[NIGHT SENTINEL SYSTEM]\n"
            "You are a cardiac monitoring specialist in night shift ICU surveillance mode. "
            "Analyze the following continuous cardiac monitoring data and provide a concise risk assessment:\n"
        ),
        "clinical_analysis": (
            "[CLINICAL CARDIOLOGY ANALYSIS]\n"
            "As a cardiologist, analyze the following patient cardiac monitoring data and provide clinical interpretation:\n"
        ),
        "quantization_calibration": (
            "[QUANTIZATION CALIBRATION]\n"
            "Process the following cardiac monitoring interpretation for model quantization calibration:\n"
        )
    }
    
    system_prompt = system_prompts.get(system_prompt_type, system_prompts["night_sentinel"])
    
    try:
        # Try to read with latin-1 encoding (more compatible with special chars)
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.info("UTF-8 decoding failed, trying latin-1 encoding...")
            with open(input_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Split the records by separator
        records = content.split("=" * 80)
        
        formatted_records = []
        record_count = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                record = record.strip()
                if not record:
                    continue
                
                # Remove the "Below is a cardiology topic:" header
                clean_record = record.replace("Below is a cardiology topic:", "").strip()
                
                if not clean_record:
                    continue
                
                # Create formatted prompt in MedGemma chat template
                formatted_prompt = (
                    "<start_of_turn>user\n"
                    f"{system_prompt}"
                    f"{clean_record}\n"
                    "<end_of_turn>\n"
                    "<start_of_turn>model\n"
                    "Clinical assessment complete. I have processed the cardiac monitoring data "
                    "and documented the rhythm analysis, vital signs, and clinical recommendations. "
                    "This record is calibrated for quantization-aware model optimization in cardiology domain.\n"
                    "<end_of_turn>\n\n"
                )
                
                f.write(formatted_prompt)
                formatted_records.append(formatted_prompt)
                record_count += 1
        
        logger.info(f"Successfully reformatted {record_count} records")
        logger.info(f"Output file: {output_path}")
        logger.info(f"File size: {output_path.stat().st_size} bytes")
        
        return True
        
    except Exception as e:
        logger.error(f"Error reformatting calibration dataset: {e}")
        return False


def reformat_all_variants(input_file, output_dir="./calibration_output"):
    """
    Generate multiple format variants for different use cases.
    
    Args:
        input_file: Path to original calibration file
        output_dir: Directory to save variant files
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    variants = {
        "night_sentinel": "medgemma_calibration_night_sentinel.txt",
        "clinical_analysis": "medgemma_calibration_clinical_analysis.txt",
        "quantization_calibration": "medgemma_calibration_quantization.txt"
    }
    
    logger.info("Generating calibration dataset variants...")
    
    for variant_type, output_filename in variants.items():
        output_file = output_path / output_filename
        success = reformat_for_medgemma_it(input_file, str(output_file), system_prompt_type=variant_type)
        if success:
            logger.info(f"✓ Generated {variant_type} variant: {output_file}")
        else:
            logger.error(f"✗ Failed to generate {variant_type} variant")
    
    logger.info("All variants generated successfully!")


# Main execution
if __name__ == "__main__":
    input_file = "calibration_output/cardiology_calibration_imatrix.txt"
    output_dir = "calibration_output"
    
    # Generate single variant (for I-matrix quantization)
    logger.info("=" * 80)
    logger.info("REFORMATTING CALIBRATION DATASET FOR MEDGEMMA-4B-IT")
    logger.info("=" * 80)
    
    success = reformat_for_medgemma_it(
        input_file,
        f"{output_dir}/medgemma_calibration_imatrix_formatted.txt",
        system_prompt_type="quantization_calibration"
    )
    
    if success:
        print("\n" + "=" * 80)
        print("SUCCESS: Calibration dataset reformatted for MedGemma-4B-IT")
        print("=" * 80)
        print(f"Output file: {output_dir}/medgemma_calibration_imatrix_formatted.txt")
        print("\nFormat: Instruction-tuned chat template")
        print("Ready for: llama.cpp I-matrix quantization with MedGemma instruction-following")
        print("=" * 80)
    else:
        print("ERROR: Reformatting failed. Check logs above.")
    
    # Optional: Generate all variants
    # reformat_all_variants(input_file, output_dir)