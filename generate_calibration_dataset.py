"""
Generate calibration dataset for llama.cpp I-Matrix quantization.
Processes HR/ECG data from adolescent and MIMIC-III patient records.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CardiacCalibrationDatasetGenerator:
    """Generate cardiology-focused calibration data for I-matrix quantization."""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        """
        Initialize generator with data paths.
        
        Args:
            data_dir: Root data directory containing 'processed' folder
            output_dir: Directory for output calibration files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.processed_dir = self.data_dir / "processed"
        self.output_dir.mkdir(exist_ok=True)
        
        self.adolescent_dir = self.processed_dir / "hr_adolescent"
        self.mimic_dir = self.processed_dir / "hr_mimic_iii"
        
        self.calibration_records = []
        self.metadata = {
            "total_records": 0,
            "populations": {
                "adolescent": {"count": 0, "records": []},
                "adult": {"count": 0, "records": []},
                "neonate": {"count": 0, "records": []}
            },
            "total_tokens": 0,
            "output_file": "cardiology_calibration_imatrix.txt"
        }
    
    def extract_adolescent_metrics(self, subject_id: int, csv_path: Path, subject_info: Dict) -> Dict:
        """
        Extract metrics from adolescent HR data.
        
        Args:
            subject_id: Subject code
            csv_path: Path to CSV file
            subject_info: Demographic information
            
        Returns:
            Dictionary with extracted metrics
        """
        try:
            df = pd.read_csv(csv_path)
            hr_values = df['heart rate [#/min]'].dropna().values
            
            if len(hr_values) == 0:
                return None
            
            metrics = {
                "subject_id": subject_id,
                "population": "adolescent",
                "age_years": subject_info.get("age_years", "unknown"),
                "gender": subject_info.get("gender", "unknown"),
                "height_cm": subject_info.get("length_cm", "unknown"),
                "weight_kg": subject_info.get("weight_kg", "unknown"),
                "hr_count": len(hr_values),
                "hr_mean": float(np.mean(hr_values)),
                "hr_std": float(np.std(hr_values)),
                "hr_min": float(np.min(hr_values)),
                "hr_max": float(np.max(hr_values)),
                "hr_median": float(np.median(hr_values)),
                "data_points": len(hr_values),
                "duration_hours": len(hr_values) / 60  # 1-minute intervals
            }
            return metrics
        except Exception as e:
            logger.error(f"Error processing adolescent subject {subject_id}: {e}")
            return None
    
    def extract_mimic_metrics(self, patient_id: str, csv_path: Path) -> Dict:
        """
        Extract metrics from MIMIC-III ECG/HR data.
        
        Args:
            patient_id: Patient identifier (e.g., 'adult_3190202n')
            csv_path: Path to CSV file
            
        Returns:
            Dictionary with extracted metrics
        """
        try:
            df = pd.read_csv(csv_path)
            population_type = "adult" if patient_id.startswith("adult") else "neonate"
            
            # Extract basic HR metrics
            hr_values = pd.to_numeric(df['HR'], errors='coerce').dropna().values
            
            if len(hr_values) == 0:
                return None
            
            # Extract additional physiological parameters (with safe column access)
            pulse = pd.to_numeric(df.get('PULSE', pd.Series()), errors='coerce').dropna()
            resp = pd.to_numeric(df.get('RESP', pd.Series()), errors='coerce').dropna()
            spo2 = pd.to_numeric(df.get('%SpO2', pd.Series()), errors='coerce').dropna()
            
            # Extract beat counts (safe access)
            normal_beats = pd.to_numeric(df.get('Normal Beat Count', pd.Series()), errors='coerce').sum()
            pvc_count = pd.to_numeric(df.get('PVC Count', pd.Series()), errors='coerce').sum()
            svpb_count = pd.to_numeric(df.get('SVPB Count', pd.Series()), errors='coerce').sum()
            paced_beats = pd.to_numeric(df.get('Paced Beat Count', pd.Series()), errors='coerce').sum()
            
            # Extract rhythm abnormalities (safe access)
            bigeminy = pd.to_numeric(df.get('Bigeminy Percent', pd.Series()), errors='coerce').max()
            trigeminy = pd.to_numeric(df.get('Trigeminy Percent', pd.Series()), errors='coerce').max()
            pvc_runs = pd.to_numeric(df.get('PVC Run Count', pd.Series()), errors='coerce').sum()
            svpb_runs = pd.to_numeric(df.get('SVPB Run Count', pd.Series()), errors='coerce').sum()
            paced_run_count = pd.to_numeric(df.get('Paced Run Count', pd.Series()), errors='coerce').sum()
            
            # HR variability (safe access)
            hr_variability = pd.to_numeric(df.get('HR Variability', pd.Series()), errors='coerce').max()
            
            # Get elapsed time (first column should be time-related)
            elapsed_time = df.iloc[-1, 0] if len(df) > 0 else 0
            
            metrics = {
                "patient_id": patient_id,
                "population": population_type,
                "hr_count": len(hr_values),
                "hr_mean": float(np.mean(hr_values)),
                "hr_std": float(np.std(hr_values)),
                "hr_min": float(np.min(hr_values)),
                "hr_max": float(np.max(hr_values)),
                "hr_median": float(np.median(hr_values)),
                "pulse_mean": float(pulse.mean()) if len(pulse) > 0 else None,
                "resp_mean": float(resp.mean()) if len(resp) > 0 else None,
                "spo2_mean": float(spo2.mean()) if len(spo2) > 0 else None,
                "data_points": len(df),
                "duration_sec": float(elapsed_time) if not pd.isna(elapsed_time) else 0,
                "normal_beats": int(normal_beats) if not pd.isna(normal_beats) else 0,
                "pvc_count": int(pvc_count) if not pd.isna(pvc_count) else 0,
                "svpb_count": int(svpb_count) if not pd.isna(svpb_count) else 0,
                "paced_beats": int(paced_beats) if not pd.isna(paced_beats) else 0,
                "bigeminy_percent": float(bigeminy) if not pd.isna(bigeminy) else 0,
                "trigeminy_percent": float(trigeminy) if not pd.isna(trigeminy) else 0,
                "pvc_run_count": int(pvc_runs) if not pd.isna(pvc_runs) else 0,
                "svpb_run_count": int(svpb_runs) if not pd.isna(svpb_runs) else 0,
                "paced_run_count": int(paced_run_count) if not pd.isna(paced_run_count) else 0,
                "max_hr_variability": float(hr_variability) if not pd.isna(hr_variability) else None
            }
            return metrics
        except Exception as e:
            logger.error(f"Error processing MIMIC patient {patient_id}: {e}")
            return None
    
    def generate_adolescent_narrative(self, metrics: Dict) -> str:
        """Generate clinical narrative for adolescent patient."""
        if metrics is None:
            return ""
        
        subject_id = metrics['subject_id']
        age = metrics['age_years']
        gender = "Female" if metrics['gender'] == 'F' else "Male"
        height = metrics['height_cm']
        weight = metrics['weight_kg']
        hr_mean = metrics['hr_mean']
        hr_std = metrics['hr_std']
        hr_min = metrics['hr_min']
        hr_max = metrics['hr_max']
        duration = metrics['duration_hours']
        
        # Assessment based on HR values
        resting_status = "normal baseline resting" if 60 <= hr_mean <= 100 else (
            "elevated baseline resting" if hr_mean > 100 else "low baseline resting"
        )
        
        variability_assessment = "excellent" if hr_std < 5 else (
            "good" if hr_std < 10 else "moderate"
        )
        
        narrative = f"""Adolescent patient, {age} years old, {gender}. Height: {height} cm, Weight: {weight} kg.

Continuous heart rate monitoring over {duration:.1f} hours shows {resting_status} heart rate of {hr_mean:.1f} bpm (±{hr_std:.1f} bpm). Heart rate range: {hr_min:.0f} to {hr_max:.0f} bpm. HR variability is {variability_assessment}, indicating {'good cardiac stability and healthy autonomic tone' if hr_std < 10 else 'adequate cardiac response to stimuli'}.

Rhythm assessment: Regular sinus rhythm throughout monitoring period with no significant arrhythmias detected. No evidence of ectopic beats or conduction abnormalities. Cardiac rate response to daily activities appears appropriate for age."""
        
        return narrative
    
    def generate_mimic_narrative(self, metrics: Dict) -> str:
        """Generate clinical narrative for MIMIC-III patient."""
        if metrics is None:
            return ""
        
        patient_id = metrics['patient_id']
        population = metrics['population']
        hr_mean = metrics['hr_mean']
        hr_std = metrics['hr_std']
        hr_min = metrics['hr_min']
        hr_max = metrics['hr_max']
        pulse_mean = metrics['pulse_mean']
        resp_mean = metrics['resp_mean']
        spo2_mean = metrics['spo2_mean']
        normal_beats = metrics['normal_beats']
        pvc_count = metrics['pvc_count']
        svpb_count = metrics['svpb_count']
        paced_beats = metrics['paced_beats']
        bigeminy = metrics['bigeminy_percent']
        trigeminy = metrics['trigeminy_percent']
        pvc_runs = metrics['pvc_run_count']
        duration_min = metrics['duration_sec'] / 60
        
        # Build population context
        population_context = (
            "pediatric ICU patient, requiring continuous cardiac monitoring"
            if population == "neonate"
            else "adult ICU patient, with complex cardiac monitoring"
        )
        
        # Vital signs assessment
        vitals_str = f"Mean heart rate {hr_mean:.1f} bpm (±{hr_std:.1f} bpm), range {hr_min:.0f}-{hr_max:.0f} bpm"
        if pulse_mean:
            vitals_str += f"; pulse {pulse_mean:.1f} bpm"
        if resp_mean:
            vitals_str += f"; respiration {resp_mean:.1f} /min"
        if spo2_mean:
            vitals_str += f"; SpO2 {spo2_mean:.1f}%"
        
        # Rhythm assessment
        total_beats = normal_beats + pvc_count + svpb_count + paced_beats
        rhythm_components = []
        
        if paced_beats > 0:
            paced_pct = (paced_beats / total_beats * 100) if total_beats > 0 else 0
            rhythm_components.append(f"paced rhythm ({paced_pct:.1f}% paced beats)")
        
        if pvc_count > 0:
            pvc_pct = (pvc_count / total_beats * 100) if total_beats > 0 else 0
            rhythm_components.append(f"premature ventricular contractions (PVC) present ({pvc_pct:.1f}%, {int(pvc_count)} total)")
        
        if svpb_count > 0:
            svpb_pct = (svpb_count / total_beats * 100) if total_beats > 0 else 0
            rhythm_components.append(f"supraventricular premature beats ({svpb_pct:.1f}%, {int(svpb_count)} total)")
        
        if bigeminy > 0:
            rhythm_components.append(f"bigeminy detected ({bigeminy:.1f}%)")
        
        if trigeminy > 0:
            rhythm_components.append(f"trigeminy detected ({trigeminy:.1f}%)")
        
        # Build narrative
        narrative = f"""{population_context.capitalize()}.

Continuous cardiac monitoring over {duration_min:.1f} minutes. {vitals_str}.

Cardiac rhythm analysis: Predominantly normal sinus rhythm with {len(rhythm_components)} abnormality features documented."""
        
        if rhythm_components:
            narrative += " " + ", ".join(rhythm_components) + "."
        else:
            narrative += "Stable cardiac rhythm without significant arrhythmias."
        
        # Clinical interpretation
        if pvc_count > 0 or svpb_count > 0 or paced_beats > 0:
            narrative += " Patient demonstrates rhythm abnormalities requiring ongoing cardiac assessment and possible intervention."
        else:
            narrative += " Cardiac rhythm remains stable with appropriate rate control."
        
        return narrative
    
    def format_calibration_prompt(self, narrative: str) -> str:
        """Format narrative as calibration prompt matching finetune style."""
        if not narrative:
            return ""
        
        prompt = f"""Below is a cardiology topic:

{narrative}
"""
        return prompt
    
    def process_adolescent_subjects(self) -> int:
        """Process all adolescent subjects and generate records."""
        logger.info("Processing adolescent subjects...")
        
        # Load subject demographics
        info_path = self.adolescent_dir / "subjects_info.json"
        with open(info_path, 'r') as f:
            subjects_info = json.load(f)
        
        subject_dict = {s['subject_code']: s for s in subjects_info}
        
        count = 0
        for subject_info in subjects_info:
            subject_id = subject_info['subject_code']
            csv_path = self.adolescent_dir / f"subject_{subject_id}_hr.csv"
            
            if not csv_path.exists():
                logger.warning(f"CSV file not found for subject {subject_id}")
                continue
            
            # Extract metrics
            metrics = self.extract_adolescent_metrics(subject_id, csv_path, subject_info)
            if metrics is None:
                continue
            
            # Generate narrative and prompt
            narrative = self.generate_adolescent_narrative(metrics)
            prompt = self.format_calibration_prompt(narrative)
            
            # Store record
            record = {
                "type": "adolescent",
                "subject_id": subject_id,
                "metrics": metrics,
                "prompt": prompt
            }
            
            self.calibration_records.append(record)
            self.metadata["populations"]["adolescent"]["records"].append({
                "subject_id": subject_id,
                "data_points": metrics['data_points'],
                "duration_hours": metrics['duration_hours']
            })
            
            count += 1
            logger.info(f"Processed adolescent subject {subject_id}")
        
        self.metadata["populations"]["adolescent"]["count"] = count
        logger.info(f"Processed {count} adolescent subjects")
        return count
    
    def process_mimic_subjects(self) -> Dict[str, int]:
        """Process all MIMIC-III subjects and generate records."""
        logger.info("Processing MIMIC-III subjects...")
        
        counts = {"adult": 0, "neonate": 0}
        
        # Get all CSV files in MIMIC directory
        csv_files = sorted(self.mimic_dir.glob("*_*.csv"))
        
        for csv_path in csv_files:
            patient_id = csv_path.stem  # e.g., 'adult_3190202n'
            
            # Extract metrics
            metrics = self.extract_mimic_metrics(patient_id, csv_path)
            if metrics is None:
                continue
            
            # Generate narrative and prompt
            narrative = self.generate_mimic_narrative(metrics)
            prompt = self.format_calibration_prompt(narrative)
            
            # Determine population type
            pop_type = "adult" if patient_id.startswith("adult") else "neonate"
            
            # Store record
            record = {
                "type": pop_type,
                "patient_id": patient_id,
                "metrics": metrics,
                "prompt": prompt
            }
            
            self.calibration_records.append(record)
            self.metadata["populations"][pop_type]["records"].append({
                "patient_id": patient_id,
                "data_points": metrics['data_points'],
                "duration_sec": metrics['duration_sec']
            })
            
            counts[pop_type] += 1
            logger.info(f"Processed {pop_type} patient {patient_id}")
        
        self.metadata["populations"]["adult"]["count"] = counts["adult"]
        self.metadata["populations"]["neonate"]["count"] = counts["neonate"]
        logger.info(f"Processed {counts['adult']} adult and {counts['neonate']} neonate subjects")
        
        return counts
    
    def save_calibration_dataset(self):
        """Save merged calibration dataset to single text file."""
        output_file = self.output_dir / "cardiology_calibration_imatrix.txt"
        
        logger.info(f"Saving calibration dataset to {output_file}")
        
        with open(output_file, 'w') as f:
            for i, record in enumerate(self.calibration_records, 1):
                f.write(record['prompt'])
                
                # Add separator between records
                if i < len(self.calibration_records):
                    f.write("\n" + "="*80 + "\n\n")
        
        # Count tokens (rough estimate: avg 4 chars per token)
        with open(output_file, 'r') as f:
            content = f.read()
            estimated_tokens = len(content) // 4
            self.metadata["total_tokens"] = estimated_tokens
        
        logger.info(f"Calibration dataset saved: {output_file}")
        logger.info(f"Total prompts: {len(self.calibration_records)}")
        logger.info(f"Estimated tokens: {estimated_tokens}")
        
        return output_file
    
    def save_metadata(self):
        """Save dataset metadata to JSON."""
        output_file = self.output_dir / "calibration_dataset_log.json"
        
        self.metadata["total_records"] = len(self.calibration_records)
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                return super().default(obj)
        
        with open(output_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Metadata saved to {output_file}")
        return output_file
    
    def generate(self):
        """Main method to generate complete calibration dataset."""
        logger.info("Starting calibration dataset generation...")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Process subjects
        adolescent_count = self.process_adolescent_subjects()
        mimic_counts = self.process_mimic_subjects()
        
        total_subjects = adolescent_count + mimic_counts['adult'] + mimic_counts['neonate']
        logger.info(f"Total subjects processed: {total_subjects}")
        
        # Save outputs
        calibration_file = self.save_calibration_dataset()
        metadata_file = self.save_metadata()
        
        logger.info("=" * 80)
        logger.info("CALIBRATION DATASET GENERATION COMPLETE")
        logger.info(f"Output file: {calibration_file}")
        logger.info(f"Metadata file: {metadata_file}")
        logger.info(f"Total records: {len(self.calibration_records)}")
        logger.info(f"Estimated tokens: {self.metadata['total_tokens']}")
        logger.info("=" * 80)
        
        return {
            "calibration_file": calibration_file,
            "metadata_file": metadata_file,
            "total_records": len(self.calibration_records),
            "estimated_tokens": self.metadata['total_tokens'],
            "metadata": self.metadata
        }

# Execute if run directly
if __name__ == "__main__":
    # Paths
    data_dir = Path("./data")
    output_dir = Path("./calibration_output")
    
    # Generate dataset
    generator = CardiacCalibrationDatasetGenerator(data_dir, output_dir)
    results = generator.generate()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Calibration file: {results['calibration_file']}")
    print(f"Metadata file: {results['metadata_file']}")
    print(f"Total records: {results['total_records']}")
    print(f"Estimated tokens: {results['estimated_tokens']}")
    print(f"\nPopulation breakdown:")
    for pop, data in results['metadata']['populations'].items():
        print(f"  {pop.capitalize()}: {data['count']} subjects")
    print("="*80)
