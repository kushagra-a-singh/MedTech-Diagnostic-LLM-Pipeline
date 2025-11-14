"""
Local patient information storage (FHIR substitute).
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PatientStorage:
    """
    Local storage for patient information (FHIR substitute).
    """

    def __init__(self, storage_path: str = "data/patient_storage"):
        """
        Initialize patient storage.

        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.patients_file = self.storage_path / "patients.json"
        self.observations_file = self.storage_path / "observations.json"
        self.exams_file = self.storage_path / "exams.json"

        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize storage files if they don't exist."""
        if not self.patients_file.exists():
            self._save_json(self.patients_file, {})

        if not self.observations_file.exists():
            self._save_json(self.observations_file, {})

        if not self.exams_file.exists():
            self._save_json(self.exams_file, {})

    def _load_json(self, file_path: Path) -> Dict:
        """Load JSON file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}

    def _save_json(self, file_path: Path, data: Dict):
        """Save JSON file."""
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")

    def create_patient(
        self,
        patient_id: str,
        name: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Create or update a patient record.

        Args:
            patient_id: Patient identifier
            name: Patient name
            age: Patient age
            gender: Patient gender
            **kwargs: Additional patient information

        Returns:
            Patient record
        """
        patients = self._load_json(self.patients_file)

        patient = {
            "patient_id": patient_id,
            "name": name,
            "age": age,
            "gender": gender,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **kwargs,
        }

        patients[patient_id] = patient
        self._save_json(self.patients_file, patients)

        logger.info(f"Created/updated patient: {patient_id}")
        return patient

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """
        Get patient information.

        Args:
            patient_id: Patient identifier

        Returns:
            Patient record or None
        """
        patients = self._load_json(self.patients_file)
        return patients.get(patient_id)

    def add_observation(
        self,
        patient_id: str,
        observation_type: str,
        value: Any,
        unit: Optional[str] = None,
        date: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Add a clinical observation.

        Args:
            patient_id: Patient identifier
            observation_type: Type of observation (e.g., "blood_pressure", "temperature")
            value: Observation value
            unit: Unit of measurement
            date: Observation date
            **kwargs: Additional observation data

        Returns:
            Observation record
        """
        observations = self._load_json(self.observations_file)

        if patient_id not in observations:
            observations[patient_id] = []

        observation = {
            "observation_id": f"obs_{len(observations[patient_id]) + 1}",
            "patient_id": patient_id,
            "type": observation_type,
            "value": value,
            "unit": unit,
            "date": date or datetime.now().isoformat(),
            **kwargs,
        }

        observations[patient_id].append(observation)
        self._save_json(self.observations_file, observations)

        logger.info(f"Added observation for patient {patient_id}: {observation_type}")
        return observation

    def get_observations(
        self, patient_id: str, observation_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get patient observations.

        Args:
            patient_id: Patient identifier
            observation_type: Optional filter by observation type

        Returns:
            List of observations
        """
        observations = self._load_json(self.observations_file)
        patient_obs = observations.get(patient_id, [])

        if observation_type:
            return [obs for obs in patient_obs if obs.get("type") == observation_type]

        return patient_obs

    def add_exam(
        self,
        patient_id: str,
        exam_type: str,
        modality: str,
        body_region: str,
        image_path: Optional[str] = None,
        report: Optional[str] = None,
        date: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Add an imaging exam record.

        Args:
            patient_id: Patient identifier
            exam_type: Type of exam
            modality: Imaging modality (CT, MRI, etc.)
            body_region: Body region imaged
            image_path: Path to image file
            report: Exam report
            date: Exam date
            **kwargs: Additional exam data

        Returns:
            Exam record
        """
        exams = self._load_json(self.exams_file)

        if patient_id not in exams:
            exams[patient_id] = []

        exam = {
            "exam_id": f"exam_{len(exams[patient_id]) + 1}",
            "patient_id": patient_id,
            "exam_type": exam_type,
            "modality": modality,
            "body_region": body_region,
            "image_path": image_path,
            "report": report,
            "date": date or datetime.now().isoformat(),
            **kwargs,
        }

        exams[patient_id].append(exam)
        self._save_json(self.exams_file, exams)

        logger.info(f"Added exam for patient {patient_id}: {exam_type} ({modality})")
        return exam

    def get_exams(
        self,
        patient_id: str,
        modality: Optional[str] = None,
        body_region: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get patient exams.

        Args:
            patient_id: Patient identifier
            modality: Optional filter by modality
            body_region: Optional filter by body region

        Returns:
            List of exams
        """
        exams = self._load_json(self.exams_file)
        patient_exams = exams.get(patient_id, [])

        filtered = patient_exams
        if modality:
            filtered = [exam for exam in filtered if exam.get("modality") == modality]
        if body_region:
            filtered = [
                exam for exam in filtered if exam.get("body_region") == body_region
            ]

        return filtered

    def get_patient_context(self, patient_id: str) -> Dict:
        """
        Get comprehensive patient context (FHIR-like query).

        Args:
            patient_id: Patient identifier

        Returns:
            Comprehensive patient context
        """
        patient = self.get_patient(patient_id)
        observations = self.get_observations(patient_id)
        exams = self.get_exams(patient_id)

        context = {
            "patient": patient or {},
            "observations": observations,
            "prior_exams": exams,
            "exam_count": len(exams),
            "observation_count": len(observations),
        }

        return context

    def search_patients(self, **filters) -> List[Dict]:
        """
        Search for patients by criteria.

        Args:
            **filters: Search filters (e.g., age, gender)

        Returns:
            List of matching patients
        """
        patients = self._load_json(self.patients_file)
        results = []

        for patient_id, patient in patients.items():
            match = True
            for key, value in filters.items():
                if patient.get(key) != value:
                    match = False
                    break
            if match:
                results.append(patient)

        return results
