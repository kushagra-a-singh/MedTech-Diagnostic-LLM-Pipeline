"""
MCP-FHIR client implementation.
"""

import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class MCPFHIRClient:
    """
    MCP-FHIR client for clinical context integration.
    """

    def __init__(self, config: Dict):
        """
        Initialize MCP-FHIR client.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fhir_server_config = config["fhir_server"]
        self.mcp_agent_config = config["mcp_agent"]
        self.auth_config = config.get("authentication", {"enabled": False})
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/fhir+json"})

        logger.info("MCP-FHIR client initialized")

    def _base_url(self) -> str:
        return self.fhir_server_config["base_url"].rstrip("/")

    def _request(self, method: str, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self._base_url()}/{path.lstrip('/')}"
        timeout = self.fhir_server_config.get("timeout", 30)
        for attempt in range(self.fhir_server_config.get("retry_attempts", 3)):
            try:
                resp = self.session.request(method, url, params=params, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"FHIR request failed (attempt {attempt+1}): {e}")
        return {}

    def get_patient_context(self, patient_id: str) -> Dict:
        """
        Get patient clinical context from FHIR.
        """
        logger.info(f"Getting clinical context for patient: {patient_id}")
        context: Dict = {"patient_id": patient_id}
      
        patient = self._request("GET", f"Patient/{patient_id}")
        if patient:
            context.update(
                {
                    "gender": patient.get("gender", "unknown"),
                    "birthDate": patient.get("birthDate", "unknown"),
                    "name": patient.get("name", [{}])[0].get("text", "unknown"),
                }
            )
        
        obs_bundle = self._request(
            "GET",
            "Observation",
            params={"subject": f"Patient/{patient_id}", "_count": 50},
        )
        observations = (
            [e["resource"] for e in obs_bundle.get("entry", [])] if obs_bundle else []
        )
      
        cond_bundle = self._request(
            "GET",
            "Condition",
            params={"subject": f"Patient/{patient_id}", "_count": 50},
        )
        conditions = (
            [e["resource"] for e in cond_bundle.get("entry", [])] if cond_bundle else []
        )
       
        med_bundle = self._request(
            "GET",
            "MedicationRequest",
            params={"subject": f"Patient/{patient_id}", "_count": 50},
        )
        medications = (
            [e["resource"] for e in med_bundle.get("entry", [])] if med_bundle else []
        )
   
        img_bundle = self._request(
            "GET",
            "ImagingStudy",
            params={"subject": f"Patient/{patient_id}", "_count": 20},
        )
        imaging_studies = (
            [e["resource"] for e in img_bundle.get("entry", [])] if img_bundle else []
        )

        context.update(
            {
                "observations": observations,
                "conditions": conditions,
                "medications": medications,
                "imaging_studies": imaging_studies,
            }
        )
        return context

    def query_fhir_resources(
        self, resource_type: str, query_params: Dict
    ) -> List[Dict]:
        """Query FHIR resources generically."""
        bundle = self._request("GET", resource_type, params=query_params)
        return [e["resource"] for e in bundle.get("entry", [])] if bundle else []

    def get_imaging_studies(self, patient_id: str) -> List[Dict]:
        return self.query_fhir_resources(
            "ImagingStudy", {"subject": f"Patient/{patient_id}", "_count": 50}
        )

    def get_diagnostic_reports(self, patient_id: str) -> List[Dict]:
        return self.query_fhir_resources(
            "DiagnosticReport", {"subject": f"Patient/{patient_id}", "_count": 50}
        )

    def get_conditions(self, patient_id: str) -> List[Dict]:
        return self.query_fhir_resources(
            "Condition", {"subject": f"Patient/{patient_id}", "_count": 50}
        )

    def get_medications(self, patient_id: str) -> List[Dict]:
        return self.query_fhir_resources(
            "MedicationRequest", {"subject": f"Patient/{patient_id}", "_count": 50}
        )
