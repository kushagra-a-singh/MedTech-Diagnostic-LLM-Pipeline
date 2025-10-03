"""
(NON-FUNTIONAL)Test script for the MedTech Diagnostic LLM Pipeline
"""

import sys
import os
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from segmentation import SegmentationPipeline
        print("✓ Segmentation module imported")
    except ImportError as e:
        print(f"✗ Segmentation module import failed: {e}")
    
    try:
        from vector_store import FAISSIndex
        print("✓ Vector store module imported")
    except ImportError as e:
        print(f"✗ Vector store module import failed: {e}")
    
    try:
        from llm import MedicalLLM
        print("✓ LLM module imported")
    except ImportError as e:
        print(f"✗ LLM module import failed: {e}")
    
    try:
        from mcp_fhir import MCPFHIRClient
        print("✓ MCP-FHIR module imported")
    except ImportError as e:
        print(f"✗ MCP-FHIR module import failed: {e}")
    
    try:
        from feedback import FeedbackCollector
        print("✓ Feedback module imported")
    except ImportError as e:
        print(f"✗ Feedback module import failed: {e}")
    
    try:
        from deployment import PipelineDeployer
        print("✓ Deployment module imported")
    except ImportError as e:
        print(f"✗ Deployment module import failed: {e}")

def test_configs():
    """Test configuration files."""
    print("\nTesting configuration files...")
    
    config_dir = Path("configs")
    config_files = [
        "segmentation_config.yaml",
        "vector_store_config.yaml",
        "llm_config.yaml",
        "mcp_fhir_config.yaml",
        "deployment_config.yaml"
    ]
    
    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"✓ {config_file} exists")
        else:
            print(f"✗ {config_file} missing")

def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("\nTesting pipeline initialization...")
    
    try:
        from main import MedTechPipeline
        
        pipeline = MedTechPipeline("configs")
        pipeline.initialize_components()
        
        status = pipeline.get_pipeline_status()
        print("✓ Pipeline initialized successfully")
        print(f"  Components: {list(status['components'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return False

def test_dummy_processing():
    """Test dummy processing."""
    print("\nTesting dummy processing...")
    
    try:
        from main import MedTechPipeline
        
        pipeline = MedTechPipeline("configs")
        pipeline.initialize_components()
        
        dummy_image = "test_dummy_image.nii.gz"
        result = pipeline.process_medical_image(dummy_image, "data/test_outputs")
        
        print(f"✓ Dummy processing completed with status: {result['status']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dummy processing failed: {e}")
        return False

def main():
    """Run all tests."""
    print("MedTech Diagnostic LLM Pipeline - Test Suite")
    print("=" * 50)
    
    test_imports()
    
    test_configs()
    
    init_success = test_pipeline_initialization()
    
    if init_success:
        test_dummy_processing()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Add your medical images to the data/ directory")
    print("3. Run the example: python examples/basic_usage.py")
    print("4. Or run the main pipeline: python main.py --help")

if __name__ == "__main__":
    main() 