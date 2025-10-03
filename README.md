# MedTech Diagnostic LLM Pipeline

An end-to-end, open-source LLM-integrated AI diagnostic pipeline for medical imaging that combines volumetric segmentation (MONAI + Swin UNETR), vector retrieval (FAISS), and retrieval-augmented medical LLM report generation. It exposes a FastAPI backend and can be paired with a Next.js frontend.

## 🎯 Project Overview

This pipeline integrates:
- **Segmentation**: MONAI + Swin UNETR for volumetric semantic segmentation
- **Vector Retrieval**: FAISS for storing and retrieving embeddings from prior reports
- **Medical LLM**: BioMistral/Hippo/Falcon models for report generation
- **Context Protocol**: FHIR integration for clinical-grade context-driven inference
- **Deployment**: FastAPI backend with Next.js frontend

## 🏗️ Architecture

```
┌─────────────────┐      ┌─────────────────┐    ┌─────────────────┐
│   DICOM/NIfTI   │───▶ │  Segmentation   │───▶│  FAISS Index    │
│     Input       │      │  (Swin UNETR)   │    │   (Embeddings)  │
└─────────────────┘      └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐     ┌─────────────────┐    ┌─────────────────┐
│  FHIR           │◀───│  Context        │◀───│  Vector         │
│  Integration    │     │  Retrieval      │    │  Retrieval      │
└─────────────────┘     └─────────────────┘    └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  Medical LLM    │
                        │  Report Gen     │
                        └─────────────────┘

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (3.9+ recommended)
- **CUDA 11.8+** (optional, for GPU acceleration)
- **Node.js 16+** (for frontend)
- **8GB+ RAM** (16GB+ recommended)
- **10GB+ disk space** (for models and data)

### Automated Setup

#### Windows
```bash
# Clone the repository
git clone <repository-url>
cd MedTech-Diagnostic-LLM-Pipeline

# Run automated setup
setup.bat
```

#### Linux/macOS
```bash
# Clone the repository
git clone <repository-url>
cd MedTech-Diagnostic-LLM-Pipeline

# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. **Create Virtual Environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

2. **Install Dependencies**
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (with CUDA support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip uninstall faiss-cpu
pip install faiss-gpu
```

3. **Download Models**
```bash
python setup_models.py --model all
```

4. **Setup Frontend** (optional)
```bash
cd frontend
npm install
npm run build
cd ..
```

## 🎮 Usage

### Start the Pipeline

#### Option 1: Start Both Services
```bash
# Windows
start_pipeline.bat

# Linux/macOS
./start_pipeline.sh
```

#### Option 2: Start Services Separately
```bash
# Start backend
# Windows: start_backend.bat
# Linux/macOS: ./start_backend.sh
python api_server.py

# Start frontend (in another terminal)
# Windows: start_frontend.bat
# Linux/macOS: ./start_frontend.sh
cd frontend && npm run dev
```

### Access the Application

- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Frontend Interface**: http://localhost:3000

### Process Medical Images

#### Via Web Interface
1. Open http://localhost:3000
2. Upload a medical image (NIfTI or DICOM)
3. Wait for processing to complete
4. View the generated report and segmentation results

#### Via API
```bash
# Process by file path
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "data/inputs/sample_image.nii.gz"}'

# Upload and process
curl -X POST "http://localhost:8000/process-upload" \
  -F "file=@path/to/your/image.nii.gz"
```

#### Via Python
```python
import requests

# Process image
response = requests.post(
    "http://localhost:8000/process",
    json={"image_path": "data/inputs/sample_image.nii.gz"}
)

result = response.json()
print(result)
```

## 📁 Project Structure

```
MedTech-Diagnostic-LLM-Pipeline/
├── src/                          # Source code
│   ├── segmentation/             # Medical image segmentation
│   │   ├── segmentation_pipeline.py
│   │   ├── swin_unetr_model.py
│   │   ├── preprocessing.py
│   │   └── postprocessing.py
│   ├── vector_store/             # FAISS vector search
│   │   ├── faiss_index.py
│   │   ├── embedding_extractor.py
│   │   └── retrieval_engine.py
│   ├── llm/                      # Medical LLM integration
│   │   └── medical_llm.py
│   ├── mcp_fhir/                 # FHIR integration
│   │   └── mcp_fhir_client.py
│   ├── feedback/                 # Expert feedback system
│   │   └── feedback_collector.py
│   └── deployment/               # Deployment utilities
│       └── pipeline_deployer.py
├── configs/                      # Configuration files
│   ├── segmentation_config.yaml
│   ├── llm_config.yaml
│   ├── vector_store_config.yaml
│   ├── mcp_fhir_config.yaml
│   └── deployment_config.yaml
├── frontend/                     # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   └── lib/
│   ├── package.json
│   └── next.config.js
├── data/                         # Data directories
│   ├── inputs/                   # Input medical images
│   ├── outputs/                  # Processing outputs
│   ├── uploads/                  # Uploaded files
│   └── models/                   # Downloaded models
├── models/                       # Model storage
├── logs/                         # Application logs
├── main.py                       # Main pipeline orchestrator
├── api_server.py                 # FastAPI backend server
├── setup_models.py               # Model download utility
├── requirements.txt              # Python dependencies
├── setup.bat                     # Windows setup script
├── setup.sh                      # Linux/macOS setup script
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# API Server
NEXT_PUBLIC_API_URL=http://localhost:8000
API_HOST=0.0.0.0
API_PORT=8000

# CUDA Settings
CUDA_VISIBLE_DEVICES=0

# Model Settings
TRANSFORMERS_CACHE=./models/transformers_cache
HF_HOME=./models/huggingface_cache

# FHIR Server
FHIR_SERVER_URL=https://hapi.fhir.org/baseR4

# Hugging Face access token (for hub loading without pre-download)
HUGGINGFACE_HUB_TOKEN=medtech-capstone
```

### Model Configuration
Edit `configs/llm_config.yaml` to choose your preferred model and hub options:
```yaml
llm:
  model_type: "biomistral"  # "biomistral", "hippo", "falcon"
  use_hf_hub: true           # load directly from Hugging Face hub
  trust_remote_code: true    # enable for instruct/chat templates
  auth_env_var: HUGGINGFACE_HUB_TOKEN
```

### Segmentation Configuration
Edit `configs/segmentation_config.yaml`:
```yaml
segmentation:
  model:
    pretrained: true
    model_path: "models/swin_unetr_pretrained.pth"
```

## 📊 Supported Formats

### Input Formats
- **NIfTI**: `.nii`, `.nii.gz`
- **DICOM**: `.dcm` files or DICOM directories
- **Image formats**: `.png`, `.jpg`, `.tiff` (2D only)

### Output Formats
- **Segmentation masks**: NIfTI format
- **Reports**: JSON, HTML, text
- **Embeddings**: NumPy arrays
- **Metadata**: JSON

## 🧪 Testing

### Run Tests
```bash
# Basic import tests
python -c "import torch; import monai; import transformers; import faiss; print('✓ All imports successful')"

# API tests
curl http://localhost:8000/health

# Process sample data
python -c "
from main import MedTechPipeline
pipeline = MedTechPipeline()
pipeline.initialize_components()
result = pipeline.process_medical_image('data/inputs/sample_image.nii.gz')
print('✓ Pipeline test successful')
"
```

## 🔍 API Endpoints

### Core Endpoints
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /status` - Pipeline status
- `POST /process` - Process image by path
- `POST /process-upload` - Upload and process image
- `GET /history` - Processing history

### Model Management
- `POST /models/download` - Download models
- `GET /models/status` - Model status
- `POST /config/reload` - Reload configuration

### Search & Feedback
- `POST /search` - Search similar cases
- `POST /feedback` - Submit expert feedback
- `GET /metrics` - Performance metrics

## 🚨 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in configs/segmentation_config.yaml
inference:
  batch_size: 1
  sw_batch_size: 2
```

#### Model Download Fails
```bash
# Download models manually
python setup_models.py --model biomistral
python setup_models.py --model swin_unetr
```

#### Frontend Won't Start
```bash
# Check Node.js version
node --version  # Should be 16+

# Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
```

#### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check virtual environment
which python  # Should point to venv
```

### Performance Optimization

#### For CPU-Only Systems
- Set `device: "cpu"` in config files
- Use smaller models (e.g., distilled versions)
- Reduce input image resolution

#### For GPU Systems
- Install CUDA 11.8+
- Use `faiss-gpu` instead of `faiss-cpu`
- Set appropriate `CUDA_VISIBLE_DEVICES`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MONAI**: Medical imaging framework
- **Hugging Face**: Model hosting and transformers
- **FAISS**: Vector similarity search
- **BioMistral**: Medical language model
- **Next.js**: Frontend framework
- **FastAPI**: Backend framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: See `docs/` directory

---

**⚠️ Disclaimer**: This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult with qualified medical professionals for medical advice.
└─────────────────┘    └─────────────────┘     └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker (for deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd medtech-diagnostic-llm-pipeline
```

2. Create venv and install dependencies:
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
# or: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. (Optional) Login to Hugging Face if required by models:
```bash
huggingface-cli login
```

5. Run the pipeline CLI:
```bash
python main.py --mode single --input data/inputs/sample.nii.gz --output-dir data/segmentation_outputs
```

### API Server
```bash
python main.py --mode deploy --deployment-type api
# http://localhost:8000
# Endpoints:
#  GET  /health
#  GET  /status
#  POST /process {"image_path": "...", "output_dir": "..."}
#  POST /process-upload (multipart file upload)
```

### Docker
```bash
docker compose up --build
# API at http://localhost:8000
```

### Next.js Frontend (optional)
1. Scaffold a Next.js app under `frontend/`:
```bash
npm create next-app@latest frontend -- --typescript --eslint --tailwind --src-dir --app
cd frontend && npm install
```
2. Create `frontend/src/app/page.tsx` with a simple uploader calling the API `/process-upload`.
3. Set `frontend/.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```
4. Run the dev server:
```bash
npm run dev
# http://localhost:3000
```

## 📁 Project Structure

```
medtech-diagnostic-llm-pipeline/
├── src/
│   ├── segmentation/          # MONAI + Swin UNETR segmentation
│   ├── vector_store/          # FAISS indexing and retrieval
│   ├── llm/                   # Medical LLM integration
│   ├── mcp_fhir/              # MCP-FHIR integration
│   ├── feedback/              # Expert feedback and fine-tuning
│   └── deployment/            # Deployment and serving
├── configs/                   # Configuration files
├── data/                      # Data storage
├── models/                    # Model checkpoints
├── tests/                     # Unit tests
├── docs/                      # Documentation
└── examples/                  # Usage examples
```

## 🔧 Configuration

The pipeline is configured through YAML files in the `configs/` directory:

- `segmentation_config.yaml`: MONAI and Swin UNETR settings
- `llm_config.yaml`: LLM model configurations
- `mcp_fhir_config.yaml`: MCP-FHIR integration settings
- `deployment_config.yaml`: Deployment and serving settings

## 📊 Performance Metrics

- **Segmentation**: Dice score ~0.89, surface distance error
- **LLM**: BLEU score, clinical correctness, expert-rated alignment
- **Retrieval**: Precision@K, recall@K for context retrieval

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This is a research-grade implementation. The models are not validated for clinical use. Use only for prototyping and research purposes, not for medical advice.

## 📚 References

- [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B)
- [Hippo-7B](https://huggingface.co/cyberiada/hippo-7b)
- [MONAI](https://monai.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [MCP-FHIR](https://github.com/modelcontextprotocol/mcp-fhir) 