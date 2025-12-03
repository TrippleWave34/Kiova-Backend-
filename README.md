# Kiova Backend API

Central backend for the Kiova Fashion Platform handling Authentication, Wardrobe, AI, and E-Commerce logic.

## Local Setup

### 1. Environment & Dependencies
Create a virtual environment and install dependencies. Python 3.11 is recommended.

**Linux / Mac:**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration
Create a .env file in the root directory. You can reference .env.example.
```bash
cp .env.example .env
```

Open .env and fill in your database credentials.


### 3. Run Server
Start the development server:
```bash
uvicorn api:app --reload
```

- API: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs

# Database Migrations (Alembic)

Use Alembic to handle database schema changes without data loss.
### Generate migration script:
Run this after modifying models.py.

```bash
alembic revision --autogenerate -m "Description of changes"
```

### Apply changes to DB:
Run this to execute the migration.
```bash
alembic upgrade head
```

# Deployment to Azure
There are two ways to deploy this API to the Azure Container App.

### Method 1: GitHub Actions (Automatic)
Recommended. The CI/CD pipeline runs automatically whenever you push code to the repository.
1. Commit and push changes to the main branch.
2. Check the Actions tab in GitHub to monitor the build.
3. The Azure Container App will automatically pull the latest image tag.
Note: Requires ACR_USERNAME and ACR_PASSWORD secrets to be configured in GitHub Settings.

### Method 2: Manual Script (Fallback)
Use this if you need to force a deployment from your local machine or if GitHub Actions is failing.
Prerequisites:
- Azure CLI installed (az login successful)
- Docker installed and running

Run the script:
```bash
chmod +x deploy.sh
./deploy.sh
```
This script will build the Docker image locally, push it to the Azure Container Registry, and force the Container App to update.