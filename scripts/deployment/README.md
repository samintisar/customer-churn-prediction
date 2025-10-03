# Deployment Scripts

This directory contains scripts for starting the production API server.

## 📁 Files

| File | Description | Platform |
|------|-------------|----------|
| `start_api.py` | Cross-platform Python startup script | All |
| `start_production_api.ps1` | PowerShell startup script | Windows |
| `start_production_api.bat` | Batch startup script | Windows |

## 🚀 Usage

### From Project Root

**PowerShell (Recommended for Windows):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\deployment\start_production_api.ps1
```

**Batch File:**
```cmd
scripts\deployment\start_production_api.bat
```

**Python Script:**
```bash
conda run -n ml-conda python scripts/deployment/start_api.py --production --port 5000
```

### Options for start_api.py

```bash
# Development mode (Flask dev server)
python scripts/deployment/start_api.py

# Production mode (Waitress/Gunicorn)
python scripts/deployment/start_api.py --production

# Custom host and port
python scripts/deployment/start_api.py --production --host 0.0.0.0 --port 8000

# Custom worker threads/processes
python scripts/deployment/start_api.py --production --workers 8

# Skip requirement checks (faster startup)
python scripts/deployment/start_api.py --production --skip-checks
```

## ℹ️ Notes

- All scripts automatically change to the project root directory before starting
- The `.ps1` and `.bat` scripts automatically use the `ml-conda` environment
- The Python script can work with any active Python environment
- For production, use `--production` flag to enable Waitress (Windows) or Gunicorn (Unix)

## 📚 Related Documentation

- Main deployment guide: `docs/DEPLOYMENT_GUIDE.md`
- Deployment success info: `docs/DEPLOYMENT_SUCCESS.md`
- Quick start guide: `docs/QUICK_START.md`
