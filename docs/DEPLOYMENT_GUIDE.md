# FileForge Deployment Guide
## Personal Production Setup

This guide covers deploying FileForge for personal production use on Windows.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running FileForge](#running-fileforge)
5. [Database Setup](#database-setup)
6. [AI/ML Dependencies](#aiml-dependencies)
7. [Folder Watching](#folder-watching)
8. [Backup & Maintenance](#backup--maintenance)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum
- **OS:** Windows 10/11 (64-bit)
- **Python:** 3.11 or 3.12
- **RAM:** 8 GB
- **Storage:** 10 GB free (plus space for database)
- **GPU:** Not required (CPU-only mode available)

### Recommended
- **RAM:** 16+ GB (for large batch processing)
- **GPU:** NVIDIA with CUDA support (for OCR/vision acceleration)
- **Storage:** SSD for database performance

---

## Installation

### Step 1: Clone Repository

```powershell
cd C:\Users\brand\Development
git clone https://github.com/bjpl/file_forge.git
cd file_forge
```

### Step 2: Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
# Core installation
pip install -e .

# Development tools (optional)
pip install -e ".[dev]"
```

### Step 4: Verify Installation

```powershell
fileforge --version
fileforge --help
```

---

## Configuration

### Config File Location

```
~/.fileforge/config.toml
C:\Users\brand\.fileforge\config.toml
```

### Generate Default Config

```powershell
fileforge config init
```

### Full Configuration Template

Create/edit `C:\Users\brand\.fileforge\config.toml`:

```toml
# =============================================================================
# FileForge Configuration - Personal Production
# =============================================================================

[database]
path = "~/.fileforge/fileforge.db"
wal_mode = true                    # Better concurrent access

[scanning]
recursive = true                   # Scan subdirectories
max_size_mb = 500                  # Max file size to process
exclude_patterns = [
    "*.tmp",
    "*.bak",
    "Thumbs.db",
    "desktop.ini",
    ".git/*",
    "node_modules/*",
    "__pycache__/*"
]

[processing]
workers = 4                        # Parallel workers (adjust to CPU cores)
batch_size = 10                    # Files per batch
timeout = 300                      # Seconds before timeout

[ocr]
engine = "paddleocr"               # "paddleocr" or "tesseract"
gpu_enabled = true                 # Set false if no NVIDIA GPU
languages = ["en"]                 # OCR languages
confidence_threshold = 0.7

[vision]
model = "yolov8n"                  # Object detection model
confidence = 0.5
device = "cuda"                    # "cuda" or "cpu"

[faces]
enabled = true
model = "VGG-Face"                 # Face recognition model
detector = "retinaface"
distance_metric = "cosine"
threshold = 0.6

[nsfw]
enabled = true                     # NSFW content detection
threshold = 0.8

[llm]
provider = "ollama"
model = "llama3.2"                 # Or "mistral", "gemma2"
base_url = "http://localhost:11434"
temperature = 0.7
max_tokens = 1000

[organization]
base_path = "D:/Organized"         # Where organized files go
strategy = "category"              # "category", "date", "hybrid"
preserve_originals = true          # Copy vs move
create_symlinks = false

# Category folder mapping
[organization.folders]
images = "Pictures"
documents = "Documents"
invoices = "Documents/Invoices"
receipts = "Documents/Receipts"
medical = "Documents/Medical"
legal = "Documents/Legal"
photos_people = "Pictures/People"
photos_places = "Pictures/Places"
screenshots = "Pictures/Screenshots"

[logging]
level = "INFO"                     # DEBUG, INFO, WARNING, ERROR
file = "~/.fileforge/fileforge.log"
max_size_mb = 50
backup_count = 3

[watch]
enabled = false                    # Enable folder watching
debounce_seconds = 5               # Wait before processing new files
paths = [
    "C:/Users/brand/Downloads",
    "C:/Users/brand/Desktop"
]
```

### Environment Variables (Optional)

Create `.env` file or set system environment variables:

```powershell
# For Ollama (if not default)
$env:OLLAMA_HOST = "http://localhost:11434"

# For GPU selection
$env:CUDA_VISIBLE_DEVICES = "0"

# For Tesseract (if using)
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## Running FileForge

### GUI Mode (Recommended)

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Run GUI
python -m fileforge.gui.app
```

Or create a shortcut:
```powershell
# Create desktop shortcut
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\FileForge.lnk")
$Shortcut.TargetPath = "C:\Users\brand\Development\file_forge\.venv\Scripts\pythonw.exe"
$Shortcut.Arguments = "-m fileforge.gui.app"
$Shortcut.WorkingDirectory = "C:\Users\brand\Development\file_forge"
$Shortcut.IconLocation = "shell32.dll,3"
$Shortcut.Save()
```

### CLI Mode

```powershell
# Scan a folder
fileforge scan "D:\Photos" --recursive

# Process scanned files
fileforge process --workers 4

# Organize files
fileforge organize --strategy category --target "D:\Organized"

# Query files
fileforge query --tag "invoice" --category "documents"

# Search by content
fileforge search "quarterly report 2024"

# Find similar faces
fileforge faces cluster --threshold 0.6

# Watch folder (continuous)
fileforge watch "C:\Users\brand\Downloads" --auto-process
```

### Common Workflows

#### 1. Initial Photo Library Organization

```powershell
# Step 1: Scan entire photo library
fileforge scan "D:\Photos" --recursive --include "*.jpg,*.png,*.heic,*.raw"

# Step 2: Process all files (OCR, faces, objects)
fileforge process --batch-size 20 --workers 4

# Step 3: Cluster faces for identification
fileforge faces cluster --min-faces 3

# Step 4: Preview organization (dry-run)
fileforge organize --strategy hybrid --target "D:\Organized" --dry-run

# Step 5: Execute organization
fileforge organize --strategy hybrid --target "D:\Organized"
```

#### 2. Document Processing

```powershell
# Scan documents folder
fileforge scan "C:\Users\brand\Documents\Unsorted" --include "*.pdf,*.docx,*.xlsx"

# Process with OCR
fileforge process --ocr-only

# Organize by category
fileforge organize --strategy category
```

#### 3. Continuous Downloads Monitoring

```powershell
# Start watching Downloads folder
fileforge watch "C:\Users\brand\Downloads" --auto-process --auto-organize
```

---

## Database Setup

### Location
```
C:\Users\brand\.fileforge\fileforge.db
```

### Initialize Database

```powershell
fileforge db init
```

### Database Maintenance

```powershell
# Check database stats
fileforge db stats

# Vacuum database (reclaim space)
fileforge db vacuum

# Export database to JSON
fileforge db export --output backup.json

# Rebuild search index
fileforge db reindex
```

### Backup Database

```powershell
# Manual backup
Copy-Item "$env:USERPROFILE\.fileforge\fileforge.db" "D:\Backups\fileforge_$(Get-Date -Format 'yyyyMMdd').db"
```

---

## AI/ML Dependencies

### Ollama (Required for LLM)

```powershell
# Install Ollama
winget install Ollama.Ollama

# Start Ollama service
ollama serve

# Pull recommended model
ollama pull llama3.2

# Verify
ollama list
```

### PaddleOCR (Recommended OCR)

Already installed via pip. For GPU acceleration:

```powershell
# Install CUDA-enabled PaddlePaddle (optional)
pip install paddlepaddle-gpu
```

### Tesseract (Alternative OCR)

```powershell
# Install via winget
winget install UB-Mannheim.TesseractOCR

# Add to PATH
$env:Path += ";C:\Program Files\Tesseract-OCR"

# Install language packs
# Download from: https://github.com/tesseract-ocr/tessdata
```

### GPU Setup (NVIDIA)

```powershell
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi
```

---

## Folder Watching

### Start Watch Daemon

```powershell
# Foreground (for testing)
fileforge watch "C:\Users\brand\Downloads" --verbose

# As background process
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m fileforge watch C:\Users\brand\Downloads --auto-process"
```

### Windows Task Scheduler Setup

1. Open Task Scheduler
2. Create Basic Task: "FileForge Watcher"
3. Trigger: At log on
4. Action: Start a program
   - Program: `C:\Users\brand\Development\file_forge\.venv\Scripts\pythonw.exe`
   - Arguments: `-m fileforge watch "C:\Users\brand\Downloads" --auto-process`
   - Start in: `C:\Users\brand\Development\file_forge`
5. Conditions: Start only if AC power
6. Settings: Allow task to be run on demand

### Watched Folder Rules

Create `~/.fileforge/watch_rules.toml`:

```toml
[[rules]]
name = "Auto-organize downloads"
pattern = "*"
action = "process_and_organize"
delay_seconds = 10

[[rules]]
name = "Skip temp files"
pattern = "*.tmp,*.crdownload,*.part"
action = "ignore"

[[rules]]
name = "Priority for invoices"
pattern = "*invoice*,*receipt*"
action = "process"
priority = "high"
category = "documents/invoices"
```

---

## Backup & Maintenance

### Automated Backup Script

Create `backup_fileforge.ps1`:

```powershell
# FileForge Backup Script
$BackupDir = "D:\Backups\FileForge"
$Date = Get-Date -Format "yyyyMMdd_HHmmss"

# Create backup directory
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null

# Backup database
Copy-Item "$env:USERPROFILE\.fileforge\fileforge.db" "$BackupDir\fileforge_$Date.db"

# Backup config
Copy-Item "$env:USERPROFILE\.fileforge\config.toml" "$BackupDir\config_$Date.toml"

# Cleanup old backups (keep last 7)
Get-ChildItem "$BackupDir\fileforge_*.db" | Sort-Object CreationTime -Descending | Select-Object -Skip 7 | Remove-Item

Write-Host "Backup complete: $BackupDir"
```

Schedule in Task Scheduler to run daily.

### Weekly Maintenance

```powershell
# Vacuum database
fileforge db vacuum

# Rebuild indexes
fileforge db reindex

# Check for orphaned entries
fileforge db check --fix-orphans

# Clear processing cache
fileforge cache clear
```

---

## Troubleshooting

### Common Issues

#### 1. "Module not found" errors

```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Reinstall
pip install -e .
```

#### 2. OCR not working

```powershell
# Check PaddleOCR
python -c "from paddleocr import PaddleOCR; print('OK')"

# Check Tesseract
tesseract --version

# Try CPU mode
# In config.toml: gpu_enabled = false
```

#### 3. GPU out of memory

```powershell
# Reduce batch size in config.toml
# batch_size = 5

# Or force CPU mode
# device = "cpu"
```

#### 4. Ollama connection failed

```powershell
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama
ollama serve

# Check model is pulled
ollama list
```

#### 5. Database locked

```powershell
# Check for running processes
Get-Process | Where-Object {$_.Path -like "*fileforge*"}

# If stuck, wait or restart
Stop-Process -Name python -Force  # Use with caution
```

#### 6. Permission errors on organize

```powershell
# Run as administrator for system folders
# Or check target folder permissions
icacls "D:\Organized"
```

### Logs

```powershell
# View logs
Get-Content "$env:USERPROFILE\.fileforge\fileforge.log" -Tail 100

# Real-time log following
Get-Content "$env:USERPROFILE\.fileforge\fileforge.log" -Wait
```

### Debug Mode

```powershell
# Run with verbose output
fileforge --verbose scan "D:\Test"

# Enable debug logging
# In config.toml: level = "DEBUG"
```

---

## Quick Reference

### Essential Commands

| Command | Description |
|---------|-------------|
| `fileforge scan <path>` | Scan folder for files |
| `fileforge process` | Process scanned files |
| `fileforge organize` | Organize processed files |
| `fileforge query` | Search database |
| `fileforge search <text>` | Full-text search |
| `fileforge watch <path>` | Monitor folder |
| `fileforge undo` | Undo last operation |
| `fileforge db stats` | Database statistics |

### GUI Navigation

| View | Shortcut | Description |
|------|----------|-------------|
| Dashboard | Ctrl+1 | Overview & quick actions |
| File Browser | Ctrl+2 | Browse & select files |
| Processing | Ctrl+3 | Queue & progress |
| Results | Ctrl+4 | View processed files |
| Settings | - | Configuration |
| Refresh | F5 | Refresh current view |
| Open Folder | Ctrl+O | Open folder dialog |

### File Locations

| Item | Path |
|------|------|
| Config | `~/.fileforge/config.toml` |
| Database | `~/.fileforge/fileforge.db` |
| Logs | `~/.fileforge/fileforge.log` |
| Cache | `~/.fileforge/cache/` |

---

## Support

- **Repository:** https://github.com/bjpl/file_forge
- **Issues:** https://github.com/bjpl/file_forge/issues

---

*Last updated: December 2024*
