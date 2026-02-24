"""
MedGemma Sentinel - Project Initialization
Creates necessary directories for the project to run
"""

import sys
from pathlib import Path


def main():
    """Create all necessary directories for the project"""
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Define directories to create
    directories = [
        project_root / "data" / "patients",
        project_root / "data" / "reports",
        project_root / "data" / "reports" / "plots",
        project_root / "data" / "synthetic",
        project_root / "llama-cpp",
    ]
    
    print("🏥 Initializing MedGemma Sentinel directories...")
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {directory.relative_to(project_root)}")
        except Exception as e:
            print(f"✗ Failed to create {directory}: {e}", file=sys.stderr)
            return 1
    
    print("\n✅ Initialization complete!")
    print(f"Project ready at: {project_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
