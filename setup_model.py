import os
import shutil
import subprocess
import sys

def main():
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        # Check standard portable winget or user local locations
        candidate_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Ollama.Ollama.Portable_Microsoft.Winget.Source_8wekyb3d8bbwe\ollama.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ollama.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            r"C:\Program Files\Ollama\ollama.exe",
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                ollama_bin = p
                break

    if not ollama_bin:
        ollama_bin = "ollama"

    cmd = [ollama_bin, "create", "jarvis-3b", "-f", "Modelfile"]
    print(f"Executing: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print("Success:")
        print(proc.stdout)
        if proc.stderr:
            print("Stderr:", proc.stderr)
    except FileNotFoundError:
        print("Error: 'ollama' executable not found on system PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error during model creation (exit code {e.returncode}):")
        print(e.stdout)
        print(e.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
