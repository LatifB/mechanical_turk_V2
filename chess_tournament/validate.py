import re
import time
import json
from typing import Dict, Any
import importlib.util
import traceback
import os
import shutil
import subprocess
import sys

def _validate_local(dir_name: str) -> dict:
    """
    Loads player.py, tries to instantiate and call get_move.
    Returns a dict with validation results (JSON-serializable).
    """
    res: Dict[str, Any] = {
        "import_ok": False,
        "class_found": False,
        "instance_ok": False,
        "move_from_fen": None,
        "valid_move_format": False,
        "error_message": None,
        "duration": None,
        "approved": False
    }

    UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"

    t0 = time.time()
    try:
        player_path = os.path.join(dir_name, "player.py")
        if not os.path.isfile(player_path):
            res["error_message"] = "player.py missing"
            return res

        # load module under unique name
        module_name = f"student_player_{os.path.basename(dir_name)}_{int(time.time()*1000)}"
        spec = importlib.util.spec_from_file_location(module_name, player_path)
        if spec is None or spec.loader is None:
            res["error_message"] = "cannot_create_spec"
            return res
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        res["import_ok"] = True

        # find class
        cls = getattr(mod, "TransformerPlayer", None)
        if cls is None:
            res["error_message"] = "TransformerPlayer class not found"
            return res
        res["class_found"] = True

        # instantiate (try common signatures)
        inst = None
        try:
            inst = cls("student_test")
        except TypeError:
            # try no-arg constructor
            try:
                inst = cls()
            except Exception:
                res["error_message"] = "instantiation_failed:\n" + traceback.format_exc()
                return res
        except Exception:
            res["error_message"] = "instantiation_raised:\n" + traceback.format_exc()
            return res

        res["instance_ok"] = True

        # call get_move
        try:
            move = inst.get_move(fen)
            res["move_from_fen"] = move
            # valid if None or matches simple UCI pattern (we allow additional text but prefer exact)
            if move is None:
                res["valid_move_format"] = True
            elif isinstance(move, str) and UCI_RE.match(move.strip()):
                res["valid_move_format"] = True
            else:
                # try to extract first UCI-like substring
                m = re.search(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", str(move))
                if m:
                    res["move_from_fen"] = m.group(1).lower()
                    res["valid_move_format"] = True
                else:
                    res["valid_move_format"] = False
        except Exception:
            res["error_message"] = "get_move_exception:\n" + traceback.format_exc()
            return res

        res["duration"] = time.time() - t0
        return res
    except Exception:
        res["error_message"] = "unexpected_child_error:\n" + traceback.format_exc()
        return res


def validate_player(repo: str) -> dict:
    """
    Validates a chess player implementation.
    Works in both Google Colab and regular Python environments.
    
    Args:
        repo: Repository URL (e.g., 'https://github.com/user/repo.git')
    
    Returns:
        dict: Validation results with 'approved' flag indicating success
    """
    original_dir = os.getcwd()
    repo_name = repo.split('/')[-1].replace('.git', '')
    cloned_path = os.path.join(original_dir, repo_name)
    
    try:
        # Clone the repository
        print(f"Cloning {repo}...")
        try:
            subprocess.run(
                ['git', 'clone', repo],
                cwd=original_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print("✓ Clone successful")
        except subprocess.CalledProcessError as e:
            print(f"✗ Git clone failed: {e.stderr}")
            return {
                "import_ok": False,
                "class_found": False,
                "instance_ok": False,
                "move_from_fen": None,
                "valid_move_format": False,
                "error_message": f"git clone failed: {e.stderr}",
                "duration": None,
                "approved": False
            }
        
        # Check for and install requirements.txt
        requirements_path = os.path.join(cloned_path, 'requirements.txt')
        if os.path.exists(requirements_path):
            print("Installing requirements.txt...")
            try:
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '-r', requirements_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("✓ Requirements installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠ Warning: pip install had issues: {e.stderr}")
                # Continue anyway - don't fail validation over pip
        
        # Run _validate_local on the directory
        print(f"Validating {repo_name}...")
        res = _validate_local(cloned_path)
        if res['import_ok'] and res['class_found'] and res['instance_ok'] and res['valid_move_format']:
            res['approved'] = True
            print("✅ Player approved!")
        else:
            print(f"❌ Player rejected: {res['error_message']}")
        
        return res
    
    finally:
        # Cleanup: remove cloned directory
        if os.path.exists(cloned_path):
            print(f"Cleaning up {repo_name}...")
            shutil.rmtree(cloned_path)
