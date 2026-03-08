"""
Submission validation for chess tournament.
"""

import os
import sys
import tempfile
import pandas as pd
from pathlib import Path
import logging

# Import validate_player from chess_tournament.validate
from chess_tournament.validate import validate_player


class SubmissionValidator:
    """
    Validates student submissions and clones repositories for tournament use.
    """
    
    def __init__(self, config, logger: logging.Logger = None):
        """
        Args:
            config: ChampionshipConfig instance
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
    
    def process_submissions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate submissions from Google Forms or similar.
        
        Args:
            df: DataFrame with columns: student_number, repo_url, etc.
                Must have at least 'student_number' and 'repo_url' columns.
        
        Returns:
            DataFrame with columns:
                - student_number
                - repo_url
                - repo_path (local path after cloning)
                - approved (bool)
                - error_msg (if validation failed)
                - import_ok, class_found, instance_ok, valid_move_format (validation details)
        """
        results = []
        
        for idx, row in df.iterrows():
            # Convert student_number to int first (to remove .0), then to string
            try:
                student_num = str(int(row.get("student_number", "")))
            except (ValueError, TypeError):
                student_num = str(row.get("student_number", "")).strip()
            
            repo_url = str(row.get("repo_url", "")).strip()
            
            if not repo_url or not student_num:
                self.logger.warning(f"Skipping row {idx}: missing student_number or repo_url")
                continue
            
            self.logger.info(f"Validating {student_num}: {repo_url}")
            
            # Create a unique temp directory for this validation
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Change to temp directory so clone happens there
                    original_cwd = os.getcwd()
                    os.chdir(temp_dir)
                    
                    # Use validate_player from chess_tournament.validate
                    validation_result = validate_player(repo_url)
                    approved = validation_result.get("approved", False)
                    
                    # Change back to original directory
                    os.chdir(original_cwd)
                    
                    if approved:
                        self.logger.info(f"✅ {student_num} APPROVED")
                        # Clone repo for tournament use
                        dest_path = self.config.submission_dir / student_num
                        if not dest_path.exists():
                            self._clone_repo(repo_url, dest_path)
                        else:
                            self.logger.info(f"Repository already exists for {student_num}, skipping clone")
                    else:
                        self.logger.warning(f"❌ {student_num} REJECTED: {validation_result.get('error_message')}")
                        dest_path = ""
                    
                    results.append({
                        "student_number": student_num,
                        "repo_url": repo_url,
                        "repo_path": str(dest_path),
                        "approved": approved,
                        "error_msg": validation_result.get("error_message", ""),
                        "import_ok": validation_result.get("import_ok", False),
                        "class_found": validation_result.get("class_found", False),
                        "instance_ok": validation_result.get("instance_ok", False),
                        "valid_move_format": validation_result.get("valid_move_format", False),
                        "duration": validation_result.get("duration", None),
                    })
                
                except Exception as e:
                    self.logger.error(f"Validation exception for {student_num}: {e}")
                    # Make sure we change back to original directory
                    try:
                        os.chdir(original_cwd)
                    except:
                        pass
                    
                    results.append({
                        "student_number": student_num,
                        "repo_url": repo_url,
                        "repo_path": "",
                        "approved": False,
                        "error_msg": str(e),
                        "import_ok": False,
                        "class_found": False,
                        "instance_ok": False,
                        "valid_move_format": False,
                        "duration": None,
                    })
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(self.config.validation_results_csv, index=False)
        
        approved_count = result_df["approved"].sum()
        self.logger.info(f"\nValidation complete: {len(result_df)} submissions, {approved_count} approved")
        
        return result_df
    
    def _clone_repo(self, repo_url: str, dest_path: Path):
        """
        Clone a Git repository to a local path.
        
        Args:
            repo_url: Git repository URL
            dest_path: Destination path for cloning
        """
        import subprocess
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            subprocess.run(
                ["git", "clone", repo_url, str(dest_path)],
                timeout=self.config.max_clone_timeout,
                capture_output=True,
                check=True
            )
            self.logger.info(f"✓ Cloned {repo_url} to {dest_path}")
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Git clone timed out for {repo_url}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git clone failed for {repo_url}: {e.stderr.decode()}")
