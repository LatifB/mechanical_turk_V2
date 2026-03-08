"""
Championship configuration management with Google Colab support.
"""

import logging
from pathlib import Path
from typing import Optional


class ChampionshipConfig:
    """
    Centralized configuration for chess championship tournaments.
    
    Automatically detects Google Drive if mounted, otherwise uses local directory.
    """
    
    def __init__(self, 
                 work_dir: Optional[Path] = None,
                 submission_dir: Path = Path("student_submissions"),
                 max_clone_timeout: int = 120):
        """
        Args:
            work_dir: Directory for tournament outputs. If None, tries Google Drive,
                     falls back to /content/chess_tournament_outputs (Colab) or
                     ./chess_tournament_outputs (local).
            submission_dir: Where to clone student repositories.
            max_clone_timeout: Git clone timeout in seconds.
        """
        # Auto-detect work directory (prefer Google Drive if mounted)
        if work_dir is None:
            drive_path = Path("/content/drive/MyDrive/chess_tournament_outputs")
            if Path("/content/drive").exists() and any(Path("/content/drive").iterdir()):
                work_dir = drive_path
            else:
                work_dir = Path("/content/chess_tournament_outputs") if Path("/content").exists() \
                          else Path("./chess_tournament_outputs")
        
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.submission_dir = Path(submission_dir)
        self.submission_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_clone_timeout = max_clone_timeout
        
        # Tournament stage files
        self.qualifiers_plan_csv = self.work_dir / "qualifiers_plan.csv"
        self.qualifiers_results_csv = self.work_dir / "qualifiers_results.csv"
        self.qualifiers_log_txt = self.work_dir / "qualifiers_log.txt"
        
        self.semifinals_plan_csv = self.work_dir / "semifinals_plan.csv"
        self.semifinals_results_csv = self.work_dir / "semifinals_results.csv"
        
        self.finals_plan_csv = self.work_dir / "finals_plan.csv"
        self.finals_results_csv = self.work_dir / "finals_results.csv"
        
        self.progress_html = self.work_dir / "progress.html"
        self.final_leaderboard_csv = self.work_dir / "leaderboard.csv"
        self.final_leaderboard_md = self.work_dir / "LEADERBOARD.md"
        self.participants_snapshot = self.work_dir / "participants_snapshot.json"
        self.validation_results_csv = self.work_dir / "validation_results.csv"


def setup_logging(config: ChampionshipConfig) -> logging.Logger:
    """
    Configure logging to file and console.
    
    Args:
        config: ChampionshipConfig instance
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("chess_championship")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(config.qualifiers_log_txt, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(fh)
    
    # Console handler (brief format)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    
    return logger
