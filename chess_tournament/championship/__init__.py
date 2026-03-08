"""
Chess Championship Framework

Provides modular tools for running multi-stage chess tournaments
with student submissions, validation, and leaderboard generation.

Example:
    from championship import ChessChampionship, ChampionshipConfig
    from chess_tournament import RandomPlayer
    
    config = ChampionshipConfig()
    baselines = {"random": {"name": "Random", "factory": lambda: RandomPlayer("Random")}}
    championship = ChessChampionship(config, baselines)
    results = championship.run(submissions_df)
"""

from .config import ChampionshipConfig, setup_logging
from .validator import SubmissionValidator
from .runner import TournamentRunner
from .leaderboard import LeaderboardGenerator
from .orchestrator import ChessChampionship

__all__ = [
    "ChampionshipConfig",
    "setup_logging",
    "SubmissionValidator",
    "TournamentRunner",
    "LeaderboardGenerator",
    "ChessChampionship"
]

__version__ = "0.1.0"