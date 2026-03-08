"""
Leaderboard generation and formatting.
"""

from datetime import datetime
from pathlib import Path
import pandas as pd


class LeaderboardGenerator:
    """Generate final leaderboard in various formats."""
    
    @staticmethod
    def write_markdown(results_df: pd.DataFrame, out_path: Path):
        """
        Write leaderboard to readable Markdown format.
        
        Args:
            results_df: Results DataFrame (participant_name, points, fallbacks, etc.)
            out_path: Path to write Markdown file
        """
        lines = [
            "# 🏆 Chess Championship Final Leaderboard",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Final Rankings",
            "",
            "| Rank | Player | Points | Fallbacks |",
            "|------|--------|--------|-----------|",
        ]
        
        for rank, (_, row) in enumerate(results_df.iterrows(), 1):
            player_name = row.get("participant_name", "N/A")
            points = float(row.get("points", 0))
            fallbacks = int(row.get("fallbacks", 0))
            lines.append(f"| {rank} | {player_name} | {points:.1f} | {fallbacks} |")
        
        lines.append("")
        lines.append("## Tie-Breaking Rules")
        lines.append("")
        lines.append("1. **Points**: Higher is better (wins/draws)")
        lines.append("2. **Buchholz**: Sum of opponents' final points (Swiss rounds)")
        lines.append("3. **Fallbacks**: Lower is better (fewer errors/crashes)")
        
        out_path.write_text("\n".join(lines))
