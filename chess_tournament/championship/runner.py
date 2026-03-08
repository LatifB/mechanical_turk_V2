"""
Tournament execution using Swiss and round-robin formats.
"""

import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from ..tournament import swiss_tournament, instantiate_participant, destroy_instance


class TournamentRunner:
    """
    Orchestrates tournament stages (qualifiers, semifinals, finals).
    
    Uses Swiss tournament for group stages and can use round-robin for finals.
    """
    
    def __init__(self, config, logger: logging.Logger, baseline_factories: Dict[str, Dict[str, Any]] = None):
        """
        Args:
            config: ChampionshipConfig instance
            logger: Logger instance
            baseline_factories: Dict of baseline factories (needed for instantiation)
        """
        self.config = config
        self.logger = logger
        self.baseline_factories = baseline_factories or {}
    
    def create_plan(self,
                   participants: List[Dict[str, Any]],
                   group_size: int,
                   out_plan_csv: Path) -> pd.DataFrame:
        """
        Create a tournament plan by assigning players to groups.
        
        Args:
            participants: List of participant descriptors
                         (from ChessChampionship._build_participants)
            group_size: Target players per group
            out_plan_csv: Path to save plan CSV
        
        Returns:
            DataFrame with plan (group_id, participant_id, etc.)
        """
        rng = random.Random(42)
        shuffled = participants.copy()
        rng.shuffle(shuffled)
        
        rows = []
        group_id = 1
        idx = 0
        
        while idx < len(shuffled):
            group_members = shuffled[idx : idx + group_size]
            for desc in group_members:
                rows.append({
                    "group_id": group_id,
                    "participant_id": desc["id"],
                    "participant_name": desc["name"],
                    "type": desc.get("type"),
                    "repo_path": desc.get("repo_path", ""),
                    "baseline_key": desc.get("baseline_key", "")
                })
            group_id += 1
            idx += group_size
        
        plan_df = pd.DataFrame(rows)
        plan_df.to_csv(out_plan_csv, index=False)
        
        num_groups = plan_df["group_id"].max()
        self.logger.info(f"Created plan: {num_groups} groups, ~{group_size} players/group → {out_plan_csv}")
        
        return plan_df
    
    def run_swiss_stage(self,
                       stage_name: str,
                       plan_csv: Path,
                       results_csv: Path,
                       n_rounds: int = 3,
                       games_per_pairing: int = 1,
                       max_half_moves: int = 200,
                       engine_break: float = 0.0) -> pd.DataFrame:
        """
        Run a Swiss-tournament stage.
        
        Args:
            stage_name: Name for logging (e.g., "Qualifiers")
            plan_csv: Path to tournament plan CSV
            results_csv: Path to save results CSV
            n_rounds: Number of Swiss rounds
            games_per_pairing: Games per matched pair
            max_half_moves: Max half-moves per game
            engine_break: Sleep time between games (for engine rate limiting)
        
        Returns:
            DataFrame with results (participant_name, points, fallbacks, etc.)
        """
        self.logger.info(f"\n╔════ {stage_name.upper()} (SWISS) ════╗")
        
        plan_df = pd.read_csv(plan_csv, dtype=str)
        plan_df["group_id"] = plan_df["group_id"].astype(int)
        
        group_ids = sorted(plan_df["group_id"].unique())
        all_results = []
        
        for group_id in group_ids:
            group_df = plan_df[plan_df["group_id"] == group_id]
            group_participants = []
            
            # Build participant descriptors for this group
            for _, row in group_df.iterrows():
                baseline_key = row.get("baseline_key", "")
                
                desc = {
                    "type": row.get("type"),
                    "id": row.get("participant_id"),
                    "name": row.get("participant_name"),
                    "repo_path": row.get("repo_path", ""),
                    "baseline_key": baseline_key
                }
                
                # Re-inject factory for baselines
                if row.get("type") == "baseline" and baseline_key in self.baseline_factories:
                    desc["factory"] = self.baseline_factories[baseline_key]["factory"]
                
                group_participants.append(desc)
            
            self.logger.info(f"\n{stage_name} GROUP {group_id}: {len(group_participants)} players, {n_rounds} rounds")
            
            try:
                # Run swiss tournament for this group (suppress_leaderboard=True)
                result = swiss_tournament(
                    participant_descs=group_participants,
                    instantiate_fn=instantiate_participant,
                    destroy_fn=destroy_instance,
                    n_rounds=n_rounds,
                    games_per_pairing=games_per_pairing,
                    max_half_moves=max_half_moves,
                    engine_break=engine_break,
                    suppress_leaderboard=True
                )
                
                # Print leaderboard for this group (AFTER all matches in group complete)
                self.logger.info("\n🏆 GROUP LEADERBOARD 🏆")
                for rank, name in enumerate(result["leaderboard"], start=1):
                    points = result["scores"][name]
                    buchholz = result["buchholz"][name]
                    byes = result["byes"][name]
                    fallbacks = result["fallbacks"][name]
                    self.logger.info(
                        f"{rank:>2}. {name:<20}  {points:>5.1f} pts "
                        f"| buchholz {buchholz:>5.1f} | byes {byes} | fallbacks {fallbacks}"
                    )
                
                # Convert results to standard format
                for rank, name in enumerate(result["leaderboard"], start=1):
                    # Find participant descriptor to get repo_path and baseline_key
                    part_desc = next((p for p in group_participants if p["name"] == name), None)
                    part_id = part_desc["id"] if part_desc else name
                    repo_path = part_desc.get("repo_path", "") if part_desc else ""
                    baseline_key = part_desc.get("baseline_key", "") if part_desc else ""
                    
                    all_results.append({
                        "group_id": group_id,
                        "rank": rank,
                        "participant_id": part_id,
                        "participant_name": name,
                        "repo_path": repo_path,
                        "baseline_key": baseline_key,
                        "points": result["scores"][name],
                        "fallbacks": result["fallbacks"][name],
                        "buchholz": result.get("buchholz", {}).get(name, 0.0),
                    })
            
            except Exception as e:
                self.logger.error(f"Error in {stage_name} group {group_id}: {e}")
                raise
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_csv, index=False)
        self.logger.info(f"\n✅ {stage_name} complete: {len(all_results)} results")
        
        return results_df
    
    def get_advancing(self, results_df: pd.DataFrame, top_k: int) -> List[Dict[str, Any]]:
        """
        Get top_k players per group from results.
        
        Args:
            results_df: Results DataFrame from run_swiss_stage
            top_k: Number of top players PER GROUP to return
        
        Returns:
            List of participant descriptors for advancing players
        """
        advancing = []
        
        # Get top_k from each group
        if "group_id" in results_df.columns:
            for group_id in sorted(results_df["group_id"].unique()):
                group_results = results_df[results_df["group_id"] == group_id]
                sorted_results = group_results.nlargest(top_k, "points")
                
                for _, row in sorted_results.iterrows():
                    participant_id = row.get("participant_id", row["participant_name"])
                    participant_name = row["participant_name"]
                    baseline_key = str(row.get("baseline_key", "")).strip()
                    repo_path = str(row.get("repo_path", "")).strip()
                    
                    # Determine type: baselines have participant_id starting with "baseline-"
                    is_baseline = str(participant_id).startswith("baseline-")
                    
                    desc = {
                        "type": "baseline" if is_baseline else "student",
                        "id": participant_id,
                        "name": participant_name,
                        "repo_path": repo_path if repo_path else "",
                        "baseline_key": baseline_key if baseline_key else ""
                    }
                    
                    # Re-inject factory for baselines
                    if is_baseline and baseline_key in self.baseline_factories:
                        desc["factory"] = self.baseline_factories[baseline_key]["factory"]
                    
                    advancing.append(desc)
        else:
            # Fallback: if no group_id, just get top_k globally
            sorted_results = results_df.nlargest(top_k, "points")
            for _, row in sorted_results.iterrows():
                participant_id = row.get("participant_id", row["participant_name"])
                participant_name = row["participant_name"]
                baseline_key = str(row.get("baseline_key", "")).strip()
                repo_path = str(row.get("repo_path", "")).strip()
                
                is_baseline = str(participant_id).startswith("baseline-")
                
                desc = {
                    "type": "baseline" if is_baseline else "student",
                    "id": participant_id,
                    "name": participant_name,
                    "repo_path": repo_path if repo_path else "",
                    "baseline_key": baseline_key if baseline_key else ""
                }
                
                if is_baseline and baseline_key in self.baseline_factories:
                    desc["factory"] = self.baseline_factories[baseline_key]["factory"]
                
                advancing.append(desc)
        
        return advancing
