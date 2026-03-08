from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
import chess
import torch
import torch.nn as nn

from chess_tournament.players import Player


class TransformerPlayer(Player):
    class ChessFormer(nn.Module):
        def __init__(self, vocab_size=13, embed_dim=256, n_heads=8, n_layers=4, dropout=0.1):
            super().__init__()
            
            # 1. Token Embeddings (13 piece types + padding/special if needed)
            self.piece_embedding = nn.Embedding(vocab_size, embed_dim)
            
            # 2. Positional Embeddings (Learnable, 64 squares)
            self.pos_embedding = nn.Parameter(torch.randn(1, 64, embed_dim))
            
            # 3. Class Token (CLS) to aggregate global board state
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            
            # 4. Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=n_heads, 
                dim_feedforward=embed_dim * 4, 
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True # Pre-Norm is often more stable
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            # 5. Prediction Head (Regression)
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh() # Force output to [-1, 1] to match our normalization
            )
            
        def forward(self, x):
            # x shape: (Batch, 64)
            B, T = x.shape
            
            # Embed Pieces
            x = self.piece_embedding(x) # (B, 64, Embed)
            
            # Add Positional Embedding
            x = x + self.pos_embedding
            
            # Append CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) # (B, 65, Embed)
            
            # Transform
            x = self.encoder(x)
            
            # Extract CLS state (index 0) for regression
            cls_out = x[:, 0, :]
            
            # Predict
            val = self.head(cls_out)
            
            # Return both value (for loss) and latent (for your experiments)
            return val, cls_out

    def __init__(self, name: str="mechanical_turk_V2", depth: int=3):
        self.name = name
        self.depth = depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model().to(self.device)

    def load_model(self):
        model = ChessFormer(vocab_size=13, embed_dim=256, n_heads=8, n_layers=4, dropout=0.1)
        model.load_state_dict(torch.load("./models/chessformer_ep3.pth", weights_only=True, map_location=torch.device(device)))
        model.eval()
        return model
    
    def parse_fens(self, fens):
        """
        Parses a list of FEN strings into a numpy array (N, 64) of int8.
        This runs in parallel processes.
        """
        PIECE_TO_INT = {
            '.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
        }

        results = []
        for fen in fens:
            board_str = fen.split(" ")[0]
            # Expand empty squares '3' -> '...'
            for i in range(1, 9):
                board_str = board_str.replace(str(i), '.' * i)
            board_str = board_str.replace('/', '')
            
            # Map to integers
            # Note: FEN starts at Rank 8 (top), standard mapping usually aligns 0=a1.
            # Here we just need a consistent 64-len sequence.
            ranks = [PIECE_TO_INT[c] for c in board_str]
            results.append(ranks)
        return torch.from_numpy(results).long().to(self.device)

    def get_status_evals(self, fens):
        """
        Given a list of FEN strings, returns their predicted evaluations.
        """
        # Parse FENs to (N, 64) int arrays
        board_arrays = self.parse_fens(fens)
        
        # Convert to torch tensor
        board_tensors = torch.from_numpy(board_arrays).long().to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            values, _ = self.model(board_tensors)
        
        return values.cpu().numpy().flatten()

    def minimax(self, board, depth, alpha, beta, maximizing, model):
        """Return evaluation of the position from the current player's perspective.

        - ``board`` is a ``chess.Board`` instance that will be modified in place
        (push/pop moves).
        - ``depth`` is the number of plies remaining.
        - ``alpha`` / ``beta`` are the pruning bounds.
        - ``maximizing`` is True if we are searching for the side to move;
        False if we are searching for the opponent (i.e. we just made a move).
        - ``model`` is the neural network used by ``get_status_eval``.
        """
        # terminal condition
        if depth == 0 or board.is_game_over():
            fen = board.fen()
            return self.get_status_eval(model, [fen])[0]

        if maximizing:
            value = -float("inf")
            for move in board.legal_moves:
                board.push(move)
                score = self.minimax(board, depth - 1, alpha, beta, False, model)
                board.pop()
                if score > value:
                    value = score
                if score > alpha:
                    alpha = score
                if beta <= alpha:
                    break
            return value
        else:
            value = float("inf")
            for move in board.legal_moves:
                board.push(move)
                score = self.minimax(board, depth - 1, alpha, beta, True, model)
                board.pop()
                if score < value:
                    value = score
                if score < beta:
                    beta = score
                if beta <= alpha:
                    break
            return value

    def find_best_move(self, board, depth, model):
        """Wrapper that returns the best move and its evaluation for ``board``.

        The evaluation is always from the side to move in ``board``.
        """
        best_move = None
        best_score = -float("inf")
        for move in board.legal_moves:
            board.push(move)
            score = self.minimax(board, depth - 1, -float("inf"), float("inf"), True, model)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score

    @abstractmethod
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        move, score = self.find_best_move(board, depth=self.depth, model=self.model)
        return move.uci() if move else None