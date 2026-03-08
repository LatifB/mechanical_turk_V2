from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
import chess
import random
import requests
import torch
import torch.nn as nn
import re
import time
import os

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

class Player(ABC):
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


class RandomPlayer(Player):
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None


class EnginePlayer(Player):
    """
    EnginePlayer now behaves like ANY Player:
    Input: FEN
    Output: move string (UCI) | "__NO_MOVES__" | None

    Internal failures are NOT visible to Game.
    """

    def __init__(
        self,
        name: str,
        blunder_rate: float = 0.0,
        ponder_rate: float = 0.0,
        base_delay: float = 0.9,
        enable_cache: bool = True,
    ):
        super().__init__(name)

        assert 0.0 <= blunder_rate <= 1.0
        assert 0.0 <= ponder_rate <= 1.0
        assert blunder_rate + ponder_rate <= 1.0

        self.blunder_rate = blunder_rate
        self.ponder_rate = ponder_rate
        self.base_delay = base_delay
        self.enable_cache = enable_cache

        self.api_key = os.environ.get("RAPIDAPI_KEY")
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY must be set")

        self.url = "https://chess-stockfish-16-api.p.rapidapi.com/chess/api"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "chess-stockfish-16-api.p.rapidapi.com",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        self.cache: Dict[str, Tuple[str, Optional[str]]] = {}

    def _sleep(self):
        time.sleep(self.base_delay)

    def _random_legal_from_fen(self, fen: str) -> Optional[str]:
        try:
            board = chess.Board(fen)
        except Exception:
            return None
        legal = list(board.legal_moves)
        if not legal:
            return None
        return random.choice(legal).uci()

    def _choose_move(self, best: str, ponder: Optional[str], fen: str) -> str:
        r = random.random()

        if r < self.blunder_rate:
            rm = self._random_legal_from_fen(fen)
            return rm if rm else best

        if r < self.blunder_rate + self.ponder_rate:
            return ponder if ponder else best

        return best

    def get_move(self, fen: str) -> Optional[str]:

        # CACHE
        if self.enable_cache and fen in self.cache:
            best, ponder = self.cache[fen]
            return self._choose_move(best, ponder, fen)

        self._sleep()

        try:
            r = requests.post(self.url, data={"fen": fen}, headers=self.headers, timeout=10)
            if r.status_code != 200:
                return None

            j = r.json()

        except Exception:
            return None

        # Engine says no moves
        result_field = j.get("result")
        if isinstance(result_field, str) and "bestmove (none)" in result_field.lower():

            rm = self._random_legal_from_fen(fen)
            if rm is None:
                return "__NO_MOVES__"

            return rm  # Game will treat as normal move

        best = j.get("bestmove")
        ponder = j.get("ponder")

        if not best:
            return None

        if self.enable_cache:
            self.cache[fen] = (best, ponder if ponder else None)

        return self._choose_move(best, ponder if ponder else None, fen)

class LMPlayer(Player):
    def __init__(
        self,
        name: str,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        quantization: Optional[str] = "4bit",
        temperature: float = 0.1,
        max_new_tokens: int = 6,
        retries: int = 5
    ):
        super().__init__(name)

        self.model_id = model_id
        self.quantization = quantization
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.retries = retries

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[{self.name}] Loading {self.model_id} on {self.device}")
        print(f"[{self.name}] Quantization mode: {self.quantization}")

        # -------------------------
        # Quantization config
        # -------------------------
        quant_config = None

        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        elif quantization is None:
            quant_config = None

        else:
            raise ValueError("quantization must be one of: None, '8bit', '4bit'")

        # -------------------------
        # Tokenizer
        # -------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # -------------------------
        # Config
        # -------------------------
        config = AutoConfig.from_pretrained(model_id)
        config.pad_token_id = self.tokenizer.pad_token_id

        # -------------------------
        # Model loading
        # -------------------------
        if quant_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                dtype=dtype,
                device_map="auto"
            )

        # -------------------------
        # UCI regex
        # -------------------------
        self.uci_re = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")

    def _build_prompt(self, fen: str) -> str:
        return f"""You are a chess engine.

Your task is to output the BEST LEGAL MOVE for the given chess position.

STRICT OUTPUT RULES:
- Output EXACTLY ONE move
- UCI format ONLY (examples: e2e4, g1f3, e7e8q)
- NO explanations
- NO punctuation
- NO extra text

Examples:

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Move: e2e4

FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
Move: f1b5

FEN: rnbqkb1r/pppp1ppp/5n2/4p3/1P6/5NP1/P1PPPP1P/RNBQKB1R b KQkq - 0 3
Move: e5e4

Now evaluate this position:

FEN: {fen}
Move:"""

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.uci_re.search(text)
        return match.group(0) if match else None

    def get_move(self, fen: str) -> Optional[str]:
        prompt = self._build_prompt(fen)

        for attempt in range(1, self.retries + 1):

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):]

            move = self._extract_move(decoded)

            if move:
                return move

        return None

class SmolPlayer(Player):
    """
    LLMAPIPlayer using InferenceClient.chat_completion()
    Compatible with chat/instruct models.
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str,
        model_id: str = 'moonshotai/Kimi-K2-Instruct',
        temperature: float = 0.2,
        max_tokens: int = 32,
    ):
        super().__init__(name)

        self.client = InferenceClient(
            model=model_id,
            token=os.environ.get("HF_TOKEN")
        )

        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_prompt(self, fen: str) -> str:
        return f"""You are a chess engine.

Your task is to output the BEST LEGAL MOVE for the given chess position.

STRICT OUTPUT RULES:
- Output EXACTLY ONE move
- UCI format ONLY (examples: e2e4, g1f3, e7e8q)
- NO explanations
- NO punctuation
- NO extra text

Examples:

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Move: e2e4

FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
Move: f1b5

FEN: rnbqkb1r/pppp1ppp/5n2/4p3/1P6/5NP1/P1PPPP1P/RNBQKB1R b KQkq - 0 3
Move: e5e4

Now evaluate this position:

FEN: {fen}
Move:"""

    def _extract_uci(self, text: str):
        if not text:
            return None

        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def get_move(self, fen: str):

        prompt = self._build_prompt(fen)

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            text = response.choices[0].message.content

            return self._extract_uci(text)

        except Exception as e:
            # Optional debug:
            print(f"[{self.name}] API error:", e)
            return None
          
