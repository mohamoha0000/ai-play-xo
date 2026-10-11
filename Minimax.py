import numpy as np
from collections import deque

# =========================================================
# 1. Game Rules & Minimax Solver (Dataset Generator)
# =========================================================
def check_winner(board):
    """Checks win or draw status on board."""
    win_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for a, b, c in win_combinations:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if all(cell != 0 for cell in board):
        return 0  # Draw
    return None  # Ongoing

minimax_memo = {}

def minimax(board, current_player, depth=0):
    """
    Minimax recursive evaluation with depth-based discounting to favor quicker wins.
    Returns: (best_score, list_of_optimal_moves)
    """
    state_key = (tuple(board), current_player)
    if state_key in minimax_memo:
        return minimax_memo[state_key]

    winner = check_winner(board)
    if winner is not None:
        if winner == current_player:
            return 10 - depth, []
        elif winner == -current_player:
            return depth - 10, []
        else:
            return 0, []

    best_score = -float('inf')
    best_moves = []

    for move in range(9):
        if board[move] == 0:
            board[move] = current_player
            # Opponent evaluates from their perspective
            opp_score, _ = minimax(board, -current_player, depth + 1)
            my_score = -opp_score
            board[move] = 0  # Undo move

            if my_score > best_score:
                best_score = my_score
                best_moves = [move]
            elif my_score == best_score:
                best_moves.append(move)

    minimax_memo[state_key] = (best_score, best_moves)
    return best_score, best_moves

def generate_minimax_dataset():
    """
    Generates all reachable non-terminal board configurations using BFS
    and labels them with their optimal Minimax move distribution.
    """
    print("🧠 Generating complete Minimax dataset for all reachable board states...")
    dataset = {}
    queue = deque([([0] * 9, 1)])
    visited = set()

    while queue:
        board, current_player = queue.popleft()
        state_key = (tuple(board), current_player)
        if state_key in visited:
            continue
        visited.add(state_key)

        winner = check_winner(board)
        if winner is not None:
            continue  # Terminal states require no actions

        # Perspective transformation: active player is always +1, opponent is -1
        player_view = tuple(board[i] * current_player for i in range(9))

        # Solve optimal moves using Minimax
        _, optimal_moves = minimax(board, current_player, depth=0)

        # Build target probability distribution over optimal moves
        target = [0.0] * 9
        for m in optimal_moves:
            target[m] = 1.0 / len(optimal_moves)

        dataset[player_view] = target

        # Expand game tree
        for m in range(9):
            if board[m] == 0:
                next_board = list(board)
                next_board[m] = current_player
                queue.append((next_board, -current_player))

    X = np.array(list(dataset.keys()), dtype=np.float32)
    Y = np.array(list(dataset.values()), dtype=np.float32)
    print(f"✅ Dataset generated! Total unique perspective states: {X.shape[0]}")
    return X, Y

# =========================================================
# 2. Neural Network Architecture (MLP with Adam Optimizer)
# =========================================================
class SupervisedNeuralNet:
    def __init__(self, layers=[9, 128, 64, 9]):
        self.W1 = np.random.randn(layers[0], layers[1]) * np.sqrt(2.0 / layers[0])
        self.b1 = np.zeros((1, layers[1]), dtype=np.float32)
        
        self.W2 = np.random.randn(layers[1], layers[2]) * np.sqrt(2.0 / layers[1])
        self.b2 = np.zeros((1, layers[2]), dtype=np.float32)
        
        self.W3 = np.random.randn(layers[2], layers[3]) * np.sqrt(2.0 / layers[2])
        self.b3 = np.zeros((1, layers[3]), dtype=np.float32)

    def forward(self, X):
        """Standard feed-forward pass with ReLU and Softmax."""
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.maximum(0, self.Z2)
        
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        
        # Numerically stable Softmax
        exp_Z = np.exp(self.Z3 - np.max(self.Z3, axis=1, keepdims=True))
        self.probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.probs

    def train(self, X, Y, epochs=400, lr=0.005):
        """Train using Cross-Entropy Loss with the Adam Optimizer."""
        print(f"🚀 Training Neural Network on Minimax Dataset for {epochs} epochs...")
        
        # Adam optimizer parameters and momentum buffers
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        mW3, vW3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        mb3, vb3 = np.zeros_like(self.b3), np.zeros_like(self.b3)

        N = X.shape[0]

        for epoch in range(1, epochs + 1):
            probs = self.forward(X)
            
            # Cross-Entropy Loss
            loss = -np.sum(Y * np.log(probs + 1e-9)) / N

            # Backpropagation
            dZ3 = (probs - Y) / N
            dW3 = np.dot(self.A2.T, dZ3)
            db3 = np.sum(dZ3, axis=0, keepdims=True)

            dA2 = np.dot(dZ3, self.W3.T)
            dZ2 = dA2 * (self.Z2 > 0)
            dW2 = np.dot(self.A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * (self.Z1 > 0)
            dW1 = np.dot(X.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # Adam updates
            for param, grad, m, v in [
                (self.W1, dW1, mW1, vW1), (self.b1, db1, mb1, vb1),
                (self.W2, dW2, mW2, vW2), (self.b2, db2, mb2, vb2),
                (self.W3, dW3, mW3, vW3), (self.b3, db3, mb3, vb3)
            ]:
                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
                m_hat = m / (1 - beta1 ** epoch)
                v_hat = v / (1 - beta2 ** epoch)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if epoch % 50 == 0 or epoch == epochs:
                # Calculate accuracy on picking an optimal move
                chosen = np.argmax(probs, axis=1)
                optimal_hits = np.sum(Y[np.arange(N), chosen] > 0)
                accuracy = (optimal_hits / N) * 100
                print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss:.4f} | Optimal Move Match: {accuracy:.2f}%")

        print("🔥 Neural Network training complete!")

    def predict_best_move(self, board, player):
        """Predicts the optimal move for a given board and active player."""
        player_view = np.array(board, dtype=np.float32) * player
        probs = self.forward(player_view.reshape(1, 9)).flatten()
        
        # Mask occupied cells with a large negative number
        for i in range(9):
            if board[i] != 0:
                probs[i] = -1e9
        return int(np.argmax(probs))

# =========================================================
# 3. Interactive Play Loop
# =========================================================
def print_board(board):
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    print("\n")
    for i in range(9):
        symbol = symbols[board[i]] if board[i] != 0 else str(i)
        print(f" {symbol} ", end="")
        if (i + 1) % 3 == 0:
            print()
            if i < 6:
                print("---+---+---")
        else:
            print("|", end="")
    print("\n")

def play_game(nn_agent):
    while True:
        print("=" * 45)
        print("🎮 TIC-TAC-TOE: HUMAN VS NEURAL NET BOT")
        print("=" * 45)
        print("Choose your turn:")
        print("1: Play First (X)")
        print("2: Play Second (O)")
        
        choice = input("Enter 1 or 2: ").strip()
        human_player = 1 if choice == "1" else -1
        bot_player = -human_player
        
        board = [0] * 9
        current_turn = 1

        while True:
            print_board(board)
            winner = check_winner(board)
            if winner is not None:
                if winner == human_player:
                    print("🏆 Unbelievable! You won!")
                elif winner == bot_player:
                    print("💀 The Neural Network AI wins!")
                else:
                    print("🤝 It's a draw! Perfectly played.")
                break

            if current_turn == human_player:
                valid_moves = [i for i in range(9) if board[i] == 0]
                while True:
                    try:
                        move = int(input(f"Your turn ({'X' if human_player == 1 else 'O'}). Enter cell (0-8): "))
                        if move in valid_moves:
                            board[move] = human_player
                            break
                        else:
                            print("Invalid cell! It is either occupied or out of range (0-8).")
                    except ValueError:
                        print("Please enter a valid integer between 0 and 8.")
            else:
                print("🤖 Neural Network is evaluating the board...")
                best_move = nn_agent.predict_best_move(board, bot_player)
                board[best_move] = bot_player

            current_turn = -current_turn

        replay = input("\nWould you like to play again? (y/n): ").strip().lower()
        if replay != 'y':
            print("Thanks for playing! Goodbye.")
            break

if __name__ == "__main__":
    X_train, Y_train = generate_minimax_dataset()
    net = SupervisedNeuralNet()
    net.train(X_train, Y_train, epochs=350, lr=0.005)
    play_game(net)