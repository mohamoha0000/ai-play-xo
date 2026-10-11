import numpy as np

# =========================================================
# 1. Game Environment Helpers
# =========================================================
def check_winner(board):
    """
    Checks if there is a winner or draw on the board.
    Returns:
        1: Player 1 (X) wins
       -1: Player -1 (O) wins
        0: Draw
     None: Game is still ongoing
    """
    win_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
        (0, 4, 8), (2, 4, 6)             # Diagonals
    ]
    for a, b, c in win_combinations:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if all(cell != 0 for cell in board):
        return 0  # Draw
    return None  # Ongoing

def print_board(board):
    """Prints the 3x3 Tic-Tac-Toe board in a clean terminal layout."""
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

# =========================================================
# 2. Minimax Tabular Q-Learning Agent
# =========================================================
class TabularQAgent:
    def __init__(self, alpha=0.3, gamma=0.95):
        # Q-table: maps state tuple (perspective view) -> 9 Q-values
        self.q_table = {}
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor

    def get_q_values(self, state):
        """Retrieve Q-values for a state, initialize with zeros if unseen."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9, dtype=np.float32)
        return self.q_table[state]

    def choose_action(self, state, valid_moves, epsilon=0.0):
        """Epsilon-greedy action selection with random tie-breaking."""
        if np.random.rand() < epsilon:
            return np.random.choice(valid_moves)
        
        q_values = self.get_q_values(state)
        # Find maximum value among valid legal moves only
        max_q = max(q_values[m] for m in valid_moves)
        best_actions = [m for m in valid_moves if q_values[m] == max_q]
        return np.random.choice(best_actions)

    def train_self_play(self, total_episodes=120000):
        """
        Train the tabular agent via self-play using zero-sum Minimax updates.
        Each player's perspective is normalized: self = 1, opponent = -1.
        """
        print(f"🚀 Training Tabular Q-Agent over {total_episodes:,} self-play episodes...")
        
        for episode in range(total_episodes):
            # Linear decay of exploration rate epsilon
            epsilon = max(0.05, 0.85 * (1.0 - (episode / total_episodes)))
            board = [0] * 9
            current_player = 1

            while True:
                # View board from perspective of current player
                state = tuple(board[i] * current_player for i in range(9))
                valid_moves = [i for i in range(9) if board[i] == 0]
                
                # Pick action
                action = self.choose_action(state, valid_moves, epsilon)
                
                # Apply action
                board[action] = current_player
                winner = check_winner(board)
                
                # Terminal state updates
                if winner == current_player:
                    # Immediate win: Reward = +1.0
                    self.get_q_values(state)[action] += self.alpha * (1.0 - self.get_q_values(state)[action])
                    break
                elif winner == 0:
                    # Draw: Reward = 0.0
                    self.get_q_values(state)[action] += self.alpha * (0.0 - self.get_q_values(state)[action])
                    break
                else:
                    # Game continues: update using opponent's best response (Minimax Bellman update)
                    opp_state = tuple(board[i] * (-current_player) for i in range(9))
                    opp_valid = [i for i in range(9) if board[i] == 0]
                    opp_q = self.get_q_values(opp_state)
                    best_opp_val = max(opp_q[m] for m in opp_valid)
                    
                    # Zero-sum property: opponent's gain is our loss
                    target = -self.gamma * best_opp_val
                    self.get_q_values(state)[action] += self.alpha * (target - self.get_q_values(state)[action])

                    # Alternate turn
                    current_player = -current_player

        print("🔥 Training completed successfully! Total discovered states:", len(self.q_table))

# =========================================================
# 3. Interactive Play Loop
# =========================================================
def play_game(agent):
    while True:
        print("=" * 45)
        print("🎮 TIC-TAC-TOE: HUMAN VS TABULAR Q-BOT")
        print("=" * 45)
        print("Choose your turn:")
        print("1: Play First (X)")
        print("2: Play Second (O)")
        
        choice = input("Enter 1 or 2: ").strip()
        human_player = 1 if choice == "1" else -1
        bot_player = -human_player
        
        board = [0] * 9
        current_turn = 1  # 1 (X) always moves first

        while True:
            print_board(board)
            winner = check_winner(board)
            if winner is not None:
                if winner == human_player:
                    print("🏆 Unbelievable! You won! (Should be mathematically impossible!)")
                elif winner == bot_player:
                    print("💀 The AI wins! Perfect defense and attack.")
                else:
                    print("🤝 It's a draw! Perfect game from both sides.")
                break

            if current_turn == human_player:
                # Human move
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
                # AI move
                print("🤖 AI is choosing its best move...")
                state = tuple(board[i] * bot_player for i in range(9))
                valid_moves = [i for i in range(9) if board[i] == 0]
                ai_action = agent.choose_action(state, valid_moves, epsilon=0.0)
                board[ai_action] = bot_player

            current_turn = -current_turn

        replay = input("\nWould you like to play again? (y/n): ").strip().lower()
        if replay != 'y':
            print("Thanks for playing! Goodbye.")
            break

if __name__ == "__main__":
    q_agent = TabularQAgent(alpha=0.3, gamma=0.95)
    q_agent.train_self_play(total_episodes=120000)
    play_game(q_agent)