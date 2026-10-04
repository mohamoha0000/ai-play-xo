import numpy as np
import random
import time


# =========================================================
# 1. Neural Network باستخدام NumPy فقط للحسابات
# =========================================================

class Brain:
    def __init__(self, layers, learning_rate=0.02):
        self.layers = layers
        self.learning_rate = learning_rate

        self.W = []
        self.b = []

        # Xavier/Glorot initialization
        for i in range(len(layers) - 1):
            fan_in = layers[i]
            fan_out = layers[i + 1]

            limit = np.sqrt(6 / (fan_in + fan_out))

            w = np.random.uniform(
                -limit,
                limit,
                (fan_in, fan_out)
            )

            b = np.zeros(fan_out)

            self.W.append(w.astype(np.float64))
            self.b.append(b.astype(np.float64))

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------
    def forward(self, x):

        a = np.asarray(x, dtype=np.float64)

        activations = [a]
        z_values = []

        for i in range(len(self.W)):

            z = a @ self.W[i] + self.b[i]

            z_values.append(z)

            # hidden layers
            if i < len(self.W) - 1:
                a = np.tanh(z)

            # output layer = linear
            else:
                a = z

            activations.append(a)

        return a, activations, z_values

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    def predict(self, x):
        output, _, _ = self.forward(x)
        return output

    # -----------------------------------------------------
    # Q-Learning update
    # -----------------------------------------------------
    def train_q(self, state, action, target):

        output, activations, z_values = self.forward(state)

        # فقط الحركة التي اخترناها يتم تحديثها
        delta = np.zeros_like(output)

        delta[action] = output[action] - target

        # Backpropagation
        for i in range(len(self.W) - 1, -1, -1):

            # gradient W
            grad_w = np.outer(
                activations[i],
                delta
            )

            grad_b = delta

            # Gradient clipping
            grad_w = np.clip(grad_w, -1.0, 1.0)
            grad_b = np.clip(grad_b, -1.0, 1.0)

            # Update
            self.W[i] -= self.learning_rate * grad_w
            self.b[i] -= self.learning_rate * grad_b

            # الذهاب للطبقة السابقة
            if i > 0:

                delta = (
                    delta @ self.W[i].T
                ) * (
                    1 - np.tanh(z_values[i - 1]) ** 2
                )

    # -----------------------------------------------------
    # اختيار الحركة
    # -----------------------------------------------------
    def choose_action(self, state, legal_moves, epsilon=0.0):

        # Exploration
        if random.random() < epsilon:
            return random.choice(legal_moves)

        q_values = self.predict(state)

        legal_values = q_values[legal_moves]

        max_value = np.max(legal_values)

        # إذا هناك أكثر من حركة لها نفس القيمة
        best_moves = [
            move
            for move, value in zip(legal_moves, legal_values)
            if np.isclose(value, max_value)
        ]

        return random.choice(best_moves)

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------
    def save(self, filename):

        data = {}

        for i, w in enumerate(self.W):
            data[f"W{i}"] = w

        for i, b in enumerate(self.b):
            data[f"b{i}"] = b

        np.savez(filename, **data)

        print("Model saved:", filename)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    def load(self, filename):

        data = np.load(filename)

        for i in range(len(self.W)):
            self.W[i] = data[f"W{i}"]

        for i in range(len(self.b)):
            self.b[i] = data[f"b{i}"]

        print("Model loaded:", filename)


# =========================================================
# 2. Tic Tac Toe
# =========================================================

WIN_COMBINATIONS = [

    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),

    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),

    (0, 4, 8),
    (2, 4, 6)
]


def check_winner(board):

    for a, b, c in WIN_COMBINATIONS:

        if board[a] != 0:

            if board[a] == board[b] == board[c]:
                return board[a]

    # draw
    if all(x != 0 for x in board):
        return 0

    # game continues
    return None


# =========================================================
# 3. تحويل Board إلى State
# =========================================================

def encode_state(board, player):

    state = []

    for cell in board:

        if cell == 0:
            state.append(0)

        elif cell == player:
            state.append(1)

        else:
            state.append(-1)

    return np.array(state, dtype=np.float64)


# =========================================================
# 4. طباعة اللوحة
# =========================================================

def print_board(board):

    print()

    for i in range(9):

        if board[i] == 1:
            char = "X"

        elif board[i] == -1:
            char = "O"

        else:
            char = str(i + 1)

        print(f" {char} ", end="")

        if (i + 1) % 3 != 0:
            print("|", end="")
        else:
            print()

            if i != 8:
                print("---+---+---")

    print()


# =========================================================
# 5. تدريب مباراة واحدة
# =========================================================

def train_game(
    brain,
    epsilon,
    gamma=0.95
):

    board = [0] * 9

    # X يبدأ
    player = 1

    while True:

        # الحالة الحالية
        state = encode_state(
            board,
            player
        )

        # الخانات القانونية
        legal_moves = [
            i for i, x in enumerate(board)
            if x == 0
        ]

        # اختيار الحركة
        action = brain.choose_action(
            state,
            legal_moves,
            epsilon
        )

        # وضع الحركة
        board[action] = player

        # هل انتهت المباراة؟
        result = check_winner(board)

        # -------------------------------------------------
        # اللاعب فاز
        # -------------------------------------------------

        if result is not None:

            if result == player:
                reward = 1.0

            else:
                # draw
                reward = 0.0

            target = reward

            brain.train_q(
                state,
                action,
                target
            )

            return result

        # -------------------------------------------------
        # المباراة مستمرة
        # -------------------------------------------------

        next_player = -player

        next_state = encode_state(
            board,
            next_player
        )

        next_legal_moves = [
            i for i, x in enumerate(board)
            if x == 0
        ]

        next_q = brain.predict(
            next_state
        )

        best_next_q = np.max(
            next_q[next_legal_moves]
        )

        # Zero-sum game
        target = -gamma * best_next_q

        # تحديث الشبكة
        brain.train_q(
            state,
            action,
            target
        )

        # اللاعب الآخر
        player = next_player


# =========================================================
# 6. اللعب ضد الإنسان
# =========================================================

def play_human(brain):

    board = [0] * 9

    print("\n")
    print("====================================")
    print("       HUMAN VS AI")
    print("====================================")

    while True:

        choice = input(
            "\nاختر X أو O: "
        ).strip().upper()

        if choice in ["X", "O"]:
            break

        print("اكتب X أو O فقط.")

    if choice == "X":
        human = 1
        bot = -1

    else:
        human = -1
        bot = 1

    # X دائما يبدأ
    player = 1

    while True:

        print_board(board)

        # =================================================
        # Human
        # =================================================

        if player == human:

            while True:

                try:
                    move = int(
                        input(
                            "اختر خانة من 1 إلى 9: "
                        )
                    ) - 1

                except ValueError:
                    print("أدخل رقما صحيحا.")
                    continue

                if move < 0 or move > 8:
                    print("اختر رقما من 1 إلى 9.")
                    continue

                if board[move] != 0:
                    print("هذه الخانة مشغولة.")
                    continue

                break

            board[move] = human

        # =================================================
        # Bot
        # =================================================

        else:

            state = encode_state(
                board,
                bot
            )

            legal_moves = [
                i for i, x in enumerate(board)
                if x == 0
            ]

            # مهم جدا:
            # epsilon = 0
            # لا يوجد exploration أثناء لعب الإنسان
            action = brain.choose_action(
                state,
                legal_moves,
                epsilon=0.0
            )

            board[action] = bot

            print(
                f"\nالبوت اختار الخانة: {action + 1}"
            )

        # =================================================
        # Check game
        # =================================================

        result = check_winner(board)

        if result is not None:

            print_board(board)

            if result == human:
                print("🎉 أنت فزت!")

            elif result == bot:
                print("🤖 البوت فاز!")

            else:
                print("🤝 تعادل!")

            break

        # next player
        player = -player

    print("\nالتدريب سيستأنف من حيث توقف.\n")

    input("اضغط Enter للعودة إلى التدريب...")


# =========================================================
# 7. البرنامج الرئيسي
# =========================================================

brain = Brain(
    [9, 64, 32, 9],
    learning_rate=0.02
)


TOTAL_GAMES = 30000

# نسبة الاستكشاف
START_EPSILON = 0.50
MIN_EPSILON = 0.05


print("====================================")
print("       TIC TAC TOE AI")
print("====================================")
print("بدء التدريب...\n")


for game in range(1, TOTAL_GAMES + 1):

    # ---------------------------------------------
    # Epsilon Decay
    # ---------------------------------------------

    progress = game / 2500

    epsilon = START_EPSILON * (1 - progress)

    epsilon = max(
        MIN_EPSILON,
        epsilon
    )

    # ---------------------------------------------
    # Training game
    # ---------------------------------------------

    result = train_game(
        brain,
        epsilon
    )

    # ---------------------------------------------
    # كل 100 مباراة
    # ---------------------------------------------

    if game % 5000 == 0:

        print("\n====================================")
        print(f"انتهت {game} مباراة تدريبية")
        print(f"Epsilon = {epsilon:.4f}")
        print("====================================")

        # سؤال المستخدم
        answer = input(
            "هل تريد اللعب ضد البوت الآن؟ (y/n): "
        ).strip().lower()

        if answer == "y":

            # =====================================
            # مهم:
            # هنا لا نستعمل train_game
            # لا يوجد أي تعديل على الشبكة
            # =====================================

            play_human(brain)

            print(
                f"\nاستئناف التدريب من المباراة {game + 1}..."
            )

        # حفظ النموذج
        brain.save(
            "tic_tac_toe_brain.npz"
        )


print("\n====================================")
print("انتهى التدريب بالكامل.")
print("====================================")

brain.save(
    "tic_tac_toe_brain_final.npz"
)