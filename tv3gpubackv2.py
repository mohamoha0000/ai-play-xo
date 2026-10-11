import cupy as cp
import time
import os

# =========================================================
# 1. دوال التفعيل (ReLU للمخفية، و Linear للمخرجات لتقييم Q)
# =========================================================
def relu(x):
    return cp.where(x > 0, x, 0.01 * x)

def relu_derivative(x):
    return cp.where(x > 0, 1.0, 0.01)

# =========================================================
# 2. شبكة تقييم الأفعال (Q-Network) المعجلة بـ GPU
# =========================================================
class QBrain:
    def __init__(self, layers):
        self.layers = layers
        self.W = []
        self.b = []
        for i in range(1, len(self.layers)):
            scale = cp.sqrt(2.0 / (self.layers[i-1] + self.layers[i]))
            w = (cp.random.randn(self.layers[i], self.layers[i-1]) * scale).astype(cp.float32)
            b = cp.zeros((self.layers[i], 1), dtype=cp.float32)
            self.W.append(w)
            self.b.append(b)
                
    def predict(self, input_data):
        self.per = [input_data]
        self.pers = [input_data]
        for i in range(len(self.W)):
            Z = cp.dot(self.W[i], self.per[-1]) + self.b[i]
            self.pers.append(Z)
            # الطبقة الأخيرة خطية بدون Softmax لإخراج قيم Q الحقيقية (-1 إلى +1)
            A = Z if (i == len(self.W) - 1) else relu(Z)
            self.per.append(A)
        return self.per[-1]
    
    def train_step(self, states, actions, targets, learning_rate=0.01):
        # تدريب MSE على الحركات المنفذة فقط
        q_values = self.predict(states)
        batch_size = states.shape[1]
        
        # التدرج صفري لجميع الحركات التي لم تلعب، والخطأ فقط للحركة المختارة
        deltas_last = cp.zeros_like(q_values)
        cols = cp.arange(batch_size)
        deltas_last[actions, cols] = q_values[actions, cols] - targets
        
        deltas = [None] * len(self.W)
        deltas[-1] = deltas_last
        
        for i in range(len(self.W) - 2, -1, -1):
            error_sum = cp.dot(self.W[i+1].T, deltas[i+1])
            deltas[i] = error_sum * relu_derivative(self.pers[i+1])

        for i in range(len(self.W)):
            grad_w = cp.dot(deltas[i], self.per[i].T) / batch_size
            grad_b = cp.sum(deltas[i], axis=1, keepdims=True) / batch_size
            self.W[i] -= learning_rate * grad_w
            self.b[i] -= learning_rate * grad_b

# =========================================================
# 3. قوانين فحص الفوز المتوازية
# =========================================================
def batched_check_winner(player, boards):
    combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    wins = cp.zeros(boards.shape[1], dtype=bool)
    for a, b, c in combinations:
        wins |= (boards[a] == player) & (boards[b] == player) & (boards[c] == player)
    return wins

def single_check_winner(player, board):
    combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for a, b, c in combinations:
        if board[a] == board[b] == board[c] == player:
            return True
    return False

def print_board(board):
    print("\n")
    for i in range(9):
        if board[i] == 0: print(f" {i} ", end="")
        elif board[i] == 1: print(" X ", end="")
        else: print(" O ", end="")
        if (i + 1) % 3 == 0: print()
        else: print("|", end="")
    print("\n")

# =========================================================
# 4. محرك التدريب بـ Minimax Q-Learning
# =========================================================
agent = QBrain([9, 128, 64, 9])

BATCH_SIZE = 10000
EPOCHS = 2000
GAMMA = 0.95

print(f"🚀 بدء تدريب Minimax Q-Learning لـ {BATCH_SIZE * EPOCHS:,} جولة عبر كارت الشاشة...")
start_time = time.time()

for epoch in range(EPOCHS):
    # خفض تدريجي للاستكشاف العشوائي
    epsilon = max(0.05, 0.60 * (1.0 - (epoch / (EPOCHS * 0.85))))

    boards = cp.zeros((9, BATCH_SIZE), dtype=cp.float32)
    active_games = cp.ones(BATCH_SIZE, dtype=bool)
    cols = cp.arange(BATCH_SIZE)

    turn = 0
    while cp.any(active_games) and turn < 9:
        for player in [1, -1]:
            if not cp.any(active_games) or turn >= 9:
                break
                
            active_idx = cp.where(active_games)[0]
            num_active = len(active_idx)
            sub_boards = boards[:, active_games]
            
            # منظور اللاعب الحالي (أحجاري = 1، الخصم = -1)
            player_view = sub_boards * player
            
            # حساب قيم Q لجميع المربعات
            q_vals = agent.predict(player_view)
            
            # حجب المربعات المشغولة بقيمة سالبة ضخمة
            masked_q = cp.where(player_view != 0, -1e9, q_vals)
            greedy_actions = cp.argmax(masked_q, axis=0)

            # استكشاف عشوائي محدود للمربعات الفارغة لضمان رؤية كل الاحتمالات
            rand_scores = cp.random.uniform(0.0, 1.0, (9, num_active))
            masked_rand = cp.where(player_view != 0, -1e9, rand_scores)
            random_actions = cp.argmax(masked_rand, axis=0)

            explore_mask = cp.random.rand(num_active) < epsilon
            chosen_actions = cp.where(explore_mask, random_actions, greedy_actions)

            # تطبيق الحركة للحصول على اللوحة التالية
            next_sub_boards = sub_boards.copy()
            next_sub_boards[chosen_actions, cp.arange(num_active)] = player

            # فحص النتيجة المباشرة
            wins = batched_check_winner(player, next_sub_boards)
            draws = cp.all(next_sub_boards != 0, axis=0) & ~wins

            # حساب الهدف عبر معادلة بلمان الصفرية:
            # منظور الخصم للوحة القادمة هو (-next_sub_boards * player)
            opp_view = -next_sub_boards * player
            opp_q = agent.predict(opp_view)
            opp_masked_q = cp.where(opp_view != 0, -1e9, opp_q)
            opp_max_q = cp.max(opp_masked_q, axis=0)

            # معادلة الصفر: الربح = +1، التعادل = 0، الاستمرار = -GAMMA * أقصى ما يملكه الخصم
            targets = cp.zeros(num_active, dtype=cp.float32)
            targets = cp.where(wins, 1.0, targets)
            targets = cp.where(draws, 0.0, targets)
            targets = cp.where(~wins & ~draws, -GAMMA * opp_max_q, targets)

            # تحديث الأوزان فوراً على هذا القرار
            agent.train_step(player_view, chosen_actions, targets, learning_rate=0.03)

            # تطبيق الحركة على اللوحات العامة وتحديث القناع
            boards[:, active_idx] = next_sub_boards
            active_games[active_idx[wins | draws]] = False
            
            turn += 1

    if (epoch + 1) % 15 == 0 or epoch == EPOCHS - 1:
        print(f"✅ انتهى Epoch {epoch + 1}/{EPOCHS} | معدل الاستكشاف: {epsilon:.2f}")

end_time = time.time()
print(f"🔥 انتهى التدريب بنجاح تام في {round(end_time - start_time, 2)} ثانية!")




while True:
    board = [0 for _ in range(9)]
    print_board(board)

    while True:
        print("الذكاء الاصطناعي يفكر...")
        cp_board = cp.array(board, dtype=cp.float32).reshape(9, 1)
        output = agent.predict(cp_board).flatten().get()
        
        valid_moves = [i for i in range(9) if board[i] == 0]
        if not valid_moves:
            print("اللعبة انتهت بالتعادل! 🤝")
            break
            
        best_move = max(valid_moves, key=lambda i: output[i])
        board[best_move] = 1
        print_board(board)

        if single_check_winner(1, board):
            print("للأسف.. لقد هزمك الذكاء الاصطناعي! 💀")
            break
            
        if all(x != 0 for x in board):
            print("اللعبة انتهت بالتعادل! 🤝")
            break

        while True:
            try:
                valid_moves = [i for i in range(9) if board[i] == 0]
                move = int(input("دورك! اختر رقم الخلية الفارغة (0-8): "))
                if move in valid_moves:
                    board[move] = -1
                    break
                else:
                    print("الخلية ممتلئة أو الرقم غير صحيح! حاول مرة أخرى.")
            except ValueError:
                print("الرجاء إدخال رقم صحيح.")

        print_board(board)

        if single_check_winner(-1, board):
            print("مستحيل! لقد تغلبت على الذكاء الاصطناعي! 🏆")
