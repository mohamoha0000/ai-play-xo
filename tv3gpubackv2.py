import cupy as cp
import time
import os

# =========================================================
# 1. دوال التفعيل (بسيطة ومباشرة للمصفوفات)
# =========================================================
def relu(x):
    return cp.where(x > 0, x, 0.01 * x)

def relu_derivative(x):
    return cp.where(x > 0, 1.0, 0.01)


# =========================================================
# 2. هندسة الشبكة العصبية المعجلة بالـ GPU
# =========================================================
class brain:
    def __init__(self, layers):
        self.layers = layers
        self.W = []
        self.b = []
        # نسخة مستقرة من الأوزان لحساب أهداف الرجوع العكسي بدقة
        self.target_W = []
        self.target_b = []
        # متغيرات محسن Adam لتسريع التعلم ومنع ضياع التدرجات النادرة
        self.m_W, self.v_W = [], []
        self.m_b, self.v_b = [], []
        self.step_count = 0

        for i in range(1, len(self.layers)):
            scale = cp.sqrt(2.0 / self.layers[i-1])
            w = (cp.random.randn(self.layers[i], self.layers[i-1]) * scale).astype(cp.float32)
            b = cp.zeros((self.layers[i], 1), dtype=cp.float32)
            
            self.W.append(w)
            self.b.append(b)
            self.target_W.append(w.copy())
            self.target_b.append(b.copy())
            
            self.m_W.append(cp.zeros_like(w))
            self.v_W.append(cp.zeros_like(w))
            self.m_b.append(cp.zeros_like(b))
            self.v_b.append(cp.zeros_like(b))

    def encode_board(self, boards):
        # تحويل اللوحة (9 خانات) إلى 3 قنوات واضحة (27 خانة: أحجاري، أحجار الخصم، الفراغات)
        # هذا يمنع ضرب المربعات الفارغة في صفر ويجعل الشبكة ترى الفراغات والتهديدات بوضوح
        return cp.vstack([boards == 1, boards == -1, boards == 0]).astype(cp.float32)

    def perdict(self, boards, use_target_net=False):
        x = self.encode_board(boards)
        self.per = [x]
        self.pers = [x]
        
        weights = self.target_W if use_target_net else self.W
        biases = self.target_b if use_target_net else self.b

        for i in range(len(weights)):
            Z = cp.dot(weights[i], self.per[-1]) + biases[i]
            self.pers.append(Z)
            
            # الطبقة الأخيرة خطية لتخرج تقييم الحركة مباشرة بين (-1 خسارة) و (+1 فوز)
            if i < len(weights) - 1:
                A = relu(Z)
            else:
                A = Z
            self.per.append(A)
            
        return self.per[-1]

    def train(self, boards, actions, targets, learning_rate=0.005):
        output = self.perdict(boards, use_target_net=False)
        batch_size = boards.shape[1]
        cols = cp.arange(batch_size)

        # نحسب الخطأ فقط للمربع الذي تم اختياره، ونترك باقي الـ 8 مربعات بـ 0 دون تشويه
        deltas_last = cp.zeros_like(output)
        deltas_last[actions, cols] = output[actions, cols] - targets

        deltas = [None] * len(self.W)
        deltas[-1] = deltas_last

        # الانتشار الخلفي للخطأ عبر الطبقات
        for i in range(len(self.W) - 2, -1, -1):
            error_sum = cp.dot(self.W[i+1].T, deltas[i+1])
            deltas[i] = error_sum * relu_derivative(self.pers[i+1])

        # تحديث الأوزان باستخدام Adam المبسط
        self.step_count += 1
        for i in range(len(self.W)):
            grad_W = cp.dot(deltas[i], self.per[i].T) / batch_size
            grad_b = cp.sum(deltas[i], axis=1, keepdims=True) / batch_size

            self.m_W[i] = 0.9 * self.m_W[i] + 0.1 * grad_W
            self.v_W[i] = 0.999 * self.v_W[i] + 0.001 * (grad_W ** 2)
            m_hat_W = self.m_W[i] / (1.0 - 0.9 ** self.step_count)
            v_hat_W = self.v_W[i] / (1.0 - 0.999 ** self.step_count)
            self.W[i] -= learning_rate * m_hat_W / (cp.sqrt(v_hat_W) + 1e-8)

            self.m_b[i] = 0.9 * self.m_b[i] + 0.1 * grad_b
            self.v_b[i] = 0.999 * self.v_b[i] + 0.001 * (grad_b ** 2)
            m_hat_b = self.m_b[i] / (1.0 - 0.9 ** self.step_count)
            v_hat_b = self.v_b[i] / (1.0 - 0.999 ** self.step_count)
            self.b[i] -= learning_rate * m_hat_b / (cp.sqrt(v_hat_b) + 1e-8)

    def sync_target_network(self, tau=0.1):
        # تحديث تدريجي لنسخة الأوزان المرجعية لضمان استقرار التدريب
        for i in range(len(self.W)):
            self.target_W[i] = (1.0 - tau) * self.target_W[i] + tau * self.W[i]
            self.target_b[i] = (1.0 - tau) * self.target_b[i] + tau * self.b[i]


# =========================================================
# 3. قوانين اللعبة وفحص الفوز المتوازي
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
# 4. حلقة اللعب الذاتي والتعلم العكسي (Self-Play & Backward RL)
# =========================================================
# المدخلات 27 (3 قنوات × 9 مربعات)، طبقتان مخفيتان، و 9 مخرجات
agent = brain([27, 128, 128, 9])

BATCH_SIZE = 15000   # 15 ألف مباراة في نفس اللحظة
EPOCHS = 250         # 250 دورة تدريبية شاملة
GAMMA = 0.95         # معامل تمرير المكافأة للوراء

print(f"🚀 بدء التدريب الذاتي لـ {BATCH_SIZE * EPOCHS:,} جولة عبر كارت الشاشة (GPU)...")
start_time = time.time()

# توزيع نسب استكشاف مختلفة لكل مباراة داخل الدفعة ليرى النموذج كل الحالات الممكنة دائماً
base_epsilons = cp.linspace(0.02, 0.80, BATCH_SIZE)

for epoch in range(EPOCHS):
    boards = cp.zeros((9, BATCH_SIZE), dtype=cp.float32)
    active_games = cp.ones(BATCH_SIZE, dtype=bool)
    cols = cp.arange(BATCH_SIZE)

    # سجل تاريخ الجولة الكامل داخل VRAM
    history_boards = cp.zeros((9, 9, BATCH_SIZE), dtype=cp.float32)
    history_next_boards = cp.zeros((9, 9, BATCH_SIZE), dtype=cp.float32)
    history_actions = cp.full((9, BATCH_SIZE), -1, dtype=cp.int32)
    
    # سجلات النتيجة المباشرة لكل خطوة
    history_won = cp.zeros((9, BATCH_SIZE), dtype=bool)
    history_drew = cp.zeros((9, BATCH_SIZE), dtype=bool)
    history_blunder = cp.zeros((9, BATCH_SIZE), dtype=bool) # حركة تركت الخصم يفوز فوراً

    # تخفيض تدريجي للاستكشاف مع الحفاظ على تنوع المباريات
    exploration_scale = max(0.25, 1.0 - (epoch / (EPOCHS * 0.85)))
    current_epsilons = base_epsilons * exploration_scale

    turn = 0
    # --- المرحلة الأولى: لعب المباريات للنهاية وتسجيل التاريخ ---
    while cp.any(active_games) and turn < 9:
        for player in [1, -1]:
            if not cp.any(active_games) or turn >= 9:
                break

            # اللوحة من منظور اللاعب الحالي (أحجاري = 1، الخصم = -1)
            player_view = boards * player

            # 1. توقع أفضل حركة حسب فهم الشبكة
            out = agent.perdict(player_view)
            masked_out = cp.where(boards != 0, -1e9, out)
            greedy_moves = cp.argmax(masked_out, axis=0)

            # 2. حركة استكشافية عشوائية في الأماكن الفارغة فقط
            rand_scores = cp.random.uniform(0.0, 1.0, (9, BATCH_SIZE))
            masked_rand = cp.where(boards != 0, -1e9, rand_scores)
            random_moves = cp.argmax(masked_rand, axis=0)

            # اختيار الحركة (مزيج بين الذكاء والاستكشاف)
            explore_mask = cp.random.rand(BATCH_SIZE) < current_epsilons
            best_moves = cp.where(explore_mask, random_moves, greedy_moves)

            # 3. حفظ المشهد الحالي والحركة قبل التنفيذ
            turn_board_slice = history_boards[turn]
            turn_board_slice[:, active_games] = player_view[:, active_games]
            history_actions[turn, active_games] = best_moves[active_games]

            # 4. تنفيذ الحركة على اللوحة
            boards[best_moves[active_games], cols[active_games]] = player

            # حفظ شكل اللوحة بعد الحركة (من منظور الخصم الذي سيلعب الدور القادم)
            opp_view_next = boards * (-player)
            turn_next_slice = history_next_boards[turn]
            turn_next_slice[:, active_games] = opp_view_next[:, active_games]

            # 5. فحص الفوز والتعادل
            wins = batched_check_winner(player, boards) & active_games
            draws = cp.all(boards != 0, axis=0) & active_games & ~wins

            history_won[turn, wins] = True
            history_drew[turn, draws] = True

            # 🔥 إذا فاز اللاعب في هذا الدور (turn)، فإن حركة خصمه في الدور السابق (turn - 1)
            # كانت خطأً دفاعياً قاتلاً لأنها تركت الفوز مفتوحاً!
            if turn > 0:
                history_blunder[turn - 1, wins] = True

            active_games &= ~(wins | draws)
            turn += 1

    # --- المرحلة الثانية: الرجوع للوراء من نهاية اللعبة للبداية وحساب المكافآت ---
    all_inputs_list = []
    all_actions_list = []
    all_targets_list = []

    for t in range(turn - 1, -1, -1):
        valid_mask = (history_actions[t] != -1)
        # تقليل تكرار اللوحة الفارغة تماماً في الدور 0 حتى لا تطغى على التكتيكات الدفاعية
        if t == 0:
            valid_mask &= (cols < 2500)

        if not cp.any(valid_mask):
            continue

        sub_inputs = history_boards[t][:, valid_mask]
        sub_next_boards = history_next_boards[t][:, valid_mask]
        sub_actions = history_actions[t, valid_mask]
        sub_won = history_won[t, valid_mask]
        sub_drew = history_drew[t, valid_mask]
        sub_blunder = history_blunder[t, valid_mask]

        # حساب أفضل ما يمكن للخصم فعله في الخطوة التالية
        opp_q = agent.perdict(sub_next_boards, use_target_net=True)
        opp_q_masked = cp.where(sub_next_boards != 0, -1e9, opp_q)
        opp_best_value = cp.clip(cp.max(opp_q_masked, axis=0), -1.0, 1.0)

        # القاعدة الرياضية البسيطة:
        # 1. إذا لم تنتهِ اللعبة، قيمة حركتي هي عكس أفضل قيمة يملكها الخصم في الدور القادم
        targets = -GAMMA * opp_best_value

        # 2. إذا كانت الحركة خطأً ترك الخصم يفوز في الدور التالي مباشرة -> عقاب صارم (-1.0)
        targets = cp.where(sub_blunder, -1.0, targets)

        # 3. إذا أدت الحركة للتعادل المباشر -> (0.0)
        targets = cp.where(sub_drew, 0.0, targets)

        # 4. إذا حققت الحركة الفوز المباشر -> مكافأة قصوى (+1.0)
        targets = cp.where(sub_won, 1.0, targets)

        all_inputs_list.append(sub_inputs)
        all_actions_list.append(sub_actions)
        all_targets_list.append(targets)

    # --- المرحلة الثالثة: تدريب الشبكة على جميع الأدوار معاً ---
    all_inputs = cp.hstack(all_inputs_list)
    all_actions = cp.concatenate(all_actions_list)
    all_targets = cp.concatenate(all_targets_list)

    # خلط البيانات وتدريبها على 4 دفعات لضمان توازن التعلم بين البداية والنهاية
    total_samples = all_inputs.shape[1]
    perm = cp.random.permutation(total_samples)
    mini_batch_size = total_samples // 4

    for b_idx in range(4):
        start_i = b_idx * mini_batch_size
        end_i = start_i + mini_batch_size if b_idx < 3 else total_samples
        batch_indices = perm[start_i:end_i]

        agent.train(
            all_inputs[:, batch_indices],
            all_actions[batch_indices],
            all_targets[batch_indices],
            learning_rate=0.005
        )

    # تحديث الشبكة المرجعية
    agent.sync_target_network(tau=0.15)

    if (epoch + 1) % 25 == 0 or epoch == EPOCHS - 1:
        print(f"✅ اكتمل Epoch {epoch + 1}/{EPOCHS} | تم تحديث الخبرة العكسية بنجاح.")

end_time = time.time()
print(f"🔥 انتهى التدريب بالكامل في {round(end_time - start_time, 2)} ثانية!")


# =========================================================
# 5. تجربة اللعب ضد النموذج (بقرار الشبكة العصبية 100%)
# =========================================================
while True:
    board = [0 for _ in range(9)]
    print("\n" + "="*45)
    print("🎮 مباراة جديدة ضد الذكاء الاصطناعي (هو X وأنت O)")
    print("="*45)
    print_board(board)

    while True:
        # ----- دور الذكاء الاصطناعي (X = 1) -----
        print("الذكاء الاصطناعي يفكر...")
        cp_board = cp.array(board, dtype=cp.float32).reshape(9, 1)
        output = agent.perdict(cp_board).flatten().get()
        
        valid_moves = [i for i in range(9) if board[i] == 0]
        if not valid_moves:
            print("اللعبة انتهت بالتعادل! 🤝")
            break
            
        # طباعة تقييم الشبكة لكل مربع فارغ لترى كيف يفكر عقله رياضياً
        eval_str = " | ".join([f"مربع {i}: {output[i]:+.2f}" for i in valid_moves])
        print(f"🧠 تقييم العقل للحركات: [{eval_str}]")

        best_move = max(valid_moves, key=lambda i: output[i])
        board[best_move] = 1
        print_board(board)

        if single_check_winner(1, board):
            print("للأسف.. لقد هزمك الذكاء الاصطناعي! 💀")
            break
            
        if all(x != 0 for x in board):
            print("اللعبة انتهت بالتعادل! 🤝")
            break

        # ----- دورك أنت (O = -1) -----
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
            break
            
        if all(x != 0 for x in board):
            print("اللعبة انتهت بالتعادل! 🤝")
            break

    again = input("\nهل تريد لعب جولة أخرى لتجربة خطة مختلفة؟ (y/n): ")
    if again.lower() != 'y':
        break