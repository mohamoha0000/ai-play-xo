import cupy as cp
import time
import os

# =========================================================
# 1. تعريف دوال التفعيل لتتعامل مع المصفوفات العملاقة في CuPy
# =========================================================
def relu(x):
    return cp.where(x > 0, x, 0.01 * x)

def relu_derivative(x):
    return cp.where(x > 0, 1.0, 0.01)

def softmax(x):
    x = x - cp.max(x, axis=0, keepdims=True)  # للاستقرار العددي
    exp_x = cp.exp(x)
    return exp_x / cp.sum(exp_x, axis=0, keepdims=True)


# =========================================================
# 2. هندسة الشبكة العصبية المعجلة بالـ GPU
# =========================================================
class brain:
    def __init__(self, layers):
        self.layers = layers
        self.W = []
        self.b = []
        for i in range(1, len(self.layers)):
            # تخصيص الأوزان مباشرة داخل VRAM
            weight_matrix = cp.round(cp.random.uniform(-0.5, 0.5, (self.layers[i], self.layers[i-1])), 2)
            bias_vector = cp.round(cp.random.uniform(-0.5, 0.5, (self.layers[i], 1)), 2)
            self.W.append(weight_matrix)
            self.b.append(bias_vector)
                
    def perdict(self, input_data):
        self.per = [input_data]
        self.pers = [input_data]
        
        for layer_index in range(len(self.W)):
            # ضرب مصفوفة الأوزان في جميع البوردات دفعة واحدة (Batch Processing)
            Z = cp.dot(self.W[layer_index], self.per[-1]) + self.b[layer_index]
            self.pers.append(Z)
            
            is_output_layer = (layer_index == len(self.W) - 1)
            if not is_output_layer:
                A = relu(Z)
            else:
                A = Z  # لا نطبق أي دالة تفعيل على الطبقة الأخيرة قبل softmax
            self.per.append(A)
            
        return softmax(self.per[-1])
    
    def train(self, input_data, target, learning_rate=0.5): # رفعنا معدل التعلم قليلاً لأننا نستخدم المتوسط
        output = self.perdict(input_data)
        deltas = [None] * len(self.W)
        
        error = output - target
        deltas[-1] = error
        
        for i in range(len(self.W)-2, -1, -1):
            error_sum = cp.dot(self.W[i+1].T, deltas[i+1])
            deltas[i] = error_sum * relu_derivative(self.pers[i+1])

        # حساب متوسط الأخطاء على عدد البوردات (لمنع انفجار الأوزان)
        batch_size = input_data.shape[1] 

        for i in range(len(self.W)):
            self.W[i] -= (learning_rate / batch_size) * cp.dot(deltas[i], self.per[i].T)
            self.b[i] -= (learning_rate / batch_size) * cp.sum(deltas[i], axis=1, keepdims=True)
            
        return output

# =========================================================
# 3. قوانين اللعبة: فحص الفوز على آلاف الجولات في نفس اللحظة
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


agent = brain([9, 36, 18, 9])

BATCH_SIZE = 100000
EPOCHS = 50

print(f"🚀 بدء التدريب الذاتي (Self-Play) لـ {BATCH_SIZE * EPOCHS:,} جولة عبر كارت الشاشة (GPU)...")
start_time = time.time()

for epoch in range(EPOCHS):
    boards = cp.zeros((9, BATCH_SIZE))
    active_games = cp.ones(BATCH_SIZE, dtype=bool)
    cols = cp.arange(BATCH_SIZE)

        # 1. مصفوفة حفظ مشهد اللوحة في كل دور (Turn, Cell, Game)
    # أقصى عدد أدوار هو 9، وكل لوحة بها 9 خانات، وعدد المباريات BATCH_SIZE
    history_boards = cp.zeros((9, 9, BATCH_SIZE), dtype=cp.float32)

    # 2. مصفوفة حفظ رقم المربع الذي تم اختياره (من 0 إلى 8)
    history_actions = cp.zeros((9, BATCH_SIZE), dtype=cp.int32)

    # 3. مصفوفة لتحديد اللاعب صاحب الحركة (1 لـ X و -1 لـ O)
    history_players = cp.zeros((9, BATCH_SIZE), dtype=cp.int8)



    history_boards.fill(0)
    history_actions.fill(-1)
    history_players.fill(0)

    turn = 0
    while cp.any(active_games):
        for player in [1, -1]: # التناوب بين اللاعب الأول (1) والثاني (-1)
            # رؤية اللوحة من منظور اللاعب الحالي (أحجاري دائماً = 1، وأحجار الخصم = -1)
            player_view = boards * player
            
            target = cp.where(boards == 0, 1.0, 0.0)
            out = agent.perdict(player_view, target)
            
            # حجب الأماكن المشغولة
            out = cp.where(boards != 0, -1000.0, out)
            best_moves = cp.argmax(out, axis=0)


            history_boards[turn, :, active_games] = player_view[:, active_games]
            history_actions[turn, active_games] = best_moves[active_games]
            history_players[turn, active_games] = player

            
            # تطبيق الحركة للمباريات النشطة
            boards[best_moves[active_games], cols[active_games]] = player
            
            # فحص الفوز للاعب الحالي
            wins = batched_check_winner(player, boards)
            active_games &= ~wins
            
            # فحص التعادل
            draws = cp.all(boards != 0, axis=0)
            active_games &= ~draws
            
            # إذا انتهت كل المباريات نخرج من الحلقة
            if not cp.any(active_games):
                break

    print(f"✅ تم الانتهاء من حزمة {epoch + 1}/{EPOCHS} (إجمالي: {(epoch+1) * BATCH_SIZE:,} جولة)")

end_time = time.time()
print(f"🔥 انتهى تدريب النموذج الواحد بنجاح في {round(end_time - start_time, 2)} ثانية فقط! 🔥")






board = [0 for _ in range(9)]
print_board(board)

while True:
    # ----- دور محمد -----
    print("محمد يفكر...")
    
    # تحويل البورد العادي إلى مصفوفة CuPy وإرسالها للتوقع فقط
    cp_board = cp.array(board, dtype=float).reshape(9, 1)
    output = agent.perdict(cp_board).flatten().get() # .get() تعيد البيانات إلى المعالج (CPU)
    
    valid_moves = [i for i in range(9) if board[i] == 0]
    if not valid_moves:
        print("اللعبة انتهت بالتعادل!")
        break
        
    # إيجاد أعلى احتمال من بين الحركات الفارغة المتاحة فقط
    best_move = max(valid_moves, key=lambda i: output[i])
    board[best_move] = 1
    print_board(board)

    if single_check_winner(1, board):
        print("للأسف.. لقد هزمك 'محمد'! 💀")
        break
        
    if all(x != 0 for x in board):
        print("اللعبة انتهت بالتعادل! 🤝")
        break

    # ----- دور اللاعب -----
    while True:
        try:
            move = int(input("دورك! اختر رقم الخلية الفارغة (0-8): "))
            if move in valid_moves and board[move] == 0:
                board[move] = -1
                break
            else:
                print("الخلية ممتلئة أو الرقم غير صحيح! حاول مرة أخرى.")
        except ValueError:
            print("الرجاء إدخال رقم صحيح.")

    if single_check_winner(-1, board):
        print_board(board)
        print("مستحيل! لقد تغلبت على الذكاء الاصطناعي! 🏆")
        break









#https://chatgpt.com/share/6ac6c134-1c60-83ea-b08e-a46daac82823   expline codeç


#https://chatgpt.com/share/6ac6c201-4164-83ea-915e-8f8e4a674027    softmax whwn have milone output and we want one corect

"""
hi softmax when have milion output and wnate just one corect use soft max but what if we want 2 not neet just one or want from 100 oautput we want 10 output corect should be 1

expline use arabic plz

use sigmod but 

كلاكن من صعب شبكة اخراج كل شيء صفر و عشر مخرجات هي التي واحد 

اعطيني حلول افضل مع شرح بطريقة منضمة

https://chatgpt.com/share/6ac6c4da-ccd8-83e9-b773-cc4c845fc6fe





شرح بنية فكرتي كيف سيتدرب ونعاقب نمودج الخ
https://chatgpt.com/share/6ac6cce7-baac-83ea-a18f-1d2a44e50bc0

"""