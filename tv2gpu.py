import cupy as cp
import time
import os

# =========================================================
# 1. تعريف دوال التفعيل لتتعامل مع المصفوفات العملاقة في CuPy
# =========================================================
def sigmoid(x):
    x = cp.clip(x, -500, 500)
    return 1 / (1 + cp.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return (s * (1 - s)) + 0.001

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
            if is_output_layer:
                A = sigmoid(Z)
            else:
                A = relu(Z)
            self.per.append(A)
            
        return self.per[-1]
    
    def train(self, input_data, target, learning_rate=0.5): # رفعنا معدل التعلم قليلاً لأننا نستخدم المتوسط
        output = self.perdict(input_data)
        deltas = [None] * len(self.W)
        
        error = self.per[-1] - target
        deltas[-1] = error * sigmoid_derivative(self.pers[-1])
        
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

# =========================================================
# 4. مرحلة التدريب السريع على كارت الشاشة (500,000 جولة)
# =========================================================
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    mohamed = brain([9, 36, 18, 9])
    youssef = brain([9, 36, 18, 9])

    BATCH_SIZE = 200000  # 50 ألف لعبة بالتوازي
    EPOCHS = 100         # 10 دورات × 50 ألف = 500,000 جولة
    
    print(f"🚀 بدء التدريب لـ {BATCH_SIZE * EPOCHS:,} جولة عبر كارت الشاشة (GPU)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        # إنشاء 50 ألف بوردة فارغة داخل VRAM دفعة واحدة: (9 صفوف, 50000 عمود)
        boards = cp.zeros((9, BATCH_SIZE))
        active_games = cp.ones(BATCH_SIZE, dtype=bool) # قناع لتتبع المباريات التي لم تنتهِ
        cols = cp.arange(BATCH_SIZE)
        
        while cp.any(active_games):
            # ----------------------------------------------------
            # دور محمد (X = 1) لجميع المباريات النشطة
            # ----------------------------------------------------
            target = cp.where(boards == 0, 1.0, 0.0) # الهدف هو اللعب في الأماكن الفارغة
            out_m = mohamed.train(boards, target)
            
            # منع محمد من اللعب في الأماكن الممتلئة (Masking) بإعطائها قيمة سالبة ضخمة
            out_m = cp.where(boards != 0, -1000.0, out_m)
            best_moves_m = cp.argmax(out_m, axis=0) # اختيار أعلى رقم لكل جولة
            
            # تطبيق الحركات للمباريات النشطة فقط
            boards[best_moves_m[active_games], cols[active_games]] = 1
            
            # تحديث حالة المباريات (الفوز أو التعادل)
            m_wins = batched_check_winner(1, boards)
            active_games &= ~m_wins # استبعاد المباريات التي فاز بها محمد
            
            draws = cp.all(boards != 0, axis=0)
            active_games &= ~draws # استبعاد المباريات المتعادلة
            
            if not cp.any(active_games): break

            # ----------------------------------------------------
            # دور يوسف (O = -1) لجميع المباريات النشطة
            # ----------------------------------------------------
            target = cp.where(boards == 0, 1.0, 0.0)
            out_y = youssef.train(boards, target)
            
            out_y = cp.where(boards != 0, -1000.0, out_y)
            best_moves_y = cp.argmax(out_y, axis=0)
            
            boards[best_moves_y[active_games], cols[active_games]] = -1
            
            y_wins = batched_check_winner(-1, boards)
            active_games &= ~y_wins
            
            draws = cp.all(boards != 0, axis=0)
            active_games &= ~draws

        print(f"✅ تم الانتهاء من حزمة {epoch + 1}/{EPOCHS} (إجمالي: {(epoch+1) * BATCH_SIZE:,} جولة)")

    end_time = time.time()
    print(f"🔥 انتهى تدريب نصف مليون جولة بنجاح في {round(end_time - start_time, 2)} ثانية فقط! 🔥")
    time.sleep(2)

    # =========================================================
    # 5. اللعب ضد الذكاء الاصطناعي
    # =========================================================
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*40)
    print("🤖 مرحباً بك! أنت الآن تلعب ضد 'محمد'")
    print("محمد يلعب بـ (X) وأنت تلعب بـ (O)")
    print("="*40)

    board = [0 for _ in range(9)]
    print_board(board)

    while True:
        # ----- دور محمد -----
        print("محمد يفكر...")
        
        # تحويل البورد العادي إلى مصفوفة CuPy وإرسالها للتوقع فقط
        cp_board = cp.array(board, dtype=float).reshape(9, 1)
        output = mohamed.perdict(cp_board).flatten().get() # .get() تعيد البيانات إلى المعالج (CPU)
        
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


#https://chatgpt.com/share/6ac1a005-6bc4-83ea-8a59-1b6fb2ec0273

#https://share.gemini.google/6OFW4ed7lEGl