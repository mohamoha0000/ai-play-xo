import numpy as np
import time

# ---------------- دوال التفعيل والشبكة كما هي تماماً ----------------
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return (s * (1 - s)) + 0.001

def relu(x):
    return np.where(x > 0, x, 0.01 * x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.01)

class brain:
    def __init__(self, layers):
        self.layers = layers
        self.W = []
        self.b = []
        for i in range(1, len(self.layers)):
            weight_matrix = np.round(np.random.uniform(-0.5, 0.5, (self.layers[i], self.layers[i-1])), 2)
            bias_vector = np.round(np.random.uniform(-0.5, 0.5, (self.layers[i], 1)), 2)
            self.W.append(weight_matrix)
            self.b.append(bias_vector)
                
    def perdict(self, input_data):
        A = np.array(input_data, dtype=float).reshape(-1, 1)
        self.per = [A]
        self.pers = [A]
        for layer_index in range(len(self.W)):
            Z = np.dot(self.W[layer_index], self.per[-1]) + self.b[layer_index]
            self.pers.append(Z)
            is_output_layer = (layer_index == len(self.W) - 1)
            if is_output_layer:
                A = sigmoid(Z)
            else:
                A = relu(Z)
            self.per.append(A)
        return self.per[-1].flatten().tolist()
    
    def train(self, input_data, target, learning_rate=0.1):
        output = self.perdict(input_data)
        Y = np.array(target, dtype=float).reshape(-1, 1)
        deltas = [None] * len(self.W)
        
        error = self.per[-1] - Y
        deltas[-1] = error * sigmoid_derivative(self.pers[-1])
        
        for i in range(len(self.W)-2, -1, -1):
            error_sum = np.dot(self.W[i+1].T, deltas[i+1])
            deltas[i] = error_sum * relu_derivative(self.pers[i+1])

        for i in range(len(self.W)):
            self.W[i] -= learning_rate * np.dot(deltas[i], self.per[i].T)
            self.b[i] -= learning_rate * deltas[i]
            
        return output

def check_winner(player, board):
    combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for a, b, c in combinations:
        if board[a] == board[b] == board[c] == player:
            return True
    return False

# دالة لطباعة البورد بشكل أنيق أثناء اللعب
def print_board(board):
    print("\n")
    for i in range(9):
        if board[i] == 0:
            print(f" {i} ", end="") # طباعة رقم الخلية لسهولة اللعب
        elif board[i] == 1:
            print(" X ", end="")
        else:
            print(" O ", end="")
        
        if (i + 1) % 3 == 0:
            print()
        else:
            print("|", end="")
    print("\n")

# =========================================================
# 1. مرحلة التدريب السريع (Fast Training)
# =========================================================
mohamed = brain([9, 36, 18, 9])
youssef = brain([9, 36, 18, 9])

EPOCHS = 200000
over = False

print(f"بدء التدريب لـ {EPOCHS} جولة... (سيستغرق ثواني معدودة)")
start_time = time.time()

for epoch in range(EPOCHS):
    
    # اطبع فقط كل 1000 جولة لتوفير الوقت
    if (epoch + 1) % 1000 == 0:
        print(f"تم تدريب {epoch + 1} جولة...")

    board = [0 for _ in range(9)]
    
    while True:
        if all(x != 0 for x in board) or over:
            over = False
            break
        
        # --- دور محمد (X) ---
        output = mohamed.train(board, [1 if _ == 0 else 0 for _ in board])
        
        # التعديل الذكي: ابحث عن أفضل الأماكن *الفارغة فقط* لتجنب التعليق
        valid_moves = [i for i in range(9) if board[i] == 0]
        if not valid_moves: break
        best_move = max(valid_moves, key=lambda i: output[i])
        board[best_move] = 1

        if check_winner(1, board):
            over = True
            continue

        if all(x != 0 for x in board): break

        # --- دور يوسف (O) ---
        output = youssef.train(board, [1 if _ == 0 else 0 for _ in board]) 
        
        valid_moves = [i for i in range(9) if board[i] == 0]
        if not valid_moves: break
        best_move = max(valid_moves, key=lambda i: output[i])
        board[best_move] = -1

        if check_winner(-1, board):
            over = True
            continue

end_time = time.time()
print(f"انتهى التدريب بنجاح في {round(end_time - start_time, 2)} ثانية فقط! 🔥")

# =========================================================
# 2. مرحلة اللعب ضد الذكاء الاصطناعي (Human vs AI)
# =========================================================
print("="*40)
print("مرحباً بك! أنت الآن تلعب ضد 'محمد' (الذي تدرب للتو)")
print("محمد يلعب بـ (X) وأنت تلعب بـ (O)")
print("="*40)

board = [0 for _ in range(9)]
print_board(board)

while True:
    # دور محمد (الذكاء الاصطناعي يبدأ أولاً)
    print("محمد يفكر...")
    time.sleep(1) # تأخير بسيط للشعور بواقعية اللعب
    
    # نستخدم دالة perdict بدلاً من train لأنه يطبق ما تعلمه ولا يتدرب الآن
    output = mohamed.perdict(board) 
    
    valid_moves = [i for i in range(9) if board[i] == 0]
    if not valid_moves:
        print("اللعبة انتهت بالتعادل!")
        break
        
    best_move = max(valid_moves, key=lambda i: output[i])
    board[best_move] = 1
    print_board(board)

    if check_winner(1, board):
        print("للأسف.. لقد هزمك 'محمد'!")
        break
        
    if all(x != 0 for x in board):
        print("اللعبة انتهت بالتعادل!")
        break

    # دور اللاعب البشري
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

    if check_winner(-1, board):
        print_board(board)
        print("مستحيل! لقد تغلبت على الذكاء الاصطناعي! 🏆")
        break