import math
import random,time
import itertools

# 1. دوال التفعيل (Sigmoid للمخرج و ReLU للطبقات المخفية)
def sigmoid(x):
    x = max(-500, min(x, 500))
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    # 🔥 إضافة سحرية: نمنع المشتقة من الموت تماماً إذا وصلت النتيجة لـ 1 أو 0
    return (s * (1 - s)) + 0.001
# 1. دالة Leaky ReLU (تسمح بتسريب خطأ صغير جداً بدلاً من القتل التام)
def relu(x):
    return x if x > 0 else 0.01 * x

# 2. مشتقة Leaky ReLU (المشتقة لن تكون صفراً أبداً!)
def relu_derivative(x):
    return 1.0 if x > 0 else 0.01

class brain:
    def __init__(self, layers):
        self.layers = layers
        self.w = []
        for x in range(1, len(self.layers)):
            self.w.append([])
            for i in range(self.layers[x]):
                # غيرنا الأرقام إلى -0.5 و 0.5
                self.w[x-1].append([round(random.uniform(-0.5, 0.5), 2) for _ in range(self.layers[x-1])] + [round(random.uniform(-0.5, 0.5), 2)])
                
    def perdict(self, input_data):
        self.per = [input_data]
        self.pers = [input_data]

        # استخدمنا enumerate هنا للحصول على layer_index بشكل صحيح
        for layer_index, layer_weights in enumerate(self.w):
            self.per.append([])
            self.pers.append([])

            is_output_layer = (layer_index == len(self.w) - 1)
            
            for neuron_weights in layer_weights:
                N = len(neuron_weights) - 1
                # 🔥 فكرتك العبقرية: تم الحفاظ على القسمة على (len(neuron_weights)-1) 🔥
                net_input = (sum([self.per[-2][e] * neuron_weights[e] for e in range(N)]) + (neuron_weights[-1]))
                self.pers[-1].append(net_input)
                
                # تطبيق Sigmoid فقط على الطبقة الأخيرة، و ReLU على الباقي
                if is_output_layer:
                    self.per[-1].append(sigmoid(net_input))
                else:
                    self.per[-1].append(relu(net_input))
                    
        return self.per[-1]
    
    def train(self, input_data, target, learning_rate=0.1):
        output = self.perdict(input_data)
        deltas = [[] for _ in range(len(self.w))]
        
        # 1. حساب خطأ المخرجات (باستخدام مشتقة Sigmoid)
        for i in range(len(output)):
            error = output[i] - target[i]
            deltas[-1].append(error * sigmoid_derivative(self.pers[-1][i]))
        
        # 2. حساب خطأ الطبقات المخفية (باستخدام مشتقة ReLU)
        for i in range(len(self.w)-2, -1, -1):
            for x in range(len(self.w[i])):
                error_sum = sum(self.w[i+1][j][x] * deltas[i+1][j] for j in range(len(self.w[i+1])))
                deltas[i].append(error_sum * relu_derivative(self.pers[i+1][x]))

        # 3. تحديث الأوزان والانحياز
        for i in range(len(self.w)):
            for e in range(len(self.w[i])):
                for x in range(len(self.w[i][e])-1):
                    self.w[i][e][x] -= learning_rate * deltas[i][e] * self.per[i][x]
                
                # تصحيح مهم: استخدام -= بدلاً من = لتحديث الانحياز (Bias)
                self.w[i][e][-1] -= learning_rate * deltas[i][e]
        return output

def check_winner(player,board):
    combinations = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6)
    ]

    for a, b, c in combinations:
        if board[a] == board[b] == board[c] == player:
            return True

    return False



# إنشاء الشبكة 
mohamed = brain([9, 36, 18, 9])
youssef = brain([9, 36, 18, 9])


over=False
print("بدء التدريب...")
for epoch in range(1500):

    if (epoch+1) % 10 == 0:
        print("round:",epoch)
        for i in range(9):
            if board[i] == 0:
                print(" ", end="")
            elif board[i] == 1:
                print("X", end="")
            else:
                print("O", end="")
            if (i + 1) % 3 == 0:
                print()
            else:
                print("|", end="")
        time.sleep(10)


    board = [0 for _ in range(9)]
    while True:


        if all(x != 0 for x in board) or over:
            print("It's a draw!")
            print(board)
            print("round:",epoch)
            time.sleep(1)
            over=False
            break
        
        while True:
            output = mohamed.train(board,[1 if _ == 0 else 0 for _ in board])
            if not board[output.index(max(output))] == 0:
                break
            board[output.index(max(output))] = 1
            break

        if check_winner(1,board):
            print("Player mohamed wins!")
            over=True
            continue

        while True:
            output = youssef.train(board,[1 if _ == 0 else 0 for _ in board]) 
            if not board[output.index(max(output))] == 0:
                break
            board[output.index(max(output))] = -1
            break

        if check_winner(-1,board):
            print("Player youssef wins!")
            over=True
            continue
