import math
import random
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

# إعداد البيانات (جميع الـ 32 حالة الممكنة)
# إعداد البيانات (جميع الـ 32 حالة الممكنة)
data = []
for combo in itertools.product([0,1], repeat=5):
    ones = sum(combo)
    if ones >= 3 and ones != 5:
        target = [1]
    else:
        target = [0]
    data.append((list(combo), target))

# 🔥 الحل السحري: تكرار الحالة الصعبة لكي لا تنساها الشبكة (Oversampling) 🔥
for _ in range(10): 
    data.append(([1, 1, 1, 1, 1], [0]))

# إنشاء الشبكة والتدريب...

# إنشاء الشبكة 
mohamed = brain([5, 8, 6, 1])

print("بدء التدريب...")
for epoch in range(1500):
    for inp, target in data:
        # يمكنك رفع learning_rate قليلاً هنا لأن فكرتك (القسمة) تصغر الأرقام
        mohamed.train(inp, target, learning_rate=0.5)

    if epoch % 300 == 0:
        print("Epoch", epoch, " | Test:", mohamed.perdict([1,1,1,0,0]))

# الاختبار النهائي
tests = [
    [0,1,1,1,0], # يجب أن تكون قريبة من 1
    [0,1,1,0,1], # يجب أن تكون قريبة من 1
    [1,0,1,1,0], # يجب أن تكون قريبة من 1
    [0,0,1,1,1], # يجب أن تكون قريبة من 1
    [1,1,1,1,0], # يجب أن تكون قريبة من 1
    [1,1,1,1,1]  # حالة خاصة: يجب أن تكون قريبة من 0
]

print("\n--- النتائج النهائية للاختبار ---")
for t in tests:
    print(t, "=>", mohamed.perdict(t))

