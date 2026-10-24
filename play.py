import tkinter as tk
from tkinter import font
import numpy as np
import os

# =========================================================
# 1. محرك الذكاء الاصطناعي (مُبسط للعمل على المعالج CPU)
# =========================================================
def relu(x):
    return np.where(x > 0, x, 0.01 * x)

class TicTacToeAI:
    def __init__(self, model_path='tictactoe_model.npz'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ملف الأوزان '{model_path}' غير موجود. قم بتدريب النموذج أولاً.")
            
        data = np.load(model_path)
        self.W = []
        self.b = []
        i = 0
        while f'W_{i}' in data:
            # تحويل البيانات إلى Numpy للعمل بكفاءة على الواجهة
            self.W.append(data[f'W_{i}'])
            self.b.append(data[f'b_{i}'])
            i += 1

    def encode_board(self, boards):
        return np.vstack([boards == 1, boards == -1, boards == 0]).astype(np.float32)

    def predict(self, board):
        x = self.encode_board(board.reshape(9, 1))
        A = x
        for i in range(len(self.W)):
            Z = np.dot(self.W[i], A) + self.b[i]
            A = relu(Z) if i < len(self.W) - 1 else Z
        return A.flatten()

# =========================================================
# 2. الواجهة الرسومية الممتعة (Tkinter)
# =========================================================
class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe AI 🧠")
        self.root.geometry("450x550")
        self.root.configure(bg="#2C3E50")
        
        self.ai = TicTacToeAI()
        self.board = np.zeros(9, dtype=np.int32)
        
        # 1 للذكاء الاصطناعي (X)، -1 للاعب (O)
        self.ai_player = 1
        self.human_player = -1
        
        self.create_widgets()
        self.reset_game()

    def create_widgets(self):
        # خطوط مخصصة
        self.title_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.btn_font = font.Font(family="Helvetica", size=36, weight="bold")
        self.ctrl_font = font.Font(family="Helvetica", size=12, weight="bold")

        # العنوان
        self.lbl_status = tk.Label(self.root, text="اختر من يبدأ اللعب!", font=self.title_font, bg="#2C3E50", fg="#ECF0F1")
        self.lbl_status.pack(pady=15)

        # لوحة الأزرار
        self.frame_board = tk.Frame(self.root, bg="#34495E")
        self.frame_board.pack(pady=10)

        self.buttons = []
        for i in range(9):
            btn = tk.Button(self.frame_board, text="", font=self.btn_font, width=4, height=1,
                            bg="#ECF0F1", activebackground="#BDC3C7", relief="raised",
                            command=lambda i=i: self.human_move(i))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.buttons.append(btn)

        # أدوات التحكم
        self.frame_controls = tk.Frame(self.root, bg="#2C3E50")
        self.frame_controls.pack(pady=15)

        self.start_var = tk.StringVar(value="Human")
        
        tk.Radiobutton(self.frame_controls, text="أنا أبدأ (O)", variable=self.start_var, value="Human", 
                       font=self.ctrl_font, bg="#2C3E50", fg="#ECF0F1", selectcolor="#34495E", 
                       command=self.reset_game).grid(row=0, column=0, padx=10)
                       
        tk.Radiobutton(self.frame_controls, text="الذكاء يبدأ (X)", variable=self.start_var, value="AI", 
                       font=self.ctrl_font, bg="#2C3E50", fg="#ECF0F1", selectcolor="#34495E", 
                       command=self.reset_game).grid(row=0, column=1, padx=10)

        self.btn_reset = tk.Button(self.root, text="🔄 إعادة اللعب", font=self.ctrl_font, bg="#E74C3C", fg="white", 
                                   activebackground="#C0392B", activeforeground="white", command=self.reset_game)
        self.btn_reset.pack(pady=5)

    def update_board_ui(self):
        for i in range(9):
            if self.board[i] == self.ai_player:
                self.buttons[i].config(text="X", fg="#E74C3C", state="disabled")
            elif self.board[i] == self.human_player:
                self.buttons[i].config(text="O", fg="#3498DB", state="disabled")
            else:
                self.buttons[i].config(text="", state="normal")

    def check_winner(self, player):
        combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for a, b, c in combinations:
            if self.board[a] == self.board[b] == self.board[c] == player:
                return True
        return False

    def handle_game_over(self):
        if self.check_winner(self.ai_player):
            self.lbl_status.config(text="الذكاء الاصطناعي فاز! 💀", fg="#E74C3C")
            self.disable_all_buttons()
            return True
        elif self.check_winner(self.human_player):
            self.lbl_status.config(text="لقد فزت! 🏆", fg="#2ECC71")
            self.disable_all_buttons()
            return True
        elif not np.any(self.board == 0):
            self.lbl_status.config(text="تعادل! 🤝", fg="#F1C40F")
            self.disable_all_buttons()
            return True
        return False

    def disable_all_buttons(self):
        for btn in self.buttons:
            btn.config(state="disabled")

    def human_move(self, index):
        if self.board[index] == 0:
            self.board[index] = self.human_player
            self.update_board_ui()
            
            if not self.handle_game_over():
                self.lbl_status.config(text="الذكاء الاصطناعي يفكر...", fg="#F39C12")
                self.root.update()
                self.root.after(400, self.ai_move) # تأخير بسيط لتبدو واقعية

    def ai_move(self):
        valid_moves = [i for i in range(9) if self.board[i] == 0]
        if not valid_moves:
            return

        # توقع الحركة من الشبكة العصبية
        predictions = self.ai.predict(self.board)
        
        # اختيار أفضل حركة متاحة
        best_move = -1
        best_score = -float('inf')
        
        for move in valid_moves:
            if predictions[move] > best_score:
                best_score = predictions[move]
                best_move = move

        self.board[best_move] = self.ai_player
        self.update_board_ui()
        
        if not self.handle_game_over():
            self.lbl_status.config(text="دورك الآن (O)", fg="#ECF0F1")

    def reset_game(self):
        self.board = np.zeros(9, dtype=np.int32)
        self.update_board_ui()
        
        if self.start_var.get() == "AI":
            self.lbl_status.config(text="الذكاء الاصطناعي يفكر...", fg="#F39C12")
            self.root.update()
            self.root.after(400, self.ai_move)
        else:
            self.lbl_status.config(text="دورك الآن (O)", fg="#ECF0F1")

if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = TicTacToeGUI(root)
        root.mainloop()
    except FileNotFoundError as e:
        import tkinter.messagebox as messagebox
        root.withdraw()
        messagebox.showerror("خطأ", str(e))