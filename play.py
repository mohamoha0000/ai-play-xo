import tkinter as tk
from tkinter import font, messagebox
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
            raise FileNotFoundError(f"ملف الأوزان '{model_path}' غير موجود.\nيرجى التأكد من وجود الملف في نفس مجلد اللعبة.")
            
        data = np.load(model_path)
        self.W = []
        self.b = []
        i = 0
        while f'W_{i}' in data:
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
# 2. الواجهة الرسومية الاحترافية (Modern Dark UI)
# =========================================================
class TicTacToeGUI:
    # لوحة الألوان العصرية (Modern Slate & Neon Palette)
    COLORS = {
        "bg_main": "#0F172A",       # خلفية رئيسية داكنة جداً
        "bg_card": "#1E293B",       # خلفية البطاقات واللوحة
        "tile_empty": "#334155",    # لون المربع الفارغ
        "tile_hover": "#475569",    # لون المربع عند تمرير الماوس
        "tile_win": "#065F46",      # خلفية مربعات الفوز (أخضر داكن)
        "tile_lose": "#7F1D1D",     # خلفية مربعات الخسارة (أحمر داكن)
        "text_primary": "#F8FAFC",  # نص أبيض ناصع
        "text_muted": "#94A3B8",    # نص فرعي رمادي
        "player_o": "#38BDF8",      # أزرق سماوي متوهج للاعب (O)
        "ai_x": "#F43F5E",          # وردي/أحمر متوهج للذكاء (X)
        "accent_green": "#10B981",  # أخضر للفوز
        "accent_amber": "#F59E0B",  # ذهبي للتعادل والتفكير
        "btn_reset": "#6366F1",     # بنفسجي/نيلي لزر الإعادة
        "btn_reset_hover": "#4F46E5"
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe AI 🧠")
        self.root.geometry("460x680")
        self.root.resizable(False, False)
        self.root.configure(bg=self.COLORS["bg_main"])
        
        self.ai = TicTacToeAI()
        self.board = np.zeros(9, dtype=np.int32)
        
        self.ai_player = 1      # X
        self.human_player = -1  # O
        self.is_ai_thinking = False
        self.game_over = False
        
        # عداد النتائج
        self.scores = {"human": 0, "ai": 0, "ties": 0}
        
        self.setup_fonts()
        self.create_widgets()
        self.reset_game()

    def setup_fonts(self):
        self.header_font = font.Font(family="Segoe UI", size=20, weight="bold")
        self.status_font = font.Font(family="Segoe UI", size=14, weight="bold")
        self.score_num_font = font.Font(family="Segoe UI", size=18, weight="bold")
        self.score_lbl_font = font.Font(family="Segoe UI", size=10, weight="bold")
        self.btn_font = font.Font(family="Segoe UI", size=34, weight="bold")
        self.ctrl_font = font.Font(family="Segoe UI", size=11, weight="bold")

    def create_widgets(self):
        # 1. عنوان اللعبة العلوي
        lbl_title = tk.Label(
            self.root, text="تحدي الذكاء الاصطناعي 🧠", 
            font=self.header_font, bg=self.COLORS["bg_main"], fg=self.COLORS["text_primary"]
        )
        lbl_title.pack(pady=(18, 8))

        # 2. لوحة النتائج (Scoreboard)
        self.frame_score = tk.Frame(self.root, bg=self.COLORS["bg_card"], padx=15, pady=10)
        self.frame_score.pack(fill="x", padx=35, pady=5)
        for col in range(3):
            self.frame_score.columnconfigure(col, weight=1)

        # نتيجة اللاعب
        tk.Label(self.frame_score, text="أنت (O)", font=self.score_lbl_font, bg=self.COLORS["bg_card"], fg=self.COLORS["player_o"]).grid(row=0, column=0)
        self.lbl_score_human = tk.Label(self.frame_score, text="0", font=self.score_num_font, bg=self.COLORS["bg_card"], fg=self.COLORS["text_primary"])
        self.lbl_score_human.grid(row=1, column=0)

        # التعادل
        tk.Label(self.frame_score, text="تعادل", font=self.score_lbl_font, bg=self.COLORS["bg_card"], fg=self.COLORS["text_muted"]).grid(row=0, column=1)
        self.lbl_score_ties = tk.Label(self.frame_score, text="0", font=self.score_num_font, bg=self.COLORS["bg_card"], fg=self.COLORS["text_primary"])
        self.lbl_score_ties.grid(row=1, column=1)

        # نتيجة الذكاء الاصطناعي
        tk.Label(self.frame_score, text="الذكاء (X)", font=self.score_lbl_font, bg=self.COLORS["bg_card"], fg=self.COLORS["ai_x"]).grid(row=0, column=2)
        self.lbl_score_ai = tk.Label(self.frame_score, text="0", font=self.score_num_font, bg=self.COLORS["bg_card"], fg=self.COLORS["text_primary"])
        self.lbl_score_ai.grid(row=1, column=2)

        # 3. شريط حالة اللعب
        self.lbl_status = tk.Label(
            self.root, text="دورك الآن (O)", 
            font=self.status_font, bg=self.COLORS["bg_main"], fg=self.COLORS["player_o"]
        )
        self.lbl_status.pack(pady=10)

        # 4. شبكة اللعب (3x3)
        self.frame_board = tk.Frame(self.root, bg=self.COLORS["bg_card"], padx=10, pady=10)
        self.frame_board.pack(pady=5)

        self.buttons = []
        for i in range(9):
            btn = tk.Button(
                self.frame_board, text="", font=self.btn_font, width=3, height=1,
                bg=self.COLORS["tile_empty"], activebackground=self.COLORS["tile_hover"],
                fg=self.COLORS["text_primary"], relief="flat", bd=0, cursor="hand2",
                command=lambda idx=i: self.human_move(idx)
            )
            btn.grid(row=i//3, column=i%3, padx=5, pady=5, ipadx=8, ipady=4)
            
            # تأثيرات تمرير الماوس (Hover Effects)
            btn.bind("<Enter>", lambda e, idx=i: self.on_tile_hover(idx, True))
            btn.bind("<Leave>", lambda e, idx=i: self.on_tile_hover(idx, False))
            self.buttons.append(btn)

        # 5. اختيار من يبدأ (أزرار تبديل عصرية بدلاً من الدوائر التقليدية)
        self.frame_controls = tk.Frame(self.root, bg=self.COLORS["bg_main"])
        self.frame_controls.pack(pady=12)

        self.start_var = tk.StringVar(value="Human")
        
        self.rb_human = tk.Radiobutton(
            self.frame_controls, text="👤 أنا أبدأ (O)", variable=self.start_var, value="Human",
            indicatoron=0, width=14, pady=6, font=self.ctrl_font, relief="flat", bd=0, cursor="hand2",
            bg=self.COLORS["bg_card"], fg=self.COLORS["text_muted"],
            selectcolor=self.COLORS["player_o"], activeforeground=self.COLORS["bg_main"],
            command=self.on_starter_change
        )
        self.rb_human.grid(row=0, column=0, padx=6)

        self.rb_ai = tk.Radiobutton(
            self.frame_controls, text="🤖 الذكاء يبدأ (X)", variable=self.start_var, value="AI",
            indicatoron=0, width=14, pady=6, font=self.ctrl_font, relief="flat", bd=0, cursor="hand2",
            bg=self.COLORS["bg_card"], fg=self.COLORS["text_muted"],
            selectcolor=self.COLORS["ai_x"], activeforeground=self.COLORS["text_primary"],
            command=self.on_starter_change
        )
        self.rb_ai.grid(row=0, column=1, padx=6)

        # 6. زر إعادة اللعب
        self.btn_reset = tk.Button(
            self.root, text="🔄 جولة جديدة", font=self.ctrl_font,
            bg=self.COLORS["btn_reset"], fg="white", activebackground=self.COLORS["btn_reset_hover"],
            activeforeground="white", relief="flat", bd=0, padx=25, pady=8, cursor="hand2",
            command=self.reset_game
        )
        self.btn_reset.pack(pady=5)

    def on_tile_hover(self, index, is_entering):
        """تغيير لون المربع عند تمرير الماوس إذا كان فارغاً واللعبة مستمرة"""
        if self.board[index] == 0 and not self.game_over and not self.is_ai_thinking:
            color = self.COLORS["tile_hover"] if is_entering else self.COLORS["tile_empty"]
            self.buttons[index].config(bg=color)

    def on_starter_change(self):
        """تحديث ألوان أزرار الاختيار وإعادة تشغيل الجولة"""
        if self.start_var.get() == "Human":
            self.rb_human.config(fg=self.COLORS["bg_main"])
            self.rb_ai.config(fg=self.COLORS["text_muted"])
        else:
            self.rb_human.config(fg=self.COLORS["text_muted"])
            self.rb_ai.config(fg=self.COLORS["text_primary"])
        self.reset_game()

    def update_board_ui(self):
        for i in range(9):
            if self.board[i] == self.ai_player:
                self.buttons[i].config(
                    text="X", fg=self.COLORS["ai_x"], 
                    disabledforeground=self.COLORS["ai_x"],
                    bg=self.COLORS["tile_empty"], state="disabled", cursor="arrow"
                )
            elif self.board[i] == self.human_player:
                self.buttons[i].config(
                    text="O", fg=self.COLORS["player_o"], 
                    disabledforeground=self.COLORS["player_o"],
                    bg=self.COLORS["tile_empty"], state="disabled", cursor="arrow"
                )
            else:
                self.buttons[i].config(
                    text="", bg=self.COLORS["tile_empty"], 
                    state="normal", cursor="hand2"
                )

    def get_winning_combo(self, player):
        """إرجاع المربعات الثلاثة الفائزة لتلوينها، أو None في حال عدم وجود فوز"""
        combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for combo in combinations:
            a, b, c = combo
            if self.board[a] == self.board[b] == self.board[c] == player:
                return combo
        return None

    def highlight_winning_line(self, combo, bg_color):
        for idx in combo:
            self.buttons[idx].config(bg=bg_color)

    def update_scoreboard(self):
        self.lbl_score_human.config(text=str(self.scores["human"]))
        self.lbl_score_ties.config(text=str(self.scores["ties"]))
        self.lbl_score_ai.config(text=str(self.scores["ai"]))

    def handle_game_over(self):
        ai_win_combo = self.get_winning_combo(self.ai_player)
        if ai_win_combo:
            self.game_over = True
            self.scores["ai"] += 1
            self.update_scoreboard()
            self.highlight_winning_line(ai_win_combo, self.COLORS["tile_lose"])
            self.lbl_status.config(text="الذكاء الاصطناعي فاز! 🤖", fg=self.COLORS["ai_x"])
            self.disable_all_buttons()
            return True

        human_win_combo = self.get_winning_combo(self.human_player)
        if human_win_combo:
            self.game_over = True
            self.scores["human"] += 1
            self.update_scoreboard()
            self.highlight_winning_line(human_win_combo, self.COLORS["tile_win"])
            self.lbl_status.config(text="رائع! لقد فزت باللقاء 🏆", fg=self.COLORS["accent_green"])
            self.disable_all_buttons()
            return True

        if not np.any(self.board == 0):
            self.game_over = True
            self.scores["ties"] += 1
            self.update_scoreboard()
            self.lbl_status.config(text="تعادل عادل! 🤝", fg=self.COLORS["accent_amber"])
            self.disable_all_buttons()
            return True

        return False

    def disable_all_buttons(self):
        for btn in self.buttons:
            btn.config(state="disabled", cursor="arrow")

    def human_move(self, index):
        # منع النقر إذا كان المربع مشغولاً أو الذكاء يفكر أو انتهت اللعبة
        if self.board[index] == 0 and not self.is_ai_thinking and not self.game_over:
            self.board[index] = self.human_player
            self.update_board_ui()
            
            if not self.handle_game_over():
                self.is_ai_thinking = True
                self.lbl_status.config(text="الذكاء الاصطناعي يفكر... ⏳", fg=self.COLORS["accent_amber"])
                self.root.after(350, self.ai_move)

    def ai_move(self):
        if self.game_over:
            return

        valid_moves = [i for i in range(9) if self.board[i] == 0]
        if not valid_moves:
            self.is_ai_thinking = False
            return

        predictions = self.ai.predict(self.board)
        best_move = max(valid_moves, key=lambda move: predictions[move])

        self.board[best_move] = self.ai_player
        self.update_board_ui()
        self.is_ai_thinking = False
        
        if not self.handle_game_over():
            self.lbl_status.config(text="دورك الآن (O)", fg=self.COLORS["player_o"])

    def reset_game(self):
        self.board = np.zeros(9, dtype=np.int32)
        self.game_over = False
        self.is_ai_thinking = False
        self.update_board_ui()
        
        # ضبط ألوان أزرار الاختيار عند البدء
        if self.start_var.get() == "Human":
            self.rb_human.config(fg=self.COLORS["bg_main"])
            self.rb_ai.config(fg=self.COLORS["text_muted"])
            self.lbl_status.config(text="دورك الآن (O)", fg=self.COLORS["player_o"])
        else:
            self.rb_human.config(fg=self.COLORS["text_muted"])
            self.rb_ai.config(fg=self.COLORS["text_primary"])
            self.is_ai_thinking = True
            self.lbl_status.config(text="الذكاء الاصطناعي يفكر... ⏳", fg=self.COLORS["accent_amber"])
            self.root.after(350, self.ai_move)


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = TicTacToeGUI(root)
        root.mainloop()
    except FileNotFoundError as e:
        root.withdraw()
        messagebox.showerror("خطأ في تحميل النموذج", str(e))