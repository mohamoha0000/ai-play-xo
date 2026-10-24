import tkinter as tk
from tkinter import font
import numpy as np
import os

class SelfPlayViewer:
    def __init__(self, root, replay_file='training_replays.npy'):
        self.root = root
        self.root.title("مختبر مراقبة تطور الذكاء الاصطناعي 🔬🧠")
        self.root.geometry("520x750")
        self.root.configure(bg="#1E272E")

        if not os.path.exists(replay_file):
            # إنشاء ملف وهمي للتجربة إذا لم تجد الملف الأصلي (لحماية الكود من التعطل)
            self.create_dummy_replays(replay_file)
            # raise FileNotFoundError(f"ملف الجولات '{replay_file}' غير موجود! شغّل التدريب أولاً.")

        # تحميل الجولات
        self.replays = np.load(replay_file)
        self.total_epochs = len(self.replays)
        
        # متغيرات الحالة
        self.current_epoch = 0
        self.current_step = 0
        self.board = np.zeros(9, dtype=int)
        self.is_playing = False
        self.after_id = None
        
        # متغيرات وضع "لقطات التطور"
        self.highlights_mode = False
        self.highlight_epochs = np.linspace(0, self.total_epochs - 1, min(10, self.total_epochs), dtype=int)
        self.highlight_index = 0

        # تحليل النتائج مسبقاً
        self.stats = {'X': 0, 'O': 0, 'Draw': 0}
        self.analyze_all_games()

        self.create_widgets()
        self.start_epoch_replay(0)

    def analyze_all_games(self):
        """تحليل جميع الجولات لمعرفة من فاز في كل جولة لإظهار الإحصائيات"""
        for moves in self.replays:
            board = np.zeros(9, dtype=int)
            winner_found = False
            for step, move in enumerate(moves):
                if move == -1 or step >= 9: break
                player = 1 if step % 2 == 0 else -1
                board[move] = player
                
                # فحص الفوز السريع
                combos = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
                for a, b, c in combos:
                    if board[a] == board[b] == board[c] == player:
                        if player == 1: self.stats['X'] += 1
                        else: self.stats['O'] += 1
                        winner_found = True
                        break
                if winner_found: break
            
            if not winner_found:
                self.stats['Draw'] += 1

    def create_widgets(self):
        title_font = font.Font(family="Helvetica", size=14, weight="bold")
        stat_font = font.Font(family="Helvetica", size=12, weight="bold")
        sub_font = font.Font(family="Helvetica", size=13, weight="bold")
        btn_font = font.Font(family="Helvetica", size=36, weight="bold")
        ctrl_font = font.Font(family="Helvetica", size=11, weight="bold")

        # --- لوحة الإحصائيات الشاملة ---
        frame_stats = tk.Frame(self.root, bg="#2C3A47", bd=2, relief="ridge")
        frame_stats.pack(pady=10, fill="x", padx=20)
        
        tk.Label(frame_stats, text="🏆 السجل التاريخي للذكاء الاصطناعي 🏆", font=title_font, bg="#2C3A47", fg="#F8EFBA").pack(pady=5)
        
        stats_inner = tk.Frame(frame_stats, bg="#2C3A47")
        stats_inner.pack(pady=5)
        
        tk.Label(stats_inner, text=f"فوز (O)\n{self.stats['O']}", font=stat_font, bg="#2C3A47", fg="#00A8FF", width=10).grid(row=0, column=0)
        tk.Label(stats_inner, text=f"التعادلات\n{self.stats['Draw']}", font=stat_font, bg="#2C3A47", fg="#E1B12C", width=10).grid(row=0, column=1)
        tk.Label(stats_inner, text=f"فوز (X)\n{self.stats['X']}", font=stat_font, bg="#2C3A47", fg="#E84118", width=10).grid(row=0, column=2)

        # --- معلومات الدورة الحالية ---
        self.lbl_ai_phase = tk.Label(self.root, text="", font=title_font, bg="#1E272E", fg="#00D8D6")
        self.lbl_ai_phase.pack(pady=(5, 2))

        self.lbl_epoch = tk.Label(self.root, text="", font=title_font, bg="#1E272E", fg="white")
        self.lbl_epoch.pack()

        self.lbl_status = tk.Label(self.root, text="مستعد...", font=sub_font, bg="#1E272E", fg="#F8EFBA")
        self.lbl_status.pack(pady=5)

        # --- رقعة اللعب ---
        frame_board = tk.Frame(self.root, bg="#485460", bd=4, relief="sunken")
        frame_board.pack(pady=5)

        self.buttons = []
        for i in range(9):
            btn = tk.Button(frame_board, text="", font=btn_font, width=4, height=1,
                            bg="#D2DAE2", disabledforeground="#1E272E", state="disabled")
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.buttons.append(btn)

        # --- أزرار التحكم ---
        frame_ctrl = tk.Frame(self.root, bg="#1E272E")
        frame_ctrl.pack(pady=10)

        self.btn_toggle = tk.Button(frame_ctrl, text="▶ تشغيل", font=ctrl_font, bg="#4CD137", fg="black",
                                     width=10, command=self.toggle_play)
        self.btn_toggle.grid(row=0, column=0, padx=5)

        tk.Button(frame_ctrl, text="🔄 إعادة", font=ctrl_font, bg="#7F8FA6", fg="white",
                  width=10, command=lambda: self.start_epoch_replay(self.current_epoch)).grid(row=0, column=1, padx=5)

        tk.Button(frame_ctrl, text="⏭ التالي", font=ctrl_font, bg="#3867D6", fg="white",
                  width=10, command=self.next_epoch).grid(row=0, column=2, padx=5)
                  
        # الزر الحماسي (لقطات التطور)
        self.btn_highlights = tk.Button(frame_ctrl, text="🌟 لقطات تطور الذكاء", font=ctrl_font, bg="#8C7AE6", fg="white",
                  width=18, command=self.toggle_highlights)
        self.btn_highlights.grid(row=0, column=3, padx=5)

        # --- شريط القفز ---
        self.epoch_slider = tk.Scale(self.root, from_=1, to=self.total_epochs, orient="horizontal",
                                     length=400, bg="#1E272E", fg="white", highlightthickness=0,
                                     command=self.on_slider_Move, label="تصفح الجولات يدوياً:")
        self.epoch_slider.pack(pady=10)

    def get_ai_phase(self, epoch):
        """يحدد مرحلة الذكاء بناءً على رقم الجولة لإضافة جو حماسي"""
        progress = epoch / self.total_epochs
        if progress < 0.3:
            return "مرحلة العشوائية والتخبط 🤪 (يبحث عن حلول)", "#FF7979"
        elif progress < 0.7:
            return "مرحلة التعلم وبناء الاستراتيجيات 🤔 (بدأ يفهم)", "#F6E58D"
        else:
            return "مرحلة الذكاء والاحتراف 🧠🤖 (مستوى متقدم)", "#BADC58"

    def check_winner(self, player):
        combos = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in combos:
            if self.board[a] == self.board[b] == self.board[c] == player:
                return (a, b, c)
        return None

    def start_epoch_replay(self, epoch_idx):
        if self.after_id:
            self.root.after_cancel(self.after_id)

        self.current_epoch = epoch_idx
        self.current_step = 0
        self.board = np.zeros(9, dtype=int)

        # تحديث الواجهة والنصوص
        self.epoch_slider.set(epoch_idx + 1)
        phase_text, phase_color = self.get_ai_phase(epoch_idx)
        self.lbl_ai_phase.config(text=phase_text, fg=phase_color)
        
        mode_text = "🌟 وضع اللقطات 🌟" if self.highlights_mode else f"الجولة: {epoch_idx + 1} / {self.total_epochs}"
        self.lbl_epoch.config(text=mode_text)
        
        self.lbl_status.config(text="يتخذ القرار...", fg="#F8EFBA")

        for btn in self.buttons:
            btn.config(text="", bg="#D2DAE2")

        if self.is_playing:
            self.after_id = self.root.after(500, self.play_next_step) # إبطاء قليل للمتعة

    def play_next_step(self):
        moves = self.replays[self.current_epoch]

        if self.current_step >= 9 or moves[self.current_step] == -1:
            self.finish_round()
            return

        move = moves[self.current_step]
        player = 1 if self.current_step % 2 == 0 else -1

        self.board[move] = player
        symbol = "X" if player == 1 else "O"
        color = "#E84118" if player == 1 else "#00A8FF"
        self.buttons[move].config(text=symbol, disabledforeground=color)

        self.current_step += 1

        win_combo = self.check_winner(player)
        if win_combo:
            for idx in win_combo:
                self.buttons[idx].config(bg="#4CD137") # لون أخضر فاقع للفوز
            self.lbl_status.config(text=f"🔥 ضربة قاضية! فوز ({symbol}) 🔥", fg="#4CD137")
            if self.is_playing:
                self.after_id = self.root.after(1500, self.auto_advance) # انتظار أطول بعد الفوز ليتذوق المستخدم اللحظة
            return

        if self.is_playing:
            self.after_id = self.root.after(400, self.play_next_step)

    def finish_round(self):
        self.lbl_status.config(text="عناد تكتيكي.. انتهت بالتعادل! 🤝", fg="#E1B12C")
        if self.is_playing:
            self.after_id = self.root.after(1200, self.auto_advance)

    def auto_advance(self):
        if self.highlights_mode:
            self.highlight_index += 1
            if self.highlight_index < len(self.highlight_epochs):
                self.start_epoch_replay(self.highlight_epochs[self.highlight_index])
            else:
                # انتهى عرض اللقطات
                self.toggle_highlights()
                self.lbl_status.config(text="انتهى عرض التطور 🎬", fg="#00D8D6")
        else:
            if self.current_epoch + 1 < self.total_epochs:
                self.start_epoch_replay(self.current_epoch + 1)
            else:
                self.is_playing = False
                self.btn_toggle.config(text="▶ تشغيل", bg="#4CD137")

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_toggle.config(text="⏸ إيقاف مؤقت", bg="#FFA801")
            self.play_next_step()
        else:
            self.btn_toggle.config(text="▶ استئناف", bg="#4CD137")
            if self.after_id:
                self.root.after_cancel(self.after_id)

    def toggle_highlights(self):
        """تفعيل/إلغاء وضع ملخص التطور الذكي"""
        self.highlights_mode = not self.highlights_mode
        if self.highlights_mode:
            self.btn_highlights.config(bg="#9C88FF", text="⏹ إيقاف اللقطات")
            self.highlight_index = 0
            self.is_playing = True
            self.btn_toggle.config(text="⏸ إيقاف مؤقت", bg="#FFA801")
            self.start_epoch_replay(self.highlight_epochs[0])
        else:
            self.btn_highlights.config(bg="#8C7AE6", text="🌟 لقطات تطور الذكاء")
            self.is_playing = False
            self.btn_toggle.config(text="▶ تشغيل", bg="#4CD137")

    def next_epoch(self):
        if self.highlights_mode:
            self.auto_advance()
        else:
            nxt = min(self.total_epochs - 1, self.current_epoch + 1)
            self.start_epoch_replay(nxt)

    def on_slider_Move(self, val):
        if self.highlights_mode:
            return # تعطيل الشريط في وضع اللقطات
        target_epoch = int(val) - 1
        if target_epoch != self.current_epoch:
            self.start_epoch_replay(target_epoch)
            
    def create_dummy_replays(self, filename):
        """دالة مساعدة لإنشاء ملف تدريب وهمي إذا لم يكن الملف موجوداً لتفادي الأخطاء"""
        dummy_data = []
        for _ in range(100): # 100 جولة عشوائية للتجربة
            moves = np.random.choice(9, 9, replace=False)
            dummy_data.append(moves)
        np.save(filename, np.array(dummy_data))

if __name__ == "__main__":
    root = tk.Tk()
    app = SelfPlayViewer(root)
    root.mainloop()