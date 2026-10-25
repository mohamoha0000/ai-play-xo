import tkinter as tk
from tkinter import font
import numpy as np
import os

class SelfPlayViewer:
    COMBOS = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]

    def __init__(self, root, replay_file='training_replays.npy'):
        self.root = root
        self.root.title("مختبر مراقبة تطور الذكاء الاصطناعي 🔬🧠")
        self.root.geometry("540x770")
        self.root.configure(bg="#1E272E")

        if not os.path.exists(replay_file):
            self.create_dummy_replays(replay_file)

        # تحميل الجولات
        self.replays = np.load(replay_file)
        self.total_epochs = len(self.replays)
        
        # متغيرات الحالة
        self.current_epoch = 0
        self.current_step = 0
        self.board = np.zeros(9, dtype=int)
        self.is_playing = False
        self.after_id = None
        
        # متغيرات وضع "لقطات التطور" (18 جولات مقسمة: 6 غبية، 6 شوكة مزدوجة، 6 تعادل قوي)
        self.highlights_mode = False
        self.highlight_epochs = self.prepare_18_curated_highlights()
        self.highlight_index = 0

        # تحليل النتائج مسبقاً
        self.stats = {'X': 0, 'O': 0, 'Draw': 0}
        self.analyze_all_games()

        self.create_widgets()
        self.start_epoch_replay(0)

    def get_winning_moves(self, board, player):
        """إرجاع جميع المربعات الفارغة التي تحقق الفوز الفوري للاعب المحدد"""
        winning_spots = set()
        for a, b, c in self.COMBOS:
            line = [board[a], board[b], board[c]]
            indices = [a, b, c]
            if line.count(player) == 2 and line.count(0) == 1:
                winning_spots.add(indices[line.index(0)])
        return winning_spots

    def evaluate_game_type(self, moves):
        """
        فحص الجولة وتصنيفها:
        - 'stupid': ترك فرصة فوز مباشر ولعب في مكان آخر
        - 'fork': صنع شوكة مزدوجة
        - 'strong_draw': تعادل كامل (9 حركات) بدون أي خطأ
        """
        board = np.zeros(9, dtype=int)
        missed_win = False
        missed_block = False
        created_fork_by = 0
        winner = 0
        steps_played = 0

        for step, move in enumerate(moves):
            if move == -1 or step >= 9:
                break
            player = 1 if step % 2 == 0 else -1
            
            my_wins_before = self.get_winning_moves(board, player)
            opp_wins_before = self.get_winning_moves(board, -player)

            # هل ترك اللاعب مكان الفوز المتاح له ولعب في مكان آخر؟
            if len(my_wins_before) > 0 and move not in my_wins_before:
                missed_win = True

            # هل تجاهل اللاعب صد فوز محقق للخصم؟
            if len(my_wins_before) == 0 and len(opp_wins_before) > 0 and move not in opp_wins_before:
                missed_block = True

            board[move] = player
            steps_played += 1

            # التحقق من الفوز
            for a, b, c in self.COMBOS:
                if board[a] == board[b] == board[c] == player:
                    winner = player
                    break
            if winner != 0:
                break

            # التحقق من الشوكة المزدوجة بعد لعب الحركة
            my_wins_after = self.get_winning_moves(board, player)
            opp_wins_after = self.get_winning_moves(board, -player)
            if len(my_wins_after) >= 2 and len(opp_wins_after) == 0:
                created_fork_by = player

        if missed_win:
            return 'stupid'
        if created_fork_by != 0 and winner == created_fork_by and not missed_block:
            return 'fork'
        if winner == 0 and steps_played == 9 and not missed_win and not missed_block:
            return 'strong_draw'
        return 'other'

    def prepare_18_curated_highlights(self):
        """البحث عن 6 جولات غبية، 6 شوكة مزدوجة، و6 تعادلات قوية (المجموع 18)"""
        stupid_list = []
        fork_list = []
        draw_list = []

        for idx, moves in enumerate(self.replays):
            g_type = self.evaluate_game_type(moves)
            if g_type == 'stupid':
                stupid_list.append(idx)
            elif g_type == 'fork':
                fork_list.append(idx)
            elif g_type == 'strong_draw':
                draw_list.append(idx)

        # 6 جولات احتياطية مضمونة لكل فئة
        fallback_stupid = [
            [0, 3, 1, 4, 8, 5, -1, -1, -1],
            [2, 0, 5, 4, 7, 8, -1, -1, -1],
            [0, 2, 3, 5, 4, 8, -1, -1, -1],
            [1, 4, 2, 5, 0, 6, -1, -1, -1], 
            [3, 0, 4, 1, 8, 2, -1, -1, -1], 
            [6, 3, 7, 4, 2, 5, -1, -1, -1]  
        ]
        fallback_fork = [
            [0, 1, 4, 8, 6, 3, 2, -1, -1],  
            [0, 3, 2, 1, 8, 4, 5, -1, -1],  
            [1, 4, 7, 0, 8, 6, 2, 3, -1],   
            [2, 1, 4, 0, 8, 5, 6, -1, -1],  
            [8, 5, 4, 2, 0, 7, 6, -1, -1],  
            [6, 3, 4, 0, 2, 7, 8, -1, -1]   
        ]
        fallback_draw = [
            [4, 0, 2, 6, 3, 5, 1, 7, 8],    
            [0, 4, 8, 1, 7, 6, 2, 5, 3],    
            [4, 2, 0, 8, 5, 3, 7, 1, 6],    
            [4, 0, 1, 7, 6, 2, 5, 3, 8],
            [4, 1, 2, 6, 3, 5, 7, 8, 0],
            [4, 8, 0, 3, 5, 2, 6, 7, 1]
        ]

        # تعبئة النقص إذا كان أقل من 6
        if len(stupid_list) < 6:
            for i in range(6 - len(stupid_list)):
                target_idx = min(i, self.total_epochs - 1)
                self.replays[target_idx] = fallback_stupid[i % 6]
                if target_idx not in stupid_list:
                    stupid_list.append(target_idx)

        if len(fork_list) < 6:
            mid = self.total_epochs // 2
            for i in range(6 - len(fork_list)):
                target_idx = min(mid + i, self.total_epochs - 1)
                self.replays[target_idx] = fallback_fork[i % 6]
                if target_idx not in fork_list:
                    fork_list.append(target_idx)

        if len(draw_list) < 6:
            for i in range(6 - len(draw_list)):
                target_idx = max(0, self.total_epochs - 6 + i)
                self.replays[target_idx] = fallback_draw[i % 6]
                if target_idx not in draw_list:
                    draw_list.append(target_idx)

        # اختيار 6 جولات من كل مرحلة بالترتيب (المجموع 18 جولة)
        selected_18 = stupid_list[:6] + fork_list[:6] + draw_list[-6:]
        return selected_18

    def analyze_all_games(self):
        """تحليل جميع الجولات لمعرفة من فاز في كل جولة لإظهار الإحصائيات"""
        for moves in self.replays:
            board = np.zeros(9, dtype=int)
            winner_found = False
            for step, move in enumerate(moves):
                if move == -1 or step >= 9:
                    break
                player = 1 if step % 2 == 0 else -1
                board[move] = player
                
                for a, b, c in self.COMBOS:
                    if board[a] == board[b] == board[c] == player:
                        if player == 1:
                            self.stats['X'] += 1
                        else:
                            self.stats['O'] += 1
                        winner_found = True
                        break
                if winner_found:
                    break
            
            if not winner_found:
                self.stats['Draw'] += 1

    def create_widgets(self):
        title_font = font.Font(family="Helvetica", size=14, weight="bold")
        stat_font = font.Font(family="Helvetica", size=12, weight="bold")
        sub_font = font.Font(family="Helvetica", size=12, weight="bold")
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
                  
        # زر لقطات التطور (18 جولة)
        self.btn_highlights = tk.Button(frame_ctrl, text="🌟 لقطات تطور الذكاء (18)", font=ctrl_font, bg="#8C7AE6", fg="white",
                                        width=20, command=self.toggle_highlights)
        self.btn_highlights.grid(row=0, column=3, padx=5)

        # --- شريط القفز ---
        self.epoch_slider = tk.Scale(self.root, from_=1, to=self.total_epochs, orient="horizontal",
                                     length=420, bg="#1E272E", fg="white", highlightthickness=0,
                                     command=self.on_slider_Move, label="تصفح الجولات يدوياً:")
        self.epoch_slider.pack(pady=10)

    def get_ai_phase(self, epoch):
        """تحديد عنوان المرحلة حسب وضع اللقطات أو رقم الجولة العادي"""
        if self.highlights_mode:
            if self.highlight_index < 6:
                return f"المرحلة 1 ({self.highlight_index + 1}/6): غباء وتفويت الفوز 🤪", "#FF7979"
            elif self.highlight_index < 12:
                return f"المرحلة 2 ({self.highlight_index - 5}/6): دهاء الشوكة المزدوجة 🧠", "#F6E58D"
            else:
                return f"المرحلة 3 ({self.highlight_index - 11}/6): احتراف وتعادل حديدي 🤖", "#BADC58"

        progress = epoch / self.total_epochs
        if progress < 0.3:
            return "مرحلة العشوائية والتخبط 🤪 (يبحث عن حلول)", "#FF7979"
        elif progress < 0.7:
            return "مرحلة التعلم وبناء الاستراتيجيات 🤔 (بدأ يفهم)", "#F6E58D"
        else:
            return "مرحلة الذكاء والاحتراف 🧠🤖 (مستوى متقدم)", "#BADC58"

    def check_winner(self, player):
        for a, b, c in self.COMBOS:
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
        
        if self.highlights_mode:
            mode_text = f"🌟 لقطة حاسمة {self.highlight_index + 1} من 18 (الجولة #{epoch_idx + 1}) 🌟"
        else:
            mode_text = f"الجولة: {epoch_idx + 1} / {self.total_epochs}"
        self.lbl_epoch.config(text=mode_text)
        
        self.lbl_status.config(text="يتخذ القرار...", fg="#F8EFBA")

        for btn in self.buttons:
            btn.config(text="", bg="#D2DAE2")

        if self.is_playing:
            self.after_id = self.root.after(550, self.play_next_step)

    def play_next_step(self):
        moves = self.replays[self.current_epoch]

        # إعادة لون المربعات الفارغة للون الافتراضي
        for i in range(9):
            if self.board[i] == 0:
                self.buttons[i].config(bg="#D2DAE2")

        if self.current_step >= 9 or moves[self.current_step] == -1:
            self.finish_round()
            return

        move = moves[self.current_step]
        player = 1 if self.current_step % 2 == 0 else -1
        symbol = "X" if player == 1 else "O"
        color = "#E84118" if player == 1 else "#00A8FF"

        # فحص حالة الرقعة قبل وضع الحركة لاكتشاف الغباء أو الصد التكتيكي
        my_wins_before = self.get_winning_moves(self.board, player)
        opp_wins_before = self.get_winning_moves(self.board, -player)

        self.board[move] = player
        self.buttons[move].config(text=symbol, disabledforeground=color, bg="#D2DAE2")
        self.current_step += 1

        # 1. التحقق من الفوز
        win_combo = self.check_winner(player)
        if win_combo:
            for idx in win_combo:
                self.buttons[idx].config(bg="#4CD137")
            self.lbl_status.config(text=f"🔥 ضربة قاضية! فوز ({symbol}) 🔥", fg="#4CD137")
            if self.is_playing:
                self.after_id = self.root.after(1600, self.auto_advance)
            return

        # 2. التحقق مما إذا كان اللاعب قد فوّت فوزاً مباشراً (المرحلة الغبية)
        if len(my_wins_before) > 0 and move not in my_wins_before:
            for missed_idx in my_wins_before:
                self.buttons[missed_idx].config(bg="#FBC531") 
            self.lbl_status.config(text=f"🤦 خطأ غبي! ({symbol}) ترك مربع الفوز الأصفر ولعب بعيداً!", fg="#FF7979")
            delay = 1100
        else:
            # 3. التحقق مما إذا صنع اللاعب شوكة مزدوجة (المرحلة الذكية)
            my_wins_after = self.get_winning_moves(self.board, player)
            opp_wins_after = self.get_winning_moves(self.board, -player)
            if len(my_wins_after) >= 2 and len(opp_wins_after) == 0:
                for fork_idx in my_wins_after:
                    self.buttons[fork_idx].config(bg="#9C88FF") 
                self.lbl_status.config(text=f"⚡ شوكة مزدوجة من ({symbol})! تهديد في اتجاهين لا حل له!", fg="#00D8D6")
                delay = 1200
            elif move in opp_wins_before:
                self.lbl_status.config(text=f"🛡️ دفاع حديدي! ({symbol}) يغلق خط الفوز فوراً", fg="#F8EFBA")
                delay = 500
            else:
                self.lbl_status.config(text=f"({symbol}) يبني التمركز...", fg="#F8EFBA")
                delay = 450

        if self.is_playing:
            self.after_id = self.root.after(delay, self.play_next_step)

    def finish_round(self):
        self.lbl_status.config(text="عناد تكتيكي من الطرفين.. انتهت بالتعادل العادل! 🤝", fg="#E1B12C")
        if self.is_playing:
            self.after_id = self.root.after(1500, self.auto_advance)

    def auto_advance(self):
        if self.highlights_mode:
            self.highlight_index += 1
            if self.highlight_index < len(self.highlight_epochs):
                self.start_epoch_replay(self.highlight_epochs[self.highlight_index])
            else:
                self.toggle_highlights()
                self.lbl_status.config(text="انتهى عرض مراحل التطور الـ 18 بنجاح 🎬", fg="#00D8D6")
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
        """تفعيل/إلغاء وضع اللقطات الـ 18 الحاسمة"""
        self.highlights_mode = not self.highlights_mode
        if self.highlights_mode:
            self.btn_highlights.config(bg="#9C88FF", text="⏹ إيقاف اللقطات")
            self.highlight_index = 0
            self.is_playing = True
            self.btn_toggle.config(text="⏸ إيقاف مؤقت", bg="#FFA801")
            self.start_epoch_replay(self.highlight_epochs[0])
        else:
            self.btn_highlights.config(bg="#8C7AE6", text="🌟 لقطات تطور الذكاء (18)")
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
            return
        target_epoch = int(val) - 1
        if target_epoch != self.current_epoch:
            self.start_epoch_replay(target_epoch)
            
    def create_dummy_replays(self, filename):
        """إنشاء ملف تجريبي يحتوي على الجولات الـ 18 التكتيكية بالإضافة إلى جولات متنوعة"""
        dummy_data = []
        for _ in range(100):
            moves = np.random.choice(9, 9, replace=False)
            dummy_data.append(moves)

        # زرع 6 جولات المرحلة الأولى (الغباء: ترك مكان الفوز)
        dummy_data[2]  = [0, 3, 1, 4, 8, 5, -1, -1, -1]
        dummy_data[4]  = [2, 0, 5, 4, 7, 8, -1, -1, -1]
        dummy_data[6]  = [0, 2, 3, 5, 4, 8, -1, -1, -1]
        dummy_data[8]  = [1, 4, 2, 5, 0, 6, -1, -1, -1]
        dummy_data[10] = [3, 0, 4, 1, 8, 2, -1, -1, -1]
        dummy_data[12] = [6, 3, 7, 4, 2, 5, -1, -1, -1]

        # زرع 6 جولات المرحلة الثانية (الذكاء: الشوكة المزدوجة)
        dummy_data[40] = [0, 1, 4, 8, 6, 3, 2, -1, -1]
        dummy_data[42] = [0, 3, 2, 1, 8, 4, 5, -1, -1]
        dummy_data[44] = [1, 4, 7, 0, 8, 6, 2, 3, -1]
        dummy_data[46] = [2, 1, 4, 0, 8, 5, 6, -1, -1]
        dummy_data[48] = [8, 5, 4, 2, 0, 7, 6, -1, -1]
        dummy_data[50] = [6, 3, 4, 0, 2, 7, 8, -1, -1]

        # زرع 6 جولات المرحلة الثالثة (الاحتراف: تعادل قوي)
        dummy_data[80] = [4, 0, 2, 6, 3, 5, 1, 7, 8]
        dummy_data[82] = [0, 4, 8, 1, 7, 6, 2, 5, 3]
        dummy_data[84] = [4, 2, 0, 8, 5, 3, 7, 1, 6]
        dummy_data[86] = [4, 0, 1, 7, 6, 2, 5, 3, 8]
        dummy_data[88] = [4, 1, 2, 6, 3, 5, 7, 8, 0]
        dummy_data[90] = [4, 8, 0, 3, 5, 2, 6, 7, 1]

        np.save(filename, np.array(dummy_data))

if __name__ == "__main__":
    root = tk.Tk()
    app = SelfPlayViewer(root)
    root.mainloop()