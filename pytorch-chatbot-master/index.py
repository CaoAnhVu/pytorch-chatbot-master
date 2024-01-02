import tkinter as tk
from tkinter import  ttk, scrolledtext, IntVar
from PIL import Image, ImageTk
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox  #Import thư viện messagebox
from model import NeuralNet
import time
import torch
import json
import random
from nltk_utils import bag_of_words, tokenize, stem

class ChatBotGUI:
    def __init__(self, master):
        self.master = master
        master.title("ChatBot")
        
        # Biến để theo dõi chủ đề hiện tại (sáng hoặc tối)
        self.is_dark_theme = IntVar()
        
        # Tạo một widget scrolled text để hiển thị cuộc trò chuyện với font lớn hơn
        self.chat_display = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=80, height=20, font=('Arial', 16))
        self.chat_display.pack(padx=20, pady=20)

        # Tải hình ảnh cho ChatBot và người dùng
        self.chatbot_img = Image.open("D:\\TRÍ TUỆ NHÂN TẠO\\pytorch-chatbot-master\\bot.png")
        self.chatbot_img = self.chatbot_img.resize((50, 50), resample=Image.LANCZOS)
        self.chatbot_img = ImageTk.PhotoImage(self.chatbot_img)

        self.user_img = Image.open("D:\\TRÍ TUỆ NHÂN TẠO\\pytorch-chatbot-master\\profile.png")
        self.user_img = self.user_img.resize((50, 50), resample=Image.LANCZOS)
        self.user_img = ImageTk.PhotoImage(self.user_img)
        
        # Tải icon máy bay 
        airplane_icon = Image.open("D:\\TRÍ TUỆ NHÂN TẠO\\pytorch-chatbot-master\\send.png")
        airplane_icon = airplane_icon.resize((20, 20), resample=Image.LANCZOS)
        self.airplane_img = ImageTk.PhotoImage(airplane_icon)

        # Tạo một ô nhập để người dùng nhập
        self.user_input = tk.Entry(master, width=50, font=('Arial', 14))
        self.user_input.pack(padx=20, pady=20, side=tk.LEFT)

        # Tạo một nút để gửi đầu vào của người dùng với icon máy bay
        self.send_button = tk.Button(master, text="Send", command=self.send_message, compound=tk.LEFT, image=self.airplane_img, width=150, height=20, font=('Arial', 12))
        self.send_button.place(relx=1, rely=1, anchor=tk.SE, x=-250, y=-20)
        
        # Tạo một thanh trượt chuyển đổi cho chủ đề sáng/tối
        style = ttk.Style()
        style.configure('ThemeToggle.TCheckbutton', indicatoron=False, font=('Arial', 12))

        self.theme_toggle = ttk.Checkbutton(master, text="Dark Theme", variable=self.is_dark_theme, command=self.apply_theme, style='ThemeToggle.TCheckbutton')
        self.theme_toggle.pack(side=tk.RIGHT, padx=10)

        # Liên kết phím Enter với chức năng gửi tin nhắn
        self.master.bind('<Return>', lambda event=None: self.send_message())
        
         # Tạo một nút cho cuộc trò chuyện mới
        self.new_chat_button = tk.Button(master, text="New Chat", command=self.create_new_chat, font=('Arial', 12))
        self.new_chat_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Load the trained model and intents
        self.load_model()
        
         # Hiển thị thông báo xin chào với hình ảnh chatbot
        self.display_message_with_typing_animation("Sam: Hi! I'm your ChatBot. How can I assist you today?", 'sam', image=self.chatbot_img, align='left')
        

    def load_model(self):
        # Load intents
        with open('D:\\TRÍ TUỆ NHÂN TẠO\\pytorch-chatbot-master\\pytorch-chatbot-master\\intents.json', 'r', encoding='utf-8') as f:
            self.intents = json.load(f)
    
        

        # Tải trạng thái model
        data = torch.load("D:\\TRÍ TUỆ NHÂN TẠO\\pytorch-chatbot-master\\data.pth")
        self.model_state = data["model_state"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]

        # Khởi tạo và tải model
        self.model = NeuralNet(len(self.all_words), 8, len(self.tags))
        self.model.load_state_dict(self.model_state)
        self.model.eval()
        
    def apply_theme(self):
        # Áp dụng màu sắc chủ đề dựa trên chủ đề hiện tại
        if self.is_dark_theme.get():
            self.master.config(bg='#333333')  # Dark background color
            self.chat_display.config(bg='#2d2d2d', fg='white')  # Dark text color
            self.user_input.config(bg='#2d2d2d', fg='white')  # Dark text color
        else:
            self.master.config(bg='white')  # Light background color
            self.chat_display.config(bg='white', fg='black')  # Light text color
            self.user_input.config(bg='white', fg='black')  # Light text color

    def create_new_chat(self):
        # Hiển thị hộp thoại xác nhận
        result = messagebox.askquestion("New Chat", "Are you sure you want to start a new chat?")

        # Xử lý kết quả từ hộp thoại xác nhận
        if result == 'yes':
            # Xóa nội dung trong hien_thi_cuoc_tro_chuyen
            self.chat_display.delete(1.0, tk.END)

            # Hiển thị thông báo xin chào mới
            self.display_message_with_typing_animation("Sam: Hi! I'm your ChatBot. How can I assist you today?", 'sam', image=self.chatbot_img)

    def send_message(self):
        # Nhận đầu vào từ người dùng
        user_message = self.user_input.get()
        self.user_input.delete(0, tk.END)

        # Hiển thị tin nhắn của người dùng với hình ảnh người dùng
        self.display_message("You: "  + user_message, 'user', image=self.user_img)

        # Xử lý đầu vào của người dùng và nhận phản hồi từ model
        response = self.get_model_response(user_message)

        # Hiển thị phản hồi của mô hình với hình ảnh chatbot
        self.display_message_with_typing_animation("Sam: " + response, 'sam', image=self.chatbot_img)

    def get_model_response(self, user_message):
        # Tokenize và tạo một túi từ cho đầu vào của người dùng
        user_words = tokenize(user_message)
        user_bag = bag_of_words(user_words, self.all_words)

        # Chuyển đổi thành tensor PyTorch
        user_input_tensor = torch.tensor(user_bag, dtype=torch.float32).unsqueeze(0)

         # Nhận dự đoán từ mô hình
        with torch.no_grad():
            output = self.model(user_input_tensor)

        # Nhận tag đã dự đoán
        predicted_tag = self.tags[torch.argmax(output).item()]

        # Nhận một phản hồi phù hợp từ ý định
        for intent in self.intents['intents']:
            if intent['tag'] == predicted_tag:
                responses = intent['responses']
                return random.choice(responses)

        return "I'm sorry, I don't understand."

    def display_message(self, message, sender, image=None, align='left'):
    # Hiển thị tin nhắn trong hiển thị cuộc trò chuyện với hình ảnh
        if image:
            self.chat_display.image_create(tk.END, image=image)
        if sender == 'sam':
            align = 'left'  # Align chatbot's message to the left
        else:
            align = 'bottom'
        self.chat_display.insert(tk.END, "\n" + message + "\n\n", (sender, align))
        self.chat_display.yview(tk.END)

        # Cấu hình các tag cho căn lề trái (chatbot) và căn lề phải (người dùng)
        self.chat_display.tag_configure('sam', justify='left')
        self.chat_display.tag_configure('user', justify='left')
        
    def display_message_with_typing_animation(self, message, sender, image=None, align='left'):
        # Hiển thị tin nhắn với hiệu ứng đánh máy
        if image:
            self.chat_display.image_create(tk.END, image=image)

        # Cấu hình các tag cho căn lề trái (chatbot) và căn lề phải (người dùng)
        self.chat_display.tag_configure('sam', justify='left')
        self.chat_display.tag_configure('user', justify='left')

        # Theo dõi người gửi cuối cùng và dòng cuối cùng cho từng người gửi
        last_sender = None
        last_line = ""

        # Lặp qua các ký tự trong thông báo và hiển thị chúng với độ trễ
        for char in message:
            # Xác định căn lề dựa trên người gửi
            if sender != last_sender:
                align = 'left' if sender == 'sam' else 'left'
                last_sender = sender

            self.chat_display.insert(tk.END, char, (sender, align))
            last_line += char

            self.chat_display.yview(tk.END)
            self.master.update()  # Buộc cập nhật để xem các thay đổi
            time.sleep(0.02)  # Điều chỉnh để có tốc độ gõ mong muốn

if __name__ == "__main__":
    root = tk.Tk()
    chatbot_gui = ChatBotGUI(root)
    root.mainloop()
