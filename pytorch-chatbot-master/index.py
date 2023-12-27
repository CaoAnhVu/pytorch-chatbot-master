import tkinter as tk
from tkinter import ttk, scrolledtext, IntVar
from PIL import Image, ImageTk
import random
import torch
import json
import time
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem

class ChatBotGUI:
    def __init__(self, master):
        self.master = master
        master.title("ChatBot Thông Minh")

        # Variable to track the current theme (light or dark)
        self.is_dark_theme = IntVar()

        # Create a scrolled text widget for the chat display with a larger font
        self.chat_display = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=80, height=20, font=('Arial', 16))
        self.chat_display.pack(padx=20, pady=20)

        # Load images for chatbot and user
        self.chatbot_img = Image.open("D:\\TRÍ TUỆ NHÂN TẠO\\pytorch-chatbot-master\\bot.png")
        self.chatbot_img = self.chatbot_img.resize((50, 50), resample=Image.LANCZOS)
        self.chatbot_img = ImageTk.PhotoImage(self.chatbot_img)

        self.user_img = Image.open("D:\\TRÍ TUỆ NHÂN TẠO\\pytorch-chatbot-master\\profile.png")
        self.user_img = self.user_img.resize((50, 50), resample=Image.LANCZOS)
        self.user_img = ImageTk.PhotoImage(self.user_img)

        # Create an entry widget for user input
        self.user_input = tk.Entry(master, width=50, font=('Arial', 14))
        self.user_input.pack(padx=20, pady=20, side=tk.LEFT)

        # Load the airplane icon
        airplane_icon = Image.open("D:\\TRÍ TUỆ NHÂN TẠO\\pytorch-chatbot-master\\send.png")
        airplane_icon = airplane_icon.resize((20, 20), resample=Image.LANCZOS)
        self.airplane_img = ImageTk.PhotoImage(airplane_icon)

        # Create a button to send the user input with an airplane icon
        self.send_button = tk.Button(master, text="Send", command=self.send_message, compound=tk.RIGHT, image=self.airplane_img, width=150, height=20, font=('Arial', 12))
        self.send_button.place(relx=1, rely=1, anchor=tk.SE, x=-20, y=-20)

        # Create a toggle slider for light/dark theme
        style = ttk.Style()
        style.configure('ThemeToggle.TCheckbutton', indicatoron=False, font=('Arial', 12))

        self.theme_toggle = ttk.Checkbutton(master, text="Dark Theme", variable=self.is_dark_theme, command=self.apply_theme, style='ThemeToggle.TCheckbutton')
        self.theme_toggle.pack(side=tk.LEFT, padx=10)

        # Bind the Enter key to the send_message function
        self.master.bind('<Return>', lambda event=None: self.send_message())

        # Load the trained model and intents
        self.load_model()

        # Display a welcome message with chatbot image
        self.display_message_with_typing_animation("Sam: Hi! I'm your ChatBot. How can I assist you today?", 'sam', image=self.chatbot_img, align='left')

    def load_model(self):
        # Load intents
        with open('D:\\TRÍ TUỆ NHÂN TẠO\\pytorch-chatbot-master\\pytorch-chatbot-master\\intents.json', 'r', encoding='utf-8') as f:
            self.intents = json.load(f)

        # Load model state
        data = torch.load("D:\\TRÍ TUỆ NHÂN TẠO\\pytorch-chatbot-master\\data.pth")
        self.model_state = data["model_state"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]

        # Initialize and load the model
        self.model = NeuralNet(len(self.all_words), 8, len(self.tags))
        self.model.load_state_dict(self.model_state)
        self.model.eval()

    def apply_theme(self):
        # Apply theme colors based on the current theme
        if self.is_dark_theme.get():
            self.master.config(bg='#333333')  # Dark background color
            self.chat_display.config(bg='#2d2d2d', fg='white')  # Dark text color
            self.user_input.config(bg='#2d2d2d', fg='white')  # Dark text color
        else:
            self.master.config(bg='white')  # Light background color
            self.chat_display.config(bg='white', fg='black')  # Light text color
            self.user_input.config(bg='white', fg='black')  # Light text color

    def send_message(self):
        # Get user input
        user_message = self.user_input.get()
        self.user_input.delete(0, tk.END)

        # Display user message with user image
        self.display_message("You: " + user_message, 'user', image=self.user_img, align='right')

        # Process user input and get the model's response
        response = self.get_model_response(user_message)

        # Display model response with chatbot image
        self.display_message_with_typing_animation("Sam: " + response, 'sam', image=self.chatbot_img, align='left')

    def get_model_response(self, user_message):
        # Tokenize and create a bag of words for user input
        user_words = tokenize(user_message)
        user_bag = bag_of_words(user_words, self.all_words)

        # Convert to PyTorch tensor
        user_input_tensor = torch.tensor(user_bag, dtype=torch.float32).unsqueeze(0)

        # Get model prediction
        with torch.no_grad():
            output = self.model(user_input_tensor)

        # Get the predicted tag
        predicted_tag = self.tags[torch.argmax(output).item()]

        # Get an appropriate response from intents
        for intent in self.intents['intents']:
            if intent['tag'] == predicted_tag:
                responses = intent['responses']
                return random.choice(responses)

        return "I'm sorry, I don't understand."

    def display_message(self, message, sender, image=None, align='left'):
        # Display messages in the chat display with images
        if image:
            self.chat_display.image_create(tk.END, image=image)
        self.chat_display.insert(tk.END, message + "\n", (sender, align))
        self.chat_display.yview(tk.END)

        # Configure tags for left (bot) and right (user) alignment
        self.chat_display.tag_configure('sam', justify='left', background='lightblue')
        self.chat_display.tag_configure('user', justify='right', background='lightgreen')

    def display_message_with_typing_animation(self, message, sender, image=None, align='left'):
        # Display messages with typing animation
        if image:
            self.chat_display.image_create(tk.END, image=image)

        # Configure tags for left (bot) and right (user) alignment
        self.chat_display.tag_configure('sam', justify='left', background='lightblue')
        self.chat_display.tag_configure('user', justify='right', background='lightgreen')

        # Keep track of the last sender and line for each sender
        last_sender = None
        last_line = ""

        # Iterate over characters in the message and display them with a delay
        for char in message:
            # Determine the alignment based on the sender
            if sender != last_sender:
                align = 'left' if sender == 'sam' else 'right'
                last_sender = sender

            self.chat_display.insert(tk.END, char, (sender, align))
            last_line += char

            self.chat_display.yview(tk.END)
            self.master.update()  # Force an update to see the changes
            time.sleep(0.02)  # Adjust the sleep duration for desired typing speed

if __name__ == "__main__":
    root = tk.Tk()
    chatbot_gui = ChatBotGUI(root)
    root.mainloop()
