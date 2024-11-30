import tkinter as tk 
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import Frame
import cv2
import mediapipe as mp
import pickle
import numpy as np
button_style = {
    "font": ("Helvetica", 12),
    "bg": "#61dafb",
    "fg": "Black",
    "activebackground": "#21a1f1",
    "activeforeground": "white",
    "relief": tk.RAISED,
    "bd": 3,
    "width": 15,
    "height": 2
}

def upload_image(image_label, right_frame):
    file_path = filedialog.askopenfilename()
    if file_path:

        image = Image.open(file_path)
        desired_width = 300
        desired_height = 400
        image = image.resize((desired_width, desired_height))
        img = cv2.imread(file_path)
        image = ImageTk.PhotoImage(image)
        image_label.config(image=image)
        image_label.image = image
        prediction = get_prediction(img=img)
        
        prediction_label = tk.Label(right_frame, text=f"predistion is -> {prediction}")
        prediction_label.place(x=225, y=450)



def go_back(root):
    # go back to mainmenu page 
    root.destroy()  
    main_menu_page()

def go_to_photo_window(root):
    root.destroy()
    upload_photo_page()

def go_to_video_window(root):
    from cam3 import start
    root.destroy()
    start()

def clear_window(window):
    for widget in window.winfo_children():
        widget.destroy()
def get_img_arr(img):
    data_aux=[]
    x_list=[]
    y_list=[]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_list.append(x)
                y_list.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_list))
                data_aux.append(y - min(y_list))

    while len(data_aux)<42:
        data_aux.append(0)

    return data_aux
def get_prediction(img):
    model_dict = pickle.load(open('model6.p', 'rb'))
    model = model_dict['model']

    img_arr = get_img_arr(img)
    data_aux_np = np.asarray(img_arr).reshape(1, -1)
    prediction = model.predict(data_aux_np)
    predicted_class = np.argmax(prediction, axis=1)
    labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'del', 'nothing', 'space']
    return labels[int(predicted_class[0])]
def upload_photo_page():

    root = tk.Tk()
    root.title("asl-classification")
    root.geometry("700x500")
    root.configure(bg="#d3d3d3") 
    root.resizable(False, False)


    left_frame = tk.Frame(root, width=450, height=500, bg="#282c34")
    left_frame.pack(side="left", fill="y")


    right_frame = tk.Frame(root, width=600, height=500, bg="white")
    right_frame.pack(side="right", fill="both", expand=True)

    # image uploaded and its prediciton
    image_label = tk.Label(right_frame)
    image_label.place(x=150, y=30)  

    upload_button = tk.Button(left_frame, text="upload photo", command=lambda: upload_image(image_label,right_frame), **button_style)
    upload_button.pack(pady=40)  

    back_button = tk.Button(left_frame, text="go back", command=lambda: go_back(root), **button_style)
    back_button.pack(pady=10) 


    root.mainloop()

def main_menu_page():# Main menu using Tkinter
    root = tk.Tk()
    root.title("Hand Sign Recognition")

    # Set window size and center it
    window_width = 400
    window_height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width / 2) - (window_width / 2)
    y = (screen_height / 2) - (window_height / 2)

    root.geometry(f'{window_width}x{window_height}+{int(x)}+{int(y)}')

    # Background color
    root.configure(bg='#282c34')

    # Title label
    title_label = tk.Label(root, text="Hand Sign Recognition", font=("Helvetica", 18, "bold"), bg='#282c34', fg='#61dafb')
    title_label.pack(pady=20)

    # Frame for buttons
    frame = tk.Frame(root, bg='#282c34')
    frame.pack(pady=20)

    # Button style
    button_style = {
        "font": ("Helvetica", 14),
        "bg": "#61dafb",
        "fg": "Black",
        "activebackground": "#21a1f1",
        "activeforeground": "white",
        "relief": tk.RAISED,
        "bd": 3,
        "width": 20,
        "height": 2
    }

    # Button to predict from video
    video_button = tk.Button(frame, text="Predict from Video", **button_style,command=lambda: go_to_video_window(root=root))
    video_button.grid(row=0, column=0, padx=10, pady=10)

    # Button to predict from photo
    photo_button = tk.Button(frame, text="Predict from Photo", **button_style,command=lambda: go_to_photo_window(root=root))
    photo_button.grid(row=1, column=0, padx=10, pady=10)

    # Start the Tkinter event loop
    root.mainloop()





main_menu_page()

