import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
def create_blue_theme(root):
    style = ttk.Style(root)
    style.theme_use('clam')

    style.configure('TFrame', background='#87CEEB')
    style.configure('TLabel', background='#87CEEB', foreground='white', font=('Arial', 16))
    style.configure('TEntry', fieldbackground='#ADD8E6', background='#ADD8E6', font=('Arial', 14))
    style.configure('TButton', background='#5F9EA0', foreground='white', font=('Arial', 14))
    

def close(cap,root):   
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()
    from main import main_menu_page
    main_menu_page()

def type_the_letter(letter,text_box):
    if letter=='space':
        letter=' '
    elif letter=='del':
        text_box.delete(len(text_box.get())-1)
        return
    text_box.insert(tk.END,letter)

def start():
    root = tk.Tk()
    root.title("the recognised letters")
    # Set the dimensions of the window (width x height)
    window_width = 583
    window_height = 200

    # Set the position of the window (x and y coordinates)
    position_x = 1000
    position_y = 400

    # Construct the geometry string
    geometry_string = f"{window_width}x{window_height}+{position_x}+{position_y}"

    # Apply the geometry settings to the window
    root.geometry(geometry_string)

    cap = cv2.VideoCapture(0)
    # Set the theme to blue
    create_blue_theme(root)

    # Create the main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Create a label
    label = ttk.Label(main_frame, text="Make the sign with one hand and press C to add it here:")
    label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

    # Create a text box
    text_box = ttk.Entry(main_frame, width=50)
    text_box.grid(row=1, column=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    # Create a button
    
    button = ttk.Button(main_frame, text="Close",command=lambda: close(cap=cap,root=root))
    button.grid(row=2, column=0, pady=20,padx=10, sticky=tk.W)

    model_dict = pickle.load(open('./model6.p', 'rb'))
    model = model_dict['model']


    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)


    labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'del', 'nothing', 'space']
    start_camera(text_box=text_box,root=root,cap=cap,hands=hands,labels=labels,model=model,mp_drawing=mp_drawing,mp_drawing_styles=mp_drawing_styles,mp_hands=mp_hands)
    root.mainloop()



def start_camera(text_box,root,cap,hands,labels,model,mp_drawing,mp_drawing_styles,mp_hands):



    ret, frame = cap.read()
    if ret:
        data_aux = []
        x_ = []
        y_ = []


        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            while len(data_aux)<42:
                data_aux.append(0)
            if len(data_aux) == 42:
                # Convert to numpy array and ensure correct shape
                data_aux_np = np.asarray(data_aux).reshape(1, -1)
                #print(f"Shape of input data: {data_aux_np.shape}")
                prediction = model.predict(data_aux_np)
                predicted_class = np.argmax(prediction, axis=1)

                predicted_character = labels[int(predicted_class[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                            cv2.LINE_AA)
                key = cv2.waitKey(1)
                if key==ord('c'):
                    type_the_letter(predicted_character,text_box=text_box)

        cv2.imshow('Alphabet Classification', frame)
    root.after(10,lambda: start_camera(text_box=text_box,root=root,cap=cap,hands=hands,labels=labels,model=model,mp_drawing=mp_drawing,mp_drawing_styles=mp_drawing_styles,mp_hands=mp_hands))
