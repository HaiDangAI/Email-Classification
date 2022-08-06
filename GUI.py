import tkinter as tk
from tkinter import *
from tkinter import filedialog
from TrainingModel import EmailPredict

window = tk.Tk()
window.title('Email Classification')
window.geometry('1000x600')

def upload_file():
    try:
        file_path=filedialog.askopenfilename()
        inputBox.delete(0, END)
        inputBox.insert(0, file_path)
    except:
        pass

def show_email():
    try:
        emailBox.delete(1.0, END)
        file_path = inputBox.get()
        inputBox.delete(0, END)
        email = open(file_path).read()
        emailBox.insert(END, email)
    except: pass

def predict():
    if inputBox.get() != '': showButton.invoke()
    email = emailBox.get(0.0, END)
    prediction = EmailPredict(email)
    predictBox.insert(END, prediction+'\n')
    predictBox.see('end')

url_label = Label(window, text='URL:', font=('Times New Roman', 14))
url_label.place(x=30 ,y=50)
email_label = Label(window, text='Email:', font=('Times New Roman', 14))
email_label.place(x=30 ,y=100)

uploadButton = Button(window, text='Upload File', command=upload_file, height=1, width=10)
uploadButton.place(x=900, y=50)
showButton = Button(window, text='Show', command=show_email, height=1, width=10)
showButton.place(x=900, y=100)
predictButton = Button(window, text='Predict', command=predict, height=1, width=10)
predictButton.place(x=900, y=150)
exitButton = Button(window, text='Exit',command=window.destroy ,height=1, width=10)
exitButton.place(x=900, y=390)


inputBox = Entry(window, font=('Times New Roman', 14))
inputBox.place(x=100 ,y=50, width=770)

emailBox = Text(window, width=96, height=28)
emailBox.place(x=100 ,y=100)

predictBox = Text(window, width=10, height=10)
predictBox.place(x=900 ,y=200)


window.mainloop()