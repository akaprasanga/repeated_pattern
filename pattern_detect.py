from tkinter import Tk, Label, Button
from xor import repeat_detect
from tkinter.filedialog import askopenfilenames
import shutil
import os

idx = 0
label_trick = 0

class MyFirstGUI:

    def __init__(self, master):
        self.master = master
        master.title("A simple GUI")

        self.label = Label(master, font="25",text="FIND REPEATED PATTER IN IMAGES")
        self.label.pack()

        # lab = Label(root)
        # lab.pack()



        self.greet_button = Button(master,font="50",pady=5,padx=10, text="Choose Images", command=self.OpenFile)
        self.greet_button.pack()

        self.label1 = Label(master, bg="white",font="50",pady=20,padx=20,text="This label Will be Updated After processing of Images Finishes")
        global label_trick
        label_trick = self.label1
        self.label1.pack()





        self.close_button = Button(master, text="Clear Ouput Folders", command=self.flush_folders)
        self.close_button.pack()

        self.label = Label(master, text="The ouput are stored in 'Output' and 'Failed' Folder of Installation Directory")
        self.label.config(font=("Courier",10))
        self.label.pack()

        self.close_button = Button(master,font="50",pady=10,text="Close Program", command=master.quit)
        self.close_button.pack(side = "bottom")
    def create_folders(self):
        if not os.path.exists('Output'):
            os.makedirs("Output")
        if not os.path.exists('Failed'):
            os.makedirs('Failed')



           
    def greet(self):
        print("Greetings!")
    def OpenFile(self):
        if not os.path.exists('Output'):
            os.makedirs("Output")
        if not os.path.exists('Failed'):
            os.makedirs('Failed')
        name = askopenfilenames(
                            filetypes =(("Image Files", "*.jpg;*.png"),("All Files","*.*")),
                            title = "Choose a file."
                            )
        print (name)
        for var in name:
            global idx
 
            print (var)
            print('Processing Image'+str(idx))
            repeat_detect.main(1,var)
            global label_trick
            label_trick.config(text= "Proceesing Finished Check Ouput Folders")


            idx +=1


            # cv2.waitKey(0)
            # lab.config(text='hi')

        print('Finished Processing')

    def flush_folders(self):
        if not os.path.exists('Output'):
            print("No any Ouput Folder")
        else:
            shutil.rmtree('Output')
            shutil.rmtree('Failed')
            
            os.makedirs("Output")
            
            os.makedirs('Failed')

           
 



root = Tk()
my_gui = MyFirstGUI(root)
root.geometry("720x480")
root.mainloop()

