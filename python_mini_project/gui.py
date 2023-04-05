from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import cv2
from keras.models import model_from_json

# reading model jsonfile
json_file = open('model_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# load weights into new model
loaded_model.load_weights("model_final.h5")
root = Tk()
root.geometry("800x600")
title_label = Label(root,text='Hand Written Equation Solver',font=('Arial',24))
title_label.pack()

def equ_eval(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img=~img
        ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ctrs,ret=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        w=int(28)
        h=int(28)
        train_data=[]
        print(len(cnt))
        rects=[]
        for c in cnt :
            x,y,w,h= cv2.boundingRect(c)
            rect=[x,y,w,h]
            rects.append(rect)
        print(rects)
        bool_rect=[]
        for r in rects:
            l=[]
            for rec in rects:  
                flag=0
                if rec!=r:
                    if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
                        flag=1 
                    l.append(flag)
                if rec==r:
                    l.append(0)
            bool_rect.append(l) 
        print(bool_rect)
        dump_rect=[]
        for i in range(0,len(cnt)): 
            for j in range(0,len(cnt)):
                if bool_rect[i][j]==1:   
                    area1=rects[i][2]*rects[i][3]
                    area2=rects[j][2]*rects[j][3]
                    if(area1==min(area1,area2)):  
                        dump_rect.append(rects[i])
        print(len(dump_rect)) 
        final_rect=[i for i in rects if i not in dump_rect]
        print(final_rect)
        for r in final_rect:
            x=r[0]
            y=r[1]
            w=r[2]
            h=r[3]
            im_crop =thresh[y:y+h+10,x:x+w+10]
            im_resize = cv2.resize(im_crop,(28,28))
            im_resize=np.reshape(im_resize,(28,28,1))
            train_data.append(im_resize)  
    equation=''
    for i in range(len(train_data)): 
        train_data[i]=np.array(train_data[i])
        train_data[i]=train_data[i].reshape(1,28,28,1)
        result=np.argmax(loaded_model.predict(train_data[i]), axis=-1)
        if(result[0]==10):
            equation = equation +'-'
        if(result[0]==11):
            equation = equation +'+'
        if(result[0]==12):
            equation = equation +'*'
        if(result[0]==13):
            equation = equation +'/'    
        if(result[0]==0):
            equation = equation +'0'
        if(result[0]==1):
            equation = equation +'1'
        if(result[0]==2):
            equation = equation +'2'
        if(result[0]==3):
            equation = equation +'3'
        if(result[0]==4):
            equation = equation +'4'
        if(result[0]==5):
            equation = equation +'5'
        if(result[0]==6):
            equation = equation +'6'
        if(result[0]==7):
            equation = equation +'7'
        if(result[0]==8):
            equation = equation +'8'
        if(result[0]==9):
            equation = equation +'9'
    return(equation) 
global eval_image                                              
# Create a file dialog to browse for image
def browse_file():
     # Remove previous image label widget, if it exists
    global image_label 
    if 'image_label' in globals():
        image_label.destroy()
    file_path = filedialog.askopenfilename()
    # Open the image file using Pillow
    image = Image.open(file_path)
    eval_image = image
    # Resize the image if necessary
    if image.width > 600 :
        scale_factor = 600 / image.width
        image = image.resize((int(scale_factor * image.width), int(scale_factor * image.height)))
    # Convert the image to Tkinter format
    tk_image = ImageTk.PhotoImage(image)
    # Create a label to display the image
    image_label = Label(root, image=tk_image)
    image_label.image = tk_image
    image_label.pack()
    equ_text.insert(END,equ_eval(file_path))
    equ = equ_text.get("1.0",END)
    res_text.insert(END,eval(equ))
    

# Create a button to browse for image
browse_button = Button(root, text="Browse", command=browse_file)
browse_button.pack()
equ_lable = Label(root,text='Equation :',font=('Arial',15)).place(x=250,y=350)
#equ_text = Text(root,height = 1, width = 10).place(x=350,y=355)
equ_text = Text(root,height = 1, width = 10)
equ_text.pack()
equ_text.place(x=350,y=355)



res_lable = Label(root,text='Result :',font=('Arial',15)).place(x=250,y=377)
res_text = Text(root,height = 1, width = 10)
res_text.pack()
res_text.place(x=350,y=382)
root.mainloop()
