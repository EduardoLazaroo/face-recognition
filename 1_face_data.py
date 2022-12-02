import utils
import numpy as np
import cv2
import os

#import das classes do utils
check_cam = utils.checkCam()
check_face = utils.checkFace()
check_error = utils.errorCam()

#adpter para utilizarmos 
check_error_adpter = utils.errorCamAdapter(check_error)
adm = utils.ADM()

#Iniciando camera e configurando as dimensões da captura de video
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

#facecascade cml
faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

#configurando face id
face_id = input('\n digite o ID do usuário e pressione ==>  ')




#Messagem de inicialização do adpter
adm.printADM(check_cam)



print("\n [INFO] Inicializando a captura de rosto. Olhe para a câmera e espere...")
count = 0

#Lógica por tras de criar a camera e realizar a leitura por video
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20, 20)
    )

    #transformando a imagem em captura cinza
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1
        
        # Salve a imagem capturada na pasta de conjuntos de dados (formato jpg)
        try:
            #Print checagem de imagem pelo adpter
            adm.printADM(check_face)
            cv2.imwrite("faces/" + str(face_id) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)
        
        except:
            #error no adpter
            adm.printADM(check_error_adpter)

    k = cv2.waitKey(100) & 0xff
    if k == 27: #Press 'ESC' para sair
        break
    elif count >= 1: #Máximo de fotos configuradas
        break 

cap.release()
cv2.destroyAllWindows()