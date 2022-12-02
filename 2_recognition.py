import face_recognition
import os, sys
import cv2
import numpy as np
import math
from utils import ListClass

#criando a função de calculo (%)
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

#declarando variaveis
class FaceRecognition:
    empty_list = ListClass()
    face_locations, face_encodings, face_names, known_face_encodings, known_face_names = empty_list.generate_list(5)

    process_current_frame = True
    
    def __init__(self):
        self.encode_faces()
    
    #percorrer imagens das pastas
    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
            
        #printando todas as faces cadastradas na pasta
        print(self.known_face_names)

    #abertua de  camera
    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            # Processe apenas todos os outros quadros de vídeo para economizar tempo
            if self.process_current_frame:
                # Redimensionando o quadro do vídeo para 1/4 do tamanho para processamento de reconhecimento facial mais rápido
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convertendo a imagem da cor BGR (que o OpenCV usa) para a cor RGB (que o reconhecimento facial usa)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Encontrando todos os rostos e codificações de rosto no quadro atual do vídeo
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # Vendo se o rosto é compatível com os rostos conhecidos
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    # Calcular a distância mais curta de cada face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Exibindo os resultados
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4 
                bottom *= 4
                left *= 4

                # Criando um frame no seu rosto
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Exibindo resultado
            cv2.imshow('Face Recognition', frame)

            # 'q' para sair
            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
