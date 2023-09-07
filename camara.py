import cv2 as cv

contador: int = 1

camara = cv.VideoCapture(0)

def change_res(width, height):
    camara.set(3, width)
    camara.set(4, height)

change_res(640, 480)

while camara.isOpened():

    ret, frame = camara.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imshow('Presionar "q" para finalizar', frame)
    if cv.waitKey(1) == ord('q'):
        contador = 0
        break
    if contador % 20 == 0:
        cv.imwrite("imagen"+str(int(contador/10))+".jpg", frame, [640, 480, 3])
    contador += 1

camara.release()
cv.destroyAllWindows()

