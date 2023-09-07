import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob

contador = 0

camara = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (int(camara.get(3)), int(camara.get(4))))
while camara.isOpened():
    ret, frame = camara.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # write the flipped frame
    out.write(frame)
    cv.imshow('Presionar "q" para finalizar', frame)
    if cv.waitKey(1) == ord('q'):
        break
    if cv.waitKey(1) == ord('a'):
        cv.imwrite("chessboard.jpg",frame)
# Release everything if job is finished
camara.release()
out.release()
cv.destroyAllWindows()

file = cv.VideoCapture('output.mp4')
while file.isOpened():
    ret, frame = file.read()
    # if frame is read correctly ret is True
    if ret:
        contador += 1
    if not ret:
        break
    if contador % 10 == 0 and contador <= 200:
        cv.imwrite("imagen{:d}.jpg".format(int(contador / 10)), frame)
        print("Printed image {:d} successfully!".format(int(contador / 10)))
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
file.release()
cv.destroyAllWindows()

images = glob.glob('imagen*.jpg')

nx = 7; ny = 7
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(nx*ny,2)

print("Primeros puntos:"); print(objp[0:3])
print("Últimos puntos:"); print(objp[-4:])

# Lectura de una imagen
I = cv.imread('chessboard.jpg', 1)
# Tamaño de la imagen
I_size = (I.shape[1], I.shape[0])
print("Tamaño de la imagen: (x,y)=({},{})".format(I_size[0], I_size[1]))
# Conversión de la imagen a escala de grises
Igray = cv.cvtColor(I, cv.COLOR_BGR2GRAY)

# Encontrar las esquinas (corners) internas del patrón. A estas esquinas se les denomina también
# "puntos de la imagen" ("image points")
retval, corners = cv.findChessboardCorners(Igray, (nx, ny), None)

# Si se encuentra los puntos (corners), mostrar la imagen con los puntos
if retval == True:
    # Añadir los puntos (corners) encontrados a la imagen
    cv.drawChessboardCorners(I, (8,6), corners, retval)
    # Mostrar la imagen
    plt.figure(figsize=(10,10))
    plt.imshow(I, cmap='gray')
    plt.axis('off')
    plt.show()

obj_points = []  # Puntos del objeto (3d)
img_points = []  # Puntos de la imagen (2d)

# Bucle a lo largo de todas las imágenes, buscando las "esquinas" (corners)
for idx, iname in enumerate(images):
    # Leer la imagen
    I = cv.imread(iname)
    # Convertir a escala de grises
    Igray = cv.cvtColor(I, cv.COLOR_BGR2GRAY)

    # Encontrar las esquinas (corners) internas del patrón
    retval, corners = cv.findChessboardCorners(Igray, (nx, ny), None)

    # Si se encuentra los puntos, añadirlos a la lista
    if retval == True:
        # Puntos del objeto y puntos de la imagen, para la imagen actual
        obj_points.append(objp)
        img_points.append(corners)

        # Añadir a la imagen los puntos (corners) encontrados
        cv.drawChessboardCorners(I, (7, 7), corners, retval)

        # Si se desea grabar las imágenes con sus "corners", descomentar la siguiente línea
        # cv2.imwrite('corners'+str(idx)+'.jpg', I)

        # Mostrar la imagen (se puede abrir una ventana adicional)
        cv.imshow('Imagen con puntos esquina', I)
        cv.waitKey(500)

cv.destroyAllWindows()

retval, M, coefs_dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, I_size, None, None)

print("Matriz de calibración:"); print(np.round(M,3))
print("\nCoeficientes de distorsión:"); print(np.round(coefs_dist,3))

# Imagen de entrada
orig = cv.imread('imagen10.jpg')

# Corregir la distorsión
sinDist = cv.undistort(orig, M, coefs_dist, None)

# Visualize undistortion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(orig)
ax1.set_title('Imagen original', fontsize=20); ax1.axis('off')
ax2.imshow(sinDist)
ax2.set_title('Imagen sin distorsión', fontsize=20); ax2.axis('off')
plt.show()
