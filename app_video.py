from objLoader import *
import numpy as np
import time
import math
import cv2
import os

#-------------------------------------------------------------------------------------------------------------
def render(img, obj, projection, model, objectName, color=False):
    vertices = obj.vertices

    if(objectName == "Tree"):
        scale_matrix = np.eye(3) * 0.03
    elif(objectName == "House"):
        scale_matrix = np.eye(3) * 0.65

    h, w = model.shape

    numberOfPoints = len(obj.faces)
    for counter,face in enumerate(obj.faces):
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)

        if color is False and counter < len(obj.faces):
            if(objectName == "Tree"):
                if(counter < numberOfPoints/1.5):
                    cv2.fillConvexPoly(img, imgpts, (27,211,50))
                else:
                    cv2.fillConvexPoly(img, imgpts, (33,67,101))
            elif(objectName == "House"):
                if(counter < 1*numberOfPoints/32):
                    cv2.fillConvexPoly(img, imgpts, (226,219,50))
                elif(counter < 2*numberOfPoints/8):
                    cv2.fillConvexPoly(img, imgpts, (250,250,250))
                elif(counter < 13*numberOfPoints/16):
                    cv2.fillConvexPoly(img, imgpts, (28,186,249))
                else:
                    cv2.fillConvexPoly(img, imgpts, (14,32,130))

        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]
            cv2.fillConvexPoly(img, imgpts, color)

    return img

#-------------------------------------------------------------------------------------------------------------
def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    return np.dot(camera_parameters, projection)

#-------------------------------------------------------------------------------------------------------------
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)

    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

#-------------------------------------------------------------------------------------------------------------

globalTime = time.time()
MIN_MATCHES = 15
homography = None 
resizeFactor = 0.25
resizeFactorImage = 0.50

camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

dir_name = os.getcwd()

objectNames = ["House","Tree"]

models = []
modelPipe = cv2.imread(os.path.join(dir_name, 'references/pipe_model.jpg'), 0)
modelPipe = cv2.resize(modelPipe,None,fx=resizeFactor,fy=resizeFactor, interpolation = cv2.INTER_AREA)
models.append(modelPipe)
modelCafe = cv2.imread(os.path.join(dir_name, 'references/cafe_model.jpg'), 0)
modelCafe = cv2.resize(modelCafe,None,fx=resizeFactor,fy=resizeFactor, interpolation = cv2.INTER_AREA) 
models.append(modelCafe)

obj = []
obj.append(OBJ(os.path.join(dir_name, 'objModels/house.obj'), swapyz=True))
obj.append(OBJ(os.path.join(dir_name, 'objModels/tree.obj'), swapyz=True))

cap = cv2.VideoCapture(0)
while True:
    start = time.time()
    ret,frame = cap.read()
    originalFrame = frame.copy()
    height, width = originalFrame.shape[:2]

    if not ret:
        print("Error with Video")
        break

    for i,model in enumerate(models):
    	try:
    	    kp_model, des_model = orb.detectAndCompute(model, None)
    	    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    	    matches = bf.match(des_model, des_frame)
    	    matches = sorted(matches, key=lambda x: x.distance)

    	    if len(matches) > MIN_MATCHES:
    	        src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    	        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    	        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    	        if homography is not None:
    	            try:
    	                projection = projection_matrix(camera_parameters, homography)  
    	                frame = render(frame, obj[i], projection, model, objectNames[i], False)
    	            except:
    	                pass
    	except:
    		pass

    cv2.putText(originalFrame,"Original",(int(width*0.05),int(height*0.95)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(150,0,0),5, cv2.LINE_8)
    cv2.putText(frame,"Augmented Reality",(int(width*0.05),int(height*0.95)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,150,0),5, cv2.LINE_8)

    combinedImage = np.zeros((height,width*2,3), np.uint8)
    combinedImage[0:height,0:width] = originalFrame
    combinedImage[0:height,width:2*width] = frame
    cv2.line(combinedImage,(width,0),(width,height),(30,30,30),3,8)
    cv2.imshow("Live", combinedImage)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print("\rTime Elapsed: %7.4fs  FPS: %7.4f" %(time.time()-globalTime, 1.0/(time.time()-start)),end='')

print("")
cap.release()
cv2.destroyAllWindows()