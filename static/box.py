import sys
import os
import cv2
import numpy as np
# import imutils
from four_point_transform import four_point_transform

fullpath=os.path.dirname(os.path.realpath(__file__))

timestamp = sys.argv[1]
file_ext = sys.argv[2]


def func(image):
	bordersize=100
	image_padded=cv2.copyMakeBorder(image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
	cpy = image_padded.copy()
	
	gray = cv2.cvtColor(cpy,cv2.COLOR_BGR2GRAY)
	#cv2.imshow("gray", imutils.resize(gray, height = 650))
	#cv2.waitKey(0)
	
	noiserem = cv2.GaussianBlur(gray, (5,5), 0)
	
	edges = cv2.Canny(noiserem,150,200,apertureSize = 3)
	#cv2.imshow("edges", imutils.resize(edges, height = 650))
	#cv2.waitKey(0)
	
	lines = cv2.HoughLines(edges,2,np.pi/180,600)
	
	for x in range(0, len(lines)):
    	#for x1,y1,x2,y2 in lines[x]:
		for rho,theta in lines[x]:
	    		a = np.cos(theta)
	    		b = np.sin(theta)
	    		x0 = a*rho
	    		y0 = b*rho
	    		
	    		x1 = int(x0 + 10000*(-b))
	    		y1 = int(y0 + 10000*(a))
	    		x2 = int(x0 - 10000*(-b))
	    		y2 = int(y0 - 10000*(a))
			
			
				
	    		cv2.line(cpy,(x1,y1),(x2,y2),(0,0,255),5)
	
	#cv2.imshow("This",imutils.resize(cpy, height = 650))
	#cv2.waitKey(0)
	x_tl = -10000
	y_tl = -10000
	
	x_tr = 10000
	y_tr = -10000
	
	x_bl = -10000
	y_bl = 10000
	
	x_br = 10000
	y_br = 10000
		
	points = {}
	#cv2.circle(cpy, (20, 1669), 50, (0,255,0), 10)
	for x in range(len(lines)):
		for y in range(x+1, len(lines)):
			#print(lines)
			r1,theta1 = lines[x][0]
			r2,theta2 = lines[y][0]
			#print(abs(theta1-theta2))
			
			if((theta1 > 10.0*np.pi/180.0 and theta1< 80.0*np.pi/180.0) or (theta1 > 100.0*np.pi/180.0 and theta1< 170.0*np.pi/180.0)):
				continue
			if((theta2 > 10.0*np.pi/180.0 and theta2< 80.0*np.pi/180.0) or (theta2 > 100.0*np.pi/180.0 and theta2< 170.0*np.pi/180.0)):
				continue
			if(abs(theta1 - theta2)<5.0*np.pi/180.0):
				continue 
			if(abs(theta1 - theta2)>175.0*np.pi/180.0):
				continue
			#print([theta1*180.0/np.pi ,theta2*180.0/np.pi])
			s1 = np.sin(theta1)
			c1 = np.cos(theta1)
			s2 = np.sin(theta2)
			c2 = np.cos(theta2)
			d = np.sin(theta2 - theta1)
			x1 = (r1*s2 - r2*s1)/d
			y1 = (r2*c1 - r1*c2)/d
			#points.append([x,y])
			cv2.circle(cpy,(x1,y1), 50, (255,0,0), 10)
			
			
			if( (0 - x1)*(0 - x1) + (0 - y1)*(0 - y1) < (0 - x_tl)*(0 - x_tl) + (0 - y_tl)*(0 - y_tl) ):
				x_tl = x1
				y_tl = y1
				
			if( (0 - x1)*(0 - x1) + (cpy.shape[0] - y1)*(cpy.shape[0] - y1) < (0 - x_bl)*(0 - x_bl) + (cpy.shape[0] - y_bl)*(cpy.shape[0] - y_bl) ):
				x_bl = x1
				y_bl = y1
				
			if( (cpy.shape[1] - x1)*(cpy.shape[1] - x1) + (0 - y1)*(0 - y1) < (cpy.shape[1] - x_tr)*(cpy.shape[1] - x_tr) + (0 - y_tr)*(0 - y_tr) ):
				x_tr = x1
				y_tr = y1
				
			if( (cpy.shape[1] - x1)*(cpy.shape[1] - x1) + (cpy.shape[0] - y1)*(cpy.shape[0] - y1) < (cpy.shape[1] - x_br)*(cpy.shape[1] - x_br) + (cpy.shape[0] - y_br)*(cpy.shape[0] - y_br) ):
				x_br = x1
				y_br = y1
			
	cv2.circle(cpy,(x_tl,y_tl), 200, (255,255,0), 10)
	cv2.circle(cpy,(x_tr,y_tr), 150, (255,255,0), 10)
	cv2.circle(cpy,(x_bl,y_bl), 100, (255,255,0), 10)
	cv2.circle(cpy,(x_br,y_br), 50, (255,255,0), 10)
	
	cv2.line(image_padded ,(x_tl,y_tl), (x_tr, y_tr), (255,0,0), 10)
	cv2.line(image_padded ,(x_tr,y_tr), (x_br, y_br), (255,0,0), 10)
	cv2.line(image_padded ,(x_br,y_br), (x_bl, y_bl), (255,0,0), 10)
	cv2.line(image_padded ,(x_bl,y_bl), (x_tl, y_tl), (255,0,0), 10)
	
	
	#cv2.imshow("This",imutils.resize(cpy, height = 650))
	#cv2.waitKey(0)
	
	points = np.array([ [x_tl, y_tl], [x_bl, y_bl], [x_br,y_br], [x_tr, y_tr]])
	return image_padded,points
			

def pre_process(image):
	cpy = image.copy()
	#cv2.imshow("original",imutils.resize(image,height = 650))
	cpy,points = func(image)
	#cv2.imshow("ouput", imutils.resize(cpy,height = 650))
	warped = four_point_transform(cpy, points)
	#cv2.imshow("warped", imutils.resize(warped, height = 650))
	#cv2.waitKey(0)
	return warped
	
img1=cv2.imread(fullpath+'/uploads/'+timestamp+'/input.'+file_ext)
warped = pre_process(img1)
img1=cv2.resize(warped,(589,821))
cv2.imwrite(fullpath+'/uploads/'+timestamp+'/preprocessed-resized.'+file_ext,img1)



ret,binary = cv2.threshold(img1,205,255,cv2.THRESH_BINARY)
cv2.imwrite(fullpath+'/uploads/'+timestamp+'/binary_for_frontend.'+file_ext, binary)



img=img1
cv_size = lambda img: tuple(img.shape[1::-1])
s=cv_size(img)
m=s[1]/821
ret,img = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
cv2.rectangle(img,(int(8*m),int(8*m)),(int(579*m),int(136*m)),(0,255,0),5)
cv2.rectangle(img,(int(8*m),int(136*m)),(int(579*m),int(204*m)),(0,255,0),5)
cv2.rectangle(img,(int(8*m),int(204*m)),(int(579*m),int(501*m)),(0,255,0),5)
cv2.rectangle(img,(int(8*m),int(501*m)),(int(579*m),int(813*m)),(0,255,0),5)
cv2.imwrite(fullpath+'/uploads/'+timestamp+'/ROIsegmented.'+file_ext,img)