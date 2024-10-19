import cv2



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3,320)
    cap.set(4,240)
    count = 0

    frame_width = int(video.get(3)) 
    frame_height = int(video.get(4)) 
    
    size = (frame_width, frame_height) 
    
    # Below VideoWriter object will create 
    # a frame of above defined The output  
    # is stored in 'filename.avi' file. 
    result = cv2.VideoWriter('filename.avi',  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            10, size) 

    while True and count < 100:
        print("round:_________", count)
        count+=1
        print("frame: ", count)
        ret, img = cap.read()
        result.write(frame)
        
        #cv2.imshow('AA', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()