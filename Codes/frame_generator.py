import cv2

path = '..\\video\\video.mp4'
savepath = '..\\frame\\'
cap = cv2.VideoCapture(path)
c = 1
while 1:
    # get a frame
    ret, frame = cap.read()
    # show a frame
    # cv2.imshow("capture", frame)
    name = str(c).rjust(4, '0')
    frame_rz = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(savepath + name + '.jpg', frame_rz)
    c = c + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
