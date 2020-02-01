# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import imutils
import os

def align_face(image):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # load the input image, resize it, and convert it to grayscale
    #image = cv2.imread(args["image"])
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    #cv2.imshow("Input", image)
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)

def Zoom(cv2Object, zoomSize):
    # Resizes the image/video frame to the specified amount of "zoomSize".
    # A zoomSize of "2", for example, will double the canvas size
    cv2Object = imutils.resize(cv2Object, width=(zoomSize * cv2Object.shape[1]))
    # center is simply half of the height & width (y/2,x/2)
    center = (cv2Object.shape[0]/2,cv2Object.shape[1]/2)
    # cropScale represents the top left corner of the cropped frame (y/x)
    cropScale = (center[0]/zoomSize, center[1]/zoomSize)
    # The image/video frame is cropped to the center with a size of the original picture
    # image[y1:y2,x1:x2] is used to iterate and grab a portion of an image
    # (y1,x1) is the top left corner and (y2,x1) is the bottom right corner of new cropped frame.
    cv2Object = cv2Object[cropScale[0]:(center[0] + cropScale[0]), cropScale[1]:(center[1] + cropScale[1])]
    return cv2Object

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

def recognize_image(path):
    # load the input image and convert it from BGR to RGB
    image = cv2.imread(path)
    scale_percent = 60  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    #rgb = Zoom(rgb,2)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb,
                                            model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    # initialize the list of names for each face detected
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
        # show the output image
        cv2.imshow("Image", image)

print(args["image"])
files = os.listdir(args["image"])

for file in files:
    fpath = os.path.join(args["image"],file)
    print(file)
    recognize_image(fpath)
    key = cv2.waitKey(0) #& 0xFF

    # if the `k` key was pressed, write the *original* frame to disk
    # so we can later process it and use it for face recognition
    if key == ord("k"):
        continue


    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
