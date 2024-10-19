# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
#from tqdm import tqdm
from os.path import join
from sys import platform
import argparse
from my_preprocess import *
sys.path.append(os.path.abspath("E:\file\openpose\build_GPU\examples\tutorial_api_python\EasyMocap-master"))
#from apps import *
from apps.demo import mv1p_my
from easymocap.socket.base_client import BaseSocketClient



def use_mv1p_my(img, annot):
    from easymocap.mytools import load_parser, parse_parser
    from easymocap.dataset import CONFIG, MV1PMF
    parser = load_parser()
    parser.add_argument('--skel', action='store_true')
    args = parse_parser(parser)
    
    dataset = MV1PMF(image = img, annot = annot, root = args.path, cams=args.sub, out=args.out,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=args.undis, no_img=False, verbose=args.verbose)
    dataset.writer.save_origin = args.save_origin

    skel_d = mv1p_my.mv1pmf_skel(dataset, check_repro=True, args=args)
    #show_img = mv1p_my.mv1pmf_smpl(dataset, args, skel_d)
    return skel_d


try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="/Users/jacoby/Downloads/EasyMocap-master/real_time_process/captured_images/image.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    #params["face"] = True
    params["render_pose"] = 1
    params["net_resolution"] = '256x176'
    

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture(r'"E:\file\openpose\build_GPU\examples\tutorial_api_python\EasyMocap-master\cam\videos\1.mp4"')
    cap.set(3,320)
    cap.set(4,240)

    count = 0
    client = BaseSocketClient('127.0.0.1', 9999)
    #from os import listdir
 
    # get the path/directory
    #folder_dir = r"E:\file\openpose\build_GPU\examples\tutorial_api_python\EasyMocap-master\cam\images\1"
    #for images in os.listdir(folder_dir):
 
        # check if the image ends with png
        #if (images.endswith(".jpg")) and count < 50:
            #print(images)

    while True:
        #print("round:_________", count)
        count+=1
        #print("frame: ", count)
        ret, img = cap.read()
        #print(img)
        cv2.imshow('AA', img)
            #name = "E:\\file\\openpose\\build_GPU\\examples\\tutorial_api_python\\EasyMocap-master\\cam\\images\\1\\" + images
            
        

        #out = '/Users/jacoby/Downloads/EasyMocap-master/real_time_process/image_for_json_check/image.jpg'
        #cv2.imwrite(out, img)
        #img = cv2.imread(name)
        #img = cv2.imread('"E:\file\EasyMocap-master\zju_is_feng\images\1\000000.jpg"')
        #print("----------")
        #print(img)
        #cv2.imshow('AA', img)
            # Process Image
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        #print(str(datum.poseKeypoints))
        #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        if str(datum.poseKeypoints) == 'None':
            print("No Person Detected")
            cv2.imshow("Result", img)
        #data = my_load_data(str(datum.poseKeypoints), str(datum.poseIds))
        #elif count % 2 == 1:
        else:
            data = convert_from_openpose(img, str(datum.poseKeypoints))
            #print(data)

            #data_2 = read_annot(data)
        skel = use_mv1p_my(img, data)
        print("BB")
        client.send_smpl(skel)
        print("cc")

        #cv2.imshow("Result", img_to_show)

            # Display Image
            #print("Body keypoints: \n" + str(datum.poseKeypoints))
            #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        #cv2.waitKey(1)
    
    #cap.release()
    cv2.destroyAllWindows()
    
    
except Exception as e:
    print(e)
    sys.exit(-1)
