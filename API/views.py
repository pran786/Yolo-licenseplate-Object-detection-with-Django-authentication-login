import imutils
import numpy as np
import sys
from pathlib import Path
from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponseRedirect
import requests
from rest_framework.response import Response 
import cv2
import torch
from .models import Image
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors,save_one_box
from rest_framework.decorators import api_view
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from yolov5.detect import *
import json
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
import base64
from yolov5.utils.general import xyxy2xywh,increment_path,check_img_size
from django.http import JsonResponse
from base64 import b64encode,b64decode
import threading
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login , logout
from django.shortcuts import render, redirect
from django.urls import reverse
import easyocr
#import redis
#red = redis.Redis('127.0.0.1', 6379, password= 'root@123')
thread_local_models = {}
model_path_2 = 'yolov5s.pt'
model_path_1 = 'LPD.pt'
project = 'EAGLE_WATCH'
name = 'crops'
apihit = None
exist_ok = True
save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
(save_dir).mkdir(parents=True, exist_ok=True) 
def load_model(model_path):
    # Initialize the YOLOv5 model
   
    model = torch.hub.load('yolov5', 'custom', path=model_path,source='local', force_reload = True)  # local custom model

    # Store the model in a thread-local storage
    thread_local_models[model_path] = model

# Define a function to initialize the YOLOv5 models in separate threads
def initialize_models():
    # Create a thread-local storage for the models
    global model_path_1,model_path_2

    # Create a thread for each model and load the model in the thread
    t1 = threading.Thread(target=load_model, args=(model_path_1,))
    t2 = threading.Thread(target=load_model, args=(model_path_2,))
    t1.start()
    t2.start()

    # Wait for the threads to finish loading the models
    t1.join()
    t2.join()

# Initialize the YOLOv5 models in a separate thread when the server starts
#initialize_models()

device = select_device('cpu')
show_vid=False,  # show results
save_txt=False,  # save results to *.txt
save_conf=False,  # save confidences in --save-txt labels
save_crop=False,  # save cropped prediction boxes
save_vid=False, 

dettype = None
global btnval
btnval = 'false'
#pred_classes = torch.tensor([0,1,2,3,10]) 
imgsz=(1280, 1280)
#uploaded_file_url = 0
source = 0
data2 = None
frame_keys = []
dnn = False
half = True
cap = None
ipcamera_url = None
reader = easyocr.Reader(['en'])
@login_required()
@api_view(['GET','POST'])
def index(request): 
    global ipcamera_url,dataset
    global dettype,apihit,model_1,model_2
    if apihit is not None:
        print('api is already working')

    if request.method=='POST' and apihit is None:
        apihit = None
        
        formdata = json.loads(json.dumps(request.data))
        #print(formdata)
        ipcamera_url = formdata['username']
        ipcamera_url  = str(ipcamera_url)
        print(ipcamera_url)
        
        
        
        if 'license' in formdata.keys():
            dettype = formdata['license']
            #model_1 = DetectMultiBackend(model_path_1, device=device, dnn=dnn, data=ipcamera_url, fp16=half)
            #imgsz = check_img_size(imgsz, s=stride)
            #stride1, names, pt = model.stride, model.names, model.pt
            #imgsz = check_img_size(imgsz, s=stride)  # check image size
            #model_1 = thread_local_models[model_path_1]
            model_1 = torch.hub.load('yolov5', 'custom', path=model_path_1,source='local', force_reload = True)  # local custom model
            model_1.to(device)
        else:
            #model_2 = DetectMultiBackend(model_path_2, device=device, dnn=dnn, data=ipcamera_url, fp16=half)
            #imgsz = check_img_size(imgsz, s=stride)
            #model_2 = thread_local_models[model_path_2]
            model_2 = torch.hub.load('yolov5', 'custom', path=model_path_2,source='local', force_reload = True)  # local custom model
            model_2.to(device)
            dettype = formdata['objects']
        
        # password = formdata['password']
        # cameraip = formdata['ip-address']
        # ipport = formdata['port']
        #ipcamera_url = f'rtsp://{username}:{password}@{cameraip}:{ipport}/11'
        try:
            
            return HttpResponseRedirect('/stream_video')
            
            #return StreamingHttpResponse(stream(request='GET'), content_type = 'multipart/x-mixed-replace; boundary=frame')
        except RuntimeError:
            pass

    return render(request, 'index.html')

#device = select_device('cpu')
#torch.hub.set_dir('yolov5')
#torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
#model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', force_reload = True)  # local custom model
#model = torch.hub.load('ultralytics/yolov5','yolov5s')
# model_1.conf = 0.20
# stride, names, pt = model_1.stride, model_1.names, model_1.pt
# imgsz = check_img_size(imgsz, s=stride)  # check image size
from django.shortcuts import render  
from django.http import HttpResponse  
import time

global json_value
json_value = {}
# #out = cv2.VideoWriter('drone_detection.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, (1366,720))

seen = 0
@torch.no_grad()
def stream(request):
    global frame_keys,ipcamera_url
    global seen,model_1,model_2 
    global data,mydata,mydata2
    global ret,cap,btnval,apihit
    mydata = {}
    btnval = 'false'
    print(ipcamera_url)
    print(cap)
    time.sleep(0.0)
    cap = cv2.VideoCapture(ipcamera_url)
    time.sleep(1)
    #initialize_models() 
    print(cap,'getting cap object')
    # model_1 = thread_local_models[model_path_1]  
    # model_2 = thread_local_models[model_path_2] 
    # if dettype is not None and dettype=='lic':
    #     model = torch.hub.load('yolov5', 'custom', path='../LPD.pt', force_reload = True)  # local custom model

    
    while True: 
        
        mydata2 = []
        if cap is None: 
            apihit = None
            return HttpResponseRedirect('/') 
        ret,frame = cap.read()  
        #print(ipcamera_url) 
        
        if ret:
            #frame = cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
            #frame = frame/255
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
            #print(frame.shape)
            imc = frame.copy()
            seen = seen+1
            
            if seen%2!=0:
                continue
            if dettype is not None and dettype=='lic':
                # frame = frame.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
                # frame = np.ascontiguousarray(frame)  # contiguous

                #frame = torch.from_numpy(frame).to(device)
                #frame /= 255  # 0 - 255 to 0.0 - 1.0
                # if len(frame.shape) == 3:
                #     frame = frame[None]  # expand for batch dim
                names = model_1.names

                results = model_1(frame,size = 640, augment= True)
                # print(results.render())
                # for i in results.render():
                #     frame = i
                    #cv2.imwrite('finalimg2.jpg',frame)

                    

                det = results.pred[0]
                    
                annotator = Annotator(frame, line_width=2,pil= not ascii)
                if det is not None and len(det):
                    mylist = []
                    for *xyxy, conf, cls in reversed(det): 
                        #print(xyxy)
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        mylist.append(names[c])
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        save_one_box(xyxy, imc, file= save_dir / f'{seen}.jpg', BGR=True)
                        
                        #result = reader.readtext(r'EAGLE_WATCH\crops\4.jpg')
                        #print(result)
                        #text = ' '.join([res[1] for res in result])
                        #print(text)
                # frame = ocr_image(frame)
                # print('this code works')
                # print(frame.shape)
                
            
            
            else:
                
                names = model_2.names
                results = model_2(frame, augment= True)
                det = results.pred[0]
                annotator = Annotator(frame, line_width=2,pil= not ascii)
                if det is not None and len(det):
                    mylist = []
                    for *xyxy, conf, cls in reversed(det): 
                        #print(xyxy)
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        mylist.append(names[c])
                        annotator.box_label(xyxy, label, color=colors(c, True))
        

                    #mydata = {'x1':int(xyxy[0]), 'y1':int(xyxy[1]), 'x2':int(xyxy[2]), 'y2':int(xyxy[3]),'class': names[c],'score':round(float(conf) * 100,2)}
                    #mydata2.append(mydata)

                
            frame = annotator.result()
            #frame = frame*255
            cv2.imwrite('finalimg.jpg',frame)
            #frame = frame*255.0
            #data = im.fromarray(frame[:,:,::-1])
            _,data = cv2.imencode('.jpg',frame)
            data = data.tobytes()
            Image.objects.create(image_data=data)
            #Image.objects.exclude(pk=Image.objects.latest('created_at').pk).delete()
            #red.hset('YF',str(seen),data)
            #json_value['frame_no ' + str(seen)] = mydata2
            if btnval=='true':
                break
                print(btnval, 'this is the stream need')
                #return HttpResponseRedirect('/')
            elif btnval=='new':
                break
                #return HttpResponseRedirect('/')

        
            try:
                yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n' 
            except:
                continue
                #cap.open(ipcamera_url)
                #continue
        

@api_view(['GET'])
@csrf_exempt
def video_feed(request):
    try:
        return StreamingHttpResponse(stream(request='GET'), content_type = 'multipart/x-mixed-replace; boundary=frame')
    except:
        pass
        # cap.open(ipcamera_url)
    

#@api_view(["GET"])   
def user_stream(request):
    #x = 0
    #Image.objects.all().delete()
    while True:
        try:
            Image.objects.exclude(pk=Image.objects.latest('created_at').pk).delete()
            #data = red.hget('YF', str(x)+'f')
            time.sleep(0.05)
            data3 = Image.objects.latest('created_at').image_data
            #data = Image.objects.last().image_data
            #x = x+1
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + data3 + b'\r\n\r\n'   
        except RuntimeError:
            continue
        except TypeError:
            continue
    
            
            
        #print(len(data2))
        
        
            
            
        
            
            
            
          
            
            


def users_video(request):
    try:
        return StreamingHttpResponse(user_stream(request), content_type = 'multipart/x-mixed-replace; boundary=frame')
    except RuntimeError:
        pass
    

   

 
@api_view(['POST'])
def stop_video(request):
    global btnval,apihit
    if request.method=='POST':
        apihit = None
        btndata = json.loads(json.dumps(request.data))
        btnval = btndata['stop']
        if btnval=='true':
            print(btnval, 'this is the val we need')
            return HttpResponseRedirect('/home')
        elif btnval=='new':
            return HttpResponseRedirect('/home')

@api_view(['GET','POST'])
def stream_video(request):
    # if request.method=='POST':
    #     btndata = json.loads(json.dumps(request.data))
    #     btnval = btndata['stop']
    #     print(btnval, 'this is the val we need')
    #     return HttpResponseRedirect('')
    return render(request, 'video_stream.html')


def login_view(request):
    if request.user.is_authenticated:
        return redirect(reverse('index'))
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        print(email,password)
        user = authenticate(request, username=email, password=password)
        print(user)
        if user is not None:
            login(request, user)
            return redirect(reverse('index'))
        else:
            return render(request, 'login.html', {'error': 'Invalid login details'})
    else:
        return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect(reverse('login'))

@api_view(['GET','POST'])
def livestream(request):
    return render(request, 'live.html')


# @torch.no_grad()
# def stream2(request):
#     bs = len(dataset)
#     print(bs)
#     if dettype is not None and dettype=='lic':
#         model = DetectMultiBackend(model_path_1, device=device, dnn=dnn, data=ipcamera_url, fp16=half)
#     else:
#         model = DetectMultiBackend(model_path_2, device=device, dnn=dnn, data=ipcamera_url, fp16=half)

#     #model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
#     seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
#     for path, im, im0s, vid_cap, s in dataset:
        
#         with dt[0]:
#             im = torch.from_numpy(im).to(model.device)
#             im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#             im /= 255  # 0 - 255 to 0.0 - 1.0
#             if len(im.shape) == 3:
#                 im = im[None]  #
#         with dt[1]:
#             #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#             pred = model(im, augment=True)

#         # NMS
#         with dt[2]:
#             pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

#         # Process predictions
#         for i, det in enumerate(pred):  # per image
#             seen += 1
            
            
#             p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             #imc = im0.copy() if save_crop else im0  # for save_crop
#             annotator = Annotator(im0, line_width=2, example=str(names))
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

#                 # Print results
                
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     # Add bbox to image
#                     c = int(cls)  # integer class
#                     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
#                     annotator.box_label(xyxy, label, color=colors(c, True))
                


#         # Stream results
#         frame = annotator.result()
#         cv2.imwrite('finalimg.jpg',frame)
#         #frame = frame*255.0
#         #data = im.fromarray(frame[:,:,::-1])
#         _,data = cv2.imencode('.jpg',frame)
#         data = data.tobytes()
#         Image.objects.create(image_data=data)
#         #Image.objects.exclude(pk=Image.objects.latest('created_at').pk).delete()
#         #red.hset('YF',str(seen),data)
#         #json_value['frame_no ' + str(seen)] = mydata2
#         if btnval=='true':
#             break
#             print(btnval, 'this is the stream need')
#             #return HttpResponseRedirect('/')
#         elif btnval=='new':
#             break
#             #return HttpResponseRedirect('/')


#         try:
#             yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n' 
#         except:
#             continue
#             #cap.open(ipcamera_url)
#             #continue
           













