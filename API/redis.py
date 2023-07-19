# #from aiohttp import request
# from re import S
# from django.shortcuts import render
# from django.http import StreamingHttpResponse
# import redis
# import requests

# from rest_framework.response import Response 



# #from utils.plots import Annotator, colors, save_one_box

# # from yolov7.utils.torch_utils import select_device

# # from yolov7.utils.torch_utils import select_device

# import cv2


# import torch
# from yolov5.utils.torch_utils import select_device
# from yolov5.utils.plots import Annotator, colors


# from rest_framework.decorators import api_view
# from django.http import HttpResponse
# from django.views.decorators.csrf import csrf_exempt
# from yolov5.detect import *

# import json
# #from django.http import JsonResponse
# show_vid=False,  # show results
# save_txt=False,  # save results to .txt
# save_conf=False,  # save confidences in --save-txt labels
# save_crop=False,  # save cropped prediction boxes
# save_vid=False, 
# pred_classes = torch.tensor([0,1,2,3,10]) 
# imgsz=(640, 640)
# #uploaded_file_url = 0
# source = 0
# red = redis.StrictRedis(host="127.0.0.1", port=6379, password='root@123')
# json_value = {}
# data2 = None
# frame_keys = []

def index(request): 
    
    return render(request, 'index.html')



# device = select_device(0)

# torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='visdbest.pt', force_reload = True)  # local custom model
# model.conf = 0.5



   


    



        
   


# stride, names, pt = model.stride, model.names, model.pt

# imgsz = check_img_size(imgsz, s=stride)  # check image size
# #print(outputs)
# #data_list = []
# #model.warmup(imgsz=(1 if pt else nr_sources, 3))  # warmup
# #global mydata
# #curr_frames, prev_frames = [None]  nr_sources, [None]  nr_sources




# from django.shortcuts import render  
# from django.http import HttpResponse  
# import time
# #cap = cv2.VideoCapture(0)

# # if webcam:

# # cap = cv2.VideoCapture(source)
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    
# # else:
# cap = cv2.VideoCapture('v3.MOV')



# #cap = cv2.VideoCapture(0)

# #out = cv2.VideoWriter('drone_detection.avi',cv2.VideoWriter_fourcc('XVID'), 20, (1366,720))

# seen = 0
# @torch.no_grad()
# def stream(request):
#     #global cap
    
    
#     global frame_keys
#     global json_value
#     global seen 
#     global data2
    
#     #json_value = {}
#     global ret
#     #mydata = {}
    
    
    
    
   
   
    
    
#     while True:
        
        
    
       
    
#         ret,frame = cap.read()
#         time.sleep(0.01)
        
        
#         if ret:
#             frame = cv2.resize(frame,(1366,720),interpolation=cv2.INTER_AREA)
            
#             seen = seen+1
#             # if seen%2!=0:
#             #     continue
        
            

            
                

            


#             results = model(frame, augment= True)
#             #print(dir(results))
#             #print(results.xyxy)
#         # for i in results.render():
#         #     data = im.fromarray(i[:,:,::-1])
#         #     curr_frames[i] = data.save('myimage.jpg')

#             det = results.pred[0]
#             #print(det)
#             annotator = Annotator(frame, line_width=2,pil= not ascii)
#             # if cfg.STRONGSORT.ECC:  # camera motion compensation
#             #     strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
#             #print(det)
#             if det is not None and len(det):
                
#                 for j in det:
#                     if int(j[5]) not in pred_classes:
#                         det = det[det.eq(j).all(dim=1).logical_not()]
#                 redis_data = []
#                 mylist = []
#                 for *xyxy, conf, cls in reversed(det):

                        
#                 #xywhs = xyxy2xywh(det[:, 0:4])
#                 #xyxy = det[:, 0:4]
#                 #confs = det[:, 4]
#                 #clss = det[:, 5]
                    
#                     c = int(cls)  # integer class
#                     label = f'{names[c]} {conf:.2f}'
#                     mylist.append(names[c])
#                     annotator.box_label(xyxy, label, color=colors(c, True))
#                     data = {'x1':int(xyxy[0]), 'y1':int(xyxy[1]), 'x2':int(xyxy[2]), 'y2':int(xyxy[3]),'class': names[c],'score':round(float(label.split(' ')[1]) * 100,2)}
#                     redis_data.append(data)
#                 red.hset('YOLOV5_OD',str(seen) , json.dumps(redis_data))
#                 json_value = red.hget('YOLOV5_OD',str(seen))
           
                
#                 # mydata = {
#                 #     'ID' : id,
#                 #     'Class name' : names[c]

#                 #             }
                            
                    
#                 #mylist.append(mydata)
#                 # newdata = {
#                 #     'frame no' : seen,
#                 #     'detections': mylist
#                 # }
#                         # if save_crop:
#                         #     txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
#                         #     save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
            

                
#             frame = annotator.result()
#             #frame = frame*255.0
#             #data = im.fromarray(frame[:,:,::-1])
#             _,data = cv2.imencode('.jpg',frame)
#             data = data.tobytes()
#             red.hset('YF',str(seen)+'f', data)
#             data2 = red.hget('YF', str(seen)+'f')
#             frame_keys.append(str(seen)+'f')
            
                
#                 #time.sleep(1)
#                 #print(frame)
#             #data.save('myimage.jpg')
            

#             # for i in results.render():
#             #     data = im.fromarray(i[:,:,::-1])
#             #     data.save('myimage.jpg')
#             #cv2.imwrite('myimage.jpg',frame)
            
                
                
#             #data.save('myimage.jpg')
#                 # print(data)
            
#             #cv2.imwrite('myimage.jpg',frame)
#             #annotator = Annotator(frame, line_width=2,pil= not ascii)
#             # frame = annotator.result()
#             # cv2.imwrite('images.jpg',frame)if _name_ == _main:
            
#             # cv2.waitKey(1)  # 1 millisecond
#             #print(data.tobytes)
#             #targets = output_to_target(results.render())
#             # print(dir(results))

#             #image_bytes = cv2.imencode('.jpg',frame)[1].tobytes()
#             # print(f"results are {results.names}")
#             #print(dir(results))
           
#             # pred = results.xyxyn
#             #mydict = {}
#             # for class_names in pred['name'].unique():
#             #     mydict[class_names] = pred[]

#             # print('###########################')
#             # #pred2 = results.pred
#             # print(pred2)
#             # print('###########################')
#             # #pred3 = results.xyxyn
#             # print(pred)
#             # print(results.tolist())

#             #print(pred)
#             #classes_names = pred['name']SSS
            
#             #from collections import Counter
#             #json_value = dict(Counter(mylist))
#             #red.hset('YJ',str(seen), json.dumps(json_value))
#             #json_value = json.dumps(json_value)
#             #json_value = red.hget('YJ',str(seen))
            
#             #print(json_value)
#             #json_value.update(newdata)
#             #json_value = json.dumps(json_value)
#                 #json_value = newdata
                
#                # print(classes_names)
#             # print(pred)

#             #data_list.append(pred)
#             # df2 = df.append(pred[0],ignore_index=True)
#             # df2.to_json('prediction.json')

            
#             #print('####################')
#             # print(open('myimage.jpg','rb').read())
            
#             #print(results.xywh)
#             #print(dir(results))
            
#             #yield (b'--frame\r\n'  b'Content-Type:image/jpeg\r\n\r\n' + open('myimage.jpg','rb').read() + b'\r\n')
#             #yield (b'--frame\r\n'  b'Content-Type:image/jpeg\r\n\r\n' + frame3 + b'\r\n')
#             try:
#                 yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + data2 + b'\r\n\r\n' 
#             except RuntimeError:
#                 continue
        
        
                  
                   



   
    
    


    

        

# @api_view(['GET'])
# @csrf_exempt

# def video_feed(request):
#     try:
        
#         return StreamingHttpResponse(stream(request='GET'), content_type = 'multipart/x-mixed-replace; boundary=frame')
#     except RuntimeError:
#         pass
    



# def streamjson(request):
   
#     yield json_value
        

# #@api_view(["GET"])   
# def user_stream(request):
#     x = 0
#     while True:
#         try:
#             data = red.hget('YF', str(x)+'f')

#             time.sleep(0.05)
#             #realtime_frame = red.hget('YF', str(seen)+'f')
#             x = x+1
#             yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n'
           
                
#         except RuntimeError:
#             continue
#         except TypeError:
#             continue
    
            
            
#         #print(len(data2))
        
        
            
            
        
            
            
            
          
            
            


# def users_video(request):
    
    
#     try:
        
    
#         return StreamingHttpResponse(user_stream(request), content_type = 'multipart/x-mixed-replace; boundary=frame')
#     except RuntimeError:
#         pass
    


# def myoutputs(request):
       
#     return StreamingHttpResponse(streamjson(request))



    


    
    
 

from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from yolov5.utils.plots import Annotator, colors
from collections import Counter
import cv2
import threading
import os
import time
import torch
import redis
import json

class VideoCamera(object):
    def __init(self):
        self.capture = cv2.VideoCapture('v3.MOV')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='visdbest.pt', force_reload = True)
        self.model.conf = 0.45
        # stride, names, pt = self.model.stride, self.model.names, self.model.pt
        #(self.grabbed, self.frame) = self.capture.read()
        self.pred_classes = [0,1,2,3,10]
        self.red = redis.Redis('192.168.2.102', 6379, password= 'root@123')
        # threading.Thread(target=self.update, args=()).start()

    def __del(self):
        self.capture.release()

    def get_frame(self):
        seen = 0
        global json_value
        json_value={}
        while True:
            seen = seen+1
            if self.capture.isOpened():
                (self.grabbed, self.frame) = self.capture.read()
                if self.grabbed:
                    frame = self.frame
                    frame = cv2.resize(frame,(1366,720),interpolation=cv2.INTER_AREA)

                    results = self.model(frame, augment= True)
                    det = results.pred[0]
                    # print(det)

                    annotator = Annotator(frame, line_width=2)
                    if len(det):
                        for j in det:
                            if int(j[5]) not in self.pred_classes:
                                det = det[det.eq(j).all(dim=1).logical_not()]
                         
                        
                        mylist = []
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            label = f'{self.model.names[c]} {conf:.2f}'
                            mylist.append(self.model.names[c])
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            #json_value = ({'Class':dict(Counter(mylist)).keys(),'Count':dict(Counter(mylist)).values()})
                            json_value=dict(Counter(mylist))
                            
                            self.red.lpush('YD',json.dumps(json_value))
                            #self.red.lpush('YD_C',str(json_value))
                          
                        #
                            
                            data = {'x1':int(xyxy[0]), 'y1':int(xyxy[1]), 'x2':int(xyxy[2]), 'y2':int(xyxy[3]),'class': self.model.names[c],'score':round(float(label.split(' ')[1]) * 100,2)}
                           
                        #print(redis_data)
                        
                            #self.red.lpush('YCD',str(data))
                        
                        #this is for getting json data like [{class: person, count:2} , {class : car, count : count:3}]
                        for key,values in json_value.items():
                            final_dict={}
                            final_dict['class']=key
                            final_dict['count_']=values
                            self.red.lpush('YD2',json.dumps(final_dict))
                            #l1.append(final_dict)
                       # print(final_dict)
                        
                        
                        res = annotator.result()
                        _,nframe = cv2.imencode('.jpg', res)
                        self.red.hset('YF',str(seen)+'f',nframe.tobytes())
                        return nframe.tobytes()

    # def update(self):
    #     while True:
    #         if self.capture.isOpened():
    #             (self.grabbed, self.frame) = self.capture.read()


def stream(camera,request):
    # red = redis.Redis(host="127.0.0.1", port=6379)
    while True:
        frame = camera.get_frame()
        # red.mset({"frame": frame, "data": data})
        try:
            

            if frame:
                yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except TypeError:
            continue
        except RuntimeError:
            continue

# @gzip.gzip_page
@api_view(["GET"])
def video_feed(request):
    try:
        video = VideoCamera()
        return StreamingHttpResponse(stream(video,request = 'GET'), content_type="multipart/x-mixed-replace;boundary=frame")
    except RuntimeError:
        pass
        # print("aborted")