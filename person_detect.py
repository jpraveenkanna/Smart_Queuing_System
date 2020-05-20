

import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys

class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
           
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

       
        
    def load_model(self):
        
        core = IECore()
        self.net = core.load_network(network=self.model, device_name=self.device, num_requests=1)
        

        
    def predict(self, image):
        
        net_input = self.preprocess_input(image)
        start=time.time()
        self.res = self.net.infer(net_input)
        inference_time=time.time()-start
        fps=100/inference_time

        output_image,coordinate = self.draw_outputs(image)
        
        cv2.putText(output_image, f"Inference time: {round(inference_time,3)}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,153,153), 2)
        
        return output_image,coordinate
        
        
    
    def draw_outputs(self,image):
        #image = cv2.imread(args.image)  #uncomment for image
        data = self.res['detection_out'][0][0]
        ih, iw = image.shape[:-1]
        tmp_image = image
        boxes=[]
        probability=[]

        for number, infer_results in enumerate(data):
            if (np.int(infer_results[1]) == 1 and infer_results[2] > self.threshold):

                xmin = np.int(iw * infer_results[3])
                ymin = np.int(ih * infer_results[4])
                xmax = np.int(iw * infer_results[5])
                ymax = np.int(ih * infer_results[6])
                boxes.append([xmin, ymin, xmax, ymax])
                probability.append([infer_results[2]])

        for box,prob in zip(boxes,probability):
            
            cv2.rectangle(tmp_image, (box[0], box[1]), (box[2], box[3]), (232, 35, 244), 2)
            mid_x,mid_y =  (box[2]+(box[0]-box[2])//2),(box[3]+(box[1]-box[3])//2)
            

        
        return tmp_image,boxes



    def preprocess_input(self, image):
        #image = cv2.imread(args.image) #uncomment for image
        n,c,h,w = self.input_shape
        images = np.ndarray(shape=(n, c, h, w))
        image_shape = (image.shape[1], image.shape[0])
        resized_input_image = cv2.resize(image, (w, h))
        resized_input_image = resized_input_image.transpose((2, 0, 1))  
        images[0] = resized_input_image
        net_input = {'data': images[0]}
        return net_input


def main(args):
    #args =arguments()
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path
    queue_param = args.queue_param


    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()

    total_model_load_time = time.time() - start_model_load_time
    print("Total_model_load_time: ",total_model_load_time)

    queue=Queue()


    try:
        queue_param=np.load(queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")


    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
         print("Cannot locate video file: ", video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    #out_video = cv2.VideoWriter( 'Manufacturing_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (initial_w, initial_h), True)

    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break

            counter+=1
            alpha = 0.7



            res_image, coords = pd.predict(frame)


            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")

            out_text=""


            
            cv2.putText(res_image, f"Total Person on screen: {len(coords)}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,153,153), 2)

            overlay = res_image.copy()

            for idx,q in enumerate(queue.queues):
                    overlay = cv2.rectangle(overlay, (q[0], q[1]), (q[2], q[3]), (0, 255, 0), 5)
                    cv2.rectangle(overlay, (q[0], q[1]), (q[0] + 240, q[1] + 80), (0,0,0), -1)
                    cv2.putText(overlay, f"Queue ID: {idx+1}", (q[0]+5, q[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),4)  #blue,green,red 

                    if idx==0:
                        v = num_people[1]
                    elif idx ==1:
                        v = num_people[2]

                    out_text += f"Persons: {v} "
                    if v >= int(max_people):
                        msg = f"Queue full; Please move to next Queue "
                        cv2.rectangle(overlay, (q[0], q[1]+100), (q[0] + 650, q[1] + 150), (0,0,0), -1)
                        cv2.putText(overlay, msg, (q[0]+5, q[1]+135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)  #blue,green,red 


                    cv2.putText(overlay, out_text, (q[0]+5, q[1]+70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    out_text=""


            res_image = cv2.addWeighted(overlay, alpha, res_image, 1 - alpha, 0)    
            res_image = cv2.resize(res_image, (initial_w, initial_h)) 
            out_video.write(res_image)

        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        out_video.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

   
    main(args)