#include "darknet.h"

// Mick #include <time.h>
#include <stdlib.h>
#include <stdio.h>

void myrobot_detection(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list"); // Mick : This file does not exist ?
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    
    // double time; // Mick : Do I really need that ?

    char buff[256];
    char *input = buff;
    int j;
    float nms=.45;
#ifdef NNPACK
    nnp_initialize();
 #ifdef QPU_GEMM
    net->threadpool = pthreadpool_create(1);
 #else
    net->threadpool = pthreadpool_create(4);
 #endif
#endif

    strncpy(input, filename, 256);  // Mick : Avoid it by passing the data

#ifdef NNPACK
    image im = load_image_thread(input, 0, 0, net->c, net->threadpool);
    image sized = letterbox_image_thread(im, net->w, net->h, net->threadpool);
#else
    image im = load_image_color(input,0,0);
    image sized = letterbox_image(im, net->w, net->h);
#endif
    layer l = net->layers[net->n-1];


    float *X = sized.data;

    // Mick time=what_time_is_it_now(); // Mick : Do I need that ?

    network_predict(net, X);
    // printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time); // Mick : To remove

    int nboxes = 0; // Mick : Number of detections
    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);

    for(int i=0;i<nboxes;i++){
        // Mick : printf("Box %d at (x,y)=(%f,%f) with (w,h)=(%f,%f)\n", i, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);

        char labelstr[4096] = {0};
        char str2float[8] = {0};
        int class = -1;
        for(j = 0; j < l.classes; ++j){
            if (dets[i].prob[j] > thresh){
                if (class < 0) {
                    strcat(labelstr, names[j]);
                    sprintf(str2float," %.0f%%",(dets[i].prob[j])*100+0.5f);
                    strcat(labelstr, str2float);
                    class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                    sprintf(str2float," %.0f%%",(dets[i].prob[j])*100+0.5f);
                    strcat(labelstr, str2float);
                }
            }
        }
        if(class >= 0){

            box b = dets[i].bbox;

            float left  = (b.x-b.w/2.);
            float right = (b.x+b.w/2.);
            float top   = (b.y-b.h/2.);
            float bot   = (b.y+b.h/2.);


            if(left < 0) left = 0;
            if(top < 0) top = 0;

            printf("\{\"category\":\"%s\",\"probability\":\"%f\",\"left\":\"%f\",\"top\":\"%f\",\"right\":\"%f\",\"bottom\":\"%f\"}\n", names[class], dets[i].prob[class], left, top, right, bot);
            fflush(stdout);
        
        }

    }

    //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    //Mick if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    //Mick draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
    
    free_detections(dets, nboxes);
    // Mick save_image(im, "predictions");

    free_image(im);
    free_image(sized);


#ifdef NNPACK
	pthreadpool_destroy(net->threadpool);
	nnp_deinitialize();
#endif
	free_network(net);
}


int main(){
    //TODO Split in 2 : Init then detect
    
    myrobot_detection("cfg/coco.data", "cfg/yolov3-tiny.cfg", "yolov3-tiny.weights", "./mypic.jpg", 0.5, 0.5);
}
