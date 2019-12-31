#include "darknet.h"

#include <stdlib.h>
#include <stdio.h>

#include <sys/ipc.h> 
#include <sys/msg.h> 

#define MSG_BUFFER_MAX  1024

typedef struct  { 
    long data_type;
    char response[MSG_BUFFER_MAX]; 
} mesg_send_buffer;

typedef struct  {  
    long data_type;  
    char message[MSG_BUFFER_MAX]; 
} mesg_reicv_buffer;

typedef struct
{
    char **names;
    network *net;
} data_init;

data_init myrobot_init_detection(char *datacfg, char *cfgfile, char *weightfile)
{
    data_init init;
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list"); // Mick : This file does not exist ?
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    

#ifdef NNPACK
    nnp_initialize();
 #ifdef QPU_GEMM
    net->threadpool = pthreadpool_create(1);
 #else
    net->threadpool = pthreadpool_create(4);
 #endif
#endif

    init.names = names;
    init.net = net;

    return init;

}

void myrobot_detection(data_init init, char *filename, float thresh, float hier_thresh, char *response)
{

    char **names;
    network *net;

    char buff[256];
    char *input = buff;
    int j;
    float nms=.45;
    char responsetmp[256];

    int size=0;

    

    net = init.net;
    names = init.names;

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

    network_predict(net, X);

    int nboxes = 0; // Mick : Number of detections
    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);

    sprintf(response, "");

    for(int i=0;i<nboxes;i++){

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

            sprintf(responsetmp, "\{\"category\":\"%s\",\"probability\":\"%f\",\"left\":\"%f\",\"top\":\"%f\",\"right\":\"%f\",\"bottom\":\"%f\"}", names[class], dets[i].prob[class], left, top, right, bot);

            if(( strlen(responsetmp) + size) <  MSG_BUFFER_MAX){
                size = size + strlen(responsetmp);
                sprintf(response,"%s%s", response, responsetmp);
            }

        }

    }

    free_detections(dets, nboxes);

    free_image(im);
    free_image(sized);

}

void myrobot_clean_detection(data_init init)
{
    network *net;

    net = init.net;

#ifdef NNPACK
	pthreadpool_destroy(net->threadpool);
	nnp_deinitialize();
#endif
	free_network(net);
}


int main(){

    int msgid, reicvid;
    mesg_send_buffer message_send;
    mesg_reicv_buffer message_reicv;

    char response[MSG_BUFFER_MAX];

    char *filename;
    char id_txt[6];
    int id, i;


    if (( msgid = msgget( (key_t)1357, 0666 | IPC_CREAT)) == -1 ){
 		perror( "msgget() failed");
		exit( 1);       
    } 

    if (( reicvid = msgget( (key_t)2468, 0666 | IPC_CREAT)) == -1 ){
		perror( "msgget() failed");
		exit( 1);
	}

    

    data_init res_init;

    res_init = myrobot_init_detection("cfg/coco.data", "cfg/yolov3-tiny.cfg", "yolov3-tiny.weights");
    
    while(1){

		if (msgrcv( reicvid, &message_reicv, sizeof( mesg_reicv_buffer) - sizeof(long), 0, 0) == -1){
			perror( "msgrcv() failed");
		}

        i=0;
        while(message_reicv.message[i] != ';'){
            printf("I: %d %c\n", i, message_reicv.message[i]);fflush(stdout);
            i++;
            if(i > 5){
                i=-1;
                break;
            }
        }

        

        if( i > 0){

            strncpy( id_txt, message_send.response, i );
            filename = message_reicv.message + i + 1;
            id = atoi(id_txt);

            myrobot_detection(res_init, filename, 0.5, 0.5, message_send.response);
            message_send.data_type = 1;


            if (msgsnd( msgid, &message_send, sizeof(mesg_send_buffer) - sizeof(long), 0) == -1)
            {
                perror( "msgsnd() failed");
                exit( 1);
            }
            
        }
    }
    
    myrobot_clean_detection(res_init);

}
