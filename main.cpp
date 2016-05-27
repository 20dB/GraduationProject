/*
 기능 설계 5/24
 
 main
 
 1. 사용자 등록(최대인원 5명제한) -> 카메라 작동 -> xml기반으로 얼굴영역찾기 -> 얼굴영역 캡쳐로 얼굴사진파일생성 (살짝왼쪽, 살짝오른쪽 옆얼굴, 정면 대략 10장정도.) -> 사진크기조정(100*149) -> 사진데이터베이스를 사용해서 평균얼굴파일생성. -> 사용자비밀번호등록
 
 2. 로그인 -> 얼굴로로그인할것인지, passward로 로그인할것인지 -> 카메라 작동 -> xml기번으로 얼굴영역찾기
 -> 얼굴영역 캡쳐로 얼굴사진파일생성(한5초동안 사진5장정도 생성) -> 사진크기조정 -> DB의등록된 평균얼굴들과 찍은사진 비교로 판단.
 
 */

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cstdio>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#define LINE_HEIGHT		12
#define MARGIN_LEFT		10
#define MARGIN_BOTTOM	10
#define VIEW_WIDTH	640
#define VIEW_HEIGHT	480


using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
void ShowMenu(IplImage* img);
void keySelectMenu(char c);
bool bVisibleMenu = 1;

/** Global variables */
CvFont* font = new CvFont;
const float font_size = 0.7f;
const int width = 640, height = 480;

/**
 카메라를 실행하는 함수.
 현재 카메라가 촬영하는 화면을 출력.
 */
 int operateCamera(){
     
     int select=0;
     int c = 0;
     IplImage *frame;
     IplImage *image = 0;
     IplImage *ori_image = 0;
     IplImage *view_image = 0;
     
     cvInitFont(font, CV_FONT_HERSHEY_PLAIN, font_size, 1.0f, 0, 1, CV_AA);
     
     CvCapture* capture = cvCaptureFromCAM(0);
     cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH, width);
     cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT, height);
     
     cvNamedWindow("Test",0);
     cvResizeWindow("Test", VIEW_WIDTH, VIEW_HEIGHT);
     
     
     while(capture) {
         ori_image = cvQueryFrame( capture );
         image = cvCloneImage(ori_image);
         
         if (!view_image) {
             view_image = cvCreateImage(cvSize(VIEW_WIDTH, VIEW_HEIGHT), image->depth, image->nChannels);
         }
         cvResize(image, view_image, CV_INTER_LINEAR);
         
//         frame = cvQueryFrame(capture);
//         cvShowImage("Test",frame);
         ShowMenu(view_image);
        
         c = cvWaitKey(10);
         if((char)c == 27 || c =='q')
             break;
         keySelectMenu(c);
         view_image->origin = image->origin;
         cvShowImage("Test", view_image);
         cvReleaseImage(&image);
     }
     
     cvReleaseCapture(&capture);
     cvDestroyWindow("Test");
     delete font;
     return 0;
}

/**
 카메라를 실행하는 함수.
 현재 카메라가 촬영하는 화면을 출력.
 */
void detectAndDisplay( Mat frame )
{

}
void keySelectMenu(char c){
    switch (c) {
        case 'l':
            
            break;
        case 'c':
            break;
        case 'm':
            bVisibleMenu = !bVisibleMenu;
            break;
        default:
            ;
    }
}
void ShowMenu(IplImage* img)
{
    int MENU_HEIGHT = MARGIN_BOTTOM;
    
    cvPutText(img, "Graduation", cvPoint(VIEW_WIDTH - 88, VIEW_HEIGHT - 23), font, CV_RGB(255,0,0));

    cvPutText(img, "Project", cvPoint(VIEW_WIDTH - 68, VIEW_HEIGHT - 10), font, CV_RGB(255,0,0));

    
    if(bVisibleMenu){
        cvPutText(img, "q - quit", cvPoint(MARGIN_LEFT, MENU_HEIGHT), font, CV_RGB(255,0,0));
        MENU_HEIGHT += LINE_HEIGHT;
        
        cvPutText(img, "m - hide menu", cvPoint(MARGIN_LEFT, MENU_HEIGHT), font, CV_RGB(255,0,0));
        MENU_HEIGHT += LINE_HEIGHT;
        
        cvPutText(img, "l - login", cvPoint(MARGIN_LEFT, MENU_HEIGHT), font, CV_RGB(255,0,0));
        MENU_HEIGHT += LINE_HEIGHT;
        cvPutText(img, "c - create Account", cvPoint(MARGIN_LEFT, MENU_HEIGHT), font, CV_RGB(255,0,0));
        MENU_HEIGHT += LINE_HEIGHT;

    } else {
        cvPutText(img, "m - show menu", cvPoint(MARGIN_LEFT, MENU_HEIGHT), font, CV_RGB(255,0,0));
        MENU_HEIGHT += LINE_HEIGHT;
    }
}

/**
 얼굴로 로그인할 경우 1
 비밀번호로 로그인할 경우 2
 */
int chooseLoginMode(){
    int mode=0;
    printf("모드를 선택합니다.\n");
    printf("1. 로그인\n2. 새로운 사용자 등록\n3. 프로그램 종료\n");
    scanf("%d", &mode);
    switch (mode) {
        case 1:
            operateCamera();
            return 1;
            break;
        case 2:
            operateCamera();
            return 1;
            break;
        case 3:
            printf("프로그램을 종료합니다 ㅃㅇ!\n");
            exit(0);
            return 1;
            break;
            
        default:
            printf("잘못된 모드 선택 입니다. 다시 선택해주세요!\n");
            chooseLoginMode();
            return 0;
            break;
    }
}

int main(int argc, const char * argv[]) {
    
    chooseLoginMode();
    
    return 0;

}