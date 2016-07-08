//
//  main.cpp
//  OpenCVTest
//
//  Created by Versatile75 on 2016. 2. 4..
//  Copyright © 2016년 Versatile75. All rights reserved.
//

#include <iostream>
#include <stdio.h>

#include <opencv2/opencv.hpp>
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
void ShowMenu(Mat img);
void keySelectMenu(char c);

/** Global variables */
CvFont* font = new CvFont;
const float font_size = 0.3f;
const int width = 640, height = 380;
bool bVisibleMenu = 1;

String face_cascade_name = "/Users/Versatile75/Documents/XcodeWorkspace/OpenCVTest/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/Users/Versatile75/Documents/XcodeWorkspace/OpenCVTest/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Face detection Test";

void keySelectMenu(char c){
    switch (c) {
        case 'l':
            
            break;
        case 'c':
            break;
        case 'h':
            bVisibleMenu = !bVisibleMenu;
            break;
        default:
            ;
    }
}
void ShowMenu(Mat img)    //  IplImage* img
{
    //    cvInitFont(font, CV_FONT_HERSHEY_PLAIN, font_size, 1.0f, 0, 1, CV_AA);
    int MENU_HEIGHT = MARGIN_BOTTOM;
    
    putText(img, "Graduation", cvPoint(VIEW_WIDTH-88, VIEW_HEIGHT-23), 2, font_size, CV_RGB(255, 0, 0));
    putText(img, "Project", cvPoint(VIEW_WIDTH - 68, VIEW_HEIGHT - 10), 2, font_size, CV_RGB(255,0,0));
    
    if(bVisibleMenu){
        putText(img, "q - quit", cvPoint(MARGIN_LEFT, MENU_HEIGHT), 2, font_size, CV_RGB(255,0,0));
        MENU_HEIGHT += LINE_HEIGHT;
        putText(img, "h - hide menu", cvPoint(MARGIN_LEFT, MENU_HEIGHT), 2, font_size, CV_RGB(255,0,0));
        MENU_HEIGHT += LINE_HEIGHT;
        putText(img, "l - login", cvPoint(MARGIN_LEFT, MENU_HEIGHT), 2, font_size, CV_RGB(255,0,0));
        MENU_HEIGHT += LINE_HEIGHT;
        putText(img, "c - create Account", cvPoint(MARGIN_LEFT, MENU_HEIGHT), 2, font_size, CV_RGB(255,0,0));
        MENU_HEIGHT += LINE_HEIGHT;
        
    } else {
        putText(img, "h - show menu", cvPoint(MARGIN_LEFT, MENU_HEIGHT), 2, font_size, CV_RGB(255,0,0));
        MENU_HEIGHT += LINE_HEIGHT;
    }
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        
        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;
        
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
    }
    //-- Show what you got
    imshow( window_name, frame );
}

//-- 2. Read the video stream
int operateCamera(){
    VideoCapture capture;
    Mat frame;
    int c=0;
    
    capture.open( -1 );
    if ( ! capture.isOpened() ){
        printf("--(!)Error opening video capture\n"); return -1;
    }
    
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        capture.set(CV_CAP_PROP_FRAME_WIDTH, width);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, height);
        resizeWindow(window_name, VIEW_WIDTH, VIEW_HEIGHT);
        
        //-- 3. Apply the classifier to the frame
        ShowMenu(frame);
        
        detectAndDisplay( frame );
        c= cvWaitKey(10);
        if ((char)c == 27 || c == 'q') {
            break;
        }
        keySelectMenu(c);
        
    }
    return 0;
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


/** @function main */
int main( void )
{
    
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    
    chooseLoginMode();
    
    return 0;
}

