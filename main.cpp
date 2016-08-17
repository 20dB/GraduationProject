//
//  main.cpp
//  GraduationProject2016
//
//  Created by Versatile75 on 2016. 8. 15..
//  Copyright © 2016년 Versatile75. All rights reserved.
//

//  알고리즘 선택과정?
//  얼굴의 좌우 선처리과정을 각각/따로?
//const bool preprocessLeftAndRightSeparately = true;
//  getPreprocessedFace 함수에서 매개변수 제거함.
//  int scaledWidth 대신 const int DETECTION_WIDTH = 320; 사용?


//  ESC키 윈도우설정?
//#if !defined WIN32 && !defined _WIN32
//#define VK_ESCAPE 0x1B      // Escape character (27)
//#endif

#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

//  LBP Cascade 분류기 설정.
//  가장 정확한게 left/righteye_2splits, 그다음이 mcs_left/righteye, 기본이 haarcascade_eye/eye_tree_eyeglasses
const char *faceCascadeFilename = "/Users/Versatile75/Documents/XcodeWorkspace/GraduationProject2016/GraduationProject2016/lbpcascade_frontalface.xml";
const char *eyeCascadeFilename1 = "/Users/Versatile75/Documents/XcodeWorkspace/GraduationProject2016/GraduationProject2016/haarcascade_lefteye_2splits.xml";
const char *eyeCascadeFilename2 = "/Users/Versatile75/Documents/XcodeWorkspace/GraduationProject2016/GraduationProject2016/haarcascade_righteye_2splits.xml";

const char *facerecAlgorithm = "FaceRecognizer.LBPH";    //  얼굴인식 알고리즘 선택 -> LBP

//  등록된 사람인지 아닌지를 결정하는 얼굴 인식 알고리즘의 신뢰도를 설정. 높아질수록 등록된 사람으로 판단
const float UNKNOWN_PERSON_THRESHOLD = 0.7f;

//  얼굴이미지의 크기를 설정.
//  getPreprocessedFace()함수가 정사각형 얼굴을 리턴하기때문에 faceWidth=faceHeight.
const int faceWidth = 70;
const int faceHeight = faceWidth;

//  카메라의 해상도를 설정. 카메라/시스템에따라 안맞을수도있음
const int CAMERA_WIDTH = 640;
const int CAMERA_HEIGHT = 480;

//  얼마나 자주 새로운 얼굴을 저장할지를 정하는 파라메터.
const double CHANGE_PARAMETER_IMAGE = 0.3;      //  트레이닝할 때, 얼마나 자주 얼굴 이미지가 바뀌어야하는지를 설정
const double CHANGE_PARAMETER_SECOND = 1.0;       //   트레이닝할 때, 시간이 얼마나 지나야 하는지를 설정.

const char *windowName = "Facial Recognition";   // GUI 화면창의 이름
const int BORDER = 8;  //   GUI 엘리먼트들과 사진의 모서리간의 경계값

//  각 과정에서의 이미지를 창으로 확인하고싶으면 true, 아니면 false
bool m_debug = false;

//  GUI상에 나타나는 모드 설정
enum USER_MODES {MODE_STARTUP=0, MODE_DETECTION, MODE_COLLECT_FACES, MODE_TRAINING, MODE_RECOGNITION, MODE_DELETE_ALL, MODE_END};
const char* MODE_NAMES[] = {"Running", "Detection", "Collect Faces", "Training", "Recognition", "Delete All", "ERROR!"};
USER_MODES m_mode = MODE_STARTUP;

//  유저수 초기화
int m_selectedUser = -1;
int m_numUsers = 0;
vector<int> m_latestFaces;

//  GUI 버튼의 위치 설정
Rect m_rectangleBottonAdd;
Rect m_rectangleBottonDel;
Rect m_rectangleBottonDebug;
int m_gui_faces_left = -1;
int m_gui_faces_top = -1;

/**
 물체 찾기에 연관된 함수들
 - 입력데이터는 빠른 탐지를 위해서, scaledWidth 만큼 축소되어있음. 얼굴찾기용으로는 240이 적당.

 1. detectObjectsCustom 함수
    주어진 파라메터값을 이용해서 얼굴과 같은 이미지 오브젝트를 찾는 함수. 여러개를 찾아서 objets로 저정함.
 2. detectLargestObject 함수
    이미지에서 가장 큰 얼굴과같은 하나의 오브젝트를 찾는 함수. 찾은 오브젝트를 largestObject에 저장함.
 3. detectManyObjects 함수
    이미지에서 모든얼굴처럼 다수의 오브젝트를 찾는 함수. 찾은 오브젝트를 objects에 저장.
 */
void detectObjectsCustom(const Mat &image, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors);
void detectLargestObject(const Mat &image, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth = 320);
//void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth = 320);

/**
 얼굴 이미지 사전 처리에 관련된 함수들
 
 1. detectEyes 함수
    주어진 이미지로부터 두눈을 찾아내는 함수. 각각의 눈의 중점 좌표를 leftEye 와 rightEye 로 리턴. 실패시 (-1, -1)로 반환.
    찾은 왼쪽 오른쪽 눈의 영역을 저장 가능... (추가해야하나..)
 2. equlizeLeftAndRightHalf 함수
    얼굴의 양쪽을 각각 히스토그램으로 평준화하는 함수. 얼굴 한쪽면에만 빛을 받을경우 이거로 평준화함.
 3. getPreprocessedFace 함수
    흑백이미지로 주어진 이미지를 변환. srcImage 매개변수는 전체 카메라 프레임의 복사본. 그래야 눈의 좌표를 그릴수 있음.
    선처리 과정에는 다음과정들이 포함됨.
        1. 눈 탐지를 통한 비율 줄이기, 회전과 트랜슬레이션.
        2. Bilateral 필터를 사용한 이미지의 노이즈 제거
        3. 히스토그램 평준화를 얼굴 왼쪽 오른쪽에 각각 적용해서 밝기를 평준화.
        4. 타원형으로 얼굴 마스크를 잘라서 배경과 머리 지우기.
    선처리된 얼굴 정사각형 이미지를 리턴. 실패시 NULL == 눈과 얼굴을 찾지 못한경우.
//    얼굴이 찾아지면, 얼굴 직사각형 좌표는 storeFaceRect에 저장하고, 눈은 storeLeftEye, storeRightEye에 각각 저장. 눈 영역은 searchedLeftEye와 searchedRightEye에 저장.
 */
void detectEyes(const Mat &faceImage, CascadeClassifier&eyeDetector1, CascadeClassifier &eyeDetector2, Point &leftEyeCenter, Point &rightEyeCenter/*, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL*/);
void equalizeLeftAndRightHalfFace(Mat &faceImage);
Mat getPreprocessedFace(Mat &srcImage, int desiredFaceWidth, CascadeClassifier &faceDetector, CascadeClassifier &eyeDetector1, CascadeClassifier &eyeDetector2, Rect *storeFaceRect = NULL, Point *storeLeftEyeCenter = NULL, Point *storeRightEyeCenter = NULL/*, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL*/);

/**
 수집한 데이터들로 얼굴인식 프로그램을 학습시키고 이를통해 인식하는 함수.
 
 1. learnCollectedFaces 함수
    수집된 얼굴들로 트레이닝 하는 함수.
 2. showTrainingDebugData 함수
    디버깅용 내부 얼굴인식 데이터를 보여주는 함수.
 3. reconstructFace 함수
    선처리된 얼굴의 아이젠벡터와 아이젠값을 사용해서 원 얼굴을 근사적으로 재건하는 함수.
 4. getSimilarity 함수
    두개의 이미지의 유사도를 비교하는 함수. L2 Error를 사용. (Square-root of sum of squared error)
 */
Ptr<FaceRecognizer> learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels, const string facerecAlgorithm = "FaceRecognizer.LBPH");
void showTrainingDebugData(const Ptr<FaceRecognizer> model, const int faceWidth, const int faceHeight);
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace);
double getSimilarity(const Mat A, const Mat B);

/**
 프로그램 초기 설정 및 GUI관련 함수
 
 1. loadXMLs 함수.
 */
void loadXMLs(CascadeClassifier &faceDetector, CascadeClassifier &eyeDetector1, CascadeClassifier &eyeDetector2);
void setCamera(VideoCapture &videoCapture, int cameraNumber);
Rect drawTextString(Mat img, string text, Point coord, Scalar color, float fontScale, int thickness, int fontFace);
Rect drawButton(Mat img, string text, Point coord, int minWidth);
bool isPointInRect(const Point pt, const Rect rc);
void onMouse(int event, int x, int y, int, void*);
void recognizeAndTrain(VideoCapture &videoCapture, CascadeClassifier &faceDetector, CascadeClassifier &eyeDetector1, CascadeClassifier &eyeDetector2);

// int나 float를 string으로 전환하는 함수
template <typename T> string toString(T t) {
    ostringstream out;
    out << t;
    return out.str();
}

template <typename T> T fromString(string t) {
    T out;
    istringstream in(t);
    in >> out;
    return out;
}

void detectObjectsCustom(const Mat &image, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors){
    cerr << "2" << endl;
    
    //  입력 이미지가 흑백이 아니므로, BGRA 형태의 흑백사진으로 바꿈
    Mat grayImage;
    if (image.channels() == 3) {  //  3채널 이미지면 BGR
        cvtColor(image, grayImage, CV_BGR2GRAY);
    }
    else if(image.channels() == 4){   //  4채널이면 BGRA
        cvtColor(image, grayImage, CV_BGRA2GRAY);
    }
    else {
        //  이미 흑백 사진이므로
        grayImage = image;
    }
    
    //  속도를 빠르게 하기위해 이미지 크기를 줄임
    Mat smallImage;
    float scale = image.cols / (float)scaledWidth;
    if (image.cols > scaledWidth) {    //  원본 사진의 비율을 유지하기 위해
        int scaleHeight = cvRound(image.rows / scale);
        resize(grayImage, smallImage, Size(scaledWidth, scaleHeight));
    }
    else {  //  이미 충분히 작으므로 그냥 진행
        smallImage = grayImage;
    }
    
    //  이미지의 밝기와 명암을 평준화
    Mat equalizedImage;
    equalizeHist(smallImage, equalizedImage);
    cerr << "3" << endl;
    
    //  전처리된 이미지에서 오브젝트 찾기
    cascade.detectMultiScale(equalizedImage, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);
    cerr <<"4" << endl;
    
    //  크기 복구
    if (image.cols > scaledWidth) {
        for (int i =0; i < (int)objects.size(); i++) {
            objects[i].x = cvRound(objects[i].x * scale);
            objects[i].y = cvRound(objects[i].y * scale);
            objects[i].width = cvRound(objects[i].width * scale);
            objects[i].height = cvRound(objects[i].height * scale);
        }
    }
    
    //  cvRound를 사용해서 확대했기때문에 혹시나 전체 사진의 범위를 넘는것을 대비
    for (int i=0; i<(int)objects.size(); i++) {
        if (objects[i].x < 0)
            objects[i].x = 0;
        if (objects[i].y < 0)
            objects[i].y = 0;
        if (objects[i].x + objects[i].width > image.cols)
            objects[i].x = image.cols - objects[i].width;
        if (objects[i].y + objects[i].height > image.rows)
            objects[i].y = image.rows - objects[i].height;
    }
    cerr << "4.1" << endl;
    
    //  함수가 종료되면, objects 벡터 배열안에 찾아진 얼굴영상들을 가진채로 리턴됨.
}

//  주어진 이미지에서 딱하나의 가장 큰 얼굴 오브젝트를 찾는 함수.
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth){
    int flags = CASCADE_FIND_BIGGEST_OBJECT;    //  모든 얼굴을 조사할지, 가장 큰 얼굴만 찾을지를 결정.

    Size minFeatureSize = Size(20, 20); //  얼굴 크기의 최소를 결정. 빠르게 검출하려면 80 X 80
    float searchScaleFactor = 1.1f; //  찾으려는 얼굴의 크기를 몇가지 다른 크기로 할지를 결정. 빠르게 검출하려면 1.2
    int minNeighbors=4;     //  false detecion 수치를 결정하는 neighbor 설정. 검출한 얼굴이 확실한지 결정. 6이면 덜 정확한 얼굴을 더 많이
    
    //  1개의 가장큰 얼굴 오브젝트 찾기
    vector<Rect> objects;
    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
    if (objects.size()>0) {
        //  오브젝트를 찾았으므로 리턴
        largestObject = (Rect)objects.at(0);
    }
    else{
        //  오브젝트를 못찾았으므로 쓰레기값 리턴
        largestObject = Rect(-1, -1, -1, -1);
    }
    cerr << "4.2" << endl;
}

//  주어진 이미지에서 다수의 얼굴 오브젝트를 찾는 함수.
//void detectManyObjects(const Mat &image, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth){
//    //  다수의 얼굴 오브젝트를 찾음.
//    int flags = CASCADE_SCALE_IMAGE;
//    //  최소크기 오브젝트 정의
//    Size minFeatureSize = Size(20, 20);
//    //  얼마나 자세히 찾을지를 결정하는 파라메터. 1.0이상
//    float searchScaleFactor = 1.1f;
//    
//    //  false detecion 수치를 결정하는 neighbor 설정. 2이면 덜 자세하게 찾음.
//    int minNeighbors=4;
//    
//    //  다수의 얼굴 오브젝트 찾기
//    detectObjectsCustom(image, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
//}

//  주어진 face 이미지에서 두 눈을 각각 찾아내는 함수. 눈을 찾으면 눈의 중심죄표 leftEyeCenter와 rightEyeCenter 리턴. 찾지 못하면 (-1, -1)반환.
//  Rect *searchedLeftEye, Rect *searchedRightEye 매개변수 삭제
void detectEyes(const Mat &faceImage, CascadeClassifier&eyeDetector1, CascadeClassifier &eyeDetector2, Point &leftEyeCenter, Point &rightEyeCenter){
    const float EYE_SX = 0.12f;
    const float EYE_SY = 0.17f;
    const float EYE_SW = 0.37f;
    const float EYE_SH = 0.36f;
    
    int leftX = cvRound(faceImage.cols * EYE_SX);
    int topY = cvRound(faceImage.rows * EYE_SY);
    int widthX = cvRound(faceImage.cols * EYE_SW);
    int heightY = cvRound(faceImage.rows * EYE_SH);
    int rightX = cvRound(faceImage.cols * (1.0 - EYE_SX-EYE_SW));
    
    Mat topLeftOfFace = faceImage(Rect(leftX, topY, widthX, heightY));
    Mat topRightOfFace = faceImage(Rect(rightX, topY, widthX, heightY));
    Rect leftEyeRectangleArea, rightEyeRectangleArea;
    //  왼쪽눈 사각형 오른쪽 눈 사각형에서 eyeDetector1으로 눈 찾기
    detectLargestObject(topLeftOfFace, eyeDetector1, leftEyeRectangleArea, topLeftOfFace.cols);
    detectLargestObject(topRightOfFace, eyeDetector1, rightEyeRectangleArea, topRightOfFace.cols);
    
    //  만약 위에서 눈을 찾지 못했다면 eyeDetector2로 눈 찾기 진행
    if (leftEyeRectangleArea.width <= 0) {
        detectLargestObject(topRightOfFace, eyeDetector2, leftEyeRectangleArea, topLeftOfFace.cols);
        if (leftEyeRectangleArea.width >0)
            cout << "2nd Eye Detector LEFT SUCCESS" << endl;
        else
            cout << "2nd Eye Detector LEFT FAILED " << endl;
    }
    else {
        cout << "1st Eye Detector LEFT SUCCESS" << endl;
    }
    
    //  만약 위에서 눈을 찾지 못했을떄 eyeDetector2로 눈 찾기 진행
    if (rightEyeRectangleArea.width <= 0) {
        detectLargestObject(topRightOfFace, eyeDetector2, rightEyeRectangleArea, topRightOfFace.cols);
        if (rightEyeRectangleArea.width >0)
            cout << "2nd Eye Detector RIGHT SUCCESS" << endl;
        else
            cout << "2nd Eye Detector RIGHT FAILED" << endl;
    }
    else {
        cout << "1nd Eye Detector RIGHT SUCCESS" << endl;
    }
    
    //  왼쪽 눈이 찾아졌는지 확인하고, 찾아지지않았으면 (-1, -1) 리턴
    if (leftEyeRectangleArea.width > 0) {
        leftEyeRectangleArea.x += leftX;    //  왼쪽눈 직사각형을 재조정, 테두리가 삭제되서
        leftEyeRectangleArea.y += topY;
        leftEyeCenter = Point(leftEyeRectangleArea.x + leftEyeRectangleArea.width/2, leftEyeRectangleArea.y + leftEyeRectangleArea.height/2);
    }
    else {
        leftEyeCenter = Point(-1, -1);
    }
    //  오른쪽 눈이 찾아졌는지 확인하고, 찾아지지않았으면 (-1, -1) 리턴
    if (rightEyeRectangleArea.width > 0) {
        rightEyeRectangleArea.x += rightX;
        rightEyeRectangleArea.y += topY;
        rightEyeCenter = Point(rightEyeRectangleArea.x + rightEyeRectangleArea.width/2, rightEyeRectangleArea.y + rightEyeRectangleArea.height/2);
    }
    else {
        rightEyeCenter = Point(-1, -1);
    }
}

//  얼굴을 좌우로 나눠서 히스토그램 평활화하는 함수
void equalizeLeftAndRightHalfFace(Mat &faceImage){
    //  전체 이미지를 한번 히스토그램 평활화를 했으므로 좌우가 상대적으로 차이가 날수 있음. 그래서 좌, 우 따로하고, 이렇게 했을때 좌우가 만나는 가운데에서 선이 나타날수 있으므로 좌, 우, 전체 평활화해서 섞음.
    int width = faceImage.cols;
    int height = faceImage.rows;
    
    //  먼저 전체 얼굴 평활화.
    Mat entireFaceImage;
    equalizeHist(faceImage, entireFaceImage);
    
    //  전체얼굴을 좌우로 나눠서 각각 평활화
    int midX = width/2;
    Mat leftFaceImage = faceImage(Rect(0, 0, midX, height));
    Mat rightFaceImage = faceImage(Rect(midX, 0, width-midX, height));
    equalizeHist(leftFaceImage, leftFaceImage);
    equalizeHist(rightFaceImage, rightFaceImage);
    
    //  좌, 우, 전체 평활화한 이미지 합치기. 화소에 직접 접근.
    for (int y=0; y < height; y++) {
        for (int x=0; x<width; x++) {
            int z;
            if (x < width/4) {  //  왼쪽의 25%는 그냥 왼쪽 이미지 사용
                z = leftFaceImage.at<uchar>(y, x);
            }
            else if (x < width*2/4) {   //  왼쪽-중앙 25%는 전체와 왼쪽 이미지 섞기
                int leftz = leftFaceImage.at<uchar>(y, x);
                int entirez = entireFaceImage.at<uchar>(y, x);
                //  가중평균으로 좀씩 자연스럽게 섞기
                float f = (x-width*1/4) / (float)(width*1/4);
                z = cvRound((1.0f - f) * leftz + (f) * entirez);
            }
            else if (x < width*3/4) {   //  오른쪽-중앙 25%는 전체와 오른쪽 이미지 섞기
                int rightz = rightFaceImage.at<uchar>(y, x-midX);
                int entirez = entireFaceImage.at<uchar>(y, x);
                //  가중평균으로 좀씩 자연스럽게 섞기
                float f = (x-width*2/4) / (float)(width*1/4);
                z = cvRound((1.0f - f) * entirez + (f) * rightz);
            }
            else {  //  오른쪽의 25% 그냥 오른쪽 이미지 사용
                z = rightFaceImage.at<uchar>(y, x-midX);
            }
            faceImage.at<uchar>(y, x) = z;
        }
    }
}

//  표준화된 사이즈와 명암 밝기를 가진 흑백 얼굴 이미지를 생성하는 함수.
Mat getPreprocessedFace(Mat &srcImage, int desiredFaceWidth, CascadeClassifier &faceDetector, CascadeClassifier &eyeDetector1, CascadeClassifier &eyeDetector2, Rect *storeFaceRect, Point *storeLeftEyeCenter, Point *storeRightEyeCenter){
    //  얼굴 이미지의 크기를 정사각형으로
    const double LEFT_EYE_X = 0.16;
    const double LEFT_EYE_Y = 0.14; //  전처리 과정이 끝나고 얼굴 몇개를 출력할건지를 결정하는 상수
    const double FACE_ELLIPSE_CY = 0.40;
    const double FACE_ELLIPSE_W = 0.50; //  적어도 0.5 이상
    const double FACE_ELLIPSE_H = 0.80; //  얼굴 마스크의 길이

    int desiredFaceHeight = desiredFaceWidth;
    
    //  얼굴 좌표와 눈좌표를 -1로 초기화. 못찾았을경우를 대비
    if (storeFaceRect)
        storeFaceRect->width = -1;
    if (storeLeftEyeCenter)
        storeLeftEyeCenter->x = -1;
    if (storeRightEyeCenter)
        storeRightEyeCenter->x = -1;
    cerr << "1" << endl;
    //  프레임에서 가장 큰 하나의 얼굴 찾기
    Rect faceRect;
    detectLargestObject(srcImage, faceDetector, faceRect);
    cerr << "5" << endl;
    
    //  얼굴이 검출되었다면
    if (faceRect.width >0) {
        //  얼굴 사각형을 호출한 함수에게 전달.
        if (storeFaceRect) {
            *storeFaceRect = faceRect;
        }
        Mat faceImage = srcImage(faceRect); //  찾은 얼굴 영역 이미지를 가져옴
        
        //  이미지를 흑백으로 바꾸는 과정.
        Mat grayImage;
        if (faceImage.channels() == 3) {
            cvtColor(faceImage, grayImage, CV_BGR2GRAY);
        }
        else if (faceImage.channels() == 4) {
            cvtColor(faceImage, grayImage, CV_BGRA2GRAY);
        }
        else {
            grayImage = faceImage;
        }
        
        //  두눈을 찾는 과정. 눈 찾기는 풀 해상도의 화면이 필요해서 줄이지 않음
        Point leftEyeCenter, rightEyeCenter;
        detectEyes(grayImage, eyeDetector1, eyeDetector2, leftEyeCenter, rightEyeCenter);
        
        //  눈의 좌표를 호출한 함수에게 전달.
        if (storeLeftEyeCenter)
            *storeLeftEyeCenter = leftEyeCenter;
        if (storeRightEyeCenter)
            *storeRightEyeCenter = rightEyeCenter;
        cerr << "5.1" << endl;
        
        //  두 눈이 찾아졌다면
        if (leftEyeCenter.x >= 0 && rightEyeCenter.x >= 0) {
            //  트레이닝 이미지와 같은 사이지로 얼굴 이미지를 조정. 두 눈을 찾았기 때문에 두 눈의 위치를 기준으로 얼굴 이미지를 조정해서 모든 얼굴 이미지의 눈의 위치를 같도록 조정.
            
            //  두 눈의 중점을 계산
            Point2f eyeCenter = Point2f((leftEyeCenter.x + rightEyeCenter.x)*0.5f, (leftEyeCenter.y + rightEyeCenter.y)*0.5f);
            //  두 눈사이의 각도를 계산
            double dy = (rightEyeCenter.y - leftEyeCenter.y);
            double dx = (rightEyeCenter.x - leftEyeCenter.x);
            double length = sqrt(dx*dx + dy*dy);
            double angle = atan2(dy, dx) * 180.0/CV_PI; //  라디안 값에서 각도로 변경.
            
            //  이상적인 오른쪽 눈의 좌표값을 계산
            const double RIGHT_EYE_X = (1.0f - LEFT_EYE_X);
            double idealLength = (RIGHT_EYE_X - LEFT_EYE_X) * desiredFaceWidth;
            double scale = idealLength / length;
            
            //  정해진 각도와 사이즈로 얼굴 이미지를 회전 및 이동을 위한 행렬 생성.
            Mat rotation_Matrix = getRotationMatrix2D(eyeCenter, angle, scale);
            
            //  원하는 위치로 눈의 중심을 이동
            double ex = desiredFaceWidth*0.5f - eyeCenter.x;
            double ey = desiredFaceHeight*LEFT_EYE_Y - eyeCenter.y;
            rotation_Matrix.at<double>(0, 2) += ex;
            rotation_Matrix.at<double>(1, 2) += ey;
            
            //  얼굴 영상을 원하는 각도, 크기, 위치로 변환, 또한 변환한 영상 배경을 기본 회색으로 설정.
            Mat warpedImage = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128)); //  출력이미지를 기본 회색으로 설정.
            warpAffine(grayImage, warpedImage, rotation_Matrix, warpedImage.size());
            imshow("warped", warpedImage);
            cerr << "5.2" << endl;
            
            //  얼굴의 왼쪽 오른쪽 히스토그램 평활화
            equalizeLeftAndRightHalfFace(warpedImage);
            imshow("equalized", warpedImage);
            cerr << "5.3" << endl;
            
            //  Bilateral 필터를 사용해서 영상의 노이즈 제거.
            Mat filteredImage = Mat(warpedImage.size(), CV_8U);
            bilateralFilter(warpedImage, filteredImage, 0, 20.0, 2.0);
            imshow("filtered", filteredImage);
            cerr << "5.4" << endl;
            
            //  타원형 얼굴 마스크로 만들기
            Mat maskImage = Mat(warpedImage.size(), CV_8U, Scalar(0));  //  빈마스크
            Point faceCenter = Point (desiredFaceWidth/2, cvRound(desiredFaceHeight*FACE_ELLIPSE_CY));
            Size size = Size(cvRound(desiredFaceWidth*FACE_ELLIPSE_W), cvRound(desiredFaceHeight*FACE_ELLIPSE_H));
            ellipse(maskImage, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
            cerr << "5.5" << endl;
            imshow("mask", maskImage);
            
            //  마스크를 사용해서 바깥쪽 코너를 제거.
            Mat completeImage = Mat(warpedImage.size(), CV_8U, Scalar(128));    //  출력할 이미지의 기본색을 회색으로 설정.
            //  마스크를 얼굴에 적용. 마스크안된 픽셀을 출력이미지에 복사.
            filteredImage.copyTo(completeImage, maskImage);
            cerr << "6"<< endl;
            imshow("finish", completeImage);
            
            return completeImage;
        }
    }
    return Mat();
}


//  XML 분류기 로딩하는 함수.
void loadXMLs(CascadeClassifier &faceDetector, CascadeClassifier &eyeDetector1, CascadeClassifier &eyeDetector2)
{
    //  얼굴 검출 xml 로딩
    try {
        faceDetector.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if ( faceDetector.empty() ) {
        cerr << "ERROR: Could not load Face Detection cascade classifier [" << faceCascadeFilename << "]!" << endl;
        exit(1);
    }
    cout << "Loaded the Face Detection cascade classifier [" << faceCascadeFilename << "]." << endl;
    
    //  눈 검출 xml 로딩
    try {
        eyeDetector1.load(eyeCascadeFilename1);
    } catch (cv::Exception &e) {}
    if ( eyeDetector1.empty() ) {
        cerr << "ERROR: Could not load 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]!" << endl;
        exit(1);
    }
    cout << "Loaded the 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]." << endl;
    
    //  눈 검출 xml 로딩
    try {
        eyeDetector2.load(eyeCascadeFilename2);
    } catch (cv::Exception &e) {}
    if ( eyeDetector2.empty() ) {
        cerr << "Could not load 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
        exit(1);
    }
    else
        cout << "Loaded the 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
}

//  카메라 구동하는 함수.
void setCamera(VideoCapture &videoCapture, int cameraNumber) {
    try {
        videoCapture.open(cameraNumber);
    } catch (cv::Exception &e) {}
    if ( !videoCapture.isOpened() ) {
        cerr << "ERROR: Could not access the camera!" << endl;
        exit(1);
    }
    cout << "Loaded camera " << cameraNumber << "." << endl;
}

//  이미지 상에 텍스트를 그림는 함수. 텍스트를 포함한 직사각형을 리턴
//  디폴트위치는 왼쪽위, x 좌표를 음수로 주면 위치를 오른쪽으로 변경가능, y 좌표를 음수로 주면 위치를 아래쪽으로 변경가능
Rect drawTextString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX) {
    //  텍스트 크기와 기준점 설정
    int baseline=0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;
    
    //  좌우, 위아래에 따라 좌표를 조정.
    if (coord.y >= 0) {
        //  좌표평면이 이미지의 좌상단에서부터 텍스트의 좌상단에 위치하므로, 1줄 밑으로보냄.
        coord.y += textSize.height;
    }
    else {
        //  좌표평면이 이미지의 우하단에서부터 텍스트의 우하단에 위치하므로, 1줄 위로보냄.
        coord.y += img.rows - baseline + 1;
    }
    // 오른쪽정렬의 경우
    if (coord.x < 0) {
        coord.x += img.cols - textSize.width + 1;
    }
    
    //  텍스트를 둘러쌀 박스를 생성
    Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);
    
    //  안티 알리아싱 된 텍스트를 그림
    putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);
    
    //  텍스트를 리턴
    return boundingRect;
}

//  drawString()함수를 사용해서 GUI상에 버튼을 그리는 함수. 그려진 버튼을 리턴, 여러개를 그릴경우, 각각을 옆에 위치시킬수 있음
//  minWidth 파라메터를 조절해서, 여러개의 같은 넓이의 버튼을 새엉 가능.
Rect drawButton(Mat img, string text, Point coord, int minWidth=0) {
    int B = BORDER;
    Point textCoord = Point(coord.x + B, coord.y + B);
    // Get the bounding box around the text.
    Rect rcText = drawTextString(img, text, textCoord, CV_RGB(0,0,0));
    // Draw a filled rectangle around the text.
    Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2*B, rcText.height + 2*B);
    // Set a minimum button width.
    if (rcButton.width < minWidth)
        rcButton.width = minWidth;
    // Make a semi-transparent white rectangle.
    Mat matButton = img(rcButton);
    matButton += CV_RGB(90, 90, 90);
    // Draw a non-transparent white border.
    rectangle(img, rcButton, CV_RGB(200,200,200), 1, CV_AA);
    
    // Draw the actual text that will be displayed, using anti-aliasing.
    drawTextString(img, text, textCoord, CV_RGB(10,55,20));
    
    return rcButton;
}

//  해당 좌표가 직사각형 안에 있는지를 판단하는 함수.
bool isPointInRect(const Point pt, const Rect rc) {
    if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
        if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
            return true;
    
    return false;
}

//  마우스 클릭 이벤트 핸들러 함수.
void onMouse(int event, int x, int y, int, void*) {
    //  왼쪽 클릭만 다룸. 오른쪽클릭이나 마우스 움직임은 무시.
    if (event != CV_EVENT_LBUTTONDOWN)
        return;
    
    Point pt = Point(x,y);
    
    //  사용자가 GUI상의 버튼을 클릭했을때
    if (isPointInRect(pt, m_rectangleBottonAdd)) {
        cout << "User clicked [Add User] button when numPersons was " << m_numUsers << endl;
        //  m_latestFaces 변수를 확인해서, 만약 이미 사용자가 있는데 수집한 얼굴이 없는 경우에는 그 사용자를 사용함.
        if ((m_numUsers == 0) || (m_latestFaces[m_numUsers-1] >= 0)) {
            //  새로운 사용자 추가.
            m_numUsers++;
            m_latestFaces.push_back(-1); // 사용자 추가를위해서 공간 할당.
            cout << "Num Users: " << m_numUsers << endl;
        }
        //  새로 추가된 사용자를 사용. 그 사용자가 비어있을경우에도 사용.
        m_selectedUser = m_numUsers - 1;
        m_mode = MODE_COLLECT_FACES;
    }
    else if (isPointInRect(pt, m_rectangleBottonDel)) {
        cout << "User clicked [Delete All] button." << endl;
        m_mode = MODE_DELETE_ALL;
    }
    else if (isPointInRect(pt, m_rectangleBottonDebug)) {
        cout << "User clicked [Debug] button." << endl;
        m_debug = !m_debug;
        cout << "Debug mode: " << m_debug << endl;
    }
    //  사용자가 버튼이 아닌 화면을 클릭했을 경우
    else {
        cout << "User clicked on the image" << endl;
        
        //  사용자가 얼굴 목록을 클릭했는지 확인
        int clickedPerson = -1;
        for (int i=0; i<m_numUsers; i++) {
            if (m_gui_faces_top >= 0) {
                Rect rcFace = Rect(m_gui_faces_left, m_gui_faces_top + i * faceHeight, faceWidth, faceHeight);
                if (isPointInRect(pt, rcFace)) {
                    clickedPerson = i;
                    break;
                }
            }
        }
        //  만약 사용자가 GUI상에서 다른 사진을 클릭했을 때, 선택한 사람을 바꿔야함.
        if (clickedPerson >= 0) {
            //  현재 클릭한 사람을, 선택한 사람으로 바꾸고, 사진을 모음
            m_selectedUser = clickedPerson;
            m_mode = MODE_COLLECT_FACES;
        }
        //  사용자가 일반 배경을 클릭했을때 모드 변경
        else {
            if (m_mode == MODE_COLLECT_FACES) {
                cout << "User wants to begin training." << endl;
                m_mode = MODE_TRAINING;
            }
        }
    }
}

//  얼굴을 찾고 학습해서 인식하는 함수.
void recognizeAndTrain(VideoCapture &videoCapture, CascadeClassifier &faceDetector, CascadeClassifier &eyeDetector1, CascadeClassifier &eyeDetector2){
    Ptr<FaceRecognizer> model;
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;
    Mat old_preprocessedFace;
    double old_time = 0;
    
    cerr << "_A" << endl;
    
    //  초기화가 완료된 상태이므로, Detection 모드로 시작
    m_mode = MODE_DETECTION;
    
    //  무한루프로 진행. 사용자가 종료키를 눌러야 종료
    while (true) {
        //  다음 카메라 프레임을 잡아서 사용. 카메라프레임 자체는 수정할 수 없음
        Mat cameraFrame;
        videoCapture >> cameraFrame;
        if (cameraFrame.empty()) {
            cerr << "Error! Could not grab the next camera frame." << endl;
            exit(1);
        }
        
        //  우리가 그릴 카메라 프레임의 복사본을 생성
        Mat displayedFrame;
        cameraFrame.copyTo(displayedFrame);
        
        //  카메라 이미지에 대한 얼굴 인식 수행. 주어진 이미지에 얼굴을 찾으면 그려야하므로, ROM이 아니게 설정
        int identity = -1;
        cerr << "A " << m_mode << endl;
        
        //  얼굴을 찾고, 사이즈와 명암과 밝기를 맞춤.
        Rect faceRect;  //  탐지된 얼굴의 위치
//        Rect searchedLeftEye, searchedRightEye; //  얼굴의 좌상단과 우상단 = 눈사각형들
        Point leftEyeCenter, rightEyeCenter;    //  찾은 눈의 중심 좌표
        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceDetector, eyeDetector1, eyeDetector2, &faceRect, &leftEyeCenter, &rightEyeCenter);
        cerr << "B" << endl;
        bool gotFaceAndEyes = false;
        if (preprocessedFace.data) {
            gotFaceAndEyes = true;
        }
        
        //  위에서 정의한 사각형을 탐지항 얼굴주위에 그림
        if (faceRect.width > 0) {
            rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);
            
            //  눈을찾아서 원을 그림
            Scalar eyeColor = CV_RGB(0, 255, 255);
            if (leftEyeCenter.x >= 0) {
                //  왼쪽눈을 찾았으면 그림
                circle(displayedFrame, Point(faceRect.x + leftEyeCenter.x, faceRect.y + leftEyeCenter.y), 6, eyeColor, 1, CV_AA);
            }
            if (rightEyeCenter.x >= 0) {
                //  오른쪽 눈을 찾았으면 그림
                circle(displayedFrame, Point(faceRect.x+rightEyeCenter.x, faceRect.y+rightEyeCenter.y), 6, eyeColor, 1, CV_AA);
            }
        }
        cerr << "C" << endl;
        if (m_mode == MODE_DETECTION) {
            cerr << "D" << endl;    //  탐지모드일때는 다른일은 하지 않음.
        }
        /*
        else if (m_mode == MODE_COLLECT_FACES){
            cerr << "E" << endl;
            //  탐지된 얼굴이 있다면
            if (gotFaceAndEyes) {
                //  탐지된 얼굴이 이전의 수집된 얼굴과 다른지 확인
                double imageDiff = 10000000000.0;
                if (old_preprocessedFace.data) {
                    imageDiff = getSimilarity(preprocessedFace, old_preprocessedFace);
                }
                
                //  기록.
                double current_time = (double)getTickCount();
                double timeDiff_seconds = (current_time - old_time)/getTickFrequency();
                
                //  일정 시간 간격과, 유사도가 적을때에만 진행.
                if ((imageDiff > CHANGE_PARAMETER_IMAGE) && (timeDiff_seconds > CHANGE_PARAMETER_SECOND)) {
                    //  해당 이미지의 미러 이미지를 트레이닝 셋에 추가. 얼굴의 왼쪽 오른쪽을 각각 다룸
                    Mat mirroredFace;
                    flip(preprocessedFace, mirroredFace, 1);
                    
                    //  탐지된 얼굴 리스트에 해당 얼굴을 추가
                    preprocessedFace.push_back(preprocessedFace);
                    preprocessedFace.push_back(mirroredFace);
                    faceLabels.push_back(m_selectedUser);
                    faceLabels.push_back(m_selectedUser);
                    
                    //  각 사용자의 최신 얼굴의 레퍼런스를 유지
                    m_latestFaces[m_selectedUser] = preprocessedFaces.size() - 2;
                    //  미러가 아닌 얼굴이미지를 가리킴.
                    
                    //  수집된 얼굴 이미지의 갯수를 출력. 미러 이미지도 같이 저장하므로, /2
                    cout << "Saved face" << (preprocessedFaces.size()/2) << " for person"<< m_selectedUser << endl;
                    
                    //  플래시, 사진 찍히는 효과. 사용자는 이걸보고 사진 찍힌걸 알수 있음
                    Mat displayedFaceRegion = displayedFrame(faceRect);
                    displayedFaceRegion += CV_RGB(90, 90, 90);
                    
                    //  다음번 반복에서 비교로 사용할 얼굴의 복사본을 유지.
                    old_preprocessedFace = preprocessedFace;
                    old_time = current_time;
                }
            }
        }
         */
        /*
        else if (m_mode == MODE_TRAINING){
            cerr << "F" << endl;
            
            //  트레이닝 할 충분한 데이터가 있는지를 확인. 적어도 2명이 필요.
            bool haveEnoughData = true;
            if (strcmp(facerecAlgorithm, "FaceRecognizer.Fisherfaces") == 0) {
                if ((m_numUsers < 2) || (m_numUsers == 2 && m_latestFaces[1] < 0)) {
                    cout << "Warning: Fisherfaces needs at least 2 people, otherwise there is nothing to differentiate! Collect more data ..."<< endl;
                    haveEnoughData = false;
                }
            }
            //  Eigenfaces일경우에는 한명이상
            if (m_numUsers < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) {
                cout << "Warning: Need some training data before it can be learnt! Collect more data ..." << endl;
                haveEnoughData = false;
            }
            
            if (haveEnoughData) {
                //  수집된 얼굴들을 사용해서 트레이닝 시작.
                model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);
                
                //  디버깅시 내부 데이터 출력
                if (m_debug) {
                    showTrainingDebugData(model, faceWidth, faceHeight);
                }
                
                //  트레이닝 종료후 인식 시작.
                m_mode = MODE_RECOGNITION;
            }
            else {
                //  충분하지 않은 트레이닝 데이터를 가진 경우, 다시 얼굴 수집 모드로 돌아감
                m_mode = MODE_COLLECT_FACES;
            }
            
        }
         */
        /*
        else if (m_mode == MODE_RECOGNITION){
            cerr << "G" << endl;
            if (gotFaceAndEyes && (preprocessedFaces.size() >0) && (preprocessedFaces.size() == faceLabels.size())) {
                
                //  얼굴 재구성.
                Mat reconstructedFace;
                reconstructedFace = reconstructFace(model, preprocessedFace);
                if (m_debug) {
                    if (reconstructedFace.data) {
                        imshow("reconstructedFace", reconstructedFace);
                    }
                }
                
                //  재구성된 얼굴이 선처리된 얼굴과 같은지를 비교.
                double similarity = getSimilarity(preprocessedFace, reconstructedFace);
                cerr << "G.1" << endl;
                
                string outputStr;
                if (similarity < UNKNOWN_PERSON_THRESHOLD) {
                    //  선처리된 얼굴의 사람을 알아냄.
                    identity = model->predict(preprocessedFace);
                    outputStr = toString(identity);
                }
                else {
                    //  신뢰도가 낮으므로 허가받지 않은 사용자로 간주
                    outputStr = "Unknown";
                }
                cout<< "Identity: " << outputStr << ". Similarity: " << similarity << endl;
                
                //  신뢰도 값을 화면 중앙 상단에 표시
                int cx = (displayedFrame.cols - faceWidth) / 2;
                Point ptBottomRight = Point(cx - 5, BORDER+faceHeight);
                Point ptTopLeft = Point(cx-15, BORDER);
                
                //  허가받지 않은 사람일경우의 쓰레쉬홀드값을 회색선으로 그림
                Point ptThreshold = Point(ptTopLeft.x, ptBottomRight.y - (1.0 - UNKNOWN_PERSON_THRESHOLD)*faceHeight);
                rectangle(displayedFrame, ptThreshold, Point(ptBottomRight.x, ptThreshold.y), CV_RGB(200, 200, 200), 1, CV_AA);
                
                //  신뢰도의 값을 0.0과 1.0사이의 값으로 잘라냄.
                double confidenceRatio = 1.0 - min(max(similarity, 0.0), 1.0);
                Point ptConfidence = Point(ptTopLeft.x, ptBottomRight.y - confidenceRatio*faceHeight);
                
                //  신뢰도바를 출력
                rectangle(displayedFrame, ptConfidence, ptBottomRight, CV_RGB(0, 255, 255), CV_FILLED, CV_AA);
                //  바의 경계를 출력
                rectangle(displayedFrame, ptTopLeft, ptBottomRight, CV_RGB(200, 200, 200), 1, CV_AA);
            }
        }
         */
        else if (m_mode == MODE_DELETE_ALL){
            //  전과정을 재시작
            m_selectedUser = -1;
            m_numUsers = 0;
            m_latestFaces.clear();
            preprocessedFaces.clear();
            faceLabels.clear();
            old_preprocessedFace = Mat();
            
            //  얼굴 탐지 모드에서 재시작
            m_mode = MODE_DETECTION;
        }
        else{
            cerr << "Error! Invalid run mode" << m_mode << endl;
            exit(1);
        }
        cerr << "H" << endl;
        
//  도움말 메시지를 출력하는 과정. 현재 저장된 사람과 얼굴숫자도 출력
        string help;
        Rect rcHelp;
        if (m_mode == MODE_DETECTION) {
            help = "Click [Add Person] when ready to collect faces.";
        }
        else if (m_mode == MODE_COLLECT_FACES) {
            help = "Click anywhere to train from your " + toString(preprocessedFaces.size()/2) + " faces of "+ toString(m_numUsers)+ " people.";
        }
        else if (m_mode == MODE_TRAINING) {
            help = "Please wait while your "+ toString(preprocessedFaces.size()/2) + " faces of "+ toString(m_numUsers)+ " people builds.";
        }
        else if (m_mode == MODE_RECOGNITION){
            help = "Click people on the right to add more faces to them, or [Add Person] for someone new.";
        }
        if (help.length() > 0) {
            //  글자색은 하얀색. 글자 음영색은 검정색으로 지정.
            //  BORDER값이 0이 되는데 음수값이 필요함으로, 2를 빼서 항상 음수값으로 유지
            float txtSize = 0.4;
            drawTextString(displayedFrame, help, Point(BORDER, -BORDER-2), CV_RGB(0, 0, 0), txtSize);
            rcHelp = drawTextString(displayedFrame, help, Point(BORDER+1, -BORDER-1), CV_RGB(255, 255, 255), txtSize);
        }
        cerr << "I" << endl;
        
//  현재 모드를 보여줌
        if (m_mode >= 0 && m_mode < MODE_END) {
            string modeStr = "MODE: "+string(MODE_NAMES[m_mode]);
            drawTextString(displayedFrame, modeStr, Point(BORDER, -BORDER-2-rcHelp.height), CV_RGB(0, 0, 0));
            drawTextString(displayedFrame, modeStr, Point(BORDER+1, -BORDER-1-rcHelp.height), CV_RGB(0, 0, 255));   //  글자색 파란색
        }
        cerr << "J" << endl;
        
//  현재의 선처리된 얼굴을 위쪽 중앙에 출력
        int cx = (displayedFrame.cols - faceWidth)/2;
        if (preprocessedFace.data) {
            //  BGR버전의 얼굴을 가져옴.
            Mat srcBGR = Mat(preprocessedFace.size(), CV_8UC3);
            cvtColor(preprocessedFace, srcBGR, CV_GRAY2BGR);
            Rect dstRC = Rect(cx, BORDER, faceWidth, faceHeight);
            Mat dstROI = displayedFrame(dstRC);
            //  src에서 dst로 픽셀 복사.
            srcBGR.copyTo(dstROI);
        }
        cerr << "K" << endl;
        //  출력할 선처리 얼굴의 테두리를 그림
        rectangle(displayedFrame, Rect(cx-1, BORDER-1, faceWidth+2, faceHeight+2), CV_RGB(200, 200, 200), 1, CV_AA);
        cerr << "L" << endl;
        
//  GUI 버튼을 그림
        m_rectangleBottonAdd = drawButton(displayedFrame, "Add User", Point(BORDER, BORDER+300));
        m_rectangleBottonDel = drawButton(displayedFrame, "Delete All User", Point(m_rectangleBottonAdd.x, m_rectangleBottonAdd.y+m_rectangleBottonAdd.height), m_rectangleBottonAdd.width);
        m_rectangleBottonDebug = drawButton(displayedFrame, "Login", Point(m_rectangleBottonDel.x, m_rectangleBottonDel.y+m_rectangleBottonDel.height), m_rectangleBottonAdd.width);
        cerr << "M" << endl;

//  화면의 우측에 1열로 각 사용자마다 가장 최근의 얼굴사진을 띄움.
        m_gui_faces_left = displayedFrame.cols - BORDER - faceWidth;
        m_gui_faces_top = BORDER;
        for (int i=0; i<m_numUsers; i++) {
            int index = m_latestFaces[i];
            if (index >= 0 && index < (int)preprocessedFaces.size()) {
                Mat srcGray = preprocessedFaces[index];
                if (srcGray.data) {
                    //  BGR버전의 얼굴을 가져옴
                    Mat srcBGR = Mat(srcGray.size(), CV_8UC3);
                    cvtColor(srcGray, srcBGR, CV_GRAY2BGR);
                    int y = min(m_gui_faces_top+i*faceHeight, displayedFrame.rows - faceHeight);
                    Rect dstRC = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                    Mat dstROI = displayedFrame(dstRC);
                    //  src에서 dst로 픽셀 복사.
                    srcBGR.copyTo(dstROI);
                }
            }
        }
        cerr << "N" << endl;
        
//  마우스로 선택된 사람의 테두리를 빨간색으로 표현하며 강조하는 과정

//  얼굴인식과정에서 인식된 사람의 테두리를 초록색으로 표현하며 강조하는 과정
        
        
        
//  스크린에 카메라 프레임 출력
        imshow(windowName, displayedFrame);
        
//  디버그 데이터를 출력할때
        
        

//  적어도 20msec은 기달려야 이미지가 스크린에 출력됨.
        int keypress = waitKey(20);
        if (keypress == 27) {   //  ESC키를 누르면 종료
            break;
        }
    }
}
int main(int argc, char *argv[]) {
    CascadeClassifier faceDetector;
    CascadeClassifier eyeDetector1;
    CascadeClassifier eyeDetector2;
    VideoCapture videoCapture;
    
    cout << "Face Detection & Face Recognition using LBP." << endl;
    cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;
    
    //  xml을 로딩
    loadXMLs(faceDetector, eyeDetector1, eyeDetector2);
    
    cout << endl;
    cout << "Hit 'Escape' in the GUI window to quit." << endl;
    
    //  디폴트 카메라 (0번)말고 다른거 사용하고 싶으면, 아규먼트로 넘겨줘서 설정 가능.
    int cameraNumber = 0;
    if (argc > 1) {
        cameraNumber = atoi(argv[1]);
    }
    
    //  카메라에 구동
    setCamera(videoCapture, cameraNumber);
    
    //  카메라 해상도를 조절. 안되는 카메라나 컴퓨터도 있음.
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
    
    //  GUI 창을 생성.
    namedWindow(windowName);
    //  사용자가 창을 클릭할때, onMouse 함수를 호출하기위한 OpenCV함수 setMouseCallback
    setMouseCallback(windowName, onMouse, 0);
    
    //  얼굴인식 실행.
    recognizeAndTrain(videoCapture, faceDetector, eyeDetector1, eyeDetector2);
    
    return 0;
}