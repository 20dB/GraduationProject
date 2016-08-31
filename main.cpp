//
//  main.cpp
//  GraduationProject2016
//
//  Created by Versatile75 on 2016. 8. 15..
//  Copyright © 2016년 Versatile75. All rights reserved.
//

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

//  가장 정확한게 mcs_left/righteye, 그다음이 left/righteye_2splits, 기본이 haarcascade_eye/eye_tree_eyeglasses
const char *faceCascadeFilename = "/Users/Versatile75/Documents/XcodeWorkspace/GraduationProject2016/GraduationProject2016/haarcascade_frontalface_default.xml";
const char *eyeCascadeFilename1 = "/Users/Versatile75/Documents/XcodeWorkspace/GraduationProject2016/GraduationProject2016/haarcascade_mcs_lefteye.xml";
const char *eyeCascadeFilename2 = "/Users/Versatile75/Documents/XcodeWorkspace/GraduationProject2016/GraduationProject2016/haarcascade_mcs_righteye.xml";

const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";    //  얼굴인식 알고리즘 선택 -> PCA

//  등록된 사람인지 아닌지를 결정하는 얼굴 인식 알고리즘의 신뢰도를 설정. 높아질수록 등록된 사람으로 판단
const float UNKNOWN_PERSON_THRESHOLD = 0.3f;

//  얼굴이미지의 크기를 설정. getPreprocessedFace()함수가 정사각형 얼굴을 리턴하기때문에 faceWidth=faceHeight.
const int faceWidth = 70;
const int faceHeight = faceWidth;

//  카메라의 해상도를 설정. 카메라/시스템에따라 안맞을수도있음
const int CAMERA_WIDTH = 640;
const int CAMERA_HEIGHT = 480;

//  얼마나 자주 새로운 얼굴을 저장할지를 정하는 파라메터.
const double THRESHOLD_OF_SIMILARITY = 0.3;      //  트레이닝할 때, 얼마나 자주 얼굴 이미지가 바뀌어야하는지를 설정하는 유사도 임계값
const double THRESHOLD_OF_TIME = 1.0;       //   트레이닝할 때, 시간이 얼마나 지나야 하는지를 설정하는 시간간격 임계값

const char *windowName = "Facial Recognition";   // GUI 화면창의 이름
const int BORDER = 8;  //   GUI 엘리먼트들과 사진의 모서리간의 경계값

//  GUI상에 나타나는 모드 설정 START,
enum USER_MODES {MODE_STARTUP=0, MODE_DETECTION, MODE_COLLECT_FACES, MODE_TRAINING, MODE_RECOGNITION, MODE_RESET, MODE_END};
const char* MODE_NAMES[] = {"Running", "Detection", "Collect Faces", "Training", "Login Existing User", "Reset", "ERROR!"};
USER_MODES m_mode = MODE_STARTUP;

//  유저수 초기화
int m_selectedUser = -1;
int m_numUsers = 0;
vector<int> m_latestFaces;

//  GUI 버튼의 위치 설정
Rect m_rectangleButtonRegisterNewUser;
Rect m_rectangleButtonLoginExistingUser;
Rect m_rectangleButtonReset;
int m_gui_faces_left = -1;
int m_gui_faces_top = -1;

const double LEFT_EYE_X = 0.16;
const double LEFT_EYE_Y = 0.14; //  전처리 과정이 끝나고 얼굴 몇개를 출력할건지를 결정하는 상수
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50; //  적어도 0.5 이상
const double FACE_ELLIPSE_H = 0.80; //  얼굴 마스크의 길이

/**
 물체 찾기에 연관된 함수들
 - 입력데이터는 빠른 탐지를 위해서, scaledWidth 만큼 축소되어있음. 얼굴찾기용으로는 240이 적당.

 1. detectObjects 함수
    주어진 파라메터값을 이용해서 얼굴과 같은 이미지 오브젝트를 찾는 함수. 여러개를 찾아서 objets로 저정함.
 2. detectLargestObject 함수
    이미지에서 가장 큰 얼굴과같은 하나의 오브젝트를 찾는 함수. 찾은 오브젝트를 largestObject에 저장함.
 */
void detectObjects(const Mat &image, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors);
void detectLargestObject(const Mat &image, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth = 320);

/**
 얼굴 이미지 사전 처리에 관련된 함수들
 
 1. detectEyes 함수
    주어진 이미지로부터 두눈을 찾아내는 함수. 각각의 눈의 중점 좌표를 leftEye 와 rightEye 로 리턴. 실패시 (-1, -1)로 반환.
    찾은 왼쪽 오른쪽 눈의 영역을 저장 가능... (추가해야하나..)
 2. equlizeLeftAndRightFace 함수
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
void detectEyes(const Mat &faceImage, CascadeClassifier&eyeDetector1, CascadeClassifier &eyeDetector2, Point &leftEyeCenter, Point &rightEyeCenter);
void equalizeLeftAndRightHalfFace(Mat &faceImage);
Mat getPreprocessedFace(Mat &srcImage, int desiredFaceWidth, CascadeClassifier &faceDetector, CascadeClassifier &eyeDetector1, CascadeClassifier &eyeDetector2, Rect *storeFaceRect = NULL, Point *storeLeftEyeCenter = NULL, Point *storeRightEyeCenter = NULL);

/**
 수집한 데이터들로 얼굴인식 프로그램을 학습시키고 이를통해 인식하는 함수.
 
 1. trainFromCollectedFaces 함수
 수집된 얼굴들로 트레이닝 하는 함수.
 3. reconstructFace 함수
 선처리된 얼굴의 아이젠벡터와 아이젠값을 사용해서 원 얼굴을 근사적으로 재건하는 함수.
 4. calcuateSimilarity 함수
 두개의 이미지의 유사도를 비교하는 함수. L2 Error를 사용. (Square-root of sum of squared error)
 */
Ptr<FaceRecognizer> trainFromCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels, const string facerecAlgorithm = "FaceRecognizer.Eigenfaces");
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace);
double calculateSimilarity(const Mat A, const Mat B);

/**
 프로그램 초기 설정 및 GUI관련 함수
 
 1. loadXMLs 함수.
 2. setCamera 함수.
 3. drawTextString 함수
 4. drawButton 함수
 5. isMousePointInRectangle 함수
 6. onMouse 함수
 7. recognizeAndTrain 함수.
 */
void loadXMLs(CascadeClassifier &faceDetector, CascadeClassifier &eyeDetector1, CascadeClassifier &eyeDetector2);
void setCamera(VideoCapture &videoCapture, int cameraNumber);
Rect drawTextString(Mat img, string text, Point coord, Scalar color, float fontScale, int thickness, int fontFace);
Rect drawButton(Mat img, string text, Point coord, int minWidth);
bool isMousePointInRectangle(const Point pt, const Rect rc);
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

void detectObjects(const Mat &image, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors){
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
    
    //  전처리된 이미지에서 오브젝트 찾기
    cascade.detectMultiScale(equalizedImage, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);
    
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
    detectObjects(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
    if (objects.size()>0) {
        //  오브젝트를 찾았으므로 리턴
        largestObject = (Rect)objects.at(0);
    }
    else{
        //  오브젝트를 못찾았으므로 쓰레기값 리턴
        largestObject = Rect(-1, -1, -1, -1);
    }
}

//  주어진 face 이미지에서 두 눈을 각각 찾아내는 함수. 눈을 찾으면 눈의 중심죄표 leftEyeCenter와 rightEyeCenter 리턴. 찾지 못하면 (-1, -1)반환.
//  Rect *searchedLeftEye, Rect *searchedRightEye 매개변수 삭제
void detectEyes(const Mat &faceImage, CascadeClassifier&eyeDetector1, CascadeClassifier &eyeDetector2, Point &leftEyeCenter, Point &rightEyeCenter){

    const float EYE_SX = 0.10f;
    const float EYE_SY = 0.19f;
    const float EYE_SW = 0.40f;
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
            int v;
            if (x < width/4) {  //  왼쪽의 25%는 그냥 왼쪽 이미지 사용
                v = leftFaceImage.at<uchar>(y, x);
            }
            else if (x < width*2/4) {   //  왼쪽-중앙 25%는 전체와 왼쪽 이미지 섞기
                int lv = leftFaceImage.at<uchar>(y, x);
                int wv = entireFaceImage.at<uchar>(y, x);
                //  가중평균으로 좀씩 자연스럽게 섞기
                float f = (x-width*1/4) / (float)(width*0.25f);
                v = cvRound((1.0f - f) * lv + (f) * wv);
            }
            else if (x < width*3/4) {   //  오른쪽-중앙 25%는 전체와 오른쪽 이미지 섞기
                int rv = rightFaceImage.at<uchar>(y, x-midX);
                int wv = entireFaceImage.at<uchar>(y, x);
                //  가중평균으로 좀씩 자연스럽게 섞기
                float f = (x-width*2/4) / (float)(width*0.25f);
                v = cvRound((1.0f - f) * wv + (f) * rv);
            }
            else {  //  오른쪽의 25% 그냥 오른쪽 이미지 사용
                v = rightFaceImage.at<uchar>(y, x-midX);
            }
            faceImage.at<uchar>(y, x) = v;
        }
    }
}

//  표준화된 사이즈와 명암 밝기를 가진 흑백 얼굴 이미지를 생성하는 함수.
Mat getPreprocessedFace(Mat &srcImage, int desiredFaceWidth, CascadeClassifier &faceDetector, CascadeClassifier &eyeDetector1, CascadeClassifier &eyeDetector2, Rect *storeFaceRect, Point *storeLeftEyeCenter, Point *storeRightEyeCenter){
    //  얼굴 이미지의 크기를 정사각형으로
    int desiredFaceHeight = desiredFaceWidth;
    
    //  얼굴 좌표와 눈좌표를 -1로 초기화. 못찾았을경우를 대비
    if (storeFaceRect)
        storeFaceRect->width = -1;
    if (storeLeftEyeCenter)
        storeLeftEyeCenter->x = -1;
    if (storeRightEyeCenter)
        storeRightEyeCenter->x = -1;
    //  프레임에서 가장 큰 하나의 얼굴 찾기
    Rect faceRect;
    detectLargestObject(srcImage, faceDetector, faceRect);
    
    //  얼굴이 검출되었다면
    if (faceRect.width > 0) {
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
            rotation_Matrix.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyeCenter.x;
            rotation_Matrix.at<double>(1, 2) += desiredFaceHeight * LEFT_EYE_Y - eyeCenter.y;
            
            //  얼굴 영상을 원하는 각도, 크기, 위치로 변환, 또한 변환한 영상 배경을 기본 회색으로 설정.
            Mat warpedImage = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128)); //  출력이미지를 기본 회색으로 설정.
            warpAffine(grayImage, warpedImage, rotation_Matrix, warpedImage.size());
            
            //  얼굴의 왼쪽 오른쪽 히스토그램 평활화
            equalizeLeftAndRightHalfFace(warpedImage);
            
            //  Bilateral 필터를 사용해서 영상의 노이즈 제거.
            Mat filteredImage = Mat(warpedImage.size(), CV_8U);
            bilateralFilter(warpedImage, filteredImage, 0, 20.0, 2.0);
            
            //  타원형 얼굴 마스크로 만들기
            Mat maskImage = Mat(warpedImage.size(), CV_8U, Scalar(0));  //  빈마스크
            Point faceCenter = Point (desiredFaceWidth/2, cvRound(desiredFaceHeight*FACE_ELLIPSE_CY));
            Size size = Size(cvRound(desiredFaceWidth*FACE_ELLIPSE_W), cvRound(desiredFaceHeight*FACE_ELLIPSE_H));
            ellipse(maskImage, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
            
            //  마스크를 사용해서 바깥쪽 코너를 제거.
            Mat completeImage = Mat(warpedImage.size(), CV_8U, Scalar(128));    //  출력할 이미지의 기본색을 회색으로 설정.
            //  마스크를 얼굴에 적용. 마스크안된 픽셀을 출력이미지에 복사.
            filteredImage.copyTo(completeImage, maskImage);
            
            return completeImage;
        }
    }
    return Mat();
}

//  XML 분류기 로딩하는 함수.
void loadXMLs(CascadeClassifier &faceDetector, CascadeClassifier &eyeDetector1, CascadeClassifier &eyeDetector2) {
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

//  영상에 텍스트를 그림는 함수. 텍스트를 포함한 직사각형을 리턴
//  디폴트위치는 왼쪽위, x 좌표를 음수로 주면 위치를 오른쪽으로 변경가능, y 좌표를 음수로 주면 위치를 아래쪽으로 변경가능
Rect drawTextString(Mat image, string text, Point coordinate, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX) {
    //  텍스트 크기와 기준점 설정
    int baseline=0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;
    
    //  좌우, 위아래에 따라 좌표를 조정.
    if (coordinate.y >= 0) {
        //  좌표평면이 이미지의 좌상단에서부터 텍스트의 좌상단에 위치하므로, 1줄 밑으로보냄.
        coordinate.y += textSize.height;
    }
    else {
        //  좌표평면이 이미지의 우하단에서부터 텍스트의 우하단에 위치하므로, 1줄 위로보냄.
        coordinate.y += image.rows - baseline + 1;
    }
    // 오른쪽정렬의 경우
    if (coordinate.x < 0) {
        coordinate.x += image.cols - textSize.width + 1;
    }
    
    //  텍스트를 둘러쌀 박스를 생성
    Rect boundingRect = Rect(coordinate.x, coordinate.y - textSize.height, textSize.width, baseline + textSize.height);
    
    //  안티 알리아싱 된 텍스트를 그림
    putText(image, text, coordinate, fontFace, fontScale, color, thickness, CV_AA);
    
    //  텍스트를 리턴
    return boundingRect;
}

//  drawString()함수를 사용해서 GUI상에 버튼을 그리는 함수. 그려진 버튼을 리턴, 여러개를 그릴경우, 각각을 옆에 위치시킬수 있음
//  minWidth 파라메터를 조절해서, 여러개의 같은 넓이의 버튼을 생성 가능.
Rect drawButton(Mat image, string text, Point coordinate, int minWidth=0) {
    int B = BORDER;
    Point textCoord = Point(coordinate.x + B, coordinate.y + B);

    //  글자의 주변에 가장자리 박스를 생성
    Rect rcText = drawTextString(image, text, textCoord, CV_RGB(0,0,0));
    //  글자 주변에 색이찬 직사각형을 그림
    Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2*B, rcText.height + 2*B);
    
    // 최소 버튼 폭을 설정.
    if (rcButton.width < minWidth)
        rcButton.width = minWidth;
    
    // 하얀 상자를 만듬
    Mat matButton = image(rcButton);
    matButton += CV_RGB(200, 200, 200);
    // 하얀 버튼 상자의 테두리를 그림.
    rectangle(image, rcButton, CV_RGB(200,200,200), 1, CV_AA);
    
    //  실제 문자열을 화면상에 출력.
    drawTextString(image, text, textCoord, CV_RGB(0,0,0));
    
    return rcButton;
}

//  해당 좌표가 직사각형 안에 있는지를 판단하는 함수.
bool isMousePointInRectangle(const Point pt, const Rect rc) {
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
    if (isMousePointInRectangle(pt, m_rectangleButtonRegisterNewUser)) {
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
    else if (isMousePointInRectangle(pt, m_rectangleButtonLoginExistingUser)) {
        cout << "User clicked [Login] button." << endl;
        m_mode = MODE_RECOGNITION;
    }
    else if (isMousePointInRectangle(pt, m_rectangleButtonReset)) {
        cout << "User clicked [Reset] button." << endl;
        m_mode = MODE_RESET;
    }
    //  사용자가 버튼이 아닌 화면을 클릭했을 경우
    else {
        cout << "User clicked on the image" << endl;
        
        //  사용자가 얼굴 목록을 클릭했는지 확인
        int clickedPerson = -1;
        for (int i=0; i<m_numUsers; i++) {
            if (m_gui_faces_top >= 0) {
                Rect rcFace = Rect(m_gui_faces_left, m_gui_faces_top + i * faceHeight, faceWidth, faceHeight);
                if (isMousePointInRectangle(pt, rcFace)) {
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
//        if (m_mode == MODE_COLLECT_FACES) {
//            cout << "User wants to begin training." << endl;
//            m_mode = MODE_TRAINING;
//        }
    }
}

//  두 이미지를 비교해서 L2 오차를 도출하는 함수. (두 영상간의 뺄셈을 제곱한 값을 합한후 제곱근얹는 방법)
double calculateSimilarity(const Mat A, const Mat B) {
    if ((A.rows > 0) && (A.rows == B.rows) && (A.cols > 0) && (A.cols == B.cols)) {
        //  두 이미지의 L2 상대 오차를 계산
        double L2error = norm(A, B, CV_L2);
        //  스케일을 변환. L2 오차가 이미지의 모든 픽셀에 걸쳐 합했으므로
        double similarity = L2error / (double)(A.rows * A.cols);
        cout << "Similarity of two matrix is " << similarity << endl;
        return similarity;
    }
    else {
        cout << "Warning!: Image have a different size in 'getSimilarity()" << endl;
        return 100000000.0; //  이상한 값 리턴
    }
}

//  백 프로젝션을 사용해서 저장되어있는 얼굴들로 얼굴을 재구성하는 함수.
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace){
    
    //  얼굴 재구성 과정은 PCA랑 피셔에서만 가능하므로 트라이캐치로 묶음.
    try {
        Mat eigenvectors = model->get<Mat>("eigenvectors");
        Mat averageFaceRow = model->get<Mat>("mean");
        int faceHeight = preprocessedFace.rows;
        
        //  입력된 얼굴들을 PCA 고유공간으로 프로젝션 실시
        Mat projection = subspaceProject(eigenvectors, averageFaceRow, preprocessedFace.reshape(1,1));
        
        //  PCA 고유 공간에서부터 반대로 얼굴을 재구성
        Mat reconstructionRow = subspaceReconstruct(eigenvectors, averageFaceRow, projection);
        
        //  단일 행 이미지가 아닌 직사각형 모양으로 이미지 변환
        Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
        //  부동 소수점 픽셀을 8비트 픽셀로 변환
        Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
        
        return reconstructedFace;
        
    } catch (cv::Exception e) {
        return Mat();
    }
}

//  수집된 얼굴들로부터 트레이닝을 실행하는 함수.
Ptr<FaceRecognizer> trainFromCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels, const string facerecAlgorithm){
    Ptr<FaceRecognizer> model;
    
    cout << "Training the collected faces..." << endl;
    
    //  FaceRecognizer 클래스는 contrib 모듈에 있음. 필요할때만 런타임으로 동적 로딩해주어야함.
    bool haveContribModule = initModule_contrib();
    if (!haveContribModule) {
        cerr << "Error! The 'contrib' module is needed for FaceRecognizer but hasn't been loaded into OpenCV" << endl;
        exit(1);
    }
    //  FaceRecognizer의 객체를 생성
    model = Algorithm::create<FaceRecognizer>(facerecAlgorithm);
    if (model.empty()) {
        cerr << "Error! The Algorithm is not available in your OpenCV version. Please update to OpenCV v.2.4.1 or newer." << endl;
        exit(1);
    }
    
    //  생성된 알고리즘 객체로 실제 트레이닝을 수행. 오래걸릴수도있음
    model->train(preprocessedFaces, faceLabels);
    
    //  생성완료된 트레이닝 모델을 xml파일로 저장.
    string savefilename = "user" + toString(m_selectedUser) + "trainModel";
    
    model->save("/Users/Versatile75/Desktop/usertrainModel.xml");
    return model;
}

//  얼굴을 찾고 학습해서 인식하는 함수.
void recognizeAndTrain(VideoCapture &videoCapture, CascadeClassifier &faceDetector, CascadeClassifier &eyeDetector1, CascadeClassifier &eyeDetector2){
    Ptr<FaceRecognizer> model;
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;
    Mat old_preprocessedFace;
    double old_time = 0;
    
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
        
        //  얼굴을 찾고, 사이즈와 명암과 밝기를 맞춤.
        Rect faceRect;  //  탐지된 얼굴의 위치
        Point leftEyeCenter, rightEyeCenter;    //  찾은 눈의 중심 좌표
        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceDetector, eyeDetector1, eyeDetector2, &faceRect, &leftEyeCenter, &rightEyeCenter);

        bool isFaceAndEyesFound = false;
        if (preprocessedFace.data)
            isFaceAndEyesFound = true;
        
        //  위에서 정의한 사각형을 탐지항 얼굴주위에 그림
        if (faceRect.width > 0) {
            rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);
            
            //  눈을찾아서 원을 그림
            Scalar eyeColor = CV_RGB(0, 255, 255);
            if (leftEyeCenter.x >= 0) //  왼쪽눈을 찾았으면 그림
                circle(displayedFrame, Point(faceRect.x + leftEyeCenter.x, faceRect.y + leftEyeCenter.y), 6, eyeColor, 1, CV_AA);
            if (rightEyeCenter.x >= 0) //  오른쪽 눈을 찾았으면 그림
                circle(displayedFrame, Point(faceRect.x+rightEyeCenter.x, faceRect.y+rightEyeCenter.y), 6, eyeColor, 1, CV_AA);
        }
        if (m_mode == MODE_DETECTION) {
            //  탐지모드일때는 다른일은 하지 않음.
        }
        else if (m_mode == MODE_COLLECT_FACES){
            //  탐지된 얼굴이 있다면
            if (isFaceAndEyesFound) {
                //  탐지된 얼굴이 이전의 수집된 얼굴과 다른지 확인
                double imageDifference = 10000000000.0;
                if (old_preprocessedFace.data) {
                    imageDifference = calculateSimilarity(preprocessedFace, old_preprocessedFace);
                }
                
                //  최소한 1초 간격으로 새로운 얼굴을 수집하도록하기위해, 시간의 경과를 측정
                double current_time = (double)getTickCount();
                double timeDifference_seconds = (current_time - old_time)/getTickFrequency();
                
                //  시간과 유사도 임계값 이상인 경우에만 얼굴 처리.
                if ((imageDifference > THRESHOLD_OF_SIMILARITY) && (timeDifference_seconds > THRESHOLD_OF_TIME)) {
                    //  해당 이미지의 미러 이미지를 트레이닝 셋에 추가. 이렇게하면 비대칭 얼굴의 문제점을 줄일수 있음. 얼굴의 왼쪽 오른쪽을 각각 다룸
                    Mat mirrorFace;
                    flip(preprocessedFace, mirrorFace, 1);
                    
                    //  탐지된 얼굴 리스트에 해당 얼굴과 미러 얼굴을 추가
                    preprocessedFaces.push_back(preprocessedFace);
                    preprocessedFaces.push_back(mirrorFace);
                    faceLabels.push_back(m_selectedUser);
                    faceLabels.push_back(m_selectedUser);
                    
                    //  각 사용자의 최신 얼굴의 레퍼런스를 유지
                    m_latestFaces[m_selectedUser] = (int)preprocessedFaces.size() - 2;
                    //  미러가 아닌 얼굴이미지를 가리킴.
                    
                    //  수집된 얼굴 이미지의 갯수를 출력. 미러 이미지도 같이 저장하므로, /2
                    cout << "Saved face" << (preprocessedFaces.size()/2) << " for person"<< m_selectedUser << endl;
                    
                    //  플래시, 사진 찍히는 효과. 사용자는 이걸보고 사진 찍힌걸 알수 있음
                    Mat flashFace = displayedFrame(faceRect);
                    flashFace += CV_RGB(90, 90, 90);
                    
                    //  다음번 반복에서 비교로 사용할 얼굴의 복사본을 유지.
                    old_preprocessedFace = preprocessedFace;
                    old_time = current_time;
                }
            }
        }
        else if (m_mode == MODE_TRAINING) {
            //  트레이닝 할 충분한 데이터가 있는지를 확인. 적어도 2명이 필요.
            bool haveEnoughData = true;
            if (strcmp(facerecAlgorithm, "FaceRecognizer.Fisherfaces") == 0) {
                if ((m_numUsers < 2) || (m_numUsers == 2 && m_latestFaces[1] < 0)) {
                    cout << "Warning: Fisherfaces needs at least 2 people, otherwise there is nothing to differentiate! Collect more data ..."<< endl;
                    haveEnoughData = false;
                }
            }
            //  Eigenfaces일 경우에는 한명이상
            if (m_numUsers < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) {
                cout << "Warning: Need some training data before it can be learnt! Collect more data ..." << endl;
                haveEnoughData = false;
            }
    
            if (haveEnoughData) {
                //  수집된 얼굴들을 사용해서 트레이닝 시작.
                model = trainFromCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);
                //  트레이닝 종료후 인식 시작.
                m_mode = MODE_DETECTION;
            }
            else {  //  충분하지 않은 트레이닝 데이터를 가진 경우, 다시 얼굴 수집 모드로 돌아감
                m_mode = MODE_COLLECT_FACES;
            }
        }
        else if (m_mode == MODE_RECOGNITION){
            if (isFaceAndEyesFound && (preprocessedFaces.size() >0) && (preprocessedFaces.size() == faceLabels.size())) {
                //  저장한 파일로부터 트레이닝 모델 로드
                Ptr<FaceRecognizer> modelFromFile;
                modelFromFile = Algorithm::create<FaceRecognizer>(facerecAlgorithm);
                Mat labels;
                
                try {
                    modelFromFile->load("/Users/Versatile75/Desktop/usertrainModel.xml");
                    labels = modelFromFile->get<Mat>("labels");
                } catch (cv::Exception &e) {
                    if (labels.rows <= 0) {
                        cerr << "Error: Couldn't load trained data from user" + toString(m_selectedUser)+ "trainModel.xml" << endl;
                        exit(1);
                    }
                }
                
                //  얼굴을 재구성해서 입력된 영상과 비교함.
                Mat reconstructedFace;
                reconstructedFace = reconstructFace(modelFromFile, preprocessedFace);

                //  재구성된 얼굴이 선처리된 얼굴과 같은지를 비교.
                double similarity = calculateSimilarity(preprocessedFace, reconstructedFace);
                
                string outputStr;
                Rect confidenceMessageRectangle;
                string confidenceMessage;
                int cx = BORDER;
                if (similarity < UNKNOWN_PERSON_THRESHOLD) {
                    //  선처리된 얼굴의 사람을 알아냄.
                    identity = modelFromFile->predict(preprocessedFace);
                    outputStr = toString(identity);
                    
                    //  신뢰도의 값을 0.0과 1.0사이의 값으로 잘라냄.
                    int confidenceRatio = (1.0 - min(max(similarity, 0.0), 1.0))*100;
                    confidenceMessage = toString(confidenceRatio) + "% Matched with User No." + toString(m_selectedUser);
                    
                    confidenceMessageRectangle = drawButton(displayedFrame, confidenceMessage, Point(cx, BORDER), 100);
                }
                else {
                    //  신뢰도가 낮으므로 허가받지 않은 사용자로 간주
                    outputStr = "Unknown";
                    confidenceMessageRectangle = drawButton(displayedFrame, outputStr, Point(cx, BORDER), 100);
                }
                cout<< "Identity: " << outputStr << ". Similarity: " << similarity << endl;
            }
        }
        else if (m_mode == MODE_RESET){
            //  전과정을 재시작
            m_selectedUser = -1;
            m_numUsers = 0;
            m_latestFaces.clear();
            preprocessedFaces.clear();
            faceLabels.clear();
            old_preprocessedFace = Mat();
        }
        else{
            cerr << "Error! Invalid run mode" << m_mode << endl;
            exit(1);
        }
        
//  도움말 메시지를 출력하는 과정. 현재 저장된 사람과 얼굴숫자도 출력
        string helpTextMessage;
        Rect rectangleHelpMessage;
        if (m_mode == MODE_DETECTION) {
            helpTextMessage = "Click [New User], then start collecting faces.";
        }
        else if (m_mode == MODE_COLLECT_FACES) {
            helpTextMessage = "Click anywhere to train from your " + toString(preprocessedFaces.size()/2) + " faces of "+ toString(m_numUsers)+ " people.";
        }
        else if (m_mode == MODE_TRAINING) {
            helpTextMessage = "Please wait while your "+ toString(preprocessedFaces.size()/2) + " faces of "+ toString(m_numUsers)+ " people builds.";
        }
        else if (m_mode == MODE_RECOGNITION){
            helpTextMessage = "Try to Login...";
        }
        else if (m_mode == MODE_RESET) {
            helpTextMessage = "Reset All User Data. Click [New User] for someone new.";
        }
        if (helpTextMessage.length() > 0) {
            //  글자색은 하얀색. 글자 음영색은 검정색으로 지정.
            //  BORDER값이 0이 되는데 음수값이 필요함으로, 2를 빼서 항상 음수값으로 유지
            float txtSize = 0.4;
            drawTextString(displayedFrame, helpTextMessage, Point(-BORDER, -BORDER-2), CV_RGB(0, 0, 0), txtSize);
            rectangleHelpMessage = drawTextString(displayedFrame, helpTextMessage, Point(-BORDER+1, -BORDER-1), CV_RGB(255, 255, 255), txtSize);
        }
        
//  현재 모드를 보여줌
        if (m_mode >= 0 && m_mode < MODE_END) {
            string modeString = "MODE: "+string(MODE_NAMES[m_mode]);
            drawTextString(displayedFrame, modeString, Point(-BORDER, -BORDER-2-rectangleHelpMessage.height), CV_RGB(0, 0, 0));
            drawTextString(displayedFrame, modeString, Point(-BORDER+1, -BORDER-1-rectangleHelpMessage.height), CV_RGB(0, 0, 255));   //  글자색 파란색
        }
        
//  GUI 버튼을 그림
        m_rectangleButtonRegisterNewUser = drawButton(displayedFrame, "New User", Point(BORDER, BORDER+350), 125);
        m_rectangleButtonLoginExistingUser = drawButton(displayedFrame, "Login Now", Point(m_rectangleButtonRegisterNewUser.x, m_rectangleButtonRegisterNewUser.y+m_rectangleButtonRegisterNewUser.height), m_rectangleButtonRegisterNewUser.width);
        m_rectangleButtonReset = drawButton(displayedFrame, "Reset All", Point(m_rectangleButtonLoginExistingUser.x, m_rectangleButtonLoginExistingUser.y+m_rectangleButtonLoginExistingUser.height), m_rectangleButtonLoginExistingUser.width);

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
 
//  마우스로 선택된 사람의 테두리를 빨간색으로 표현하며 강조하는 과정
        if (m_mode == MODE_COLLECT_FACES) {
            if (m_selectedUser >= 0 && m_selectedUser < m_numUsers) {
                int y = min(m_gui_faces_top+m_selectedUser*faceHeight, displayedFrame.rows - faceHeight);
                Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                rectangle(displayedFrame, rc, CV_RGB(255, 0, 0), 3, CV_AA);
            }
        }
        
//  얼굴인식과정에서 인식된 사람의 테두리를 파란색으로 표현하며 강조하는 과정
        if (identity >= 0 && identity < 1000) {
            int y = min(m_gui_faces_top + identity*faceHeight, displayedFrame.rows - faceHeight);
            Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
            rectangle(displayedFrame, rc, CV_RGB(0, 0, 255), 3, CV_AA);
        }
        
//  스크린에 카메라 프레임 출력
        imshow(windowName, displayedFrame);

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
    
    //  xml을 로딩
    loadXMLs(faceDetector, eyeDetector1, eyeDetector2);
    
    cout << "Click 'Esc' in the GUI window to quit." << endl;
    
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