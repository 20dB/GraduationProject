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

const char *facerecAlgorithm = "FaceRecognizer.LBPH";    //  얼굴인식 알고리즘 선택 -> LBP

//  등록된 사람인지 아닌지를 결정하는 얼굴 인식 알고리즘의 신뢰도를 설정. 높아질수록 등록된 사람으로 판단
const float UNKNOWN_PERSON_THRESHOLD = 0.7f;

//  LBP Cascade 분류기 설정.
//  가장 정확한게 left/righteye_2splits, 그다음이 mcs_left/righteye, 기본이 haarcascade_eye/eye_tree_eyeglasses
const char *faceCascadeFilename = "/Users/Versatile75/Documents/XcodeWorkspace/GraduationProject2016/GraduationProject2016/lbpcascade_frontalface.xml";
const char *eyeCascadeFilename1 = "/Users/Versatile75/Documents/XcodeWorkspace/GraduationProject2016/GraduationProject2016/haarcascade_lefteye_2splits.xml";
const char *eyeCascadeFilename2 = "/Users/Versatile75/Documents/XcodeWorkspace/GraduationProject2016/GraduationProject2016/haarcascade_righteye_2splits.xml";

//  얼굴의 차원수(Dimension)를 설정.
//  getPreprocessedFace()함수가 정사각형 얼굴을 리턴하기때문에 faceWidth=faceHeight.
const int faceWidth = 70;
const int faceHeight = faceWidth;

//  카메라의 해상도를 설정. 카메라/시스템에따라 안맞을수도있음
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

//  얼마나 자주 새로운 얼굴을 저장할지를 정하는 파라메터.
const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;      //  트레이닝할 때, 얼마나 자주 얼굴 이미지가 바뀌어야하는지를 설정
const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;       //   트레이닝할 때, 시간이 얼마나 지나야 하는지를 설정.

const char *windowName = "Facial Recognition";   // GUI 화면창의 이름
const int BORDER = 8;  //   GUI 엘리먼트들과 사진의 모서리간의 경계값

//  각 과정에서의 이미지를 창으로 확인하고싶으면 true, 아니면 false
bool m_debug = false;

//  GUI상에 나타나는 모드 설정
enum MODES {MODE_STARTUP=0, MODE_DETECTION, MODE_COLLECT_FACES, MODE_TRAINING, MODE_RECOGNITION, MODE_DELETE_ALL, MODE_END};
const char* MODE_NAMES[] = {"Running", "Detection", "Collect Faces", "Training", "Recognition", "Delete All", "ERROR!"};
MODES m_mode = MODE_STARTUP;

int m_selectedUser = -1;
int m_numUsers = 0;
vector<int> m_latestFaces;

//  GUI 버튼의 위치 설정
Rect m_rcBtnAdd;
Rect m_rcBtnDel;
Rect m_rcBtnDebug;
int m_gui_faces_left = -1;
int m_gui_faces_top = -1;

/**
 물체 찾기에 연관된 함수들
 - 입력데이터는 빠른 탐지를 위해서, scaledWidth 만큼 축소되어있음. 얼굴찾기용으로는 240이 적당.
 
 1. detectLargestObject 함수
    이미지에서 가장 큰 얼굴과같은 하나의 오브젝트를 찾는 함수. 찾은 오브젝트를 largestObject에 저장함.
 2. detectManyObjects 함수
    이미지에서 모든얼굴처럼 다수의 오브젝트를 찾는 함수. 찾은 오브젝트를 objects에 저장.
 */
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth = 320);
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth = 320);

/**
 얼굴 이미지 사전 처리에 관련된 함수들
 
 1. detectBothEyes 함수
    주어진 이미지로부터 두눈을 찾아내는 함수. 각각의 눈의 중점 좌표를 leftEye 와 rightEye 로 리턴. 실패시 (-1, -1)로 반환.
    찾은 왼쪽 오른쪽 눈의 영역을 저장 가능... (추가해야하나..)
 2. equlizeLeftAndRightHalf 함수
    얼굴의 양쪽을 각각 히스토그램으로 평준화하는 함수. 얼굴 한쪽면에만 빛을 받을경우 이거로 평준화함.
 3. getPreprocessedFace 함수
    흑백이미지로 주어진 이미지를 변환. srcImg 매개변수는 전체 카메라 프레임의 복사본. 그래야 눈의 좌표를 그릴수 있음.
    선처리 과정에는 다음과정들이 포함됨.
        1. 눈 탐지를 통한 비율 줄이기, 회전과 트랜슬레이션.
        2. Bilateral 필터를 사용한 이미지의 노이즈 제거
        3. 히스토그램 평준화를 얼굴 왼쪽 오른쪽에 각각 적용해서 밝기를 평준화.
        4. 타원형으로 얼굴 마스크를 잘라서 배경과 머리 지우기.
    선처리된 얼굴 정사각형 이미지를 리턴. 실패시 NULL == 눈과 얼굴을 찾지 못한경우.
    얼굴이 찾아지면, 얼굴 직사각형 좌표는 storeFaceRect에 저장하고, 눈은 storeLeftEye, storeRightEye에 각각 저장. 눈 영역은 searchedLeftEye와 searchedRightEye에 저장.
 */
void detectBothEyes(const Mat &face, CascadeClassifier&eyeCascade1, CascadeClassifier &eyeCascade2, Point &leftEye, Point &rightEye, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL);
void equalizeLeftAndRightHalf(Mat &faceImg);
Mat getPreprocessedFace(Mat &srcImg, int desiredFaceWidth, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, Rect *storeFaceRect = NULL, Point *storeLeftEye = NULL, Point *storeRightEye = NULL, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL);

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


//  XML 분류기 로딩하는 함수.
void loadXMLs(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    //  얼굴 검출 xml 로딩
    try {
        faceCascade.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if ( faceCascade.empty() ) {
        cerr << "ERROR: Could not load Face Detection cascade classifier [" << faceCascadeFilename << "]!" << endl;
        exit(1);
    }
    cout << "Loaded the Face Detection cascade classifier [" << faceCascadeFilename << "]." << endl;
    
    //  눈 검출 xml 로딩
    try {
        eyeCascade1.load(eyeCascadeFilename1);
    } catch (cv::Exception &e) {}
    if ( eyeCascade1.empty() ) {
        cerr << "ERROR: Could not load 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]!" << endl;
        exit(1);
    }
    cout << "Loaded the 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]." << endl;
    
    //  눈 검출 xml 로딩
    try {
        eyeCascade2.load(eyeCascadeFilename2);
    } catch (cv::Exception &e) {}
    if ( eyeCascade2.empty() ) {
        cerr << "Could not load 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
        //  두번째 눈 검출 xml이 없어도 종료되지 않음. 왜냐하면 눈검출 xml을 하나만 사용한다는 뜻이기 때문에..
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
Rect drawButton(Mat img, string text, Point coord, int minWidth = 0) {
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
    if (isPointInRect(pt, m_rcBtnAdd)) {
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
    else if (isPointInRect(pt, m_rcBtnDel)) {
        cout << "User clicked [Delete All] button." << endl;
        m_mode = MODE_DELETE_ALL;
    }
    else if (isPointInRect(pt, m_rcBtnDebug)) {
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
void recognizeAndTrain(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2){
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
        Rect searchedLeftEye, searchedRightEye; //  얼굴의 좌상단과 우상단 = 눈사각형들
        Point leftEye, rightEye;    //  찾은 눈의 좌표
        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);
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
            if (leftEye.x >= 0) {
                //  왼쪽눈을 찾았으면 그림
                circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
            }
            if (rightEye.x >= 0) {
                //  오른쪽 눈을 찾았으면 그림
                circle(displayedFrame, Point(faceRect.x+rightEye.x, faceRect.y+rightEye.y), 6, eyeColor, 1, CV_AA);
            }
        }
        cerr << "C" << endl;
        if (m_mode == MODE_DETECTION) {
            cerr << "D" << endl;    //  탐지모드일때는 다른일은 하지 않음.
        }
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
                if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {
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
        m_rcBtnAdd = drawButton(displayedFrame, "Add User", Point(BORDER, BORDER+300));
        m_rcBtnDel = drawButton(displayedFrame, "Delete All User", Point(m_rcBtnAdd.x, m_rcBtnAdd.y+m_rcBtnAdd.height), m_rcBtnAdd.width);
        m_rcBtnDebug = drawButton(displayedFrame, "Login", Point(m_rcBtnDel.x, m_rcBtnDel.y+m_rcBtnDel.height), m_rcBtnAdd.width);
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
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;
    VideoCapture videoCapture;
    
    cout << "Face Detection & Face Recognition using LBP." << endl;
    cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;
    
    //  xml을 로딩
    loadXMLs(faceCascade, eyeCascade1, eyeCascade2);
    
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
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);
    
    //  GUI 창을 생성.
    namedWindow(windowName);
    //  사용자가 창을 클릭할때, onMouse 함수를 호출하기위한 OpenCV함수 setMouseCallback
    setMouseCallback(windowName, onMouse, 0);
    
    //  얼굴인식 실행.
    recognizeAndTrain(videoCapture, faceCascade, eyeCascade1, eyeCascade2);
    
    return 0;
}