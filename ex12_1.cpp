//
//  ex12_1.cpp
//  OpenCVTest
//
//  Created by Versatile75 on 2016. 5. 1..
//  Copyright © 2016년 Versatile75. All rights reserved.
//  얼굴 데이터베이스에서 얼굴 찾기

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cstdio>

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>

int main(){
    const int M = 8;    //  전체 사용되는 이미지 갯수
    const int nEigens = 7;  //  고유 오브젝트의 갯수
    int k;
    char buf[80];
    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 8, 0.05);
    IplImage* objects[M];
    IplImage* eigenObjects[M];
    IplImage* proj; //  고유공간상에서 분해되어 투영된 영상
    float eigVals[M];
    IplImage* avg;
    
    for (k=1; k <=M; k++) {
        sprintf(buf, "/Users/Versatile75/Desktop/facedatabase/face%d.bmp", k);  //  트레이닝 샘플들 로딩
        objects[k-1] = cvLoadImage(buf, 0);
    }
    
    CvSize size = cvGetSize(objects[0]);
    
    avg = cvCreateImage(size, IPL_DEPTH_32F, 1);
    for (k=0; k<M; k++) {   //  고유 얼굴 계산해서 eigenObjects에 넣음
        eigenObjects[k] = cvCreateImage(size, IPL_DEPTH_32F, 1);
    }
    cvCalcEigenObjects(M, objects, eigenObjects, 0, 0, 0, &criteria, avg, eigVals);
    
    //  평균 얼굴
    IplImage* out = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvConvertScale(avg, out, 1, 0);
    
    sprintf(buf, "/Users/Versatile75/Desktop/facedatabase/avg_face.bmp");
    cvSaveImage(buf, out);
    
    //  1부터 6까지의 고유얼굴을 사용해서 보여줌
    IplImage* newone = cvLoadImage("/Users/Versatile75/Desktop/face6.bmp", 0);
    float coeffs[nEigens];
    proj = cvCreateImage(size, IPL_DEPTH_8U, 1);
    
    //  분해 계수 계산.
    cvEigenDecomposite(newone, nEigens, eigenObjects, 0, 0, avg, coeffs);
    
    //  영상 재구성
    cvEigenProjection(eigenObjects, nEigens, 0, 0, coeffs, avg, proj);
    
    sprintf(buf, "/Users/Versatile75/Desktop/facedatabase/project_face6.bmp");
    cvSaveImage(buf, proj);
    cvReleaseImage(&proj);
}