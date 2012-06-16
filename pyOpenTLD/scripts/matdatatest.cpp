#include<opecv2/highgui/highgui.hpp>
#include<stdio.h>
using namespace cv
int main()
{
    Mat img = imread("~/Pictures/python.jpg",CV_LOAD_IMAGE_COLOR)
    d = img.data
    return 0;
}
