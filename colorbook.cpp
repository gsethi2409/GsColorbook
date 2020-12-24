#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <iostream>
using namespace cv;

class ColorBook{
    private:
        Mat image;
        double sigmaX;
        double sigmaY;
    public:
        ColorBook(Mat img);
        Mat preblur(Mat img, double sigmaX, double sigmaY);
        Mat normalize_image(Mat img);
        Mat color_quantize(Mat img);  
};

ColorBook::ColorBook(Mat img)
{
    image = img;
    sigmaX = 2;
    sigmaY = 0;
    Mat blurred = preblur(img, sigmaX, sigmaY);
    imshow( "Blurred Image", blurred);
    waitKey(5000);
    Mat normalized_blur = normalize_image(blurred);
    imshow( "Normalized Image", normalized_blur);
    waitKey(5000);
    imshow("Quantized", color_quantize(normalized_blur));
    waitKey(5000);
}

Mat ColorBook::preblur(Mat img, double sigmaX, double sigmaY)
{
    Mat blurred_img;
    GaussianBlur( img, blurred_img, Size( 0, 0 ), sigmaX, sigmaY );
    return blurred_img;
}

Mat ColorBook::normalize_image(Mat img)
{
    Mat dst;
    normalize(img, dst, 1, 0, NORM_MINMAX, CV_32F);
    return dst;
}

Mat ColorBook::color_quantize(Mat img)
{
    Mat data;
    img.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    Mat labels, centers;
    kmeans(data, 8, labels, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);

    Vec3f *p = data.ptr<Vec3f>();
    for (size_t i=0; i<data.rows; i++) {
        int center_id = labels.at<int>(i);
        p[i] = centers.at<Vec3f>(center_id);
    }

    img = data.reshape(3, img.rows);
    img.convertTo(img, CV_8U);
    
    return img;
}

int main(int argc, char *argv[]){
    if(argv[1]){
        Mat src = imread(argv[1]);
        ColorBook book(src);
    }
    else{
        Mat img = imread( "/home/gsethi2409/Personal-Projects/image-processing/ColorBooked/examples/barrhorn/barrhorn_colorized12.png", 1);
        ColorBook book(img);
    }
    return 0;
}  