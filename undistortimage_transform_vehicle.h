#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm> 

struct CameraConfig
{
    Eigen::Matrix3d K;         // 内参矩阵
    Eigen::Matrix3d R;         // 旋转矩阵
    Eigen::Vector3d T;         // 平移向量
    Eigen::Vector3d T_vehicle; // 车辆坐标系下的平移向量
};

enum CameraPosition
{
    BACK_CAMERA,
    FRONT_CAMERA,
    LEFT_CAMERA,
    RIGHT_CAMERA
};

Eigen::Vector3d UndistortImageTransformVehicle(const Eigen::Vector2d &pixel, CameraPosition position);
Eigen::Matrix3d eulerToRotationMatrix(const Eigen::Vector3d &euler_angle);
Eigen::Vector3d transformUndistortPointTOVehiclePoint(const cv::Point2d &pixel, const CameraConfig &config);
CameraConfig createCameraConfig(const Eigen::Vector3d &euler_angles_deg, const Eigen::Vector3d &T, const Eigen::Vector3d &T_vehicle, const Eigen::Matrix3d &K);