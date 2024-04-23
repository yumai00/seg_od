#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <cmath>
#include "undistortimage_transform_vehicle.h"
// #include "samples/run_infer/CameraTransforms.h"
CameraConfig create_camera_config(const Eigen::Vector3d &euler_angles_deg, const Eigen::Vector3d &T, const Eigen::Vector3d &T_vehicle, const Eigen::Matrix3d &K)
{
    CameraConfig config;
    config.K = K;
    config.T = T;
    config.T_vehicle = T_vehicle;
    config.R = eulerToRotationMatrix(euler_angles_deg);
    return config;
}

Eigen::Matrix3d eulerToRotationMatrix(const Eigen::Vector3d &euler_angle)
{
    double theta_x = euler_angle[0] * M_PI / 180;
    double theta_y = euler_angle[1] * M_PI / 180;
    double theta_z = euler_angle[2] * M_PI / 180;

    Eigen::AngleAxisd rollAngle(theta_x, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(theta_y, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(theta_z, Eigen::Vector3d::UnitZ());

    Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;
    return q.matrix();
}

// Eigen::Vector3d transformUndistortPointTOVehiclePoint(const cv::Point2d &pixel, const CameraConfig &config)
// {
//     // todo  config['K'][(0, 0)] *= 0.8  config['K'][(1, 1)] *=0.8
//     Eigen::Vector3d normalized_pixel = config.K.inverse() * Eigen::Vector3d(pixel.x, pixel.y, 1.0);
//     Eigen::Vector3d point_in_cam = config.R * normalized_pixel;

//     double scale = -config.T.z() / point_in_cam.z();
//     Eigen::Vector3d point_in_world = point_in_cam * scale + config.T;

//     Eigen::Vector3d point_in_vehicle_transformed = config.R.inverse() * (point_in_world - config.T);
//     Eigen::Vector3d point_in_vehicle = config.R * point_in_vehicle_transformed + config.T_vehicle;

//     return point_in_vehicle;
// }

Eigen::Vector3d transformUndistortPointTOVehiclePoint(const cv::Point2d &pixel, const CameraConfig &config)
{
    // Create a local copy of K to modify
    Eigen::Matrix3d modifiedK = config.K;

    // Apply the scaling as per the todo instructions
    modifiedK(0, 0) *= 0.8; // Scale the (0, 0) element of K
    modifiedK(1, 1) *= 0.8; // Scale the (1, 1) element of K

    // Calculate the normalized pixel coordinates using the modified intrinsic matrix
    Eigen::Vector3d normalized_pixel = modifiedK.inverse() * Eigen::Vector3d(pixel.x, pixel.y, 1.0);

    // Transform the normalized pixel coordinates to camera coordinates
    Eigen::Vector3d point_in_cam = config.R * normalized_pixel;

    // Project the point onto the world coordinate system
    double scale = -config.T.z() / point_in_cam.z();
    Eigen::Vector3d point_in_world = point_in_cam * scale + config.T;

    // Transform the world coordinates back to the vehicle coordinate system
    Eigen::Vector3d point_in_vehicle_transformed = config.R.inverse() * (point_in_world - config.T);
    Eigen::Vector3d point_in_vehicle = config.R * point_in_vehicle_transformed + config.T_vehicle;

    return point_in_vehicle;
}

Eigen::Vector3d UndistortImageTransformVehicle(const Eigen::Vector2d &pixel, CameraPosition position)
{
    CameraConfig back_cam = create_camera_config(
        Eigen::Vector3d(-121.16317, -0.868073344, 90.9461212),
        Eigen::Vector3d(-2286.2369233895188, -6.3787465925465696, 914.2119986738741),
        Eigen::Vector3d(-1208, 0, 583),
        (Eigen::Matrix3d() << 441.81600952148438 / 2.5, 0.0, 958.458984375 / 2.5,
         0.0, 441.77301025390625 / 2.5, 648.8740234375 / 2.5,
         0.0, 0.0, 1.0)
            .finished());

    CameraConfig front_cam = create_camera_config(
        Eigen::Vector3d(-1.06509781e+02, 7.31059790e-01, -8.97128220e+01),
        Eigen::Vector3d(2.2862369233895188e+03, -5.9252149861791210e+00, 7.7872672778567824e+02),
        Eigen::Vector3d(3545, 0, 398),
        (Eigen::Matrix3d() << 4.4443701171875000e+02 / 2.5, 0, 9.5931500244140625e+02 / 2.5,
         0, 4.4425399780273438e+02 / 2.5, 6.4693200683593750e+02 / 2.5,
         0, 0, 1)
            .finished());

    CameraConfig right_cam = create_camera_config(
        Eigen::Vector3d(-1.30116196e+02, -2.60405123e-01, -1.79918472e+02),
        Eigen::Vector3d(6.6813442520183480e+02, -1.0637217875840397e+03, 1.1587540232902436e+03),
        Eigen::Vector3d(1930, -1056, 793),
        (Eigen::Matrix3d() << 4.4294400024414062e+02 / 2.5, 0, 9.5840600585937500e+02 / 2.5,
         0, 4.4273800659179688e+02 / 2.5, 6.4796502685546875e+02 / 2.5,
         0, 0, 1)
            .finished());

    CameraConfig left_cam = create_camera_config(
        Eigen::Vector3d(-1.28602432e+02, 3.31160843e-01, -8.45068693e-02),
        Eigen::Vector3d(8.9314087652075648e+02, 1.1356942216086977e+03, 1.1605765889153931e+03),
        Eigen::Vector3d(2364.33, 1115.18, 1237.11),
        (Eigen::Matrix3d() << 4.4378799438476562e+02 / 2.5, 0, 9.5816400146484375e+02 / 2.5,
         0, 4.4360900878906250e+02 / 2.5, 6.4745501708984375e+02 / 2.5,
         0, 0, 1)
            .finished());

    cv::Point2d cv_pixel(pixel.x(), pixel.y());

    switch (position)
    {
    case BACK_CAMERA:
        return transformUndistortPointTOVehiclePoint(cv_pixel, back_cam);
    case FRONT_CAMERA:
        return transformUndistortPointTOVehiclePoint(cv_pixel, front_cam);
    case LEFT_CAMERA:
        return transformUndistortPointTOVehiclePoint(cv_pixel, left_cam);
    case RIGHT_CAMERA:
        return transformUndistortPointTOVehiclePoint(cv_pixel, right_cam);
    default:
        std::cerr << "Invalid camera position." << std::endl;
        return Eigen::Vector3d::Zero(); // 返回一个默认值以避免编译错误
    }
}
