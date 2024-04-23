#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <utility>
#include <nlohmann/json.hpp>
#include "undistortimage_transform_vehicle.h"

// 将字符串转换为 CameraPosition 枚举
CameraPosition stringToCameraPosition(const std::string &camera_side)
{
    if (camera_side == "back")
        return BACK_CAMERA;
    if (camera_side == "front")
        return FRONT_CAMERA;
    if (camera_side == "left")
        return LEFT_CAMERA;
    if (camera_side == "right")
        return RIGHT_CAMERA;
    throw std::runtime_error("Invalid camera position string");
}
struct PointHash
{
    size_t operator()(const cv::Point &point) const
    {
        return std::hash<int>{}(point.x) ^ (std::hash<int>{}(point.y) << 1);
    }
};

struct PointEquals
{
    bool operator()(const cv::Point &a, const cv::Point &b) const
    {
        return a.x == b.x && a.y == b.y;
    }
};

cv::Mat loadBinMap(const std::string &cameraSide, const std::string &binFilePath, const cv::Vec3i &dims)
{
    std::ifstream file(binFilePath + "/" + cameraSide + ".bin", std::ios::binary);
    if (!file)
    {
        std::cerr << "Error opening bin file" << std::endl;
        return cv::Mat();
    }

    // Calculate total data size
    size_t dataSize = dims[0] * dims[1] * dims[2] * sizeof(short);
    std::vector<short> buffer(dataSize / sizeof(short));

    // Read data
    file.read(reinterpret_cast<char *>(buffer.data()), dataSize);
    if (!file.good())
    {
        std::cerr << "Error reading bin file" << std::endl;
        return cv::Mat();
    }

    // Create Mat object for mapping
    cv::Mat map(dims[0], dims[1], CV_16SC2, buffer.data());
    return map.clone(); // Return a copy to ensure data remains valid
}

std::unordered_map<cv::Point, std::pair<int, int>, PointHash, PointEquals> createLookupTable(const cv::Mat &mapping)
{
    std::unordered_map<cv::Point, std::pair<int, int>, PointHash, PointEquals> lut;
    for (int y = 0; y < mapping.rows; ++y)
    {
        for (int x = 0; x < mapping.cols; ++x)
        {
            cv::Vec2s point = mapping.at<cv::Vec2s>(y, x);
            cv::Point key(point[0], point[1]);
            lut[key] = {x, y};
        }
    }
    return lut;
}

// Global map to store LUTs for each camera side
std::map<std::string, std::unordered_map<cv::Point, std::pair<int, int>, PointHash, PointEquals>> globalLUTs;

void initializeLUTs(const std::string &folder_map, const cv::Vec3i &dims)
{
    std::vector<std::string> cameraSides = {"front", "back", "left", "right"};
    for (const auto &side : cameraSides)
    {
        cv::Mat mapping = loadBinMap(side, folder_map, dims);
        if (!mapping.empty())
        {
            globalLUTs[side] = createLookupTable(mapping);
            std::cout << "LUT for " << side << " created successfully." << std::endl;
        }
        else
        {
            std::cerr << "Failed to create LUT for " << side << std::endl;
        }
    }
}

// struct CameraConfig
// {
//     Eigen::Matrix3d K;         // 内参矩阵
//     Eigen::Vector3d T;         // 世界坐标系中的平移向量
//     Eigen::Vector3d T_vehicle; // 车辆坐标系中的平移向量
//     Eigen::Matrix3d R;         // 旋转矩阵
//     Eigen::Vector4d D;         // 畸变系数
// };

// CameraConfig createCameraConfig(const std::vector<double> &euler_angles_deg, const std::vector<double> &T,
//                                 const std::vector<double> &T_vehicle, const std::vector<double> &K,
//                                 const std::vector<double> &D)
// {
//     Eigen::Vector3d eulerAnglesRad = Eigen::Vector3d(euler_angles_deg[0], euler_angles_deg[1], euler_angles_deg[2]) * M_PI / 180.0;
//     Eigen::Matrix3d R;
//     R = Eigen::AngleAxisd(eulerAnglesRad[2], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(eulerAnglesRad[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(eulerAnglesRad[0], Eigen::Vector3d::UnitX());

//     Eigen::Matrix3d K_matrix;
//     K_matrix << K[0], 0, K[2], 0, K[4], K[5], 0, 0, 1;

//     CameraConfig config = {K_matrix, Eigen::Vector3d(T[0], T[1], T[2]),
//                            Eigen::Vector3d(T_vehicle[0], T_vehicle[1], T_vehicle[2]),
//                            R, Eigen::Vector4d(D[0], D[1], D[2], D[3])};

//     return config;
// }

std::map<std::string, CameraConfig> camera_configs;
struct Point
{
    int16_t x;
    int16_t y;
};

// void fillCameraConfigs()
// {
//     // Back camera configuration
//     camera_configs["back"] = createCameraConfig(
//         {-120.780266, -0.971939385, 91.1036224},
//         {-2285.7464076665333, -14.481644303487741, 908.91618035169768},
//         {-1208, 0, 583},
//         {441.81600952148438 / 2.5, 0, 958.458984375 / 2.5,
//          0, 441.77301025390625 / 2.5, 648.8740234375 / 2.5,
//          0, 0, 1},
//         {0.21072100102901459, -0.04376740085926056, -0.0024176898878067732, 0.0013886999804526567});

//     // Front camera configuration
//     camera_configs["front"] = createCameraConfig(
//         {-105.772903, 0.700113535, -89.5732880},
//         {2285.7464076665333, -1.5845991150219252, 779.27410154366055},
//         {3545, 0, 398},
//         {444.43701171875 / 2.5, 0, 959.31500244140625 / 2.5,
//          0, 444.25399780273438 / 2.5, 646.9320068359375 / 2.5,
//          0, 0, 1},
//         {0.21345600485801697, -0.049494501203298569, 0.0014258699957281351, 0.00073903700103983283});

//     // Right camera configuration
//     camera_configs["right"] = createCameraConfig(
//         {-130.144974, -0.203830391, -179.883392},
//         {670.87355598817055, -1062.4154654553231, 1157.3517852750153},
//         {1930, -1056, 793},
//         {442.94400024414062 / 2.5, 0, 958.406005859375 / 2.5,
//          0, 442.73800659179688 / 2.5, 647.96502685546875 / 2.5,
//          0, 0, 1},
//         {0.21177799999713898, -0.046675201505422592, -0.0000052041900744370651, 0.00099931203294545412});

//     // Left camera configuration
//     camera_configs["left"] = createCameraConfig(
//         {-128.587631, 0.241358072, 0.219709948},
//         {682.32138691587022, 1062.4154654553231, 1161.3495086951327},
//         {1946, 1065, 794},
//         {443.78799438476562 / 2.5, 0, 958.16400146484375 / 2.5,
//          0, 443.6090087890625 / 2.5, 647.45501708984375 / 2.5,
//          0, 0, 1},
//         {0.21273100376129150, -0.048056099563837051, 0.00082143797772005200, 0.00079764798283576965});
// }
float distance(const cv::Point2f &point1, const cv::Point2f &point2)
{
    return std::sqrt(std::pow(point1.x - point2.x, 2) + std::pow(point1.y - point2.y, 2));
}
std::pair<std::vector<cv::Point2f>, std::vector<std::pair<cv::Point2f, cv::Point2f>>> calculateMidpoints(const std::vector<cv::Point> &contour)
{
    cv::RotatedRect rotRect = cv::minAreaRect(contour);
    cv::Point2f box[4];
    rotRect.points(box);

    std::vector<cv::Point2f> points(box, box + 4);
    std::sort(points.begin(), points.end(), [](const cv::Point2f &a, const cv::Point2f &b)
              { return a.y > b.y; });
    points.resize(3); // 保留前三个点

    std::vector<cv::Point2f> midpoints;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> linePoints;

    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = i + 1; j < 3; ++j)
        {
            cv::Point2f midpoint = (points[i] + points[j]) * 0.5f;
            midpoints.push_back(midpoint);
            linePoints.push_back({points[i], points[j]});
        }
    }
    return {midpoints, linePoints};
}
void displayImageWithPointsAndLines(const cv::Mat &image, const std::pair<std::vector<cv::Point2f>, std::vector<std::pair<cv::Point2f, cv::Point2f>>> &data)
{
    cv::Mat displayImage = image.clone();
    if (displayImage.channels() == 1)
    {
        cv::cvtColor(displayImage, displayImage, cv::COLOR_GRAY2BGR);
    }

    // 绘制中点
    for (const auto &point : data.first)
    {
        cv::circle(displayImage, point, 5, cv::Scalar(0, 255, 0), -1); // 使用绿色绘制中点
    }

    // 绘制线段
    for (const auto &line : data.second)
    {
        cv::line(displayImage, line.first, line.second, cv::Scalar(0, 0, 255), 2); // 使用红色绘制线段
    }

    cv::imshow("Processed Image", displayImage);
    cv::waitKey(0);
}
// 假设函数，实际实现需要根据process_image的具体逻辑
// std::vector<cv::Point> processImage(const cv::Mat &imageGray, const std::string &cameraSide)
// {
//     std::map<int, int> sizeFilters = {{2, 800}, {7, 300}}; //{{3, 100}};
//     cv::Point2f imageCenter(imageGray.cols / 2.0f, imageGray.rows);

//     std::vector<cv::Point2f> selectedMidpoint;
//     std::vector<std::pair<cv::Point2f, cv::Point2f>> selectedLineSegment;
//     std::vector<cv::Point> allPoints;

//     for (const auto &filter : sizeFilters)
//     {
//         int maskValue = filter.first;
//         cv::Mat maskImage;
//         cv::inRange(imageGray, cv::Scalar(maskValue), cv::Scalar(maskValue), maskImage); // 创建mask

//         std::vector<std::vector<cv::Point>> contours;
//         cv::findContours(maskImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

//         for (const auto &contour : contours)
//         {
//             if (cv::contourArea(contour) > filter.second)
//             {

//                 // auto [midpoints, lines] = calculateMidpoints(contour);
//                 auto result = calculateMidpoints(contour);
//                 std::vector<cv::Point2f> midpoints = result.first;
//                 std::vector<std::pair<cv::Point2f, cv::Point2f>> lines = result.second;
//                 float minDist = std::numeric_limits<float>::max();
//                 cv::Point2f closestMidpoint;
//                 std::pair<cv::Point2f, cv::Point2f> closestLine;

//                 for (size_t i = 0; i < midpoints.size(); ++i)
//                 {
//                     float dist = cv::norm(midpoints[i] - imageCenter);
//                     if (dist < minDist)
//                     {
//                         minDist = dist;
//                         closestMidpoint = midpoints[i];
//                         closestLine = lines[i];
//                     }
//                 }

//                 if (minDist != std::numeric_limits<float>::max())
//                 {
//                     // selectedMidpoint.push_back(closestMidpoint);
//                     // selectedLineSegment.push_back(closestLine);
//                     allPoints.push_back(closestMidpoint); // Add the midpoint
//                     // allPoints.push_back(closestLine.first); // Add the first endpoint of the line segment
//                     // allPoints.push_back(closestLine.second);
//                 }
//             }
//         }
//     }
//     // std::sort(allPoints.begin(), allPoints.end(), [](const cv::Point &a, const cv::Point &b)
//     //           { return a.x < b.x; });
//     // Debug print to console
//     // std::cout << "Camera Side: " << cameraSide << ", Total Points: " << allPoints.size() << std::endl;

//     return allPoints;
// }
std::map<int, std::vector<cv::Point>> processImage(const cv::Mat &imageGray, const std::string &cameraSide)
{
    // 定义筛选规则，key是mask值，value是面积阈值
    std::map<int, int> sizeFilters = {{2, 800}, {7, 300}};
    // 计算图像中心点
    cv::Point2f imageCenter(imageGray.cols / 2.0f, imageGray.rows);

    // 用来存储每个maskValue对应的点集
    std::map<int, std::vector<cv::Point>> pointsByMaskValue;

    for (const auto &filter : sizeFilters)
    {
        int maskValue = filter.first;
        cv::Mat maskImage;
        // 根据maskValue创建掩码图像
        cv::inRange(imageGray, cv::Scalar(maskValue), cv::Scalar(maskValue), maskImage);

        std::vector<std::vector<cv::Point>> contours;
        // 查找掩码图像中的轮廓
        cv::findContours(maskImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto &contour : contours)
        {
            // 如果轮廓的面积大于阈值
            if (cv::contourArea(contour) > filter.second)
            {
                // 计算轮廓的中点
                auto result = calculateMidpoints(contour);
                std::vector<cv::Point2f> midpoints = result.first;
                float minDist = std::numeric_limits<float>::max();
                cv::Point2f closestMidpoint;

                // 寻找最接近图像中心的中点
                for (const auto &midpoint : midpoints)
                {
                    float dist = cv::norm(midpoint - imageCenter);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        closestMidpoint = midpoint;
                    }
                }

                // 如果找到了最近的中点
                if (minDist != std::numeric_limits<float>::max())
                {
                    // 将最近的中点添加到相应maskValue的点集中
                    pointsByMaskValue[maskValue].push_back(closestMidpoint);
                }
            }
        }
    }

    // 返回按maskValue组织的点集
    return pointsByMaskValue;
}

// 转换点至矫正图坐标
std::pair<int, int> transformCameraPointToUndistortImagePoint(const cv::Point2f &target, const cv::Mat &mapping)
{
    // 遍历映射表找到目标点
    for (int y = 0; y < mapping.rows; ++y)
    {
        for (int x = 0; x < mapping.cols; ++x)
        {
            cv::Vec2s point = mapping.at<cv::Vec2s>(y, x);
            if (point[0] == static_cast<short>(target.x) && point[1] == static_cast<short>(target.y))
            {
                return {x, y};
            }
        }
    }

    std::cerr << "Target not found in the map." << std::endl;
    return {-1, -1}; // 未找到目标点
}
// std::vector<std::pair<int, int>> transformCameraPointsToUndistortImagePoints(
//     const std::string &cameraSide,
//     const std::vector<cv::Point> &targets)
// {
//     std::vector<std::pair<int, int>> results;
//     if (globalLUTs.find(cameraSide) == globalLUTs.end())
//     {
//         std::cerr << "No LUT found for camera side: " << cameraSide << std::endl;
//         return results; // Return empty if no LUT is found for the camera side
//     }

//     const auto &lut = globalLUTs[cameraSide];
//     results.reserve(targets.size());

//     for (const auto &target : targets)
//     {
//         auto iter = lut.find(target);
//         if (iter != lut.end())
//         {
//             results.push_back(iter->second);
//         }
//         else
//         {
//             std::cerr << "Target not found in the map: " << target << std::endl;
//             results.push_back({-1, -1}); // Point not found
//         }
//     }

//     return results;
// }
// std::vector<std::pair<int, int>> transformCameraPointsToUndistortImagePoints(
//     const std::string &cameraSide,
//     const std::vector<cv::Point> &targets)
// {
//     std::vector<std::pair<int, int>> results;
//     if (globalLUTs.find(cameraSide) == globalLUTs.end())
//     {
//         // std::cerr << "No LUT found for camera side: " << cameraSide << std::endl;
//         return results; // Return empty if no LUT is found for the camera side
//     }
//     const auto &lut = globalLUTs[cameraSide];
//     results.reserve(targets.size());

//     for (const auto &target : targets)
//     {
//         if (auto iter = lut.find(target); iter != lut.end())
//         {
//             results.push_back(iter->second);
//         }
//         else
//         {
//             // 尝试在目标点的2x2邻域内查找
//             bool found = false;
//             for (int dx = -1; dx <= 1 && !found; ++dx)
//             {
//                 for (int dy = -1; dy <= 1 && !found; ++dy)
//                 {
//                     cv::Point neighbor(target.x + dx, target.y + dy);
//                     if (auto neighborIter = lut.find(neighbor); neighborIter != lut.end())
//                     {
//                         results.push_back(neighborIter->second);
//                         found = true;
//                     }
//                 }
//             }
//             if (!found)
//             {
//                 // std::cerr << "Target not found in the map and no neighbors found: " << target << std::endl;
//                 results.push_back({-1, -1}); // Point not found
//             }
//         }
//     }

//     return results;
// }
// std::vector<std::pair<int, int>> transformCameraPointsToUndistortImagePoints(
//     const std::string &cameraSide,
//     const std::vector<cv::Point> &targets)
// {
//     std::vector<std::pair<int, int>> results;
//     if (globalLUTs.find(cameraSide) == globalLUTs.end())
//     {
//         return results; // Return empty if no LUT is found for the camera side
//     }
//     const auto &lut = globalLUTs[cameraSide];
//     results.reserve(targets.size());

//     for (const auto &target : targets)
//     {
//         auto iter = lut.find(target); // Moved declaration outside of the if statement
//         if (iter != lut.end())
//         {
//             results.push_back(iter->second);
//         }
//         else
//         {
//             // 尝试在目标点的2x2邻域内查找
//             bool found = false;
//             for (int dx = -1; dx <= 1 && !found; ++dx)
//             {
//                 for (int dy = -1; dy <= 1 && !found; ++dy)
//                 {
//                     cv::Point neighbor(target.x + dx, target.y + dy);
//                     auto neighborIter = lut.find(neighbor); // Moved declaration outside of the if statement
//                     if (neighborIter != lut.end())
//                     {
//                         results.push_back(neighborIter->second);
//                         found = true;
//                     }
//                 }
//             }
//             if (!found)
//             {
//                 results.push_back({-1, -1}); // Point not found
//             }
//         }
//     }

//     return results;
// }
std::map<int, std::vector<std::pair<int, int>>> transformCameraPointsToUndistortImagePoints(
    const std::string &cameraSide,
    const std::map<int, std::vector<cv::Point>> &maskValueAndPoints)
{
    std::map<int, std::vector<std::pair<int, int>>> results;

    if (globalLUTs.find(cameraSide) == globalLUTs.end())
    {
        return results; // Return empty if no LUT is found for the camera side
    }
    const auto &lut = globalLUTs[cameraSide];

    for (const auto &pair : maskValueAndPoints)
    {
        int maskValue = pair.first;
        const std::vector<cv::Point> &targets = pair.second;
        std::vector<std::pair<int, int>> maskResults;
        maskResults.reserve(targets.size());

        for (const auto &target : targets)
        {
            auto iter = lut.find(target);
            if (iter != lut.end())
            {
                maskResults.push_back(iter->second);
            }
            else
            {
                // 尝试在目标点的2x2邻域内查找
                bool found = false;
                for (int dx = -1; dx <= 1 && !found; ++dx)
                {
                    for (int dy = -1; dy <= 1 && !found; ++dy)
                    {
                        cv::Point neighbor(target.x + dx, target.y + dy);
                        auto neighborIter = lut.find(neighbor);
                        if (neighborIter != lut.end())
                        {
                            maskResults.push_back(neighborIter->second);
                            found = true;
                        }
                    }
                }
                if (!found)
                {
                    maskResults.push_back({-1, -1}); // Point not found
                }
            }
        }

        results[maskValue] = maskResults;
    }

    return results;
}
void drawAndShowPoints(cv::Mat &image, const std::vector<cv::Point2f> &points)
{
    // 遍历所有点
    for (const auto &point : points)
    {
        // 在图像上绘制点
        cv::circle(image, point, 5, cv::Scalar(0, 255, 0), -1); // 使用绿色圆圈绘制点，半径为5，实心圆
    }

    // 显示图像
    cv::imshow("Image with Points", image);
    cv::waitKey(0); // 等待用户按键后关闭窗口
}

std::pair<double, double> transformVehiclePointsTOSrvpoint(const Eigen::Vector3d &point_in_vehicle, const std::pair<double, double> &srv_vehicle_position, double scale_factor = 0.4)
{
    // 计算转换后的坐标
    double transformed_x = srv_vehicle_position.first - (point_in_vehicle.y() * scale_factor);
    double transformed_y = srv_vehicle_position.second - (point_in_vehicle.x() * scale_factor);

    // 返回转换后的点
    return std::make_pair(transformed_x, transformed_y);
}

std::string findMatchingFile(const std::string &directory, const std::string &target_prefix)
{
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(directory.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            std::string file_name = ent->d_name;
            // 检查文件名是否以目标前缀开始
            if (file_name.find(target_prefix) == 0)
            {
                closedir(dir);
                return directory + "/" + file_name;
            }
        }
        closedir(dir);
    }
    else
    {
        // 无法打开目录
        std::cerr << "Unable to open directory: " << directory << std::endl;
    }

    return ""; // 没有找到匹配的文件
}

/*pad json*/
struct Polygon
{
    std::vector<cv::Point> points; // 使用cv::Point类型来存储点
};

std::vector<Polygon> extractOccupiedParkingSpots(const std::string &filename)
{
    std::ifstream file(filename);
    nlohmann::json j;
    file >> j;

    std::vector<Polygon> result;

    for (const auto &shape : j["shapes"])
    {
        if (shape["label"] == "occupied_parkset")
        {
            Polygon polygon;
            for (const auto &point : shape["points"])
            {
                polygon.points.push_back(cv::Point(static_cast<int>(point[0]), static_cast<int>(point[1])));
            }
            result.push_back(polygon);
        }
    }

    return result;
}
std::pair<double, double> transformSrvpointToVehiclePoints(const std::pair<double, double> &srv_point, const std::pair<double, double> &srv_vehicle_position, double scale_factor = 0.4)
{
    double vehicle_x = (srv_vehicle_position.second - srv_point.second) / scale_factor;
    double vehicle_y = (srv_vehicle_position.first - srv_point.first) / scale_factor;
    return std::make_pair(vehicle_x, vehicle_y);
}

int main()
{
    std::string output_base_folder = "/home/hao/work/apa/images/vehicle/2023.4/0412/png_classify";
    std::string display_base_folder = "/home/hao/work/apa/images/vehicle/2023.4/0412/jpg_classify";
    std::string folder_map = "/home/hao/work/apa/images/vehicle/2023.4/ur_2881/map_bin_file/04";
    std::string display_undistort_folder = "/home/hao/work/apa/images/vehicle/2023.4/0412/undistort_img";
    std::string srv_directory = "/home/hao/work/apa/images/vehicle/2023.4/0412/20240412125433_srv_jpg";
    std::string json_folder = "/home/hao/work/apa/images/vehicle/2023.4/0412/json";
    const cv::Vec3i dims(520, 768, 2);                                // 映射表的维度
    std::pair<double, double> srv_vehicle_position = {319.5, 369.84}; // 相机位置319.5， 369.84
    std::map<std::string, CameraConfig> camera_configs;               // 相机配置
    std::map<std::string, std::vector<Eigen::Vector2d>> maps;         // 相机视角的映射表
    // std::map<std::string, CameraConfig> camera_configs;
    // fillCameraConfigs();

    initializeLUTs(folder_map, dims);

    for (const std::string &camera_side : {"back", "left", "right", "front"})
    {
        // 假设加载映射表和相机配置的函数
        std::string folder_path = output_base_folder + "/" + camera_side;
        std::string display_folder_path = display_base_folder + "/" + camera_side;
        std::string display_undistort_path = display_undistort_folder + "/" + camera_side;
        std::string bin_file_path = folder_map + "/" + camera_side + ".bin";
        // const cv::Mat &mapping = loadBinMap(camera_side, folder_map, dims);
        // auto config = camera_configs[camera_side];
        CameraPosition position = stringToCameraPosition(camera_side);
        //
        DIR *dir = opendir(folder_path.c_str()); // 打开目录
        if (!dir)
        {
            std::cerr << "目录不存在: " << folder_path << std::endl;
            continue;
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            if (entry->d_type == DT_REG)
            {
                std::string file_name = entry->d_name;

                if (file_name != "20240412125522_63848494509979_2_1_1042.979980_398.369995_20.590000_100.png")
                {
                    continue;
                }

                std::string image_path = folder_path + "/" + file_name;
                std::string display_image_path = display_folder_path + "/" + file_name.substr(0, file_name.find_last_of(".")) + ".jpg";
                std::string display_undistort_image_path = display_undistort_path + "/" + file_name.substr(0, file_name.find_last_of(".")) + ".jpg";
                cv::Mat mask = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
                cv::Mat display_raw_image = cv::imread(display_image_path);
                cv::Mat display_undistort_image = cv::imread(display_undistort_image_path);
                // 提取匹配键
                std::string target_prefix = file_name.substr(0, file_name.find('_', file_name.find('_') + 1));
                std::string matched_file_path = findMatchingFile(srv_directory, target_prefix);
                cv::Mat display_srv_image = cv::imread(matched_file_path);

                std::string json_file_path = json_folder + "/" + file_name.substr(0, file_name.find_last_of(".")) + ".json";
                // if (!matched_file.empty())
                // {
                //     std::cout << "Found matching file: " << matched_file << std::endl;
                // }
                // else
                // {
                //     std::cout << "No matching file found." << std::endl;
                // }

                if (!mask.empty())
                {
                    if (!mask.empty())
                    {
                        auto points = processImage(mask, camera_side);

                        for (const auto &[maskValue, pts] : points)
                        {
                            for (const auto &pt : pts)
                            { // Now 'pt' is of type 'cv::Point'
                                // std::cout << "(" << pt.x << ", " << pt.y << ")" << std::endl; // Correctly access the x and y components of cv::Point
                                cv::circle(display_raw_image, pt, 1, cv::Scalar(0, 255, 0), -1); // Draw each point with a green circle
                            }
                        }
                        // std::vector<std::pair<int, int>> corrected_points = transformCameraPointsToUndistortImagePoints(camera_side, points);
                        // std::vector<std::pair<double, double>> srv_points;
                        std::map<int, std::vector<std::pair<int, int>>> srv_points;
                        std::map<int, std::vector<std::pair<int, int>>> maskvauleandcorrected_points = transformCameraPointsToUndistortImagePoints(camera_side, points);
                        for (const auto &pair : maskvauleandcorrected_points)
                        {
                            int maskValue = pair.first;                                            // 从pair中获取maskValue
                            const std::vector<std::pair<int, int>> &correctedPoints = pair.second; // 从pair中获取correctedPoints
                            std::vector<std::pair<int, int>> tempPoints;                           // 临时存储每个maskValue对应的转换点
                            for (const auto &point : correctedPoints)
                            {
                                int x = point.first;  // 获取点的x坐标
                                int y = point.second; // 获取点的y坐标

                                Eigen::Vector2d pixel(static_cast<double>(x), static_cast<double>(y)); // 转换为double进行处理
                                Eigen::Vector3d vehicle_point = UndistortImageTransformVehicle(pixel, position);
                                std::cout << "Original mm Vehicle Point: (" << vehicle_point.x() << ", " << vehicle_point.y() << ")" << std::endl;

                                vehicle_point.x() /= 10;
                                vehicle_point.y() /= 10;
                                std::pair<int, int> srv_point = transformVehiclePointsTOSrvpoint(vehicle_point, srv_vehicle_position);
                                // 从服务点转换回车辆坐标
                                std::pair<double, double> converted_vehicle_point = transformSrvpointToVehiclePoints({srv_point.first, srv_point.second}, srv_vehicle_position);

                                // 打印转换前后的坐标
                                std::cout << "Converted to SRV Point: (" << srv_point.first << ", " << srv_point.second << ")" << std::endl;
                                std::cout << "Original cm Vehicle Point: (" << vehicle_point.x() << ", " << vehicle_point.y() << ")" << std::endl;

                                std::cout << "Converted back to cm  Vehicle Point: (" << converted_vehicle_point.first << ", " << converted_vehicle_point.second << ")" << std::endl
                                          << std::endl;
                                tempPoints.push_back(srv_point); // 将转换后的点加入到临时向量中

                                // 绘制点在校正后的图像上
                                cv::Point cvPoint(x, y);
                                cv::circle(display_undistort_image, cvPoint, 2, cv::Scalar(0, 255, 0), -1);

                                // 绘制点在服务图像上
                                cv::Point2f srv_point_display(static_cast<float>(srv_point.first), static_cast<float>(srv_point.second));
                                cv::circle(display_srv_image, srv_point_display, 2, cv::Scalar(0, 255, 0), -1);
                            }
                            srv_points[maskValue] = tempPoints; // 将临时向量存储到srv_points中对应的maskValue键下
                        }

                        // 遍历矫正后的点并传递给 UndistortImageTransformVehicle
                        // for (const auto &pt : corrected_points)
                        // {
                        //     Eigen::Vector2d pixel(pt.first, pt.second);
                        //     std::cout << "corrected_point: (" << pt.first << ", " << pt.second << ") " << std::endl;
                        //     Eigen::Vector3d vehicle_point = UndistortImageTransformVehicle(pixel, position);

                        //     std::pair<double, double> srv_point = transformVehiclePointsTOSrvpoint(vehicle_point, srv_vehicle_position);

                        //     cv::Point2f point(pt.first, pt.second);
                        //     cv::circle(display_undistort_image, point, 2, cv::Scalar(0, 255, 0), -1); // Draw the point with a green circle

                        //     cv::Point2f srv_point_display(srv_point.first, srv_point.second);
                        //     cv::circle(display_srv_image, srv_point_display, 2, cv::Scalar(0, 255, 0), -1); // Draw the point with a green circle

                        //     std::cout << camera_side << "Vehicle coordinates: (" << vehicle_point.x() << ", " << vehicle_point.y() << ", " << vehicle_point.z() << ")" << std::endl;
                        //     std::cout << "Transformed point in camera view: (" << srv_point.first << ", " << srv_point.second << ")" << std::endl;
                        // }
                        // for (const auto &pt : corrected_points)
                        // {
                        //     Eigen::Vector2d pixel(pt.first, pt.second);
                        //     Eigen::Vector3d vehicle_point = UndistortImageTransformVehicle(pixel, position);
                        //     std::pair<double, double> srv_point = transformVehiclePointsTOSrvpoint(vehicle_point, srv_vehicle_position);
                        //     srv_points.push_back(srv_point);
                        // }
                        // for (const auto &point : srv_points)
                        // {
                        //     std::cout << "SRV Point: (" << point.first << ", " << point.second << ")" << std::endl;
                        // }
                        // auto occupiedSpots = extractOccupiedParkingSpots(json_file_path);
                        // for (const auto &polygon : occupiedSpots)
                        // {
                        //     bool containsSrvPoint = false;

                        //     // 遍历srv_points的每个分类
                        //     for (const auto &[maskValue, points] : srv_points)
                        //     {
                        //         for (const auto &point : points)
                        //         {
                        //             // 创建cv::Point用于pointPolygonTest
                        //             cv::Point testPoint(point.first, point.second);
                        //             double result = cv::pointPolygonTest(polygon.points, testPoint, false);
                        //             if (result >= 0)
                        //             { // 点在多边形内部或边界上
                        //                 containsSrvPoint = true;
                        //                 break; // 找到一个点在多边形内部后即退出内层循环
                        //             }
                        //         }
                        //         if (containsSrvPoint)
                        //             break; // 如果已确定有点在多边形内，则退出外层循环
                        //     }

                        //     const cv::Scalar color = containsSrvPoint ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0); // 点在多边形内使用红色，否则使用黄色
                        //     cv::polylines(display_srv_image, polygon.points, true, color, 2);                            // 绘制多边形
                        // }
                        auto occupiedSpots = extractOccupiedParkingSpots(json_file_path);

                        // for (auto &polygon : occupiedSpots)
                        // {
                        //     for (auto &point : polygon.points)
                        //     {
                        //         // 将SRV点转换为车辆点
                        //         auto vehicle_point = transformSrvpointToVehiclePoints({static_cast<double>(point.x), static_cast<double>(point.y)}, srv_vehicle_position);

                        //         // 将车辆点转换回SRV点
                        //         Eigen::Vector3d vehicle_point_3d(vehicle_point.first, vehicle_point.second, 0);
                        //         auto srv_point = transformVehiclePointsTOSrvpoint(vehicle_point_3d, srv_vehicle_position);

                        //         // 输出转换后的结果以验证
                        //         std::cout << "Original SRV Point: (" << point.x << ", " << point.y << ")" << std::endl;
                        //         std::cout << "Converted Vehicle Point: (" << vehicle_point.first << ", " << vehicle_point.second << ")" << std::endl;
                        //         std::cout << "Converted Back to SRV Point: (" << srv_point.first << ", " << srv_point.second << ")" << std::endl;
                        //     }
                        // }
                        for (const auto &polygon : occupiedSpots)
                        {
                            bool containsSrvPoint = false;

                            // 遍历srv_points的每个分类
                            for (const auto &[maskValue, points] : srv_points)
                            {
                                for (const auto &point : points)
                                {
                                    // 创建cv::Point用于pointPolygonTest
                                    cv::Point testPoint(point.first, point.second);
                                    // 使用cv::pointPolygonTest检测点与多边形的关系，设置第三个参数为true以计算实际距离
                                    double result = cv::pointPolygonTest(polygon.points, testPoint, true);
                                    if (result >= 0 || result >= -5)
                                    { // 点在多边形内部、边界上，或离边界不到5像素
                                        containsSrvPoint = true;
                                        break; // 找到一个符合条件的点后即退出内层循环
                                    }
                                }
                                if (containsSrvPoint)
                                    break; // 如果已确定有符合条件的点在多边形内，则退出外层循环
                            }

                            // 根据是否有点在多边形内或靠近边界来设置绘制颜色
                            const cv::Scalar color = containsSrvPoint ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0); // 点在多边形内或靠近边界使用红色，否则使用黄色
                            cv::polylines(display_srv_image, polygon.points, true, color, 2);                            // 绘制多边形
                        }

                        std::cout << file_name << std::endl;
                        cv::imshow(file_name, display_undistort_image);
                        cv::imshow(target_prefix, display_srv_image);
                        cv::imshow("Display Image", display_raw_image);

                        cv::waitKey(0);
                        cv::destroyWindow(file_name);
                        cv::destroyWindow(target_prefix);
                        cv::destroyWindow("Display Image");
                    }

                    // cv::imshow("Display Image", display_srv_image);

                    // for (const auto &polygon : occupiedSpots)
                    // {
                    //     const cv::Scalar color(255, 255, 0);                                      // 黄色
                    //     const int thickness = 2;                                                  // 线条粗细
                    //     cv::polylines(display_srv_image, polygon.points, true, color, thickness); // 绘制多边形

                    //     std::cout << "Occupied parking spot points:" << std::endl;
                    //     for (const auto &point : polygon.points)
                    //     {
                    //         std::cout << "(" << point.x << ", " << point.y << ") ";
                    //     }
                    //     std::cout << std::endl;
                    // }
                    // Display the image
                }
            }
        }
        closedir(dir);
    }
    return 0;
}
