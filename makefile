CXX = g++
CXXFLAGS = -g -std=c++11 -Wall `pkg-config --cflags opencv4`
LDLIBS = `pkg-config --libs opencv4`

# 定义目标
TARGET = seg

# 默认目标
all: $(TARGET)

# 链接最终目标 CameraTransforms_N7.o CameraTransforms_D9
$(TARGET): seg_obstacle_main.o undistortimage_transform_vehicle.o
	$(CXX) -o $@ $^ $(LDLIBS)

# 编译 VehiclesDirections_main.cpp
seg_obstacle_main.o: seg_obstacle_main.cpp undistortimage_transform_vehicle.h 
	$(CXX) $(CXXFLAGS) -c $<

# 编译 CameraTransforms.cpp   CameraTransforms_N7
undistortimage_transform_vehicle.o: undistortimage_transform_vehicle.cpp  undistortimage_transform_vehicle.h #VehiclesDirections.h
	$(CXX) $(CXXFLAGS) -c $<

# 清理生成的文件
clean:
	rm -f *.o $(TARGET)
