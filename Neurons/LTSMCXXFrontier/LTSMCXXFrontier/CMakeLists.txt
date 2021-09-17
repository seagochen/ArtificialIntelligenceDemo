# CMakeList.txt: LTSMCXXFrontier 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

find_package(Torch REQUIRED)

# 将源代码添加到此项目的可执行文件。
add_executable (LTSMCXXFrontier "LTSMCXXFrontier.cpp" "OHVEncoder.cpp")

# 连接库
target_link_libraries(LTSMCXXFrontier "${TORCH_LIBRARIES}")

# 使用C++14编译标准
set_property(TARGET LTSMCXXFrontier PROPERTY CXX_STANDARD 14)

# TODO: 如有需要，请添加测试并安装目标。
