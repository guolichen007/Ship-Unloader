# 工程构建入口

现场 ROS 单帧工具使用 Python，无需编译 C++，先读[仓库 README](../README.md)和[现场运行说明](../docs/02_ROS现场一键识别运行说明.md)。本页用于 C++ 开发与 V1.4 复验。

## Ubuntu20 开发构建

目标环境为 Ubuntu20.04、GCC9.4、CMake3.16.3、Eigen3.3.7、PCL1.10.0。固定 small_gicp 源码随仓库提供，配置构建不自动下载；Python 依赖见根 requirements-ros.txt。

```bash
sudo apt-get install build-essential cmake git python3 libeigen3-dev libpcl-dev pkg-config
cmake -S ship_perception -B Ship-Unloader-Work/build/current \
  -DCMAKE_BUILD_TYPE=Release -DM0_WITH_PCL=ON
cmake --build Ship-Unloader-Work/build/current --parallel 2
ctest --test-dir Ship-Unloader-Work/build/current --output-on-failure -LE quick_acceptance
```

需要显式指定 Eigen 时，选择实际存在的系统目录 `/usr/lib/cmake/eigen3` 或 `/usr/share/eigen3/cmake`。非 Quick 不等于完整验收，见[测试说明](../docs/04_测试与验收说明.md)。

## Windows 开发构建

使用 C++17 MSVC 与 Eigen3.3.7；PCL-OFF 只作开发验证，不代表 Ubuntu 正式通过。

```powershell
cmake -S ship_perception -B Ship-Unloader-Work/build/current -G 'Visual Studio 17 2022' -A x64 -DEigen3_DIR='<Eigen3Config.cmake directory>' -DM0_WITH_PCL=OFF
cmake --build Ship-Unloader-Work/build/current --config Release --parallel 2
ctest --test-dir Ship-Unloader-Work/build/current -C Release --output-on-failure -LE quick_acceptance
```

## 正式历史基线

V1.4 正式 Tag 为 `baseline-v1.4-ubuntu20-20260919`。[实现说明](../docs/V1.4_船体稳定跟踪实现说明.md)与[Ubuntu20 验证及复验要求](../docs/V1.4_Ubuntu20验证记录.md)保留精确 SHA 和门槛。

配置、编译 SHA 与实际依赖必须一致；更新代码后先重新配置构建，G5/Full 检查工作树和编译新鲜度。所有新产物进入忽略的 Work/validation 目录。
