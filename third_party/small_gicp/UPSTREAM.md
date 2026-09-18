# small_gicp 上游来源

- 项目：small_gicp
- 上游：https://github.com/koide3/small_gicp
- 标签：v1.0.0
- 提交：fd29d8cf94cf05ed7ad21c81c27b65963110adb5
- 许可证：MIT；部分文件还保留其原有 BSD 声明，随源码原样提供。
- 导入日期：2026-09-18
- 范围：include、src/small_gicp/registration、cmake、CMakeLists.txt、LICENSE。
- 原始源码修改：无。文件来自该提交的 git show 原始 blob，逐字节校验值见 SOURCE_SHA256.json。
- 构建方式：项目自有 CMake wrapper 链接已解析的 Eigen；适配器实例化核心 Registration<GICPFactor, SerialReduction> 模板，不编译带 OpenMP 的 helper，不执行上游自动下载逻辑。
- PCL wrapper、TBB、OpenMP、march-native、Python、上游示例与 benchmark：均不启用。

以后修改上游文件必须提供独立 patch、原始与修改后校验值及原因。不要直接覆盖此快照。
