# 检测结果坐标

检测时的图像可能经过旋转、缩放、窄区裁剪和方向对齐。不要直接把结果图上的 y 当作原始文件的 y。

- **detection**：送入检测算法的图像坐标，包含 ROI 偏移，不含标题。
- **overlay**：结果 PNG 中的位置，包含顶部标题与实际预览缩放。
- **input**：用户输入文件中的位置，已撤销手动旋转及几何预处理的仿射变换。

具体标题高度、预览比例与变换矩阵保存在 `render_metadata.json` 和 `geometry_preprocessing` 中，
不应在调试程序里写死标题高度。

## regions

`detection_boxes.csv/json` 按框导出。`detection_xyxy` 是检测图上的矩形，x2/y2 为排他边界；
`input_polygon` 是矩形还原到输入图后的四角，`input_xyxy` 是该四边形的外接框。
外接框与物理纸身轮廓不是同一概念。`aligned_xyxy` 表示方向对齐空间的区段，
`boundary_samples` 保存部分列上的映射边界采样。

右侧未标注图只是左右镜像的显示对照，镜像不参与检测，也不改变左侧坐标。
若框本身重叠，统一批处理会在 `batch_geometry_validation.json` 中列出框号与重叠高度。

## tracks

`detection_tracks.csv` 每条最终轨迹一行，中心 y 是汇总位置，不保证位于全局信号峰。
`detection_points.csv` 保存实测条带点，x 为条带代表位置，不能解释为精确的纸边像素。
`detection_coordinates.json` 保存两者以及坐标变换元数据。

各条带观测、插值曲线和最终轨迹的含义不同。不能把曲线插值点当成独立检测证据。
