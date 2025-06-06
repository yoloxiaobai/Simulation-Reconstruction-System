import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import cv2
import h5py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSlider, QComboBox, QPushButton, QFileDialog, QGridLayout,
                             QMessageBox, QTabWidget, QDialog, QCheckBox, QTextBrowser, QProgressBar
                             )
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from typing import Optional
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from PIL import Image
import json
from matplotlib.backends.backend_pdf import PdfPages

# 设置 matplotlib 中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 优先使用简体，兼容其他系统
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# 参数常量默认设置
IMAGE_SIZES = [16, 32, 64, 128]                                              # 支持的图像尺寸列表
DEFAULT_IMAGE_SIZE = 2                                                       # 默认尺寸64*64
DEFAULT_SAMPLING_RATE = 0.5                                                  # 默认采样率
DEFAULT_GAUSSIAN_NOISE_STD = 0.1                                             # 高斯噪声标准差
DEFAULT_POISSON_NOISE_LAMBDA = 10.0                                          # λ取值
DEFAULT_OMP_K = 100                                                          # 默认稀疏度以及迭代次数
DEFAULT_TVAL3_LAMBDA = 0.01                                                  # 默认正则化参数λ
DEFAULT_TVAL3_MAX_ITER = 200                                                 # 默认迭代次数

class DataStorageSpec:
    """定义实验数据存储以及格式，确保成像软件和分析软件数据互通以及后续实验数据迁移进深度学习"""
    @staticmethod
    def get_experiment_dir(root: str) -> str:
        """实验目录生成"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(root, f"experiment_{timestamp}")

    @staticmethod
    def get_metadata(params: dict, metrics: dict) -> dict:
        """生成元数据结构"""
        return {
            "experiment_id": f"experiment_{time.strftime('%Y%m%d_%H%M%S')}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            # 使用时间戳确保唯一性，快速定位实验
            "params": {
                "image_size": params.get("image_size"),
                "sampling_rate": params.get("sampling_rate"),
                "noise_type": params.get("noise_type"),
                "noise_param": params.get("noise_param"),
                "matrix_generation_method": params.get("matrix_generation_method"),
                "reconstruction_algorithm": params.get("reconstruction_algorithm")
            },
            # 实验参数
            "metrics": {
                "psnr": metrics.get("psnr", "N/A"),
                "ssim": metrics.get("ssim", "N/A"),
                "recon_time": metrics.get("recon_time", "N/A")
            },
            # 评估指标
            "data_paths": {
                "measurement_matrix": "sampled_measurement_matrix.npy",
                "bucket_signals": "bucket_signals.npy",
                "ground_truth": "ground_truth_image.npy",
                "ground_truth_vis": "ground_truth_image.png",
                "reconstructed_image":"reconstructed_image.png",
            }
            # 定义实验数据的存储文件名
        }

class SpeckleVisualizationDialog(QDialog):
    """全散斑和子散斑的可视化"""
    def __init__(self, parent=None, sampling_speckle: Optional[np.ndarray] = None,
                 full_speckle: Optional[np.ndarray] = None, matrix_size: int = 64,
                 matrix_type_str: str = "未知类型"):                                     # 显示采样散斑矩阵、完整散斑矩阵的统计信息，通过current_speckle_index切换不同的散斑图案
        super().__init__(parent)
        self.setWindowTitle("散斑可视化")
        self.setGeometry(100, 100, 1000, 1000)                                         # 位置及大小
        self.sampling_speckle = sampling_speckle                                       # 子散斑
        self.full_speckle = full_speckle                                               # 全散斑
        self.matrix_size = matrix_size                                                 # 矩阵边长
        self.matrix_dim = matrix_size * matrix_size                                    # 散斑尺寸
        self.matrix_type_str = matrix_type_str                                         # 类型，用于PDF命名

        self.current_speckle_index = 0                                                 # 记录当前显示的散斑图案索引，初始化为 0

        self._setup_ui()                                                               # 设置 UI 界面并根据数据状态更新显示内容
        if self.sampling_speckle is not None and self.sampling_speckle.shape[0] > 0:   # 检查 self.sampling_speckle 是否存在且非空
            self._update_speckle_display()                                             # 满足条件将散斑图案显示在界面上
        if self.full_speckle is not None and self.full_speckle.shape[0] > 0:           # 检查 self.full_speckle是否存在且非空
            self._update_speckle_stats()                                               # 满足条件显示散斑的统计信息（如均值、方差）

    def _setup_ui(self):
        main_layout = QVBoxLayout()
        # 主选项卡
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 子散斑
        individual_speckle_widget = QWidget()
        individual_speckle_layout = QVBoxLayout()
        individual_speckle_widget.setLayout(individual_speckle_layout)
        self.tab_widget.addTab(individual_speckle_widget, "单个散斑")

        # 导航控件
        nav_layout = QHBoxLayout()
        self.btn_prev_speckle = QPushButton("上一张")
        self.btn_prev_speckle.clicked.connect(self._prev_speckle)
        num_sampled_speckles = self.sampling_speckle.shape[0] if self.sampling_speckle is not None else 0
        self.label_speckle_index = QLabel(f"当前散斑: {self.current_speckle_index + 1} / {num_sampled_speckles}")
        self.btn_next_speckle = QPushButton("下一张")
        self.btn_next_speckle.clicked.connect(self._next_speckle)
        nav_layout.addWidget(self.btn_prev_speckle)
        nav_layout.addWidget(self.label_speckle_index)
        nav_layout.addWidget(self.btn_next_speckle)
        individual_speckle_layout.addLayout(nav_layout)

        # 子散斑显示
        self.fig_individual = Figure()
        self.canvas_individual = FigureCanvas(self.fig_individual)
        individual_speckle_layout.addWidget(self.canvas_individual)
        individual_speckle_layout.addWidget(NavigationToolbar(self.canvas_individual, self))

        # 保存子散斑PDF按钮
        self.btn_save_sampled_speckles_pdf = QPushButton("保存子散斑PDF")
        self.btn_save_sampled_speckles_pdf.clicked.connect(self._save_sampled_speckles_pdf)
        individual_speckle_layout.addWidget(self.btn_save_sampled_speckles_pdf)

        # 全散斑
        mosaic_speckle_widget = QWidget()
        mosaic_speckle_layout = QVBoxLayout()
        mosaic_speckle_widget.setLayout(mosaic_speckle_layout)
        self.tab_widget.addTab(mosaic_speckle_widget, "所有散斑")

        # 全散斑显示
        self.fig_mosaic = Figure()
        self.canvas_mosaic = FigureCanvas(self.fig_mosaic)
        mosaic_speckle_layout.addWidget(self.canvas_mosaic)
        mosaic_speckle_layout.addWidget(NavigationToolbar(self.canvas_mosaic, self))

        if self.full_speckle is not None:
            self._plot_all_speckles_mosaic()

        # 统计卡
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        stats_widget.setLayout(stats_layout)
        self.tab_widget.addTab(stats_widget, "统计分析")

        self.fig_stats = Figure()
        self.canvas_stats = FigureCanvas(self.fig_stats)
        stats_layout.addWidget(self.canvas_stats)
        stats_layout.addWidget(NavigationToolbar(self.canvas_stats, self))

        # 控制显示
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn)
        self.setLayout(main_layout)
        self.fullscreen_button = QPushButton("全屏显示")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        main_layout.addWidget(self.fullscreen_button)

    def _save_sampled_speckles_pdf(self):
        # 检查是否有采样散斑数据可供保存，如果没有数据，弹出警告对话框并退出函数
        if self.sampling_speckle is None or self.sampling_speckle.shape[0] == 0:
            QMessageBox.warning(self, "保存失败", "没有可供保存的采样散斑数据。")
            return
        # 清理矩阵类型字符串以确保文件名安全（仅保留字母、数字、空格、下划线和短横线）
        safe_matrix_type_str = "".join(
            c if c.isalnum() or c in (' ', '_', '-')
              else '_' for c in self.matrix_type_str).replace(' ', '_')
        default_filename = f"{safe_matrix_type_str}_sampled_speckles.pdf"                                              # 构造默认文件名
        filePath, _ = QFileDialog.getSaveFileName(self, "保存子散斑PDF", default_filename, "PDF Files (*.pdf)")          # 弹出保存文件对话框供用户选择保存路径

        if filePath:                                                                                                   # 如果用户选择了文件路径
            try:
                with PdfPages(filePath) as pdf:                                                                        # 使用 PdfPages 将图像保存为 PDF 文件
                    fig = Figure(figsize=(8, 8))                                                                       # 调整图像大小以适应需求
                    # 计算要显示的散斑数量以及行数和列数
                    num_speckles = self.sampling_speckle.shape[0]
                    rows = int(np.sqrt(num_speckles))
                    cols = int(np.ceil(num_speckles / rows))

                    # 确保至少有1行1列
                    if rows == 0 or cols == 0:
                        rows = cols = 1

                    # 创建一个空白的拼贴图像
                    mosaic_height = rows * self.matrix_size
                    mosaic_width = cols * self.matrix_size
                    mosaic_image = np.zeros((mosaic_height, mosaic_width), dtype=self.sampling_speckle.dtype)

                    # 将每个散斑放置到拼贴图像中
                    for i in range(min(num_speckles, rows * cols)):
                        row_idx = i // cols                                                                            # 当前行索引
                        col_idx = i % cols                                                                             # 当前列索引
                        x_start = col_idx * self.matrix_size                                                           # 当前散斑在拼贴中的起始x坐标
                        y_start = row_idx * self.matrix_size                                                           # 当前散斑在拼贴中的起始y坐标
                        speckle = self.sampling_speckle[i].reshape(self.matrix_size, self.matrix_size)                 # 将当前散斑重塑为矩阵形式
                        mosaic_image[y_start:y_start + self.matrix_size, x_start:x_start + self.matrix_size] = speckle # 将散斑放置到拼贴图像的正确位置

                    # 在图像上绘制拼贴图
                    ax = fig.add_subplot(111)
                    ax.imshow(mosaic_image, cmap='gray')                                                               # 显示灰度图像
                    ax.set_title(f"采样散斑 ({self.matrix_type_str}) - {num_speckles} 个模式")                            # 设置标题
                    ax.axis('off')                                                                                     # 关闭坐标轴
                    fig.tight_layout()                                                                                 # 自动调整布局以防止重叠
                    pdf.savefig(fig)                                                                                   # 将图像保存到 PDF 文件
                    plt.close(fig)                                                                                     # 关闭图像以释放内存
                QMessageBox.information(self, "保存成功", f"子散斑已保存到: {filePath}")                                   # 成功保存后弹出信息对话框
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存PDF时发生错误: {str(e)}")                                     # 如果发生错误，弹出错误对话框并显示错误信息

    def toggle_fullscreen(self):
        """显示控制"""
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_button.setText("全屏显示")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("退出全屏")

    def _plot_speckle(self, ax, speckle_data, title="散斑模式"):
        """散斑绘制"""
        ax.cla()
        ax.imshow(speckle_data.reshape(self.matrix_size, self.matrix_size), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    def _plot_histogram(self, ax, speckle_data, title="强度直方图",
                        bins=50, color='skyblue', edgecolor='black',
                        intensity_range=(0, 1), log_scale=False):
        """
        绘制散斑数据的强度直方图。
        参数:
            ax (matplotlib.axes.Axes): 用于绘制直方图的轴对象
            speckle_data (array-like): 二维散斑图像数据，通常是数组或矩阵
            title (str, optional): 直方图的标题，默认为"强度直方图"
            bins (int, optional): 直方图的分箱数量，默认为50
            color (str, optional): 直方图柱形的填充颜色，默认为'skyblue'
            edgecolor (str, optional): 直方图柱形边框的颜色，默认为'black'
            intensity_range (tuple, optional): 强度值的范围，默认为(0, 1)
            log_scale (bool, optional): 是否使用对数刻度显示y轴，默认为False
        """
        # 检查输入数据是否可以展平（要求是类似数组的对象）
        if not hasattr(speckle_data, 'flatten'):
            raise ValueError("speckle_data必须是可展平的数组")
        ax.cla()                                                                                                    # 清空当前轴上的内容
        flattened_data = speckle_data.flatten()                                                                     # 展平输入数据以进行直方图统计

        # 绘制直方图，并获取相关统计数据（如每个bin的数量、bin的边界等）
        n, bins_edges, patches = ax.hist(
            flattened_data,
            bins=bins,
            color=color,
            edgecolor=edgecolor,
            range=intensity_range
        )

        # 计算强度数据的均值和标准差
        mean_val = flattened_data.mean()
        std_val = flattened_data.std()

        # 在直方图上绘制均值和±标准差的参考线
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'均值: {mean_val:.3f}')
        ax.axvline(mean_val + std_val, color='green', linestyle='dashed', linewidth=1, label=f'±σ: {std_val:.3f}')
        ax.axvline(mean_val - std_val, color='green', linestyle='dashed', linewidth=1)

        # 设置图表的标题和坐标轴标签
        ax.set_title(title)
        ax.set_xlabel("强度值")
        ax.set_ylabel("像素数量")

        # 如果启用对数刻度，则将y轴设置为对数刻度
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.7)                                                                      # 添加网格线，提升可视化效果
        ax.legend()                                                                                                   # 显示图例，标注均值和标准差信息

    def _update_speckle_display(self):
        """
        更新散斑图像的显示界面。
        该方法会根据当前索引显示对应的散斑图像及其强度直方图。
        """
        # 如果没有采样散斑数据或者数据为空，清空显示并退出
        if self.sampling_speckle is None or self.sampling_speckle.shape[0] == 0:
            self.label_speckle_index.setText("当前散斑: N/A")
            self.fig_individual.clear()
            self.canvas_individual.draw()
            return

        # 更新当前散斑的索引信息
        self.label_speckle_index.setText(
            f"当前散斑: {self.current_speckle_index + 1} / {self.sampling_speckle.shape[0]}")

        # 清空当前图形，并创建新的子图布局
        self.fig_individual.clear()
        gs = self.fig_individual.add_gridspec(1, 2)                              # 创建一个1行2列的网格布局

        # 添加两个子图：一个用于显示散斑图像，另一个用于显示强度直方图
        ax_speckle = self.fig_individual.add_subplot(gs[0, 0])                                # 散斑图像子图
        ax_hist = self.fig_individual.add_subplot(gs[0, 1])                                   # 直方图子图
        current_speckle = self.sampling_speckle[self.current_speckle_index]                   # 获取当前索引对应的散斑数据

        # 调用绘图函数分别绘制散斑图像和强度直方图
        self._plot_speckle(ax_speckle, current_speckle, f"散斑 {self.current_speckle_index + 1}")
        self._plot_histogram(ax_hist, current_speckle, f"散斑 {self.current_speckle_index + 1} 强度直方图")

        self.fig_individual.tight_layout()                                                     # 自动调整子图布局以防止重叠
        self.canvas_individual.draw()                                                          # 刷新画布以显示更新后的图像

    def _prev_speckle(self):
        # 切换上一个散斑
        if self.current_speckle_index > 0:
            self.current_speckle_index -= 1
            self._update_speckle_display()

    def _next_speckle(self):
        # 切换下一散斑
        if (self.sampling_speckle is not None and self.current_speckle_index
                < self.sampling_speckle.shape[
                    0] - 1):
            self.current_speckle_index += 1
            self._update_speckle_display()

    def _plot_all_speckles_mosaic(self):
        """拼接绘制所有散斑"""
        if self.full_speckle is None or self.full_speckle.shape[0] == 0:
            self.canvas_mosaic.draw()
            return

        self.fig_mosaic.clear()                                                    # 清空当前画布上的所有内容
        ax = self.fig_mosaic.add_subplot(111)                                      # 创建一个占据整个画布的子图用于显示

        # 计算布局参数，散斑数据总数以及行、列
        num_speckles = self.full_speckle.shape[0]
        rows = int(np.sqrt(num_speckles))
        cols = int(np.ceil(num_speckles / rows))
        if rows == 0 or cols == 0: rows = cols = 1

        # 创建画布基地
        mosaic_height = rows * self.matrix_size
        mosaic_width = cols * self.matrix_size
        mosaic_image = np.zeros((mosaic_height, mosaic_width), dtype=self.full_speckle.dtype)

        # 遍历所有散斑，将每个散斑图像填充到对应位置
        for i in range(min(num_speckles, rows * cols)):
            row_idx = i // cols
            col_idx = i % cols
            x_start = col_idx * self.matrix_size
            y_start = row_idx * self.matrix_size
            speckle = self.full_speckle[i].reshape(self.matrix_size, self.matrix_size)                          # 将一维散斑数据重塑为二维图像矩阵
            mosaic_image[y_start:y_start + self.matrix_size, x_start:x_start + self.matrix_size] = speckle      # 将散斑图像复制到马赛克图的对应区域

        # 显示图像
        ax.imshow(mosaic_image, cmap='gray')
        ax.set_title(f"所有 {num_speckles} 散斑模式")
        ax.axis('off')

        # 优化布局并刷新画布
        self.fig_mosaic.tight_layout()
        self.canvas_mosaic.draw()

    def _update_speckle_stats(self):
        # 检查散斑数据有效性，无数据时清空画布并返回
        if self.full_speckle is None or self.full_speckle.shape[0] == 0:
            self.fig_stats.clear()
            self.canvas_stats.draw()
            return
        all_speckle_values = self.full_speckle.flatten()                                       # 将所有散斑数据展平为一维数组

        # 创建网格布局
        self.fig_stats.clear()
        gs = self.fig_stats.add_gridspec(2, 1)
        ax_hist_all = self.fig_stats.add_subplot(gs[0, 0])
        ax_stats_text = self.fig_stats.add_subplot(gs[1, 0])
        self._plot_histogram(ax_hist_all, all_speckle_values, "所有散斑强度直方图")          # 绘制所有散斑的强度直方图

        # 计算统计量
        mean_val = np.mean(all_speckle_values)
        std_val = np.std(all_speckle_values)
        min_val = np.min(all_speckle_values)
        max_val = np.max(all_speckle_values)
        median_val = np.median(all_speckle_values)

        # 格式化统计文本
        stats_text = f"所有散斑强度统计:\n" \
                     f"  平均值: {mean_val:.4f}\n" \
                     f"  标准差: {std_val:.4f}\n" \
                     f"  最小值: {min_val:.4f}\n" \
                     f"  最大值: {max_val:.4f}\n" \
                     f"  中位数: {median_val:.4f}"
        ax_stats_text.text(0.05, 0.95, stats_text, transform=ax_stats_text.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax_stats_text.axis('off')

        # 优化布局并刷新画布
        self.fig_stats.tight_layout()
        self.canvas_stats.draw()


class DataGenerator:
    """生成和处理计算鬼成像所需的各类数据，包括图像加载、测量矩阵生成和桶探测信号计算等"""
    # 存储处理过程中的关键数据，包括原始图像、调整大小后的图像、测量矩阵和桶探测信号等初始值均为None，在相应处理步骤完成后被赋值
    def __init__(self):
        self.original_image: Optional[np.ndarray] = None
        self.resized_image: Optional[np.ndarray] = None
        self.measurement_matrix: Optional[np.ndarray] = None
        self.full_rank_measurement_matrix: Optional[np.ndarray] = None
        self.bucket_signals: Optional[np.ndarray] = None
        self.matrix_generator = self.MeasurementMatrixGenerator()

    def load_and_resize_image(self, file_path: str, target_size: int) -> Optional[np.ndarray]:   # 从指定路径加载图像并调整为目标尺寸
        """从指定路径加载图像并调整为目标尺寸"""
        try:
            pil_image = Image.open(file_path).convert('L')                                       # 打开图像文件并转换为灰度图（单通道）
            resized_pil_image = pil_image.resize((target_size, target_size),
                                                 Image.Resampling.LANCZOS)                       # 使用LANCZOS算法将图像调整为指定的目标尺寸
            self.original_image = np.array(resized_pil_image,
                                           dtype=np.float32) / 255.0                             # 将PIL图像转换为NumPy数组，设置数据类型为float32并归一化到[0,1]范围，除以255.0是因为PIL图像的像素值范围通常为0-255
            self.resized_image = self.original_image                                             # 初始状态下，调整大小后的图像与原始图像相同
            return self.resized_image
        # 错误提示
        except FileNotFoundError:
            QMessageBox.critical(None, "图像加载失败", f"文件未找到: {file_path}")
        except Exception as e:
            QMessageBox.critical(None, "图像加载失败", f"加载或调整图像大小失败: {str(e)}")
        self.resized_image = self.original_image = None
        return None

    class MeasurementMatrixGenerator:
        """测量矩阵生成器，负责生成和处理各种类型的测量矩阵，支持随机散斑、Hadamard散斑、Golay散斑和经过Walsh变换的Hadamard散斑矩阵矩阵类型 """

        def __init__(self):
            # 初始化测量矩阵生成器
            pass

        def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
            """
            将矩阵归一化到 [0, 1] 范围
            参数:
                matrix (np.ndarray): 输入矩阵
            返回:
                np.ndarray: 归一化后的矩阵
            """
            min_val, max_val = matrix.min(), matrix.max()                                       # 计算矩阵的最小值和最大值
            if max_val - min_val > 1e-9:                                                        # 检查最大值和最小值的差是否足够大（避免除零错误）
                return (matrix - min_val) / (max_val - min_val)                                 # 执行标准归一化：(X - min) / (max - min)
            return np.zeros_like(matrix) if min_val == 0 else np.ones_like(
                matrix) * min_val                                                               # 处理特殊情况：当矩阵中的所有元素几乎相同时如果所有元素都是0，则返回全0矩阵，否则返回所有元素都等于最小值的矩阵

        def _validate_image_size_for_hadamard(self, image_size: int):
            """
            验证图像尺寸是否为2的幂，这是生成Hadamard矩阵的必要条件
            参数:
                image_size (int): 图像尺寸
            抛出:
                ValueError: 当图像尺寸不是2的幂时
            备注:
                Hadamard矩阵的生成要求矩阵维度必须为2的幂，
                此验证确保后续生成过程的正确性
            """
            if not (image_size > 0 and (image_size & (image_size - 1) == 0)):
                raise ValueError(f"图像边长 ({image_size}) 必须是2的幂。")

        def _generate_random_matrix(self, total_pixels: int) -> np.ndarray:
            """
            生成随机高斯矩阵并归一化
            参数:
                total_pixels (int): 像素总数
            返回:
                np.ndarray: 归一化后的随机矩阵
            算法原理:
                1. 使用标准正态分布(N(0,1))生成随机矩阵
                2. 通过线性变换将矩阵值归一化到[0,1]区间
            """
            matrix = np.random.randn(total_pixels, total_pixels).astype(np.float32)
            return self._normalize_matrix(matrix)

        def _generate_hadamard_matrix(self, total_pixels: int) -> np.ndarray:
            """
            生成Hadamard矩阵并归一化到 [0, 1] 范围
            参数:
                total_pixels (int): 像素总数
            返回:
                np.ndarray: 归一化后的Hadamard矩阵
            数学原理:
                Hadamard矩阵是由+1和-1元素组成的正交矩阵，
                满足 H·H^T = n·I，其中n为矩阵阶数
            转换公式:
                将传统Hadamard矩阵的[-1,1]值域转换为[0,1]:
                normalized_value = (original_value + 1) / 2
            """
            hadamard_matrix = scipy.linalg.hadamard(total_pixels).astype(np.float32)
            return (hadamard_matrix + 1) / 2

        def _generate_golay_matrix(self, total_pixels: int) -> np.ndarray:
            """
            生成Golay散斑矩阵，并通过阈值二值化
            参数:
                total_pixels (int): 像素总数
            返回:
                np.ndarray: Golay散斑矩阵
            算法步骤:
                1. 生成均匀分布的随机矩阵(元素范围[0,1])
                2. 使用0.8作为阈值进行二值化:值小于0.8的元素设为0，值大于等于0.8的元素设为1
            """
            matrix = np.random.rand(total_pixels, total_pixels).astype(np.float32)
            matrix[matrix < 0.8] = 0
            matrix[matrix >= 0.8] = 1
            return matrix

        def _generate_walsh_transformed_hadamard(self, image_size: int) -> np.ndarray:
            """
            生成经过Walsh变换的Hadamard散斑矩阵。

            参数:
                image_size (int): 图像尺寸，必须是2的幂（如2、4、8、16等）。

            返回:
                np.ndarray: 经过Walsh变换的Hadamard散斑矩阵，归一化到[0,1]范围。

            算法流程:
                1. 验证图像尺寸是否为2的幂：如果不是2的幂，则抛出异常或处理错误。
                2. 生成1D Hadamard矩阵作为基函数：使用 `scipy.linalg.hadamard` 生成指定尺寸的Hadamard矩阵。
                3. 通过外积运算生成2D空间图案：外积操作将每一行与另一行进行点乘，生成二维图案，将所有生成的二维图案展平为一维向量。
                4. 将所有2D图案展平并组合成完整矩阵：将所有展平后的图案按顺序堆叠形成一个完整的矩阵。
                5. 归一化矩阵到[0,1]范围：确保输出矩阵的值在[0,1]之间，方便后续处理。
            """
            self._validate_image_size_for_hadamard(image_size)                                     # 验证图像尺寸是否为2的幂
            w_1d = scipy.linalg.hadamard(image_size).astype(np.float32)                            # 生成1D Hadamard矩阵作为基函数
            patterns_2d = [                                                                        # 通过外积运算生成2D空间图案对于每一行i和行j，计算外积并将结果展平
                np.outer(w_1d[i, :], w_1d[j, :]).flatten()                                         # 计算外积并展平
                for i in range(image_size) for j in range(image_size)
            ]
            matrix = np.array(patterns_2d)                                                         # 将所有2D图案展平并组合成完整矩阵，将patterns_2d中的所有图案堆叠成一个二维数组
            return self._normalize_matrix(matrix)                                                  # 归一化矩阵到[0,1]范围#，调用_normalize_matrix方法对矩阵进行归一化处理

        def _generate_fourier_speckle_matrix(self, image_size: int) -> np.ndarray:
            """
            生成傅里叶散斑矩阵
            通过在频域生成随机相位，然后进行傅里叶逆变换得到空间域散斑
            参数:
                image_size (int): 图像尺寸
            返回:
                np.ndarray: 傅里叶散斑矩阵
            """
            speckle_vectors = []                                                                 # 存储生成的散斑图案向量
            num_patterns = image_size * image_size                                               # 散斑图案的数量，等于像素总数
            for _ in range(num_patterns):
                random_phases = np.random.uniform(0, 2 * np.pi,
                                                  size=(image_size, image_size))                 # 频域生成随机相位：每个位置的相位值在0到2π之间随机分布
                freq_domain = np.cos(random_phases) + 1j * np.sin(
                    random_phases)                                                               # 构造复数频域图案：复数的模固定为1，相位由上面的随机值决定，这样可以保证每个图案的能量相同，但相位分布不同
                freq_centered = np.fft.ifftshift(freq_domain)                                    # 逆傅里叶变换到空间域
                spatial_result = np.fft.ifft2(freq_centered)                                     # 执行二维傅里叶逆变换，从频域转换到空间域
                spatial_real = np.real(spatial_result)                                           # 取实部作为最终的空间域图案（傅里叶散斑）
                speckle_vectors.append(spatial_real.reshape(-1))                                 # 将二维散斑图案展平为一维向量并存储

            # 将所有散斑向量组合成完整矩阵并归一化
            full_matrix = np.stack(speckle_vectors).astype(np.float32)
            return self._normalize_matrix(full_matrix)

        def generate_measurement_matrix(
                self,
                matrix_type: str,
                image_size: int,
                sampling_number: float
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            根据指定类型生成测量矩阵
            返回完整秩矩阵和采样后的矩阵
            参数:
                matrix_type (str): 矩阵类型，支持已定义的散斑类型
                image_size (int): 图像尺寸
                sampling_number (float): 采样率
            返回:
                tuple[np.ndarray, np.ndarray]: 完整秩矩阵和采样后的矩阵
            抛出:
                ValueError: 当矩阵类型不支持时
            """
            try:
                total_pixels = image_size * image_size
                num_measurements = int(total_pixels * sampling_number)
                matrix_full_rank = None
                # 根据不同类型生成对应矩阵
                if matrix_type == "随机散斑":
                    matrix_full_rank = self._generate_random_matrix(total_pixels)
                elif matrix_type == "Hadamard散斑 (随机采样)" or matrix_type == "Hadamard散斑 (顺序采样)" or matrix_type == "Hadamard散斑":
                    matrix_full_rank = self._generate_hadamard_matrix(total_pixels)
                elif matrix_type == "Golay散斑":
                    matrix_full_rank = self._generate_golay_matrix(total_pixels)
                elif matrix_type == "经过Walsh变换的Hadamard散斑":
                    matrix_full_rank = self._generate_walsh_transformed_hadamard(image_size)
                elif matrix_type == "傅里叶散斑":
                    matrix_full_rank = self._generate_fourier_speckle_matrix(image_size)
                else:
                    raise ValueError(f"不支持的矩阵类型: {matrix_type}")

                # 根据测量数量和矩阵类型生成测量矩阵
                num_measurements = min(num_measurements, matrix_full_rank.shape[0])                               # 确保测量数量不超过矩阵的行数
                if matrix_type == "Hadamard散斑 (顺序采样)":                                                        # 如果矩阵类型是"Hadamard散斑 (顺序采样)"，则按顺序选择索引
                    chosen_indices = np.arange(num_measurements)                                                  # 使用np.arange生成从0到num_measurements-1的连续索引
                else:                                                                                             # 否则，随机选择索引
                    chosen_indices = np.random.choice(matrix_full_rank.shape[0], num_measurements, replace=False) # 从矩阵的总行数中选择,需要选择的数量,不允许重复选择
                measurement_matrix = matrix_full_rank[chosen_indices, :]                                          # 使用选定的索引构建测量矩阵
                return matrix_full_rank, measurement_matrix                                                       # 返回完整的矩阵和测量矩阵
            except Exception as e:                                                                                # 捕获异常并重新抛出，保留原始错误信息
                raise e

    def generate_measurement_matrix(
            self,
            matrix_type: str,
            image_size: int,
            sampling_rate: float
    ) -> Optional[np.ndarray]:
        """
        调用嵌套的矩阵生成器并设置DataGenerator的属性
        参数:
            matrix_type (str): 矩阵类型，支持"随机散斑"、"Hadamard散斑"、"Golay散斑"等预定义类型
            image_size (int): 图像尺寸
            sampling_rate (float): 采样率（范围：0 < sampling_rate ≤ 1），用于控制生成测量矩阵的行数
        返回:
            Optional[np.ndarray]: 采样后的测量矩阵（形状为[sampled_rows, image_size²]），若生成过程中发生异常则返回None
        """
        # 调用嵌套的矩阵生成器生成完整的矩阵和采样后的矩阵
        try:
            full_rank_matrix, sampled_matrix = self.matrix_generator.generate_measurement_matrix(
                matrix_type, image_size, sampling_rate
            )
            self.full_rank_measurement_matrix = full_rank_matrix                                               # 将生成的矩阵赋值给当前对象的属性
            self.measurement_matrix = sampled_matrix                                                           # 采样后的矩阵
            return self.measurement_matrix                                                                     # 返回采样后的测量矩阵
        except Exception as e:                                                                                 # 异常处理，捕获生成过程中的错误并重新抛出
            raise e

    def generate_bucket_signals(self, noise_type: str, noise_param: float) -> Optional[np.ndarray]:
        """
        根据测量矩阵与图像生成带噪声的桶信号
        参数:
            noise_type (str): 噪声类型（如 "高斯噪声" 或 "泊松噪声"）
            noise_param (float): 噪声参数，用于控制噪声强度
        返回:
            Optional[np.ndarray]: 带噪声的桶信号数组，或在异常时返回 None
        """
        # 检查测量矩阵和图像是否为空
        if self.measurement_matrix is None or self.resized_image is None:
            raise ValueError("测量矩阵或图像数据未初始化，请检查输入数据。")
        try:
            # 展平图像数据并计算理想桶信号
            flat_image = self.resized_image.flatten()
            ideal_signals = np.dot(self.measurement_matrix, flat_image)
            noisy_signals = ideal_signals.copy()
            # 添加指定类型的噪声
            if noise_type == "高斯噪声":
                # 生成高斯噪声并叠加到信号中
                gaussian_noise = np.random.normal(loc=0, scale=noise_param, size=ideal_signals.shape)
                noisy_signals += gaussian_noise
            elif noise_type == "泊松噪声":
                # 计算泊松噪声并进行缩放处理
                scaled_values = ideal_signals * noise_param
                non_negative_scaled = np.maximum(scaled_values, 0)
                poisson_noise = np.random.poisson(non_negative_scaled)
                noisy_signals = (
                    poisson_noise / noise_param if noise_param > 0 else poisson_noise
                )
            # 确保信号非负，并更新实例属性
            noisy_signals = np.clip(noisy_signals, a_min=0, a_max=None)
            self.bucket_signals = noisy_signals
            return self.bucket_signals
        except Exception as e:
            # 捕获异常并重新抛出，便于调用者定位问题
            raise RuntimeError(f"生成桶信号失败: {e}")

    def _save_speckle_mosaic_as_pdf(self, speckle_data: np.ndarray, matrix_size: int, output_path: str, title: str):
        """
           辅助函数：将散斑数据图保存为PDF文件
           参数:
               speckle_data (np.ndarray): 散斑数据，形状为[num_speckles, matrix_size * matrix_size]，
                                          其中每个散斑是一个展平的一维数组
               matrix_size (int): 单个散斑矩阵的尺寸（假设为正方形矩阵）
               output_path (str): 输出PDF文件的路径
               title (str): 图表的标题
           返回:
               None
           """
        # 检查输入的散斑数据是否为空或无效
        if speckle_data is None or speckle_data.shape[0] == 0:
            print(f"警告: 无散斑数据可保存到 {output_path}")
            return
        # 使用 PdfPages 将图表保存为PDF文件
        try:
            with PdfPages(output_path) as pdf:                                          # 创建一个Figure对象并设置其大小
                fig = Figure(figsize=(10, 10))                                          # 按需调整图表尺寸
                num_speckles = speckle_data.shape[0]                                    # 获取散斑的数量
                # 创建一个近似正方形的网格
                rows = int(np.floor(np.sqrt(num_speckles)))
                if rows == 0: rows = 1
                cols = int(np.ceil(num_speckles / rows))
                if cols == 0: cols = 1
                # 创建空白的图像
                mosaic_height = rows * matrix_size
                mosaic_width = cols * matrix_size
                mosaic_image = np.zeros((mosaic_height, mosaic_width), dtype=speckle_data.dtype)
                # 将每个散斑填充到空白的图像中
                for i in range(min(num_speckles, rows * cols)):
                    row_idx = i // cols
                    col_idx = i % cols
                    x_start = col_idx * matrix_size                                                       # 计算当前散斑在空白图像中的位置
                    y_start = row_idx * matrix_size
                    # 将一维散斑数据重塑为二维矩阵，并填充到马赛克图像中
                    speckle = speckle_data[i].reshape(matrix_size, matrix_size)
                    mosaic_image[y_start:y_start + matrix_size, x_start:x_start + matrix_size] = speckle
                # 在图像上绘制图并添加标题
                ax = fig.add_subplot(111)                                                                # 添加子图
                ax.imshow(mosaic_image, cmap='gray', vmin=0, vmax=1)                                     # 假设散斑已被归一化到 [0,1] 范围
                ax.set_title(title)                                                                      # 设置图表标题
                ax.axis('off')                                                                           # 关闭坐标轴显示
                fig.tight_layout()                                                                       # 自动调整子图参数以避免重叠
                # 将图表保存到PDF文件
                pdf.savefig(fig)
                plt.close(fig)
            print(f"散斑图已保存到: {output_path}")
        except Exception as e:
            print(f"错误：保存散斑图到PDF时失败 ({output_path}): {str(e)}")

    def save_reconstruction_data(self, root_dir: str, params: dict, metrics: dict, options: dict,recon_image: Optional[np.ndarray] = None) -> str:
        """
        扩展保存功能，按规范存储实验数据。
        参数:
            root_dir (str): 实验数据根目录路径。
            params (dict): 实验参数字典。
            metrics (dict): 实验指标字典。
            options (dict): 配置选项，用于控制保存哪些数据。
        返回:
            str: 保存过程的日志信息。
        """
        log_entries = []  # 用于记录保存过程中的日志信息
        experiment_folder = DataStorageSpec.get_experiment_dir(root_dir)

        try:
            # 创建实验目录
            os.makedirs(experiment_folder, exist_ok=True)
            log_entries.append(f"实验目录已创建: {experiment_folder}")

            # 保存元数据
            metadata_content = DataStorageSpec.get_metadata(params, metrics)
            metadata_filepath = os.path.join(experiment_folder, "metadata.json")
            with open(metadata_filepath, 'w', encoding='utf-8') as file:
                json.dump(metadata_content, file, indent=2, ensure_ascii=False)
            log_entries.append("元数据文件已保存 (metadata.json)")

            # 保存测量矩阵
            if options.get("measurement_matrix") and self.measurement_matrix is not None:
                measurement_matrix_path = os.path.join(experiment_folder, "sampled_measurement_matrix.npy")
                np.save(measurement_matrix_path, self.measurement_matrix)
                log_entries.append("采样测量矩阵已保存。")

            # 保存桶信号
            if options.get("bucket_signals") and self.bucket_signals is not None:
                bucket_signals_path = os.path.join(experiment_folder, "bucket_signals.npy")
                np.save(bucket_signals_path, self.bucket_signals)
                log_entries.append("桶信号已保存。")

            # 保存原始图像（数值+可视化）
            if options.get("original_image") and self.original_image is not None:
                original_image_npy_path = os.path.join(experiment_folder, "ground_truth_image.npy")
                original_image_png_path = os.path.join(experiment_folder, "ground_truth_image.png")
                np.save(original_image_npy_path, self.original_image)
                cv2.imwrite(original_image_png_path, (self.original_image * 255).astype(np.uint8))
                log_entries.append("原始图像（数值+可视化）已保存。")

            # 保存重建图像（数值+可视化）
            if options.get("reconstructed_image") and recon_image is not None:
                recon_image_npy_path = os.path.join(experiment_folder, "reconstructed_image.npy")
                recon_image_png_path = os.path.join(experiment_folder, "reconstructed_image.png")
                recon_image_normalized = np.clip(recon_image, 0, 1)                                # 确保重建图像在[0,1]范围内
                np.save(recon_image_npy_path, recon_image_normalized)                                           # 保存为npy文件

                # 保存为PNG图像（转换为0-255范围）
                recon_image_uint8 = (recon_image_normalized * 255).astype(np.uint8)
                cv2.imwrite(recon_image_png_path, recon_image_uint8)

                log_entries.append("重建图像（数值+可视化）已保存。")

                # 在元数据中添加重建图像路径
                metadata_content["data_paths"]["reconstructed_image"] = "reconstructed_image.npy"
                metadata_content["data_paths"]["reconstructed_image_vis"] = "reconstructed_image.png"

                # 重新写入更新后的元数据文件
                with open(metadata_filepath, 'w', encoding='utf-8') as file:
                    json.dump(metadata_content, file, indent=2, ensure_ascii=False)
                log_entries.append("元数据已更新重建图像路径。")

            # 保存采样散斑PDF
            if options.get("sampled_speckles_pdf") and self.measurement_matrix is not None:
                matrix_gen_method = params.get("matrix_generation_method", "unknown_matrix_type")
                safe_matrix_name = "".join(
                    c if c.isalnum() or c in (' ', '_', '-') else '_' for c in matrix_gen_method).replace(' ', '_')
                pdf_filename = f"{safe_matrix_name}_sampled_speckles_mosaic.pdf"
                pdf_filepath = os.path.join(experiment_folder, pdf_filename)

                image_s = params.get("image_size", 64)                                                          # 从参数中获取图像尺寸
                title = f"采样散斑 ({matrix_gen_method}) - {self.measurement_matrix.shape[0]} 个模式"
                self._save_speckle_mosaic_as_pdf(self.measurement_matrix, image_s, pdf_filepath, title)
                log_entries.append(f"采样散斑马赛克图PDF已保存: {pdf_filename}")


            # 保存HDF5格式的数据
            if options.get("diffusion_data"):
                hdf5_filepath = os.path.join(experiment_folder, "ghost_imaging_data.h5")
                with h5py.File(hdf5_filepath, 'w') as hdf_file:
                    if self.full_rank_measurement_matrix is not None:
                        hdf_file.create_dataset('full_rank_measurement_matrix', data=self.full_rank_measurement_matrix)
                    if self.measurement_matrix is not None:
                        hdf_file.create_dataset('sampled_measurement_matrix', data=self.measurement_matrix)
                    if self.bucket_signals is not None:
                        hdf_file.create_dataset('bucket_signals', data=self.bucket_signals)
                    if self.original_image is not None:
                        hdf_file.create_dataset('ground_truth_image', data=self.original_image)
                    if recon_image is not None and options.get("reconstructed_image"):
                        hdf_file.create_dataset('reconstructed_image', data=recon_image_normalized)
                log_entries.append("HDF5数据文件已保存。")

        except Exception as e:
            # 捕获异常并记录错误信息
            error_message = f"保存过程中发生错误: {str(e)}"
            log_entries.append(error_message)
            print(f"CRITICAL SAVE ERROR: {error_message}")
        return "\n".join(log_entries)

class ImageReconstructor:
    """
       图像重建器类，实现多种计算成像重建算法，支持二阶关联鬼成像、差分鬼成像、正交匹配追踪(OMP)、TVAL3等重建方法
    """
    def __init__(self):
        pass

    def _covariance_gi(self, a: np.ndarray, b_flat: np.ndarray) -> np.ndarray:
        """
        二阶关联鬼成像 (Covariance Ghost Imaging) 算法实现
        参数:
            A (np.ndarray): 测量矩阵，形状为 [测量次数, 像素总数]
            b_flat (np.ndarray): 扁平化的桶探测器信号，形状为 [测量次数]
        返回:
            np.ndarray: 重建的图像向量，形状为 [像素总数]
        数学原理:
            基于光场的二阶关联特性，计算公式为:
            I(x) = <[B(m) - <B>][A(m,x) - <A(x)>]>
            其中:
                - B(m) 是第m次测量的桶信号
                - A(m,x) 是第m次测量对应的散斑图案在x位置的值
                - <·> 表示对所有测量次数求平均
        实现步骤:
            1. 计算桶信号的全局平均值 B_avg
            2. 计算每个像素位置的散斑图案平均值 A_col_avg
            3. 计算桶信号和散斑图案的涨落量 delta_B 和 delta_A
            4. 通过矩阵乘法计算涨落的相关性
            5. 对测量次数归一化，得到最终重建图像
        性能优化:
            使用矩阵运算代替循环，提高计算效率
            利用NumPy的广播机制处理多维数组运算
        """
        b_avg = np.mean(b_flat)                                             # 计算桶信号的全局平均值
        a_col_avg = np.mean(a, axis=0)                                      # 计算每个像素位置的散斑图案平均值（沿测量次数维度）
        delta_b = b_flat - b_avg                                            # 计算桶信号的涨落（与平均值的偏差）
        delta_a = a - a_col_avg                                             # 计算每个像素位置的散斑图案涨落
        return np.dot(delta_b, delta_a) / len(b_flat)                       # 计算涨落的关联：对每个像素，将桶信号涨落与对应位置的散斑图案涨落相乘并求和，最后除以测量次数进行归一化，得到最终重建图像

    def _differential_gi(self, a: np.ndarray, b_flat: np.ndarray) -> np.ndarray:
        """
        差分鬼成像算法实现，基于一阶关联的快速鬼成像方法，适用于实时成像场景
        参数:
            A (np.ndarray): 测量矩阵，形状为 [测量次数, 像素总数]，每一行代表一次测量的散斑图案（扁平化向量）
            b_flat (np.ndarray): 扁平化桶信号，形状为 [测量次数]，表示每次测量的总光强信号
        返回:
            np.ndarray: 重建的图像向量，形状为 [像素总数]，元素值反映各像素的光强分布，需归一化后显示
        数学原理:
            核心公式为一阶关联计算： I(x) = < B(m)·A(m,x) > - < B >·< A(x) >
            其中：
            - < B(m)·A(m,x) > 是桶信号与散斑图案的点积平均
            - < B > 是桶信号的平均值，
            < A(x) > 是散斑图案的列平均值
            公式等价于协方差计算，但省略了归一化常数（假设测量次数为1）
        实现步骤:
            1. 计算桶信号的全局平均值 B_avg
            2. 计算桶信号与平均值的偏差 delta_B = b_flat - B_avg
            3. 通过矩阵乘法计算 delta_B 与测量矩阵 A 的关联
            4. 对测量次数归一化，得到重建图像向量
        性能优势:
            - 相比二阶关联鬼成像 (Covariance GI) 减少约50%计算量
            - 无需计算散斑图案的列平均值 (A_col_avg)，直接利用原始矩阵
            - 时间复杂度为 O(M·N)，其中 M=测量次数，N=像素总数
        """
        b_avg = np.mean(b_flat)
        delta_b = b_flat - b_avg                                                              # 桶信号与均值的差
        return np.dot(delta_b, a) / len(b_flat)                                               # 直接计算关联信号

    def _omp(self, a: np.ndarray, b: np.ndarray, n_nonzero_coefs: int) -> np.ndarray:
        """
        正交匹配追踪算法 (OMP) 实现
        参数:
            a (np.ndarray): 测量矩阵，形状为 [测量次数, 像素总数]
            b (np.ndarray): 桶信号向量，形状为 [测量次数]
            n_nonzero_coefs (int): 稀疏解的最大非零系数数，需满足 1 ≤ n_nonzero_coefs ≤ min(A.shape)
        返回:
            np.ndarray: 重建的稀疏向量（扁平化图像），形状为 [像素总数]
        算法原理:
            1. 迭代选择与残差最相关的测量矩阵列（原子）
            2. 通过最小二乘法更新稀疏解
            3. 更新残差直至收敛或达到最大迭代次数
        约束条件:
            - 测量矩阵A需满秩或近似满秩
            - 信号需具有稀疏性（非零系数数远小于像素总数）
        异常处理:
            - 当最小二乘法求解失败时（矩阵奇异），回退上一步操作
        """
        x = np.zeros(a.shape[1], dtype=np.float32)                                           # 初始化重建信号向量。形状为[像素总数]，与测量矩阵列数一致，数据类型：float32（节省内存，满足大多数成像精度需求）
        residual = np.copy(b)                                                                # 初始化残差向量为桶信号的副本,残差表示当前重建信号与真实信号的差异，迭代过程中逐步减小
        support = []                                                                         # 支持集记录当前选中的测量矩阵列索引，用于跟踪稀疏解中非零系数的位置
        n_nonzero_coefs = min(n_nonzero_coefs, a.shape[1],
                              a.shape[0])                                                    # 限制最大非零系数数：不能超过像素总数（A.shape[1]），不能超过测量次数（A.shape[0]），保证优化问题有可行解

        for _ in range(n_nonzero_coefs):
            if np.linalg.norm(residual) < 1e-9:                                              # 残差收敛判断
                break
            correlations = np.abs(a.T @ residual)                                            # 计算相关性并选择最大索引（排除已选原子）
            correlations[support] = -1                                                       # 已选原子相关性设为-1
            if np.all(correlations == -1):                                                   # 无可用原子
                break
            max_idx = np.argmax(correlations)
            if max_idx in support:                                                           # 避免重复选择
                break

            support.append(max_idx)
            support.sort()
            a_support = a[:, support]

            try:
                x_s, _, _, _ = np.linalg.lstsq(a_support, b, rcond=None)                    # 最小二乘法求解稀疏系数（使用lstsq避免矩阵求逆的不稳定性）
            except np.linalg.LinAlgError:                                                   # 矩阵奇异时回退
                support.pop()
                break

            # 更新重建信号：先清零整个向量，将当前支持集对应的系数设置为最小二乘解，再更新残差
            x.fill(0)
            x[support] = x_s
            residual = b - a @ x
        return x

    def _TVAL3(self, a: np.ndarray, b_flat: np.ndarray,
               tval3_lambda: float, image_size: int,
               max_iter: int) -> np.ndarray:
        """
        TVAL3 图像重建方法实现（仅支持 CPU）
        参数:
            a (np.ndarray): 测量矩阵，形状为 [测量次数, 像素总数]
            b_flat (np.ndarray): 扁平化桶信号，形状为 [测量次数]
            tval3_lambda (float): 正则化参数 λ
            image_size (int): 重建图像边长（单位：像素）
            max_iter (int): 最大迭代次数
        返回:
            np.ndarray: 重建的图像向量（扁平化），形状为 [像素总数]
            数学模型: min_x { 0.5·||A·x - b||₂² + λ·TV(x) }
        其中:
        - ||A·x - b||₂² 是数据保真项，衡量重建结果与观测数据的匹配程度
        - TV(x) = ∑√[(∂x/∂x)² + (∂x/∂y)²] 是全变分正则项，促进图像梯度稀疏性
        - λ 是正则化参数，控制平滑度与细节的平衡
        """

        def objective_function(x_flat, a_obj, b_target, tv_lambda_param, img_shape_tuple):
            """
            目标函数计算（数据保真项 + 全变分正则项）
            """
            # 数据保真项
            ax_minus_b = a_obj.dot(x_flat) - b_target                                                 # 计算测量残差
            fidelity_term = 0.5 * np.sum(ax_minus_b ** 2)                                             # 计算L2范数平方的一半

            # 全变分正则项
            img_x = x_flat.reshape(img_shape_tuple)                                                   # 重塑为2D图像
            grad_y, grad_x = np.gradient(img_x)                                                       # 计算水平和垂直方向的梯度
            tv_term = np.sum(np.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-12))                              # 添加1e-12防止梯度为零时的数值不稳定问题
            return fidelity_term + tv_lambda_param * tv_term                                          # 返回总目标函数值

        # 预处理：归一化测量矩阵
        scaler = MinMaxScaler()
        a_normalized = scaler.fit_transform(a)

        # 将测量矩阵转换为稀疏格式
        a_sparse = csr_matrix(a_normalized)

        # 初始化猜测解
        initial_guess = np.zeros(a.shape[1], dtype=np.float32)
        bounds = [(0, 1)] * a.shape[1]

        # 优化求解
        result = minimize(
            objective_function, initial_guess,
            args=(a_sparse, b_flat, tval3_lambda, (image_size, image_size)),
            method='L-BFGS-B',
            options={'maxiter': max_iter // 2, 'gtol': 1e-4, 'disp': False},
            bounds=bounds
        )

        if not result.success:
            print(f"TVAL3 优化消息: {result.message}. 使用当前迭代结果。")

        return result.x

    def reconstruct_image(self, reconstruction_method: str,
                          measurement_matrix: np.ndarray,
                          bucket_signals: np.ndarray,
                          image_size: int,
                          omp_k: int = DEFAULT_OMP_K,
                          tval3_lambda: float = DEFAULT_TVAL3_LAMBDA,
                          tval3_max_iter: int = DEFAULT_TVAL3_MAX_ITER) -> Optional[tuple[np.ndarray, float]]:
        """
        根据指定方法重建图像。
        参数:
            reconstruction_method (str): 重建方法名称。
            measurement_matrix (np.ndarray): 测量矩阵。
            bucket_signals (np.ndarray): 桶信号。
            image_size (int): 图像尺寸。
            omp_k (int): OMP 的非零系数数量。
            tval3_lambda (float): TVAL3 的正则化参数。
            tval3_max_iter (int): TVAL3 的最大迭代次数。
        返回:
            Optional[Tuple[np.ndarray, float]]: 重建图像和耗时。
        """
        if measurement_matrix is None or bucket_signals is None or measurement_matrix.shape[0] == 0:
            raise ValueError("测量矩阵或桶信号为空。")                                                      # 输入合法性校验：测量矩阵不能为空，且行数（测量次数）需>0，桶信号不能为空，需与测量矩阵行数一致
        b_flat = bucket_signals.flatten()                                                               # 展平桶信号：支持多维输入，统一转为一维向量形状变为 [测量次数]，与测量矩阵行数匹配
        a = measurement_matrix
        n_pixels = a.shape[1]                                                                           # 提取关键维度：n_pixels = 像素总数 = image_size²后续重建结果维度需与此一致

        reconstructed_image_flat = np.zeros(n_pixels, dtype=np.float32)
        reconstruction_start_time = time.perf_counter()

        try:  # 算法调度
            if reconstruction_method == "二阶关联":
                reconstructed_image_flat = self._covariance_gi(a, b_flat)
            elif reconstruction_method == "差分鬼成像 (DGI)":
                reconstructed_image_flat = self._differential_gi(a, b_flat)
            elif reconstruction_method == "正交匹配追踪 (OMP)":
                reconstructed_image_flat = self._omp(a, b_flat, omp_k)
            elif reconstruction_method == "TVAL3":
                reconstructed_image_flat = self._TVAL3(a, b_flat, tval3_lambda, image_size, tval3_max_iter)
            else:
                raise ValueError(f"不支持的重建方法: {reconstruction_method}")

            # 归一化重建图像
            min_val, max_val = reconstructed_image_flat.min(), reconstructed_image_flat.max()
            # 检查最大值和最小值之间的差值是否大于一个很小的阈值（1e-9），以避免除零错误
            # 如果最大值和最小值之间有足够的差异，则进行归一化操作
            # 将像素值从[min_val, max_val]线性映射到[0, 1]范围
            if max_val - min_val > 1e-9:
                reconstructed_image_flat = (reconstructed_image_flat - min_val) / (max_val - min_val)
            else:
             # 如果最大值和最小值非常接近（即图像几乎为常数）
             # 如果所有像素值均为0，则将图像设置为全零
                if min_val == 0:
                    reconstructed_image_flat = np.zeros_like(reconstructed_image_flat)
                elif min_val != 0 and max_val == min_val:
             # 如果所有像素值相同且不为0，则将图像设置为全1，但需要根据像素值调整归一化的结果
             # 如果像素值在 [0, 1] 范围内，直接使用该值；否则，使用默认值 0.5
                    reconstructed_image_flat = np.ones_like(reconstructed_image_flat) * ((min_val - 0) / (
                                1 - 0) if min_val >= 0 and min_val <= 1 else 0.5)
            # 确保所有像素值被限制在 [0, 1] 范围内，防止出现溢出或异常值
            reconstructed_image_flat = np.clip(reconstructed_image_flat, 0, 1)
        except Exception as e:
            raise ValueError(f"重建过程中发生错误: {str(e)}")
        # 计算重建过程所花费的时间
        reconstruction_time = time.perf_counter() - reconstruction_start_time
        reconstructed_image = reconstructed_image_flat.reshape(image_size, image_size)
        return reconstructed_image, reconstruction_time                                                           # 返回重建后的图像和重建时间

class ImageReconstructionTask(QThread):
                                                                                                                  # 定义信号，用于通知图像重建完成或发生错误
    reconstruction_finished = pyqtSignal(object, float, float, float, int, str)                                   # 使用 object 类型表示重建后的图像
    reconstruction_error = pyqtSignal(str)
    def __init__(self, reconstructor: ImageReconstructor, method: str,
                 matrix: np.ndarray, signals: np.ndarray, original_img: Optional[np.ndarray],
                 img_size: int, omp_k: int, tval3_lambda: float, tval3_max_iter: int):
        super().__init__()
        # 初始化参数
        self.reconstructor = reconstructor                                                                       # 图像重建器实例
        self.method = method                                                                                     # 重建方法名称
        self.matrix = matrix                                                                                     # 测量矩阵
        self.signals = signals                                                                                   # 桶信号数据
        self.original_img = original_img                                                                         # 原始图像（可选）
        self.img_size = img_size                                                                                 # 图像尺寸
        self.omp_k = omp_k                                                                                       # OMP 算法的稀疏度参数
        self.tval3_lambda = tval3_lambda                                                                         # TVAL3 算法的正则化参数
        self.tval3_max_iter = tval3_max_iter                                                                     # TVAL3 算法的最大迭代次数

    def run(self):
        """
        执行图像重建任务。
        """
        try:
            # 调用重建器进行图像重建
            reconstructed_img, recon_time = self.reconstructor.reconstruct_image(
                self.method, self.matrix, self.signals, self.img_size,
                omp_k=self.omp_k,
                tval3_lambda=self.tval3_lambda,
                tval3_max_iter=self.tval3_max_iter
            )

            # 初始化 PSNR 和 SSIM 为 NaN
            current_psnr = float('nan')
            current_ssim = float('nan')

            # 如果重建成功且原始图像存在，则计算 PSNR 和 SSIM
            if reconstructed_img is not None and self.original_img is not None:
                original_img_f64 = self.original_img.astype(np.float64)
                reconstructed_img_f64 = reconstructed_img.astype(np.float64)

                # 防御性检查：确保重建图像和原始图像的尺寸一致
                if original_img_f64.shape != reconstructed_img_f64.shape:
                    self.reconstruction_error.emit(
                        f"重建图像与原始图像尺寸不匹配: {reconstructed_img_f64.shape} vs {original_img_f64.shape}")
                    return
                current_psnr = psnr(original_img_f64, reconstructed_img_f64, data_range=1.0)
                current_ssim = ssim(original_img_f64, reconstructed_img_f64, data_range=1.0, channel_axis=None,
                                    win_size=min(7, self.img_size if self.img_size >= 7 else 7))                   # 窗口大小最小为 7x7，但如果图像尺寸小于 7，则使用图像的实际尺寸作为窗口大小

            # 发出重建完成信号
            self.reconstruction_finished.emit(reconstructed_img, recon_time, current_psnr, current_ssim, self.img_size,
                                              self.method)
        except Exception as e:
            # 捕获异常并发出错误信号
            self.reconstruction_error.emit(f"重建线程错误 ({self.method}): {str(e)}")


class DataGenerationTask(QThread):
    # 定义信号，用于通知数据生成完成或更新进度
    data_generation_finished = pyqtSignal(bool, str)
    progress_updated = pyqtSignal(int)

    def __init__(self, generator: DataGenerator, matrix_type: str, image_size: int,
                 sampling_rate: float, noise_type: str, noise_param: float):
        super().__init__()
        # 初始化参数
        self.generator = generator                                                                      # 数据生成器实例
        self.matrix_type = matrix_type                                                                  # 测量矩阵类型
        self.image_size = image_size                                                                    # 图像尺寸
        self.sampling_rate = sampling_rate                                                              # 采样率
        self.noise_type = noise_type                                                                    # 噪声类型
        self.noise_param = noise_param                                                                  # 噪声参数

    def run(self):
        """
        执行数据生成任务。
        """
        try:
            # 更新进度为 0%
            self.progress_updated.emit(0)

            # 生成测量矩阵
            matrix_start_time = time.perf_counter()
            self.generator.generate_measurement_matrix(self.matrix_type, self.image_size, self.sampling_rate)
            matrix_end_time = time.perf_counter()

            # 更新进度为 50%
            self.progress_updated.emit(50)

            # 生成桶信号
            signals_start_time = time.perf_counter()
            self.generator.generate_bucket_signals(self.noise_type, self.noise_param)
            signals_end_time = time.perf_counter()

            # 更新进度为 100%
            self.progress_updated.emit(100)

            # 构建日志消息
            log_message = (f"数据生成完成！\n"
                           f"  测量矩阵生成时间: {matrix_end_time - matrix_start_time:.4f} 秒\n"
                           f"  桶信号生成时间: {signals_end_time - signals_start_time:.4f} 秒\n"
                           f"  测量矩阵尺寸: {self.generator.measurement_matrix.shape if self.generator.measurement_matrix is not None else 'N/A'}\n"
                           f"  桶信号数量: {self.generator.bucket_signals.shape[0] if self.generator.bucket_signals is not None else 'N/A'}")

            # 发出数据生成完成信号
            self.data_generation_finished.emit(True, log_message)
        except Exception as e:
            # 捕获异常并发出错误信号
            self.data_generation_finished.emit(False, f"数据生成线程错误: {str(e)}")


class GhostImagingApp(QMainWindow):
    """计算鬼成像模拟系统主窗口类
    实现图像加载、数据生成、图像重建的全流程控制，包含UI界面和业务逻辑
       """

    def __init__(self):
        super().__init__()
        self.data_generator = DataGenerator()                                                            # 数据生成器实例
        self.image_reconstructor = ImageReconstructor()                                                  # 图像重建器实例
        self.current_matrix_size: int = IMAGE_SIZES[DEFAULT_IMAGE_SIZE]                                  # 当前图像尺寸（像素）
        self.last_loaded_image_path: Optional[str] = None                                                # 最后加载的图像路径
        self.reconstructed_image: Optional[np.ndarray] = None                                            # 重建后的图像数据
        self.use_sequential_sampling = False                                                             # 是否启用顺序采样（仅对Hadamard散斑有效） - This flag is not actively used, matrix_type string handles it.
        self.init_ui()                                                                                   # 初始化UI界面
        self._initialize_parameters()                                                                    # 初始化参数设置
        self._connect_signals()                                                                          # 连接信号与槽函数

    def init_ui(self):
        """初始化用户界面，包含图像加载、数据生成、重建控制和结果显示模块"""
        self.setWindowTitle("计算鬼成像模拟系统")
        self.setGeometry(100, 100, 1600, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)

        # 图像加载模块
        img_load_group = QGridLayout()
        img_load_group.addWidget(QLabel("1. 图像加载"), 0, 0, 1, 2)
        self.btn_load_image = QPushButton("加载图像")
        img_load_group.addWidget(self.btn_load_image, 1, 0, 1, 2)
        img_load_group.addWidget(QLabel("图像尺寸:"), 2, 0)
        self.cb_image_size = QComboBox()
        self.cb_image_size.addItems([f"{s}x{s}" for s in IMAGE_SIZES])
        img_load_group.addWidget(self.cb_image_size, 2, 1)
        control_panel.addLayout(img_load_group)

        # 数据生成模块
        data_gen_group = QGridLayout()
        data_gen_group.addWidget(QLabel("2. 数据生成"), 0, 0, 1, 2)
        data_gen_group.addWidget(QLabel("测量矩阵类型:"), 1, 0)
        self.cb_matrix_type = QComboBox()
        self.cb_matrix_type.addItems(["随机散斑",
                                      "Hadamard散斑 (随机采样)",
                                      "Hadamard散斑 (顺序采样)",
                                      "Golay散斑",
                                      "经过Walsh变换的Hadamard散斑",
                                      "傅里叶散斑"])
        data_gen_group.addWidget(self.cb_matrix_type, 1, 1)
        data_gen_group.addWidget(QLabel("采样率:"), 2, 0)
        self.slider_sampling_rate = QSlider(Qt.Horizontal)
        self.slider_sampling_rate.setRange(1, 100)
        self.label_sampling_rate = QLabel()
        data_gen_group.addWidget(self.slider_sampling_rate, 2, 1)
        data_gen_group.addWidget(self.label_sampling_rate, 2, 2)
        data_gen_group.addWidget(QLabel("噪声类型:"), 3, 0)
        self.cb_noise_type = QComboBox()
        self.cb_noise_type.addItems(["无噪声", "高斯噪声", "泊松噪声"])
        data_gen_group.addWidget(self.cb_noise_type, 3, 1)
        data_gen_group.addWidget(QLabel("噪声参数:"), 4, 0)
        self.slider_noise_param = QSlider(Qt.Horizontal)
        self.label_noise_param = QLabel()
        data_gen_group.addWidget(self.slider_noise_param, 4, 1)
        data_gen_group.addWidget(self.label_noise_param, 4, 2)
        self.btn_generate_data = QPushButton("生成数据")
        data_gen_group.addWidget(self.btn_generate_data, 5, 0, 1, 3)
        control_panel.addLayout(data_gen_group)

        # 图像重建模块
        recon_group = QGridLayout()
        recon_group.addWidget(QLabel("3. 图像重建"), 0, 0, 1, 2)
        recon_group.addWidget(QLabel("重建方法:"), 1, 0)
        self.cb_reconstruction_method = QComboBox()
        self.cb_reconstruction_method.addItems(["二阶关联", "差分鬼成像 (DGI)",
                                                "正交匹配追踪 (OMP)", "TVAL3"])
        recon_group.addWidget(self.cb_reconstruction_method, 1, 1)

        # OMP算法参数
        recon_group.addWidget(QLabel("OMP Sparsity (K):"), 2, 0)
        self.slider_omp_k = QSlider(Qt.Horizontal)
        self.slider_omp_k.setRange(1, 1000)  # Wider range for K
        self.label_omp_k = QLabel()
        recon_group.addWidget(self.slider_omp_k, 2, 1)
        recon_group.addWidget(self.label_omp_k, 2, 2)

        # TVAL3 算法参数
        recon_group.addWidget(QLabel("TVAL3 Lambda:"), 3, 0)
        self.slider_tval3_lambda = QSlider(Qt.Horizontal)
        self.slider_tval3_lambda.setRange(0, 1000)  # 0.000 to 1.000
        self.label_tval3_lambda = QLabel()
        recon_group.addWidget(self.slider_tval3_lambda, 3, 1)
        recon_group.addWidget(self.label_tval3_lambda, 3, 2)
        recon_group.addWidget(QLabel("TVAL3 Max Iter:"), 4, 0)
        self.slider_tval3_max_iter = QSlider(Qt.Horizontal)
        self.slider_tval3_max_iter.setRange(10, 5000)
        self.label_tval3_max_iter = QLabel()
        recon_group.addWidget(self.slider_tval3_max_iter, 4, 1)
        recon_group.addWidget(self.label_tval3_max_iter, 4, 2)
        self.btn_reconstruct = QPushButton("重建图像")
        recon_group.addWidget(self.btn_reconstruct, 5, 0, 1, 3)
        control_panel.addLayout(recon_group)
        control_panel.addSpacing(15)

        # 操作按钮
        action_buttons_layout = QHBoxLayout()
        self.btn_view_speckles = QPushButton("查看散斑")
        self.btn_save_data = QPushButton("保存数据")
        self.btn_clear_logs = QPushButton("清除日志/结果")
        action_buttons_layout.addWidget(self.btn_view_speckles)
        action_buttons_layout.addWidget(self.btn_save_data)
        action_buttons_layout.addWidget(self.btn_clear_logs)
        control_panel.addLayout(action_buttons_layout)
        control_panel.addStretch(1)
        main_layout.addLayout(control_panel, 1)

        # 显示与日志模块
        display_log_panel = QVBoxLayout()
        self.tab_widget = QTabWidget()
        display_log_panel.addWidget(self.tab_widget)

        # 原始图像显示
        original_img_widget = QWidget()
        self.original_canvas_widget = FigureCanvas(Figure(figsize=(6, 6)))
        QVBoxLayout(original_img_widget).addWidget(self.original_canvas_widget)
        QVBoxLayout(original_img_widget).addWidget(NavigationToolbar(self.original_canvas_widget, self))
        self.tab_widget.addTab(original_img_widget, "原始图像")

        # 重建图像显示
        recon_img_widget = QWidget()
        self.recon_canvas_widget = FigureCanvas(Figure(figsize=(6, 6)))
        QVBoxLayout(recon_img_widget).addWidget(self.recon_canvas_widget)
        QVBoxLayout(recon_img_widget).addWidget(NavigationToolbar(self.recon_canvas_widget, self))
        self.tab_widget.addTab(recon_img_widget, "重建图像")

        # 评估指标显示
        metrics_layout = QHBoxLayout()
        self.label_reconstruction_time = QLabel("重建时间: N/A")
        self.label_psnr = QLabel("PSNR: N/A")
        self.label_ssim = QLabel("SSIM: N/A")
        metrics_layout.addWidget(self.label_reconstruction_time)
        metrics_layout.addWidget(self.label_psnr)
        metrics_layout.addWidget(self.label_ssim)
        metrics_layout.addStretch(1)
        display_log_panel.addLayout(metrics_layout)

        # 进度条与日志
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        display_log_panel.addWidget(self.progress_bar)

        display_log_panel.addWidget(QLabel("操作日志:"))
        self.log_text_browser = QTextBrowser()
        self.log_text_browser.setMinimumHeight(150)
        display_log_panel.addWidget(self.log_text_browser)
        main_layout.addLayout(display_log_panel, 2)
        self.statusBar()
        self._set_controls_enabled(False)
        self.btn_load_image.setEnabled(True)

    def _initialize_parameters(self):
        """初始化UI控件的参数默认值，建立控件状态与默认参数的映射关系"""
        self.cb_image_size.setCurrentIndex(
            DEFAULT_IMAGE_SIZE)                                                   # 初始化图像尺寸选择框：DEFAULT_IMAGE_SIZE=2，对应IMAGE_SIZES[2]=64，即默认尺寸为64x64像素，控件显示为"64x64"，可通过下拉菜单选择其他尺寸（16, 32, 128）
        self.cb_matrix_type.setCurrentIndex(
            0)                                                                    # 初始化测量矩阵类型选择框，索引0对应"随机散斑"，为默认选择的矩阵类型其他选项包括：Hadamard散斑、Golay散斑、经过Walsh变换的Hadamard散斑、傅里叶散斑
        self.slider_sampling_rate.setValue(int(DEFAULT_SAMPLING_RATE * 100))
        self.label_sampling_rate.setText(
            f"{DEFAULT_SAMPLING_RATE:.0%}")                                       # 初始化采样率滑动条和标签：DEFAULT_SAMPLING_RATE=0.5（50%），滑动条范围1-100，setValue(50)对应50%采样率，标签显示"50%"，与滑动条数值同步
        self.cb_noise_type.setCurrentIndex(0)
        self._update_noise_param_label(0)                                         # “基于‘无噪声’的初始噪声参数更新”

        self.cb_reconstruction_method.setCurrentIndex(0)                          # 初始化重建方法选择框
        self.slider_omp_k.setValue(DEFAULT_OMP_K)
        self.label_omp_k.setText(str(DEFAULT_OMP_K))                              # 初始化OMP算法参数：
        self.slider_tval3_lambda.setValue(int(DEFAULT_TVAL3_LAMBDA * 1000))
        self.label_tval3_lambda.setText(f"{DEFAULT_TVAL3_LAMBDA:.3f}")
        self.slider_tval3_max_iter.setValue(DEFAULT_TVAL3_MAX_ITER)
        self.label_tval3_max_iter.setText(str(DEFAULT_TVAL3_MAX_ITER))            # 初始化TVAL3算法参数
        self._update_reconstruction_sliders_enabled()                             # 根据默认重建方法（二阶关联）更新参数控件状态：二阶关联和DGI无需参数调整，因此OMP和TVAL3的参数滑动条默认禁用，当选择OMP或TVAL3时，对应滑动条自动启用（在_update_reconstruction_sliders_enabled中实现）

    def _connect_signals(self):
        """将UI控件的信号连接到对应的槽函数，实现用户交互逻辑"""
        self.btn_load_image.clicked.connect(self._load_image)                     # 图像加载模块
        self.cb_image_size.currentIndexChanged.connect(self._on_image_size_changed)
        self.slider_sampling_rate.valueChanged.connect(
            lambda val: self.label_sampling_rate.setText(f"{val}%"))               # 数据生成模块
        self.cb_noise_type.currentIndexChanged.connect(self._on_noise_type_changed)
        self.slider_noise_param.valueChanged.connect(self._update_noise_param_label)
        self.btn_generate_data.clicked.connect(self._generate_data)
        self.cb_reconstruction_method.currentIndexChanged.connect(self._update_reconstruction_sliders_enabled)  # 图像重建模块
        self.slider_omp_k.valueChanged.connect(lambda val: self.label_omp_k.setText(f"{val}"))
        self.slider_tval3_lambda.valueChanged.connect(lambda val: self.label_tval3_lambda.setText(f"{val / 1000:.3f}"))
        self.slider_tval3_max_iter.valueChanged.connect(lambda val: self.label_tval3_max_iter.setText(str(val)))
        self.btn_reconstruct.clicked.connect(self._reconstruct_image)              # 操作按钮模块
        self.btn_view_speckles.clicked.connect(self._view_speckles)
        self.btn_save_data.clicked.connect(self._save_data)
        self.btn_clear_logs.clicked.connect(self._clear_logs_and_results)

    def _set_controls_enabled(self, enable: bool):
        """统一控制UI控件的启用状态，避免无效操作"""
        self.btn_load_image.setEnabled(True)                                                           # 始终启用“加载图像”按钮（允许随时加载新图像）
        controls_to_toggle = [
            self.cb_image_size, self.cb_matrix_type, self.slider_sampling_rate,
            self.label_sampling_rate, self.cb_noise_type, self.btn_clear_logs
        ]
        for ctrl in controls_to_toggle: ctrl.setEnabled(enable)                                        # 可批量控制的控件组
        is_noise_active = self.cb_noise_type.currentText() != "无噪声"                                  # 检查当前选择的噪声类型是否为 "无噪声"，以判断噪声是否处于激活状态
        self.slider_noise_param.setEnabled(enable and is_noise_active)                                 # 根据噪声是否激活以及整体的 "enable" 状态，设置滑块和标签的可用性
        self.label_noise_param.setEnabled(enable and is_noise_active)                                  # 如果 "enable" 为 False 或者噪声未激活，则禁用滑块和标签
        if not enable:                                                                                 # 检查滑块是否被禁用
            if not self.slider_noise_param.isEnabled():                                                # 如果噪声类型为 "无噪声"，将标签文本设置为 "N/A"，表示不适用
                self.label_noise_param.setText(
                    "N/A" if self.cb_noise_type.currentText() == "无噪声" else self.label_noise_param.text())
        else:                                                                        # 如果整体的 "enable" 为 True，更新噪声参数标签以反映当前滑块值
            self._update_noise_param_label(self.slider_noise_param.value())
        self.btn_generate_data.setEnabled(enable and self.data_generator.resized_image is not None)     # 数据生成按钮：仅当已加载图像时可用
        has_data = self.data_generator.bucket_signals is not None
        self.btn_reconstruct.setEnabled(enable and has_data)
        self.cb_reconstruction_method.setEnabled(enable and has_data)
        self._update_reconstruction_sliders_enabled()                                                   # 重建相关控件：仅当已生成数据时可用

        self.btn_view_speckles.setEnabled(
            enable and self.data_generator.full_rank_measurement_matrix is not None)                    # 散斑查看按钮：仅当存在完整测量矩阵时可用
        can_save = any([self.data_generator.measurement_matrix is not None,
                        self.data_generator.bucket_signals is not None,
                        self.data_generator.original_image is not None,
                        self.reconstructed_image is not None])
        self.btn_save_data.setEnabled(enable and can_save)                                               # 数据保存按钮：仅当存在可保存数据时可用

    def _update_reconstruction_sliders_enabled(self):
        """根据当前选择的重建方法，动态启用或禁用对应的参数滑动条"""
        method = self.cb_reconstruction_method.currentText()
        base_enabled = self.btn_reconstruct.isEnabled()                                                         # 获取当前选择的重建方法和重建按钮的状态

        # 正交匹配追踪(OMP)算法需要调整稀疏度参数K
        is_omp = (method == "正交匹配追踪 (OMP)")
        self.slider_omp_k.setEnabled(is_omp and base_enabled)
        self.label_omp_k.setEnabled(is_omp and base_enabled)

        # TVAL3算法需要调整lambda参数和最大迭代次数
        is_tval3 = (method == "TVAL3")
        self.slider_tval3_lambda.setEnabled(is_tval3 and base_enabled)
        self.label_tval3_lambda.setEnabled(is_tval3 and base_enabled)
        self.slider_tval3_max_iter.setEnabled(is_tval3 and base_enabled)
        self.label_tval3_max_iter.setEnabled(is_tval3 and base_enabled)

    def _append_log(self, message: str):
        """向操作日志中添加带时间戳的消息"""
        self.log_text_browser.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _clear_logs_and_results(self):
        """清除所有日志和重建结果，重置系统状态"""
        self.log_text_browser.clear()                                                             # 清除日志显示
        self.label_reconstruction_time.setText("重建时间: N/A")
        self.label_psnr.setText("PSNR: N/A")
        self.label_ssim.setText("SSIM: N/A")                                                      # 重置评估指标显示

        # 内部函数：清空画布并显示空白图像
        def clear_canvas(canvas: FigureCanvas, title_text: str):
            if canvas.figure is None: return
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            size_to_use = self.current_matrix_size if hasattr(self, 'current_matrix_size') else 64
            ax.imshow(np.zeros((size_to_use, size_to_use)), cmap='gray', vmin=0, vmax=1)
            ax.set_title(title_text)
            ax.axis('off')
            canvas.draw()

        # 清空原始图像和重建图像显示
        clear_canvas(self.original_canvas_widget, "原始图像 (未加载)")
        clear_canvas(self.recon_canvas_widget, "重建图像 (无)")

        # 清除数据生成器中的所有数据
        self.reconstructed_image = None
        if self.data_generator:
            self.data_generator.measurement_matrix = None
            self.data_generator.full_rank_measurement_matrix = None
            self.data_generator.bucket_signals = None
            self.data_generator.original_image = None
            self.data_generator.resized_image = None

        # 保留最后加载的图像路径，允许快速重新加载 禁用大部分控件，仅保留图像加载按钮可用
        self._set_controls_enabled(False)
        self.btn_load_image.setEnabled(True)

    def _display_image(self, canvas: FigureCanvas, image_data: Optional[np.ndarray], title: str):
        """在指定画布上显示图像，处理图像数据为空的情况"""
        if canvas.figure is None: return
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        size_to_use = self.current_matrix_size if hasattr(self, 'current_matrix_size') else 64

        # 处理图像数据为空的情况，显示黑色图像并添加错误提示
        if image_data is None:
            image_data = np.zeros((size_to_use, size_to_use))
            title += " (数据错误或丢失)"
        elif image_data.shape != (size_to_use, size_to_use):
            try:
                if image_data.size == size_to_use * size_to_use:
                    image_data = image_data.reshape(size_to_use, size_to_use)
                else:
                    self._append_log(
                        f"警告: 图像 '{title}' 尺寸 {image_data.shape} 与预期 {size_to_use}x{size_to_use} 不符。显示占位符。")
                    image_data = np.zeros((size_to_use, size_to_use))
                    title += " (尺寸不匹配)"
            except Exception as e_reshape:
                self._append_log(f"警告: 调整图像 '{title}' 尺寸时出错: {e_reshape}. 显示占位符。")
                image_data = np.zeros((size_to_use, size_to_use))
                title += " (尺寸错误)"

        ax.imshow(image_data, cmap='gray', vmin=0, vmax=1)                                       # 显示图像，设置灰度色彩映射和值范围
        ax.set_title(title)
        ax.axis('off')                                                                           # 关闭坐标轴显示
        canvas.draw()                                                                            # 触发画布重绘

    def _load_image_with_path(self, file_path: str):
        """
           根据指定路径加载并显示图像
           调用数据生成器处理图像加载和调整大小，并更新UI显示
           参数:
               file_path (str): 图像文件的完整路径
           """
        self._append_log(f"正在加载图像: {file_path}...")
        self.statusBar().showMessage("正在加载图像...", 2000)
        img = self.data_generator.load_and_resize_image(file_path, self.current_matrix_size)    # 调用数据生成器加载并调整图像大小

        # 图像加载成功，更新UI状态
        if img is not None:
            self.last_loaded_image_path = file_path
            self._display_image(self.original_canvas_widget, img, "原始图像")
            self._append_log("图像加载成功。")
            self.statusBar().showMessage("图像加载完成。", 3000)
            self._set_controls_enabled(True)
        # 图像加载失败，显示错误信息
        else:
            self._append_log("图像加载失败。")
            self.statusBar().showMessage("图像加载失败。", 3000)
            self._set_controls_enabled(False)
            self.btn_load_image.setEnabled(True)

    def _load_image(self):
        """
           打开文件选择对话框，让用户选择图像文件
           选择后清除先前状态并加载新图像
           """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "",
                                                   "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")  # 打开文件选择对话框，限制为常见图像格式
        # 选择文件，清除先前状态并加载新图像
        if file_path:
            if file_path != self.last_loaded_image_path:
                self._clear_logs_and_results()
                self._append_log("加载新图像，清除之前的结果和数据。")
            self._load_image_with_path(file_path)

    def _on_image_size_changed(self):
        """
          处理图像尺寸下拉菜单变更事件
          根据用户选择调整图像尺寸，并在必要时重新加载图像
          """
        new_size = int(self.cb_image_size.currentText().split('x')[0])                                 # 获取新选择的图像尺寸
        # 如果尺寸未变化，不执行任何操作
        if new_size == self.current_matrix_size:
            return
        self._append_log(f"图像尺寸更改为 {new_size}x{new_size}。")

        # 已有加载的图像，询问是否重新加载以匹配新尺寸
        if self.last_loaded_image_path and self.data_generator.original_image is not None:
            reply = QMessageBox.question(self, "图像尺寸改变",
                                         "图像尺寸已更改。为了确保数据一致性，已加载的图像将重新加载并调整大小以匹配新尺寸。\n"
                                         "这会清除当前数据和重建结果。是否继续？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            # 用户同意，清除数据并重新加载图像
            if reply == QMessageBox.Yes:
                original_last_path = self.last_loaded_image_path
                self._clear_logs_and_results()
                self.current_matrix_size = new_size
                self._load_image_with_path(original_last_path)
            # 用户取消，恢复原来的尺寸选择
            else:
                old_size_text = f"{self.current_matrix_size}x{self.current_matrix_size}"
                old_index = self.cb_image_size.findText(old_size_text)
                if old_index != -1:
                    self.cb_image_size.blockSignals(True)
                    self.cb_image_size.setCurrentIndex(old_index)
                    self.cb_image_size.blockSignals(False)
                self._append_log("图像尺寸更改被取消，保留原始尺寸。")
        # 没有已加载的图像，直接更新尺寸设置
        else:
            self.current_matrix_size = new_size
            self._append_log(f"图像尺寸已设为 {new_size}x{new_size}。请加载图像以应用新的尺寸设置。")
            self._clear_logs_and_results()

    def _on_noise_type_changed(self):
        """响应噪声类型选择变更事件，重置噪声参数并更新界面"""
        noise_type = self.cb_noise_type.currentText()
        current_value = 0  # Default for "无噪声"

        # 根据选择的噪声类型设置默认参数值
        if noise_type == "高斯噪声":
            current_value = int(DEFAULT_GAUSSIAN_NOISE_STD * 1000)
        elif noise_type == "泊松噪声":
            current_value = int(DEFAULT_POISSON_NOISE_LAMBDA * 10)
        self.slider_noise_param.setValue(current_value)
        self._update_noise_param_label(current_value)

    def _update_noise_param_label(self, value: int):
        """根据噪声类型和滑动条值，更新参数标签显示并配置控件状态"""
        noise_type = self.cb_noise_type.currentText()
        general_enable_from_parent = self.cb_noise_type.isEnabled()
        is_specific_noise_type = noise_type != "无噪声"
        slider_should_be_enabled = general_enable_from_parent and is_specific_noise_type
        tooltip = "无噪声添加"
        display_text = "N/A"

        # 配置高斯噪声参数（范围0-1.0，滑动条步长0.001）
        if noise_type == "高斯噪声":
            self.slider_noise_param.setRange(0, 1000)
            scaled_value = value / 1000.0                                                     # 转换为实际标准差（0.000~1.000）
            display_text = f"{scaled_value:.3f}"                                              # 显示3位小数
            tooltip = "高斯噪声标准差（σ），值越大噪声越强"

        # 配置泊松噪声参数（范围0.1-20.0，滑动条步长0.1）
        elif noise_type == "泊松噪声":
            self.slider_noise_param.setRange(1, 200)
            if value < self.slider_noise_param.minimum() and self.slider_noise_param.isEnabled():
                value = self.slider_noise_param.minimum()
                self.slider_noise_param.setValue(value)
            scaled_value = value / 10.0                                                       # 转换为实际λ值（0.1~20.0）
            display_text = f"{scaled_value:.1f}"                                              # 显示1位小数
            tooltip = "泊松噪声强度λ，值越大信号强度和噪声越高"
        # 无噪声，禁用滑动条并清空显示
        else:  # "无噪声"
            self.slider_noise_param.setRange(0, 0)
            display_text = "N/A"
            tooltip = "无噪声添加"
        self.label_noise_param.setText(display_text)
        self.slider_noise_param.setToolTip(tooltip)
        self.slider_noise_param.setEnabled(slider_should_be_enabled)
        self.label_noise_param.setEnabled(slider_should_be_enabled)

    def _generate_data(self):
        """触发测量矩阵和桶信号生成流程，包含参数校验和异步处理"""
        # 校验：必须先加载图像
        if self.data_generator.resized_image is None:
            QMessageBox.warning(self, "数据生成失败", "请先加载图像。")
            return
        matrix_type = self.cb_matrix_type.currentText()
        image_s = self.current_matrix_size
        sampling_rate = self.slider_sampling_rate.value() / 100.0                           # 转换为0-1的小数
        noise_type = self.cb_noise_type.currentText()
        noise_param_text = self.label_noise_param.text()
        noise_param = 0.0
        if noise_param_text != "N/A":
            try:
                noise_param = float(noise_param_text)
            except ValueError:
                QMessageBox.critical(self, "参数错误", f"无效的噪声参数值: {noise_param_text}")
                return
        elif noise_type != "无噪声":
            QMessageBox.critical(self, "参数错误", f"噪声类型为 {noise_type} 但噪声参数为 N/A。请检查。")
            return

        # Hadamard相关矩阵的尺寸校验
        total_pixels = image_s * image_s
        if matrix_type == "Hadamard散斑 (随机采样)" or matrix_type == "Hadamard散斑 (顺序采样)" or matrix_type == "Hadamard散斑":
            if not (total_pixels > 0 and (total_pixels & (total_pixels - 1) == 0)):
                QMessageBox.critical(
                    self, f"{matrix_type} 错误",
                    f"{matrix_type} 要求像素总数（{total_pixels} = {image_s}x{image_s}）必须是2的幂 (例如 64, 256, 1024, 4096...对图像边长 {image_s} 意味着边长本身也需要是2的幂的开方，如 8, 16, 32, 64...)."
                )
                return
        elif matrix_type == "经过Walsh变换的Hadamard散斑":
            if not (image_s > 0 and (image_s & (image_s - 1) == 0)):
                QMessageBox.critical(
                    self, f"{matrix_type} 错误",
                    f"{matrix_type} 要求图像边长（{image_s}）必须是2的幂（如8, 16, 32, 64...）。"
                )
                return

        # 禁用控件并显示进度条
        self._set_controls_enabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage("正在生成数据...", 0)
        self._append_log("开始生成数据...")

        # 启动异步数据生成线程
        self.data_gen_thread = DataGenerationTask(
            self.data_generator,
            matrix_type,
            image_s,
            sampling_rate,
            noise_type,
            noise_param
        )
        self.data_gen_thread.data_generation_finished.connect(self._on_data_generation_finished)
        self.data_gen_thread.progress_updated.connect(self.progress_bar.setValue)
        self.data_gen_thread.start()

    def _on_data_generation_finished(self, success: bool, message: str):
        """处理数据生成完成事件，更新界面状态并反馈结果"""
        self.progress_bar.setVisible(False)                                                              # 隐藏进度条
        self.progress_bar.setFormat("%p%")                                                               # 恢复进度条格式
        # 数据生成成功的处理逻辑
        if success:
            self.statusBar().showMessage("数据生成完成。", 3000)                             # 状态栏提示成功
            self._append_log(message)                                                                    # 将日志信息添加到操作日志
            self.reconstructed_image = None                                                              # 重置重建图像数据
            self._display_image(self.recon_canvas_widget, None, "重建图像 (请运行重建)")     # 显示提示信息到重建图像画布
            # 重置评估指标显示
            self.label_reconstruction_time.setText("重建时间: N/A")
            self.label_psnr.setText("PSNR: N/A")
            self.label_ssim.setText("SSIM: N/A")
        else:
            # 数据生成失败的处理逻辑
            self.statusBar().showMessage("数据生成失败。", 3000)                             # 状态栏提示失败
            QMessageBox.critical(self, "数据生成失败", message)                                             # 弹出错误对话框
            self._append_log(f"数据生成失败: {message}")                                                    # 记录错误日志
        self._set_controls_enabled(True)                                                                  # 恢复控件交互状态

    def _reconstruct_image(self):
        """触发图像重建流程，包含数据校验和异步处理"""
        # 校验：必须先生成测量矩阵和桶信号
        if self.data_generator.measurement_matrix is None or self.data_generator.bucket_signals is None:
            QMessageBox.warning(self, "重建失败", "请先生成数据。")
            return

        self._set_controls_enabled(False)                                                                  # 禁用交互控件避免干扰
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setVisible(True)                                                                 # 显示进度条
        self.statusBar().showMessage("正在重建图像...", 0)
        method = self.cb_reconstruction_method.currentText()                                               # 获取当前重建方法
        self._append_log(f"开始重建图像 ({method})...")                                                      # 记录日志
        self.progress_bar.setFormat("重建中...")

        # 初始化重建任务线程
        self.recon_thread = ImageReconstructionTask(
            self.image_reconstructor,                                                                      # 图像重建器实例
            method,                                                                                        # 重建方法名称
            self.data_generator.measurement_matrix,                                                        # 测量矩阵
            self.data_generator.bucket_signals,                                                            # 桶信号
            self.data_generator.original_image,                                                            # 原始图像（用于计算评估指标）
            self.current_matrix_size,                                                                      # 图像尺寸
            self.slider_omp_k.value(),                                                                     # OMP算法的K值（稀疏度）
            self.slider_tval3_lambda.value() / 1000.0,                                                     # TVAL3的λ参数（转换为实际值）
            self.slider_tval3_max_iter.value()                                                             # TVAL3的最大迭代次数
        )

        # 连接线程信号到回调函数
        self.recon_thread.reconstruction_finished.connect(self._on_reconstruction_finished)
        self.recon_thread.reconstruction_error.connect(self._on_reconstruction_error)
        self.recon_thread.start()                                                                          # 启动重建线程

    def _on_reconstruction_finished(self, reconstructed_img: Optional[np.ndarray], recon_time: float,
                                    psnr_val: float, ssim_val: float, matrix_size: int, method: str):
        """处理图像重建完成事件，更新界面显示并记录结果"""
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)                                                                 # 隐藏进度条
        self.progress_bar.setFormat("%p%")                                                                  # 恢复进度条默认格式
        self.reconstructed_image = reconstructed_img                                                        # 保存重建图像数据

        # 显示重建图像并更新界面信息
        if reconstructed_img is not None:
            self._display_image(self.recon_canvas_widget, reconstructed_img, f"重建图像 ({method})")
            self.label_reconstruction_time.setText(f"重建时间: {recon_time:.4f} 秒")                          # 显示耗时（4位小数）

            log_msg = f"图像重建完成 ({method})。时间: {recon_time:.4f} 秒"                                     # 构建日志消息
            # 计算并显示PSNR和SSIM（有原始图像时）
            if self.data_generator.original_image is not None:
                if not np.isnan(psnr_val):
                    self.label_psnr.setText(f"PSNR: {psnr_val:.2f} dB")
                else:
                    self.label_psnr.setText("PSNR: 计算错误")
                if not np.isnan(ssim_val):
                    self.label_ssim.setText(f"SSIM: {ssim_val:.4f}")
                else:
                    self.label_ssim.setText("SSIM: 计算错误")
                log_msg += f", PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}"
            # 无原始图像时显示N/A
            else:
                self.label_psnr.setText("PSNR: N/A (无原图)")
                self.label_ssim.setText("SSIM: N/A (无原图)")

            self._append_log(log_msg)  # 添加日志
            self.statusBar().showMessage("图像重建完成。", 3000)                                # 状态栏提示
        # 异常处理：重建结果为空
        else:
            self._on_reconstruction_error(f"{method} 重建返回空结果。")

        self._set_controls_enabled(True)                                                                     # 恢复控件交互状态

    def _on_reconstruction_error(self, error_message: str):
        """处理图像重建错误事件，显示错误信息并重置界面"""
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("%p%")
        QMessageBox.critical(self, "重建错误", error_message)                                                  # 弹出错误对话框
        self._append_log(f"重建错误: {error_message}")                                                         # 记录错误日志
        self.statusBar().showMessage("重建错误。", 3000)                                        # 状态栏提示
        self._display_image(self.recon_canvas_widget, None, "重建图像 - 错误")                  # 显示错误状态
        self._set_controls_enabled(True)                                                                     # 恢复控件交互状态

    def _view_speckles(self):
        """打开散斑可视化对话框，显示采样散斑和完整散斑矩阵"""
        if self.data_generator.full_rank_measurement_matrix is None:
            QMessageBox.warning(self, "查看散斑失败", "请先生成测量矩阵。")
            return

        current_matrix_gen_type = self.cb_matrix_type.currentText()

        # 创建散斑可视化对话框
        dialog = SpeckleVisualizationDialog(
            self,
            sampling_speckle=self.data_generator.measurement_matrix,                                       # 采样后的散斑矩阵（用于单个显示）
            full_speckle=self.data_generator.full_rank_measurement_matrix,                                 # 完整散斑矩阵（用于马赛克和统计）
            matrix_size=self.current_matrix_size,                                                          # 图像尺寸（用于散斑重塑）
            matrix_type_str=current_matrix_gen_type                                                        # 传递矩阵类型字符串
        )
        dialog.exec_()                                                                                     # 显示对话框

    def _parse_metric(self, label_widget: QLabel, metric_name: str) -> any:
        """Helper to parse metric values from labels, handling 'N/A' or errors."""
        text = label_widget.text()
        try:
            if "N/A" in text or "错误" in text:
                return "N/A"
            value_str = text.split(":")[1].strip().split(" ")[0]
            return float(value_str)
        except (IndexError, ValueError) as e:
            self._append_log(f"解析指标 {metric_name} ('{text}') 失败: {e}")
            return "N/A"

    def _save_data(self):
        """保存实验数据、参数和评估指标到指定目录"""
        save_dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not save_dir_path:
            return                                                                                         # 用户取消选择时直接返回

        # 收集实验参数（当前界面设置）
        params = {
            "image_size": self.current_matrix_size,
            "sampling_rate": self.slider_sampling_rate.value() / 100.0,
            "noise_type": self.cb_noise_type.currentText(),
            "noise_param": float(self.label_noise_param.text()) if self.label_noise_param.text() != "N/A" else 0.0,
            "matrix_generation_method": self.cb_matrix_type.currentText(),
            "reconstruction_algorithm": self.cb_reconstruction_method.currentText()
        }

        # 收集评估指标（处理N/A情况）
        metrics = {
            "psnr": self._parse_metric(self.label_psnr, "PSNR"),
            "ssim": self._parse_metric(self.label_ssim, "SSIM"),
            "recon_time": self._parse_metric(self.label_reconstruction_time, "重建时间")
        }

        # 保存选项对话框
        options_dialog = QDialog(self)
        options_dialog.setWindowTitle("选择保存选项")
        layout = QVBoxLayout(options_dialog)

        # 添加标题和复选框
        layout.addWidget(QLabel("选择要保存的数据类型:"))
        chk_matrix = QCheckBox("采样测量矩阵 (.npy)")
        chk_signals = QCheckBox("桶信号 (.npy)")
        chk_image = QCheckBox("原始图像 (.npy, .png)")
        chk_sampled_speckles_pdf = QCheckBox("采样散斑图 (PDF Mosaic)")
        chk_recon_image = QCheckBox("重建图像 (.npy, .png)")
        chk_hdf5 = QCheckBox("HDF5数据文件 (推荐, 包含数值数据)")

        chk_matrix.setChecked(True)
        chk_signals.setChecked(True)
        chk_image.setChecked(True)
        chk_sampled_speckles_pdf.setChecked(True)
        chk_recon_image.setChecked(self.reconstructed_image is not None)
        chk_hdf5.setChecked(True)

        for chk in [chk_matrix, chk_signals, chk_image, chk_sampled_speckles_pdf,chk_recon_image, chk_hdf5]:
            layout.addWidget(chk)

        # 根据数据存在性动态禁用不可用选项
        chk_matrix.setEnabled(self.data_generator.measurement_matrix is not None)
        chk_signals.setEnabled(self.data_generator.bucket_signals is not None)
        chk_image.setEnabled(self.data_generator.original_image is not None)
        chk_sampled_speckles_pdf.setEnabled(
            self.data_generator.measurement_matrix is not None)
        chk_recon_image.setEnabled(self.reconstructed_image is not None)
        can_save_to_hdf5 = any([
            self.data_generator.full_rank_measurement_matrix is not None,
            self.data_generator.measurement_matrix is not None,
            self.data_generator.bucket_signals is not None,
            self.data_generator.original_image is not None,
        ])
        chk_hdf5.setEnabled(can_save_to_hdf5)
        if not any([chk_matrix.isEnabled(), chk_signals.isEnabled(), chk_image.isEnabled(),
                    chk_sampled_speckles_pdf.isEnabled(), chk_hdf5.isEnabled()]):
            QMessageBox.warning(self, "无数据", "无数据可保存。")
            return

        # 添加确定和取消按钮
        btn_ok = QPushButton("确定")
        btn_cancel = QPushButton("取消")
        btn_ok.clicked.connect(options_dialog.accept)
        btn_cancel.clicked.connect(options_dialog.reject)
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

        # 显示对话框并处理用户选择
        if options_dialog.exec_() == QDialog.Accepted:
            self.statusBar().showMessage("正在保存数据...", 0)
            QApplication.processEvents()

            try:
                log_msg = self.data_generator.save_reconstruction_data(
                    save_dir_path,
                    params=params,
                    metrics=metrics,
                    options={
                        "measurement_matrix": chk_matrix.isChecked(),
                        "bucket_signals": chk_signals.isChecked(),
                        "original_image": chk_image.isChecked(),
                        "sampled_speckles_pdf": chk_sampled_speckles_pdf.isChecked(),
                        "diffusion_data": chk_hdf5.isChecked(),
                        "reconstructed_image": chk_recon_image.isChecked()
                    },
                recon_image = self.reconstructed_image
                )
                QMessageBox.information(self, "保存成功", f"数据保存到 '{save_dir_path}':\n{log_msg}")
                self._append_log(f"数据保存到 '{save_dir_path}'。\n{log_msg}")
                self.statusBar().showMessage("数据保存成功。", 3000)
            except Exception as e:
                QMessageBox.critical(self, "保存失败", str(e))
                self._append_log(f"保存失败: {str(e)}")
                self.statusBar().showMessage(f"保存失败: {str(e)}", 5000)
        else:
            self._append_log("数据保存已取消。")
            self.statusBar().showMessage("数据保存已取消。", 3000)

def main():
    app = QApplication(sys.argv)
    window = GhostImagingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()