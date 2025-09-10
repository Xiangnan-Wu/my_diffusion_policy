from typing import Tuple
import math
import cv2
import numpy as np

def draw_reticle(img, u, v, label_color):
    """
    Draws a reticle (cross-hair) on the image at the given position on top of
    the original image.
    @param img (In/Out) uint8 3 channel image
    @param u X coordinate (width)
    @param v Y coordinate (height)
    @param label_color tuple of 3 ints for RGB color used for drawing.
    """
    # Cast to int.
    u = int(u)
    v = int(v)

    white = (255, 255, 255)
    cv2.circle(img, (u, v), 10, label_color, 1)
    cv2.circle(img, (u, v), 11, white, 1)
    cv2.circle(img, (u, v), 12, label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 3), white, 1)
    cv2.line(img, (u + 1, v), (u + 3, v), white, 1)
    cv2.line(img, (u, v - 1), (u, v - 3), white, 1)
    cv2.line(img, (u - 1, v), (u - 3, v), white, 1)


def draw_text(
    img,
    *,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline.
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]


def get_image_transform(
        input_res: Tuple[int,int]=(1280,720),   # 输入图像分辨率 (宽度, 高度)
        output_res: Tuple[int,int]=(640,480),   # 输出图像分辨率 (宽度, 高度)
        bgr_to_rgb: bool=False):                # 是否将BGR格式转换为RGB格式

    # 解析输入和输出分辨率
    iw, ih = input_res    # 输入图像宽度(input width)和高度(input height)
    ow, oh = output_res   # 输出图像宽度(output width)和高度(output height)
    
    # 初始化调整后的尺寸变量
    rw, rh = None, None   # 调整后的宽度(resized width)和高度(resized height)
    
    # 默认使用INTER_AREA插值方法（适合缩小图像）
    interp_method = cv2.INTER_AREA

    # 计算输入和输出的宽高比，决定如何调整尺寸以保持纵横比
    if (iw/ih) >= (ow/oh):
        # 情况1: 输入图像相对更宽（或宽高比相等）
        # 以目标高度为准，按比例计算宽度
        rh = oh                           # 调整后高度 = 目标高度
        rw = math.ceil(rh / ih * iw)      # 按比例计算调整后宽度，向上取整
        
        # 如果是放大操作，使用线性插值（更适合放大）
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        # 情况2: 输入图像相对更高
        # 以目标宽度为准，按比例计算高度
        rw = ow                           # 调整后宽度 = 目标宽度
        rh = math.ceil(rw / iw * ih)      # 按比例计算调整后高度，向上取整
        
        # 如果是放大操作，使用线性插值
        if ow > iw:
            interp_method = cv2.INTER_LINEAR
    
    # 计算裁剪区域：将调整后的图像居中裁剪到目标尺寸
    # 宽度方向的裁剪
    w_slice_start = (rw - ow) // 2       # 宽度裁剪起始位置（居中）
    w_slice = slice(w_slice_start, w_slice_start + ow)  # 宽度裁剪切片
    
    # 高度方向的裁剪
    h_slice_start = (rh - oh) // 2       # 高度裁剪起始位置（居中）
    h_slice = slice(h_slice_start, h_slice_start + oh)  # 高度裁剪切片
    
    # 颜色通道处理
    c_slice = slice(None)                # 默认保持原有颜色通道顺序
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)  # 如果需要BGR转RGB，反转颜色通道顺序

    def transform(img: np.ndarray):
        """
        实际执行图像变换的内部函数
        
        参数:
            img: 输入图像，形状应为 (高度, 宽度, 3)
            
        返回:
            变换后的图像，形状为 (目标高度, 目标宽度, 3)
        """
        # 验证输入图像尺寸是否符合预期
        assert img.shape == ((ih,iw,3)), f"期望图像形状 {(ih,iw,3)}，实际形状 {img.shape}"
        
        # 步骤1: 调整图像大小（保持纵横比）
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        
        # 步骤2: 居中裁剪到目标尺寸，同时处理颜色通道
        img = img[h_slice, w_slice, c_slice]
        
        return img
    
    # 返回配置好的变换函数
    return transform

def optimal_row_cols(
        n_cameras,
        in_wh_ratio,
        max_resolution=(1920, 1080)
    ):
    out_w, out_h = max_resolution
    out_wh_ratio = out_w / out_h
    
    n_rows = np.arange(n_cameras,dtype=np.int64) + 1
    n_cols = np.ceil(n_cameras / n_rows).astype(np.int64)
    cat_wh_ratio = in_wh_ratio * (n_cols / n_rows)
    ratio_diff = np.abs(out_wh_ratio - cat_wh_ratio)
    best_idx = np.argmin(ratio_diff)
    best_n_row = n_rows[best_idx]
    best_n_col = n_cols[best_idx]
    best_cat_wh_ratio = cat_wh_ratio[best_idx]

    rw, rh = None, None
    if best_cat_wh_ratio >= out_wh_ratio:
        # cat is wider
        rw = math.floor(out_w / best_n_col)
        rh = math.floor(rw / in_wh_ratio)
    else:
        rh = math.floor(out_h / best_n_row)
        rw = math.floor(rh * in_wh_ratio)
    
    # crop_resolution = (rw, rh)
    return rw, rh, best_n_col, best_n_row
