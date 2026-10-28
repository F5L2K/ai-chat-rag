"""OCR：对扫描页做文字识别。

软依赖设计：rapidocr-onnxruntime / pymupdf 没装、或模型加载失败时，
本模块只返回空字符串，不抛异常——PDF 主流程退回原来的 needs_ocr 标记，
绝不因为 OCR 模块挂了就把整个上传链路搞崩（和 rerank.py / bm25.py 同风格）。

RapidOCR 基于 PaddleOCR 模型但跑在 onnxruntime 上，不需要装 paddlepaddle，
中文识别效果够用；首次调用会自动下模型（几十 MB），之后走本地缓存。
"""

from app.core.config import OCR_DPI, OCR_ENABLED
from app.core.logging import logger

_engine = None
_init_failed = False


def _get_engine():
    global _engine, _init_failed
    if _init_failed:
        return None
    if _engine is not None:
        return _engine
    if not OCR_ENABLED:
        _init_failed = True
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
        logger.info("OCR 引擎初始化成功")
        return _engine
    except Exception as e:
        _init_failed = True
        logger.warning(
            "OCR 引擎初始化失败（未装 rapidocr-onnxruntime？），"
            "扫描页将保持 needs_ocr 标记：%s",
            e,
        )
        return None


def render_pdf_page(raw_pdf: bytes, page_no: int):
    """把 PDF 第 page_no 页（1-based）渲染成 RGB numpy 数组，失败返回 None。"""
    if not OCR_ENABLED:
        return None
    try:
        import fitz  # PyMuPDF
        import numpy as np
    except ImportError as e:
        logger.warning("未装 pymupdf，无法渲染扫描页：%s", e)
        return None

    try:
        doc = fitz.open(stream=raw_pdf, filetype="pdf")
        try:
            page = doc[page_no - 1]
            zoom = OCR_DPI / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            # RGBA / 灰度统一转 RGB，RapidOCR 只认三通道
            if pix.n == 4:
                img = img[:, :, :3]
            elif pix.n == 1:
                img = img.repeat(3, axis=-1)
            return img
        finally:
            doc.close()
    except Exception as e:
        logger.warning("PDF 第 %d 页渲染失败：%s", page_no, e)
        return None


def recognize(image) -> str:
    """对一张图片做 OCR，返回拼接后的纯文本；失败返回空串。"""
    engine = _get_engine()
    if engine is None or image is None:
        return ""
    try:
        result, _ = engine(image)
        if not result:
            return ""
        # result 形如 [[box, text, score], ...]
        return "\n".join(line[1] for line in result if line and len(line) >= 2)
    except Exception as e:
        logger.warning("OCR 识别失败：%s", e)
        return ""
