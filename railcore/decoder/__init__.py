"""
Decoder moduli
"""
from railcore.decoder.base import VideoDecoder
from railcore.decoder.gst_nvdec import GStreamerNVDECDecoder
from railcore.decoder.ffmpeg_cpu import FFMPEGCPUDecoder
from railcore.logging_setup import setup_logger

logger = setup_logger(__name__)

def create_decoder(source: str) -> VideoDecoder:
    """
    Video decoder yaratish (GStreamer yoki FFMPEG)
    
    Args:
        source: Video manba
    
    Returns:
        VideoDecoder: Decoder instance
    """
    # Avval GStreamer NVDEC'ni sinab ko'rish
    try:
        decoder = GStreamerNVDECDecoder(source)
        if decoder.is_opened():
            logger.info("GStreamer NVDEC ishlatilmoqda")
            return decoder
    except Exception as e:
        logger.warning(f"GStreamer NVDEC ishlamadi: {e}")
    
    # Fallback - FFMPEG CPU
    logger.info("FFMPEG CPU decoder ishlatilmoqda")
    return FFMPEGCPUDecoder(source)

__all__ = ['VideoDecoder', 'GStreamerNVDECDecoder', 'FFMPEGCPUDecoder', 'create_decoder']