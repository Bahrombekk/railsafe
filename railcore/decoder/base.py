"""
Decoder base interface
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class VideoDecoder(ABC):
    """Video decoder interface"""
    
    @abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Frame o'qish
        
        Returns:
            Tuple[bool, np.ndarray]: (success, frame)
        """
        pass
    
    @abstractmethod
    def reopen(self) -> bool:
        """
        Decoderni qayta ochish (reconnect)
        
        Returns:
            bool: Muvaffaqiyatli ochilsa True
        """
        pass
    
    @abstractmethod
    def release(self):
        """Decoderni yopish"""
        pass
    
    @abstractmethod
    def get_properties(self) -> dict:
        """
        Decoder xususiyatlarini olish
        
        Returns:
            dict: width, height, fps
        """
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """
        Decoder ochiq ekanligini tekshirish
        
        Returns:
            bool: Ochiq bo'lsa True
        """
        pass