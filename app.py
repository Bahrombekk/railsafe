"""
RailSafe - Multi-Camera Polygon Monitoring System
Asosiy kirish nuqtasi
"""
import sys
import traceback
from railcore.system import MultiCameraSystem
from railcore.logging_setup import setup_logger

logger = setup_logger(__name__)

def main():
    try:
        logger.info("RailSafe tizimi ishga tushirilmoqda...")
        system = MultiCameraSystem('config/config.yaml')
        system.start()
    except KeyboardInterrupt:
        logger.info("Dastur foydalanuvchi tomonidan to'xtatildi")
    except Exception as e:
        logger.error(f"Fatal xato: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()