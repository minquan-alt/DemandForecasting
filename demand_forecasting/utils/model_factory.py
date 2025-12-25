from models._models import (
    SimpleAverageNet,
    ExponentialSmoothingNet,
    SimpleMovingAverageNet,
)


def get_model(name):
    """Factory function để tạo baseline model"""
    name = name.lower()
    if name == "odoo_basic":
        return SimpleAverageNet()
    elif name == "odoo_sota":
        return ExponentialSmoothingNet()
    elif name == "sap_basic":
        return SimpleMovingAverageNet(window_size=3)
    elif name == "arima":
        return "arima"  # Special marker
    else:
        raise ValueError(f"❌ Unknown baseline model: {name}. Use: odoo_basic, sap_basic, arima")
