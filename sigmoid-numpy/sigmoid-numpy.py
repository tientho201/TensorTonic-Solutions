import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function using NumPy.
    """
    # Chuyển đổi x thành mảng NumPy (nếu x đang là list) để đảm bảo tính toán mảng
    x = np.asarray(x)
    
    # Tính toán vectorized: nhanh hơn và ngắn gọn hơn
    return 1 / (1 + np.exp(-x))