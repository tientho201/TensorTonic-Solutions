import numpy as np 

def elu(x, alpha):
    """
    Optimized ELU activation function.
    """
    # Ép kiểu sang float để tránh lỗi khi mảng gốc là kiểu int
    x = np.asarray(x, dtype=np.float64)
    
    # Tạo mặt nạ cho các phần tử <= 0
    mask = (x <= 0)
    
    # Chỉ tính toán exp trên những phần tử thực sự cần thiết
    x[mask] = alpha * (np.exp(x[mask]) - 1)
    
    return x.tolist()