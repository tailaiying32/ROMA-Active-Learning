
import numpy as np

def inside_box(q, box):
    q1, q2 = q[..., 0], q[..., 1]
    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']
    return (q1 >= q1_min) & (q1 <= q1_max) & (q2 >= q2_min) & (q2 <= q2_max)

def box_margin_original(q, box):
    q1, q2 = q[..., 0], q[..., 1]
    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']

    d_q1_min = q1 - q1_min
    d_q1_max = q1_max - q1
    d_q2_min = q2 - q2_min
    d_q2_max = q2_max - q2

    margin = np.minimum.reduce([d_q1_min, d_q1_max, d_q2_min, d_q2_max])
    
    return margin

def box_margin_user_change(q, box):
    q1, q2 = q[..., 0], q[..., 1]
    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']

    d_q1_min = q1 - q1_min
    d_q1_max = q1_max - q1
    d_q2_min = q2 - q2_min
    d_q2_max = q2_max - q2

    margin = np.minimum.reduce([d_q1_min, d_q1_max, d_q2_min, d_q2_max])
    
    # User change
    is_inside = inside_box(q, box)
    margin = np.where(is_inside, margin, -margin)
    return margin

def main():
    box = {'q1_range': (-10, 10), 'q2_range': (-10, 10)}
    
    # Test points
    points = np.array([
        [0, 0],      # Center (Inside) -> Margin should be 10
        [5, 5],      # Inside -> Margin should be 5
        [10, 0],     # Edge -> Margin should be 0
        [12, 0],     # Outside Right -> Margin should be -2 (original)
        [-15, 0],    # Outside Left -> Margin should be -5 (original)
        [15, 15]     # Outside Corner -> Margin should be -5 (original) (min dist to either edge)
    ])
    
    print(f"Box: {box}")
    print(f"{'Point':<15} | {'Original':<10} | {'User Change':<10} | {'Status'}")
    print("-" * 50)
    
    orig = box_margin_original(points, box)
    mod = box_margin_user_change(points, box)
    
    for p, o, m in zip(points, orig, mod):
        status = "Inside" if inside_box(np.array([p]), box)[0] else "Outside"
        print(f"{str(p):<15} | {o:<10.2f} | {m:<10.2f} | {status}")

if __name__ == "__main__":
    main()
