import cv2
import math 
import numpy as np
from collections import deque
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

'''
NOTE: for loop is neeeded for socring
NOTE: seperate the scoring logic function and the drawing function
FIXME: how to seperate scoring logic function and the draw_traj function
'''

def plot_data(arr, index_range, filename, score_frames):
    """
    Plots the given array 'arr' against the indices specified by 'index_range'.
    
    Parameters:
    - arr (list or ndarray): The data points to be plotted.
    - index_range (range): The range of indices to use as the x-axis.
    """
    
    # Check if the length of the array matches the length of the index range
    if len(arr) != len(index_range):
        raise ValueError("The length of the data array and the index range must match.")
    
    plt.figure(figsize=(8, 5))
    plt.plot(index_range, arr, linestyle='-', color='b')  # 折線圖，沒有圓形點
    plt.fill_between(index_range, arr, color='pink', alpha=0.5)  # 填充半透明粉紅色
    plt.title('Data vs. Index')
    plt.yscale('log')
    plt.xlabel('Index')
    plt.ylabel('Data Value')
    plt.grid(True)

    for frame in score_frames:
            plt.axvline(x=frame, color='red', linestyle='--', linewidth=1)

    plt.savefig(filename)
    
# def load_coordinates_from_file(filename):
#     """
#     Load and convert coordinate data from a text file into a NumPy array.
    
#     Args:
#         filename (str): The path to the text file containing the coordinates.
        
#     Returns:
#         numpy.ndarray: A 2D NumPy array where each row is a pair of integers.
#     """
#     # Read the entire content of the file
#     with open(filename, 'r') as file:
#         data_string = file.read().strip()

#     # Remove unwanted characters and split by commas
#     clean_data = data_string.replace('(', '').replace(')', '').split(',')

#     # Convert to a list of integers
#     integer_data = [int(num.strip()) for num in clean_data]

#     # Reshape the flat list into a 2D array of pairs
#     numpy_array = np.array(integer_data).reshape(-1, 2)
    
#     return numpy_array

#NOTE: new version
def load_coordinates_from_file(filename):
    """
    Load and convert coordinate data from a text file into a NumPy array in clockwise order.

    Args:
        filename (str): The path to the text file containing the coordinates.

    Returns:
        numpy.ndarray: A 2D NumPy array where each row is a pair of integers representing a coordinate in clockwise order.
    """
    # Read the entire content of the file
    with open(filename, 'r') as file:
        data_string = file.read().strip()

    # Split the data by new lines to separate each coordinate
    lines = data_string.splitlines()

    # Split each coordinate by ';' and convert to integers
    coordinates = [tuple(map(int, line.split(';'))) for line in lines]

    # Convert the list of tuples into a 2D NumPy array
    numpy_array = np.array(coordinates)

    # Reorder coordinates to be in clockwise order: left-top, right-top, right-bottom, left-bottom
    # Assuming the input order is: left-top, left-bottom, right-bottom, right-top
    clockwise_order = [0, 3, 2, 1]
    numpy_array = numpy_array[clockwise_order]
    
    return numpy_array

def get_half_court_coordinates(filename):
    # Load court corner coordinates
    court_corners = load_coordinates_from_file(filename).astype(np.float32)

    # Ideal court coordinates (assuming width is 610 and height is 1340)
    ideal_court = np.array([[0, 0], [610, 0], [610, 1340], [0, 1340]], dtype=np.float32)

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(court_corners, ideal_court)

    # Ideal half-court coordinates
    upper_half = np.array([[0, 0], [610, 0], [610, 670], [0, 670]], dtype=np.float32)
    lower_half = np.array([[0, 670], [610, 670], [610, 1340], [0, 1340]], dtype=np.float32)

    # Transform the half-court coordinates back to the original image coordinates
    upper_half_transformed = cv2.perspectiveTransform(upper_half[np.newaxis], np.linalg.inv(matrix))[0].astype(np.int32)
    lower_half_transformed = cv2.perspectiveTransform(lower_half[np.newaxis], np.linalg.inv(matrix))[0].astype(np.int32)

    # Return the coordinates in clockwise order
    return upper_half_transformed, lower_half_transformed

def compute_distance(x_pred, y_pred, i, bias):
    """Compute the distance between consecutive points if previous point exists."""
    if i > 0 and x_pred[i-bias] is not None and y_pred[i-bias] is not None:
        
        if x_pred[i]==y_pred[i]==0:
            return 0
        elif x_pred[i-bias]==y_pred[i-bias]==0:
            return 0
        # Ensure the current points are also valid
        if x_pred[i] is not None and y_pred[i] is not None:
            return math.sqrt((x_pred[i] - x_pred[i-bias]) ** 2 + (y_pred[i] - y_pred[i-bias]) ** 2) 
    return 0 

def is_point_in_polygon(polygon, point):
    count = 0
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        if ((polygon[i][1] > point[1]) != (polygon[j][1] > point[1])):
            if point[0] < (polygon[j][0] - polygon[i][0]) * (point[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]:
                count += 1
    return count % 2 == 1

def scoring(frame_list, x_pred, y_pred, fps=0):
    
    score = 0
    score_1 = 0
    score_2 = 0
    you_can_check_zero_speed = False
    done = False
    speed_arr = []
    acceleration_arr = []
    scoring_frame_list = []
    sequential_high_speed_count = 0
    sequential_low_speed_count = 0
    sequential_zero_speed_count = 0
    score_flag = -1 # -1 不是計分的時候 0 沒得分 1 有得分
    score_flag_count = 0 #用於連續output球落地位置和匡線
    #球的落點
    end_x = 0
    end_y = 0
    polygon = load_coordinates_from_file("court.txt")
    upper_half, lower_half = get_half_court_coordinates("court.txt")
    # low_speed = 400
    low_speed = 350
    zero_speed = 150

    scoring_events = [] #save the scoring frame list
    sec_count = 0 #count the second after scoring
    allow_score = False
    sec = fps * 8

    for i, frame in tqdm(enumerate(frame_list), desc="Scoring"):
        
        # 固定每隔bias個幀數顯示在畫面上
        bias = int(fps/fps)
        if (i%bias == 0):
            distance = compute_distance(x_pred, y_pred, i, bias)
            speed = int(distance * (fps/bias)) # distance / (1/(fps/bias))
            
        speed_arr.append(speed)

        # if i < 1500:
        #     print("%d: %d" %(i, speed_arr[i]))
        
        if not allow_score:
            sec_count += 1
            if sec_count >= sec:
                allow_score = True
                sec_count = 0

        if i > 0:
            acceleration = speed_arr[i] - speed_arr[i-1]
            acceleration_arr.append(acceleration)
        else:
            acceleration_arr.append(0)  # 第一幀的加速度設為0
        
        if(speed<low_speed and speed!=0 and you_can_check_zero_speed==False and done==False and allow_score):
            sequential_low_speed_count+=1
            if (sequential_low_speed_count>=10):
                you_can_check_zero_speed = True
            else:
                you_can_check_zero_speed = False      

        if(you_can_check_zero_speed == True): #NOTE: only check if ball satisfy sequential_low_speed_count>=10
            if(speed<zero_speed): #NOTE: if ball is stationary. 150 for test, org should be speed==0
                sequential_zero_speed_count += 1
            else:
                sequential_zero_speed_count = 0 #reset
                
            if(sequential_zero_speed_count >= 10): #NOTE: if sequential_zero_speed_count >= 10 -> Scoring
                
                #NOTE: reset
                allow_score = False
                you_can_check_zero_speed = False
                done = True
                sequential_low_speed_count = 0
                sequential_zero_speed_count = 0 
                
                j = i
                while j >= 0 and speed_arr[j] <= low_speed:
                    j -= 1
                
                # 從該frame開始檢查後續10個幀，找出局部最小值
                start_frame = j - 1 if j > 0 else j #NOTE: 從 low_speed 的前一個frame開始找
                end_frame = min(j + 10, len(acceleration_arr))
                local_min_frame = start_frame
                local_min_speed = acceleration_arr[start_frame]
                for k in range(start_frame, end_frame):
                    if acceleration_arr[k] < local_min_speed:
                        local_min_speed = acceleration_arr[k]
                        local_min_frame = k
                
                scoring_events.append(local_min_frame)
                
                # 繪製在local_min_frame的位置的圖像上
                local_min_image = scoring_frame_list[local_min_frame]  # 確保我們在正確的幀圖像上進行繪製
                
                if(is_point_in_polygon(polygon, (x_pred[local_min_frame], y_pred[local_min_frame]))):
                    if (y_pred[local_min_frame] > 480):  # down side loss score
                        score_1 += 1
                        score_flag = 1
                        cv2.polylines(local_min_image, [lower_half], isClosed=True, color=(0, 255, 0), thickness=7)  # Green polygon
                    else:  # up side loss score
                        score_2 += 1
                        score_flag = 2
                        cv2.polylines(local_min_image, [upper_half], isClosed=True, color=(0, 255, 0), thickness=7)  # Green polygon
                    cv2.circle(local_min_image, (x_pred[local_min_frame], y_pred[local_min_frame]), 10, (0, 0, 255), 4)
                else:
                    if (y_pred[local_min_frame] > 480):
                        score_2 += 1
                        score_flag = 3
                        cv2.polylines(local_min_image, [lower_half], isClosed=True, color=(0, 0, 255), thickness=7)  # Red polygon
                    else:
                        score_1 += 1
                        score_flag = 4
                        cv2.polylines(local_min_image, [upper_half], isClosed=True, color=(0, 0, 255), thickness=7)  # Red polygon
                    cv2.circle(local_min_image, (x_pred[local_min_frame], y_pred[local_min_frame]), 10, (0, 0, 255), 4)
                
        #NOTE: keep drawing the ball's position and the court line for 10 frames:   score_flag = -1 (dont draw) 0 (draw red/out) 1 (draw green/in)
        if(score_flag != -1):
            if(score_flag_count<10):
                score_flag_count += 1
                if score_flag==1:
                    cv2.circle(scoring_frame_list[local_min_frame+score_flag_count], (x_pred[local_min_frame], y_pred[local_min_frame]), 10, (0, 0, 255), 4)
                    cv2.polylines(scoring_frame_list[local_min_frame+score_flag_count], [lower_half], isClosed=True, color=(0, 255, 0), thickness=7)  # Green polygon
                elif score_flag==2:
                    cv2.circle(scoring_frame_list[local_min_frame+score_flag_count], (x_pred[local_min_frame], y_pred[local_min_frame]), 10, (0, 0, 255), 4)
                    cv2.polylines(scoring_frame_list[local_min_frame+score_flag_count], [upper_half], isClosed=True, color=(0, 255, 0), thickness=7)
                elif score_flag==3:
                    cv2.circle(scoring_frame_list[local_min_frame+score_flag_count], (x_pred[local_min_frame], y_pred[local_min_frame]), 10, (0, 0, 255), 4)
                    cv2.polylines(scoring_frame_list[local_min_frame+score_flag_count], [lower_half], isClosed=True, color=(0, 0, 255), thickness=7)
                elif score_flag==4:
                    cv2.circle(scoring_frame_list[local_min_frame+score_flag_count], (x_pred[local_min_frame], y_pred[local_min_frame]), 10, (0, 0, 255), 4)
                    cv2.polylines(scoring_frame_list[local_min_frame+score_flag_count], [upper_half], isClosed=True, color=(0, 0, 255), thickness=7) 
                    
            else:
                score_flag_count = 0
                score_flag = -1   
                
        if(speed>=low_speed):
            done = False
            you_can_check_zero_speed = False   
            sequential_low_speed_count = 0
            sequential_zero_speed_count = 0 

        # cv2.putText(frame, f'Frame: {i}', (10, 30), 
        #             cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
        """
        cv2.putText(frame, f'Score: {score}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        """
        # cv2.putText(frame, f'sequential_low_speed_count: {sequential_low_speed_count}', (10, 60), 
        #             cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, f'sequential_zero_speed_count: {sequential_zero_speed_count}', (10, 90), 
        #             cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (10, 20), (410, 120), (255, 255, 255), -1)
        cv2.putText(frame, f'Player1 (white): {score_2}', (30, 60), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f'Player2 (blue): {score_1}', (30, 95), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)

        scoring_frame_list.append(frame)
        # plot_data(inner_product_arr, list(range(len(frame_list))), "inner_product.png")
    plot_data(speed_arr, list(range(len(frame_list))), "./prediction/speed_Canada.png", score_frames=scoring_events)
    np.savetxt("./prediction/speed_Canada.csv", speed_arr, delimiter=",", fmt='%d')

    return scoring_frame_list
