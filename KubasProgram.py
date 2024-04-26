import os
import numpy as np
import pandas as pd
import cv2
from collections import Counter
    #Class & Precision & Recall & F1-Score & Support \\
    #0          1           2       3           4
report_hue_59 = [
    ['astilbe', 0.26, 0.24, 0.25, 226],
    ['bellflower', 0.43, 0.13, 0.20, 253],
    ['black_eyed_susan', 0.25, 0.39, 0.30, 302],
    ['calendula', 0.30, 0.30, 0.30, 293],
    ['california_poppy', 0.23, 0.12, 0.16, 322],
    ['carnation', 0.32, 0.34, 0.33, 288],
    ['common_daisy', 0.31, 0.40, 0.35, 306],
    ['coreopsis', 0.23, 0.27, 0.25, 297],
    ['daffodil', 0.37, 0.35, 0.36, 321],
    ['dandelion', 0.25, 0.19, 0.22, 297],
    ['iris', 0.57, 0.59, 0.58, 321],
    ['magnolia', 0.33, 0.38, 0.35, 316],
    ['rose', 0.36, 0.22, 0.27, 293],
    ['sunflower', 0.33, 0.48, 0.39, 286],
    ['tulip', 0.30, 0.32, 0.31, 300],
    ['water_lily', 1.00, 0.96, 0.98, 301],
]
report_ycbcr_17 = [
    ['astilbe', 0.29, 0.28, 0.28, 226],
    ['bellflower', 0.24, 0.18, 0.21, 253],
    ['black_eyed_susan', 0.29, 0.53, 0.38, 302],
    ['calendula', 0.46, 0.28, 0.35, 293],
    ['california_poppy', 0.26, 0.20, 0.22, 322],
    ['carnation', 0.33, 0.33, 0.33, 288],
    ['common_daisy', 0.27, 0.43, 0.34, 306],
    ['coreopsis', 0.30, 0.25, 0.27, 297],
    ['daffodil', 0.37, 0.31, 0.34, 321],
    ['dandelion', 0.27, 0.18, 0.22, 297],
    ['iris', 0.56, 0.52, 0.54, 321],
    ['magnolia', 0.36, 0.47, 0.41, 316],
    ['rose', 0.34, 0.26, 0.30, 293],
    ['sunflower', 0.42, 0.47, 0.44, 286],
    ['tulip', 0.38, 0.38, 0.38, 300],
    ['water_lily', 1.00, 0.99, 0.99, 301],
]
def calculate_accuracy(flower, report_table):
    accuracy=0
    for r in range(len(report_table)):
        if report_table[r][0] == flower:
            accuracy = "{:.4f}".format(((report_table[r][1] * report_table[r][4])+(report_table[r][2] * report_table[r][4]))/(2 * report_table[r][4]))
    return accuracy
# example
# acc = calculate_accuracy('water lily', report_hue_59)
class KubasProgram_hue:

    def __init__(self) -> None:
        print("Kubas program hue is working")
    def find_flower(self, image_path,histogram_path):
        
        # Convert image to an array of 16 for HSB hue histogram values
        def get_hue_histogram(file_path):
            img = cv2.imread(file_path)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist_hue, _ = np.histogram(img_hsv[:,:,0], bins=16, range=(0, 180))
            return hist_hue
        

        file_path = histogram_path
        df = pd.read_csv(file_path)
        X = df.iloc[:, 2:]  # Assuming columns 2 to 17 are your feature vectors
        Spiecies = df['Species']
        X = X.to_numpy()
        Spiecies = Spiecies.to_numpy()

        # Path to one given test JPEG file
        Test_photo = get_hue_histogram(image_path)
      

        # euklidian_distance = math.sqrt((X[0][0]-X[1][0])**2 + (X[0][1]-X[1][1])**2 + (X[0][2]-X[1][2])**2 + (X[0][3]-X[1][3])**2 ... )
        # euklidian_distance = np.sqrt(np.sum((X[0] - X[1])**2))
        # ^^^ similar way to calculate euklidian distance, I use 2nd shorter version 
        tab_distance = np.empty((len(X)))
        for i in range(0,len(X)):
            tab_distance[i] = np.sqrt(np.sum((Test_photo - X[i])**2))

        tab_distance_and_flower = np.column_stack((Spiecies,tab_distance))
        # [['astilbe' 29436.951268771023]
        #  ['tulip' 25603.452970253835]     <- tab looks like this
        #  ['tulip' 32489.025901063884]]  

        print(tab_distance_and_flower[np.argmin(tab_distance),0]) # main output, f.ex. 'tulip'       
        
        
        # Sortowanie tabeli według drugiej kolumny (indeks 1)
        sorted_table = tab_distance_and_flower[np.argsort(tab_distance_and_flower[:, 1])]

        # Wybieranie 59 pierwszych wierszy po posortowaniu
        selected_rows = sorted_table[:59] # K=59

        # Liczenie najczęściej występującego kwiata
        most_common_flower = Counter(selected_rows[:, 0]).most_common(1)[0][0]

    
        print("HUE Most common flower:")
        print(most_common_flower)

        #flower = tab_distance_and_flower[np.argmin(tab_distance),0]
        return (most_common_flower, str(calculate_accuracy(most_common_flower,report_hue_59)))
    
 
class KubasProgram_YCbCr:

    def __init__(self) -> None:
        print("Kuba program YCbCr is working")

    def find_flower(self, image_path,histogram_path):
        
        # Convert image to an array of 16 for HSB hue histogram values
     
        def get_ycbcr_histogram(file_path):
            img = cv2.imread(file_path)
            img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            
            hist_cb, _ = np.histogram(img_ycbcr[:,:,1], bins=16, range=(0, 180))
            hist_cr, _ = np.histogram(img_ycbcr[:,:,2], bins=16, range=(0, 180))
            
            return np.concatenate((hist_cb, hist_cr), axis=0)
        # file_path = os.path.abspath("C:\\Users\\jakub\\OneDrive\\Pulpit\\Kuba\\programing\\python\\test_histogram\\wynikhue.csv")
        # file_path = os.path.abspath("database\histogramy_hue.csv")
        file_path = histogram_path


        #print(file_path)
        df = pd.read_csv(file_path)
        X = df.iloc[:, 2:]  # Assuming columns 2 to 65 are your feature vectors
        Spiecies = df['Species']
        X = X.to_numpy()
        Spiecies = Spiecies.to_numpy()

        # Path to one given test JPEG file
        # path_to_test_image = "C:\\Users\\jakub\\OneDrive\\Pulpit\\Kuba\\programing\\python\\database\\archive\\flowers\\astilbe\\210983713_f4bb1020d9_c.jpg"
        # path_to_test_image = "myPhotos\\me.jpg"
        Test_photo = get_ycbcr_histogram(image_path)
        # Hudecs_photo = [128,13368,3062,863,2032,4204,290,5330,2335,5333,84,44,235,1215,2200,243]
        # Hudecs_photo = np.array(Hudecs_photo)
        

        # print(X[0,:])
        # print(X[1,1])

        # euklidian_distance = math.sqrt((X[0][0]-X[1][0])**2 + (X[0][1]-X[1][1])**2 + (X[0][2]-X[1][2])**2 + (X[0][3]-X[1][3])**2 ... )
        # euklidian_distance = np.sqrt(np.sum((X[0] - X[1])**2))
        # ^^^ similar way to calculate euklidian distance, I use 2nd shorter version 
        tab_distance = np.empty((len(X)))
        for i in range(0,len(X)):
            tab_distance[i] = np.sqrt(np.sum((Test_photo - X[i])**2))

        tab_distance_and_flower = np.column_stack((Spiecies,tab_distance))
        # [['astilbe' 29436.951268771023]
        #  ['tulip' 25603.452970253835]     <- tab looks like this
        #  ['tulip' 32489.025901063884]]  

        print(tab_distance_and_flower[np.argmin(tab_distance),0]) # main output, f.ex. 'tulip'       
        
        #TODO znalezc 59 najbliższych kwiatków i podliczyć jakich jest najwięcej 
        # Sortowanie tabeli według drugiej kolumny (indeks 1)
        sorted_table = tab_distance_and_flower[np.argsort(tab_distance_and_flower[:, 1])]

        # Wybieranie K pierwszych wierszy po posortowaniu
        selected_rows = sorted_table[:17]

        # Liczenie najczęściej występującego kwiata
        most_common_flower = Counter(selected_rows[:, 0]).most_common(1)[0][0]

        # Wyświetlanie wyników
        # print("Posortowana tabela:")
        # print(sorted_table)
        # print("\nWybrane dwa wiersze:")
        # print(selected_rows)
        print("YCBCR Najczęściej występujący kwiat:")
        print(most_common_flower)

        #flower = tab_distance_and_flower[np.argmin(tab_distance),0]
        return (most_common_flower, str(calculate_accuracy(most_common_flower,report_ycbcr_17)))
        # return (most_common_flower, str(1))

