import pandas as pd
import matplotlib.pyplot as plt

def box_plot(data_dict, ylim):
    # 여러 그룹의 데이터 생성
    key_list = [format(float(key), ".0e") for key in data_dict.keys()]
    data_list = data_dict.values
    # 여러 박스 플롯 생성
    plt.figure(figsize=(10, 8))
    plt.boxplot(data_list, whis=1.5)

    plt.ylim(ylim)
    plt.xticks(list(range(1, len(key_list)+1)), key_list, rotation=0)


    plt.show()