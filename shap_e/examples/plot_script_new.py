import matplotlib.pyplot as plt
import json

def main():
    # file_list=['base40M-textvec-out.json','base40M-out.json','base300M-out.json','base1B-out.json']
    file_list=[]
    label_dict={}
    for w1 in ['text300M']: # ['img','text']
        for w2 in ['decoder','transmitter']:
            for w3 in ['nerf','stf']:
                temp_str1=w1+"-"+w2+"-"+w3+".json"
                file_list.append(temp_str1)
                temp_str2=w1+", "+w2+", "+w3
                label_dict[temp_str1]=temp_str2
    plt.figure(figsize=(10,6))
    # print(label_dict)
    for file_name in file_list:
        # Read the contents of the file
        with open(file_name, "r") as file:
            json_str = file.read()

        # Deserialize the JSON string to a dictionary
        temp_dict = json.loads(json_str)
        data=temp_dict['mem']
        
        timestamps = [t for t, _ in data]
        measured_val = [m for _, m in data]
        plt.plot(timestamps, measured_val,label=label_dict[file_name],linewidth=3)
    
    dot_size=60
    # # data points for img diffusion model start
    # plt.scatter(215.6581175327301, 4773,c='black',s=dot_size,zorder=10,marker='s') # dec nerf
    # plt.scatter(151.8005805015564, 4773,c='black',s=dot_size,zorder=10,marker='s') # dec stf
    # plt.scatter(97.3896234035492, 5611,c='black',s=dot_size,zorder=10,marker='s') # trans nerf
    # plt.scatter(111.4258680343628, 5611,c='black',s=dot_size,zorder=10,marker='s') # trans stf
    
    # # data points for img decoder model start
    # plt.scatter(275.10576915740967, 5569,c='black',s=dot_size,zorder=10,marker='^') # dec nerf
    # plt.scatter(211.07834100723267, 5569,c='black',s=dot_size,zorder=10,marker='^') # dec stf
    # plt.scatter(157.81653428077698, 6409,c='black',s=dot_size,zorder=10,marker='^') # trans nerf
    # plt.scatter(170.2492916584015, 6409,c='black',s=dot_size,zorder=10,marker='^') # trans stf

    # data points for text diffusion model start
    plt.scatter(89.03706073760986, 4781,c='black',s=dot_size,zorder=10,marker='s') # dec nerf
    plt.scatter(163.02393865585327, 4781,c='black',s=dot_size,zorder=10,marker='s') # dec stf
    plt.scatter(155.35383653640747, 5609,c='black',s=dot_size,zorder=10,marker='s') # trans nerf
    plt.scatter(72.77255082130432, 5609,c='black',s=dot_size,zorder=10,marker='s') # trans stf
    
    # data points for text decoder model start
    plt.scatter(118.16228699684143, 5429,c='black',s=dot_size,zorder=10,marker='^') # dec nerf
    plt.scatter(189.1307864189148, 5429,c='black',s=dot_size,zorder=10,marker='^') # dec stf
    plt.scatter(180.62367486953735, 6129,c='black',s=dot_size,zorder=10,marker='^') # trans nerf
    plt.scatter(98.05828189849854, 6259,c='black',s=dot_size,zorder=10,marker='^') # trans stf

    plt.scatter(241.634019613266, 5841,c='black',s=20,zorder=10,marker='^')
    
    # plt.xlim(0,300)
    plt.xlim(0,100)
    plt.ylim(0,13000)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=12,loc='lower right')
    plt.grid(True)
    # plt.savefig('./shap-e-mem-plot.png')
    # plt.savefig('./shap-e-mem-plot-img.png')
    plt.savefig('./shap-e-mem-plot-text.png')
    plt.show()

if __name__ == "__main__":
    main()