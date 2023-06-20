import matplotlib.pyplot as plt
import json

def main():
    # file_list=['base40M-textvec-out.json','base40M-out.json','base300M-out.json','base1B-out.json']
    file_list=[]
    label_dict={}
    for w1 in ['image300M']: # ['img','text']
        for w2 in ['transmitter','decoder']: # 'decoder',
            for w3 in ['nerf','stf']: # ,'stf'
                temp_str1=w1+"-"+w2+"-"+w3+".json"
                file_list.append(temp_str1)
                if w1=='text300M':
                    w10='300M, text-only'
                else:
                    w10='300M'
                if w2=='decoder':
                    w20='Decoder 2'
                else:
                    w20='Decoder 1'
                if w3=='nerf':
                    w30='NeRF'
                else:
                    w30='STF'
                # temp_str2=w10+", "+w20+"("+w30+")"
                temp_str2=w20+" ("+w30+")"
                label_dict[temp_str1]=temp_str2
    plt.figure(figsize=(10,6))
    # print(label_dict)
    for file_name in file_list:
        # Read the contents of the file
        if 'decoder' in file_name:
            continue
        with open(file_name, "r") as file:
            json_str = file.read()

        # Deserialize the JSON string to a dictionary
        temp_dict = json.loads(json_str)
        data=temp_dict['mem']
        
        timestamps = [t for t, _ in data]
        measured_val = [m for _, m in data]
        
        lines='-'
        odr=5
        if 'nerf' in file_name:
            lines='--'
            odr=10
        plt.plot(timestamps, measured_val,label=label_dict[file_name],linestyle=lines,linewidth=3,zorder=odr)
    
    for file_name in file_list:
        # Read the contents of the file
        if 'transmitter' in file_name:
            continue
        with open(file_name, "r") as file:
            json_str = file.read()

        # Deserialize the JSON string to a dictionary
        temp_dict = json.loads(json_str)
        data=temp_dict['mem']
        
        timestamps = [t for t, _ in data]
        measured_val = [m for _, m in data]
        
        lines='-'
        odr=5
        if 'nerf' in file_name:
            lines='--'
            odr=10
        plt.plot(timestamps, measured_val,label=label_dict[file_name],linestyle=lines,linewidth=3,zorder=odr)
    
    dot_size=60
    # data points for img diffusion model start
    plt.scatter(39.9751398563385, 4709,c='black',s=dot_size,zorder=10,marker='s') # dec nerf
    plt.scatter(40.46715545654297, 4709,c='black',s=dot_size,zorder=10,marker='s') # dec stf
    plt.scatter(43.11763572692871, 5546,c='black',s=dot_size,zorder=10,marker='s') # trans nerf
    plt.scatter(44.48019242286682, 5546,c='black',s=dot_size,zorder=10,marker='s') # trans stf
    
    # data points for img decoder model start
    plt.scatter(95.14208674430847, 5569,c='black',s=dot_size,zorder=10,marker='^') # dec nerf
    plt.scatter(94.41113615036011, 5569,c='black',s=dot_size,zorder=10,marker='^') # dec stf
    plt.scatter(98.7299256324768, 6349,c='black',s=dot_size,zorder=10,marker='^') # trans nerf
    plt.scatter(98.4405107498169, 6349,c='black',s=dot_size,zorder=10,marker='^') # trans stf

    # # data points for text diffusion model start
    # plt.scatter(38.56686091423035, 4717,c='black',s=dot_size,zorder=10,marker='s') # dec nerf
    # plt.scatter(38.940399169921875, 4717,c='black',s=dot_size,zorder=10,marker='s') # dec stf
    # plt.scatter(44.130261182785034, 5545,c='black',s=dot_size,zorder=10,marker='s') # trans nerf
    # plt.scatter(45.02914643287659, 5545,c='black',s=dot_size,zorder=10,marker='s') # trans stf
    
    # # data points for text decoder model start
    # plt.scatter(62.75246572494507, 5267,c='black',s=dot_size,zorder=10,marker='^') # dec nerf
    # plt.scatter(62.1445369720459, 5267,c='black',s=dot_size,zorder=10,marker='^') # dec stf
    # plt.scatter(67.60589289665222, 6129,c='black',s=dot_size,zorder=10,marker='^') # trans nerf
    # plt.scatter(68.661550283432, 6259,c='black',s=dot_size,zorder=10,marker='^') # trans stf

    # plt.scatter(241.634019613266, 5841,c='black',s=20,zorder=10,marker='^')
    
    plt.xlim(0,110)
    plt.ylim(0,13000)
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=27)
    plt.legend(fontsize=25,loc='upper left')
    plt.grid(True)
    # plt.savefig('./shap-e-mem-plot-gcp.png')
    plt.savefig('./shap-e-mem-plot-img-gcp.png')
    # plt.savefig('./shap-e-mem-plot-text-gcp.png')
    plt.show()

if __name__ == "__main__":
    main()