import matplotlib.pyplot as plt
import json

def main():
    # file_list=['base40M-textvec-out.json','base40M-out.json','base300M-out.json','base1B-out.json']
    file_list=[]
    label_dict={}
    for w1 in ['img']: # ['img','text']
        for w2 in ['dec','trans']:
            for w3 in ['nerf','stf']:
                temp_str1=w1+"-"+w2+"-"+w3+"-out.json"
                file_list.append(temp_str1)
                temp_str2=""
                if w1=='img':
                    temp_str2+='image300M, '
                else:
                    temp_str2+='text300M, '
                if w2=='dec':
                    temp_str2+='decoder, '
                else:
                    temp_str2+='transmitter, '
                temp_str2+=w3
                label_dict[temp_str1]=temp_str2
    # file_list=['base40M-textvec-out.json','base40M-out.json','base300M-out.json']
    # label_dict={'base40M-textvec-out.json':'base40M, text-only','base40M-out.json':'base40M','base300M-out.json':'base300M'}
    plt.figure(figsize=(10,6))
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
    
    # data points for img based model
    plt.scatter(46.63408136367798, 2891,c='black',s=20,zorder=10,marker='s')
    plt.scatter(46.63408136367798, 2891,c='black',s=20,zorder=10,marker='s')
    plt.scatter(46.63408136367798, 2891,c='black',s=20,zorder=10,marker='s')
    plt.scatter(46.63408136367798, 2891,c='black',s=20,zorder=10,marker='s')

    # plt.scatter(241.634019613266, 5841,c='black',s=20,zorder=10,marker='^')
    
    plt.xlim(0,500)
    plt.ylim(0,13000)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12,loc='lower right')
    plt.grid(True)
    plt.savefig('./shap-e-mem-plot.png')
    plt.show()

if __name__ == "__main__":
    main()