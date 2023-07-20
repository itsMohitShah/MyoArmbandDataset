import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time
from datetime import timedelta


number_of_vector_per_example = 52
number_of_canals = 8
number_of_classes = 7
size_non_overlap = 5



def format_data_to_train(vector_to_format):
    dataset_example_formatted = []
    example = []
    emg_vector = []
    for value in vector_to_format:
        emg_vector.append(value)
        if (len(emg_vector) >= 8):
            if (len(example) == 0):
                example = emg_vector
            else:
                example = np.row_stack((example, emg_vector))
            emg_vector = []
            if (len(example) >= number_of_vector_per_example): #number of vector per example =52. Len (example) =8x1, 8x2....8*7 >>>7x8(appended to dataset example formatted)>>>>8x7
                example = example.transpose()
                dataset_example_formatted.append(example)
                example = example.transpose()
                example = example[size_non_overlap:]
    return dataset_example_formatted



def plot_generator(list_path_allclasses):
    total_count=0
    for path in tqdm(list_path_allclasses):
        print("Generating plot...")
        data_read_from_file = np.fromfile(path, dtype=np.int16)
        data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
        # print(f"Gathering dataset of {path}")
        dataset_example = format_data_to_train(data_read_from_file)
        print(f"Gathered dataset of {path}")
        count = 0
        for samples in range(len(dataset_example)):
            training_plot = dataset_example[samples]
            fig,ax = plt.subplots(nrows=8,ncols=1, figsize=(10,10),sharex=True)
            for channel in range(np.shape(training_plot)[0]):
                ax[channel].plot(training_plot[channel])
                # ax[channel].set_title(f"Channel : {channel}")
            plt.tight_layout()
            name_plotfolder=path.split('.')[0]
            name_plotfolder=name_plotfolder.split('\\')[-1]
            name_plotfolder="plots_"+name_plotfolder
            path_plotfolder=os.path.join(os.path.dirname(path),name_plotfolder)
            os.makedirs(path_plotfolder,exist_ok=True)
            plot_path=os.path.join(path_plotfolder,f'class0_{count}.png')
            fig.savefig(plot_path)
            # plt.show()
            plt.close()
            count+=1
            total_count+=1
        
        print(f"Plots generated of {path}")
    print(f"Total Number of images generated : {total_count}")


def path_retriever(root):
    print("Retrieving paths...")
    list_humans = os.listdir(root)
    list_path_allclasses=[]
    for each_human in tqdm(list_humans):
        path_each_human=os.path.join(root,each_human)
        list_trainingfolders=os.listdir(path_each_human)
        for each_trainingfolder in list_trainingfolders:
            path_trainingfolder = os.path.join(path_each_human,each_trainingfolder)
            if os.path.isdir(path_trainingfolder):
                list_classes=os.listdir(path_trainingfolder)
                for each_insidetrainingfolder in list_classes:
                    path_each_class = os.path.join(path_trainingfolder,each_insidetrainingfolder)
                    if os.path.isfile(path_each_class):
                        list_path_allclasses.append(path_each_class)
                    else:
                        pass
            else:
                pass
    print("Paths retrieved")
    return list_path_allclasses



start = time.time()
root = r'PreTrainingDataset'
list_path_allclasses = path_retriever(root)
plot_generator(list_path_allclasses)

elapsed_time = time.time() - start

print(str(timedelta(seconds=elapsed_time)))