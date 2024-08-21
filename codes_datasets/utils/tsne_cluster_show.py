
import requests
from tqdm import tqdm
import argparse
import json
import random
# import multiprocessing as mp
import os
import readline
import time
import glob
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def dim_match_tsne(args):
    
    src_files = sorted(glob.glob(os.path.join(args.input_dir,"*.jsonl"), recursive=True))
    print(f"src_files:{src_files}")
    
    data_list = []
    target_list = []
    for index,xfile in tqdm(enumerate(src_files),total=len(src_files)):
        print(f"\nreading {xfile}")
        with open(xfile, 'r',encoding='utf-8') as fin:
            for idx, line in enumerate(tqdm(fin)):
                line = line.strip()
                if len(line) < 10: continue
                try:
                    js_dict = json.loads(line)
                    embed_768 = js_dict['big_dog_embed'][0]
                    #if len(embed_768) > 768: continue
                    #if 'publish_id' not in js_dict: continue
                    target_list.append(js_dict['source'])
                    data_list.append(embed_768)
                    if len(data_list) > args.max_num: break
                except Exception as e:
                    continue
        if len(data_list) > args.max_num: break

    assert len(target_list) == len(data_list), "data_list != target_list, Error"
    
    index_pos_list = list(range(0,len(data_list)))
    random.shuffle(index_pos_list)
    index_pos_list = index_pos_list[0:args.scatter_num]

    target_list_str2id_dict = {}
    pid = 1
    for item in target_list:
        if item not in target_list_str2id_dict:
            target_list_str2id_dict[item] = pid
            pid += 1
    
    new_data_list = []
    new_target_list = []
    for pos in index_pos_list:
        new_data_list.append(data_list[pos])
        new_target_list.append(target_list_str2id_dict.get(target_list[pos]))
    assert len(new_data_list) == len(new_target_list), "new_data_list != new_target_list, Error"
    
    print(f"data_list: {len(data_list)}")
    print(f"new_data_list: {len(new_data_list)}")
    
    try:
        data_array = np.asarray(new_data_list, dtype=float)
        print("data_array.shape:", data_array.shape)  # 应输出 (150, 768)
    except ValueError as e:
        print(f"Error converting data_list to numpy array: {e}")
        data_array = None
    
    if data_array is None:
        print("----Error-----")
        return
    print("data_array.shape:",data_array.shape)#(150,768)

    try:
        tsne = TSNE(n_components=2, learning_rate=100, perplexity=35, random_state=3517)
        transformed_data = tsne.fit_transform(data_array)
        print("TSNE transformation complete. Shape:", transformed_data.shape)
    except Exception as e:
        print(f"Error during TSNE transformation: {e}")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(111)
    
    dot_size = [1]*len(new_target_list)
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1],s=dot_size,c=new_target_list)
    plt.colorbar()
    plt.title('sftExp8.3_{}_points'.format(args.scatter_num))
    output_img = os.path.join(args.output_dir,"sftExp8.3_tsne_{}.png".format(args.scatter_num))
    if os.path.exists(output_img): os.remove(output_img)
    plt.savefig(output_img,format='png',dpi = 100, bbox_inches='tight')
    plt.show()

    
def argparers():
    parser = argparse.ArgumentParser(description='Process JSON data.')
    parser.add_argument('--input_dir',
                        default='/temp/workspace/train_dataset_selected',
                        required=False,
                        help='Path to JSON file with job and CV data.')
    parser.add_argument('--output_dir',
                        default='./',
                        required=False,
                        help='Path to output jsonl file.')
    parser.add_argument('--done_file_name',
                        default='done_embed.txt',
                        required=False,
                        help='')
    parser.add_argument('--max_size',
                        type=int,
                        default=128 * 1024 * 1024,
                        help="max chunk size")
    parser.add_argument('--scatter_num',
                        type=int,
                        default=100000)
    parser.add_argument('--max_num',
                        type=int,
                        default=258063)#258063
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argparers()

    # if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)

    dim_match_tsne(args)

