import glob
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch


script_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_directory)
df = pd.read_csv("flower_labels.csv")

image_names = list(glob.glob('flower_images/*.png'))[:10]
image_names = [name.split("\\")[-1] for name in image_names]
data = [(1, 1, 2), (0.9, 4, 2), (0.5, 5, 2), (0.6, 3, 4)]

def create_csv(image_names, data, df):
    new_df = pd.DataFrame()
    same_label = []
    scores = []
    for score, image_id1, image_id2 in data:
        same_class = df.loc[df['file'] == image_names[image_id1], 'label'].values[0] == df.loc[df['file'] == image_names[image_id2], 'label'].values[0]
        same_label.append(1 if same_class else 0)
        scores.append(score)
    new_df["similarity"]=scores
    new_df["same_class"]=same_label
    new_df.to_csv("similarity_per_class.csv", index=False)

def metrics():
    df = pd.read_csv("similarity_per_class.csv")
    # Sort the DataFrame by the "value" column in ascending order
    df = df.sort_values(by='similarity')

    df_true = df[df['same_class'] == 1]
    df_false = df[df['same_class'] == 0]

    print("For same class:")
    print("Mean Similarity:",  df_true['similarity'].mean())
    print("Minimum Similarity:", df_true['similarity'].min())
    print("Maximum Similarity:", df_true['similarity'].max())
    print("Standard Deviation of Similarity:", df_true['similarity'].std())


    print("For different class:")
    print("Mean Similarity:",  df_false['similarity'].mean())
    print("Minimum Similarity:", df_false['similarity'].min())
    print("Maximum Similarity:", df_false['similarity'].max())
    print("Standard Deviation of Similarity:", df_false['similarity'].std())

    # Plot the scatter plot with different colors for 1 and 0 values
    sns.scatterplot(x=range(len(df)), y='similarity', hue='same_class', data=df, marker='o', alpha=0.7)

    # Set labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Scatter Plot with Boolean Values (0 and 1)')
    plt.legend()

    # Show the plot
    plt.show()

def find_entry_by_key_value(lst, key, value):
    for entry in lst:
        if entry.get(key) == value:
            return entry
    return None  # Return None if not found

def similarity(id1, id2, encoded_image):
    threshold = 0.9
    indices_to_keep = torch.tensor([id1, id2])
    pair_img = encoded_image[indices_to_keep]
    sim, _, _ = util.paraphrase_mining_embeddings(pair_img)[0]    
    match = 1 if sim >= threshold else 0        
    return sim, match


def test_similarity():
    path_img = "flower_images/"
    image_names = list(glob.glob(f'{path_img}*.png'))
    image_names = [name.split("\\")[-1] for name in image_names]
    file_path_text = "similarity_values.txt"
    if os.path.exists(file_path_text):
        processed_images = []
        with open(file_path_text, 'r') as file:
            for line in file:
                #processed_images.append(tuple(map(int, line.strip().split(','))))
                processed_images.append(eval(line.strip()))
        print("Loaded List of Tuples")
    else:
        print('Loading CLIP Model...')
        model = SentenceTransformer('clip-ViT-B-32')   
        encoded_image = model.encode([Image.open(f"{path_img}{filepath}") for filepath in image_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
        processed_images = util.paraphrase_mining_embeddings(encoded_image, max_pairs=len(image_names)**2)
        with open(file_path_text, 'w') as file:
            for item in processed_images:
                file.write(','.join(map(str, item)) + '\n')           
    #sim_dic = [{'{}_{}'.format(min(id1, id2), max(id1, id2)): val} for val, id1, id2 in processed_images]
    sim_dic = {}
    for val, id1, id2 in processed_images:
        key = '{}_{}'.format(min(id1, id2), max(id1, id2))
        sim_dic[key]=val
    for _ in range(15):
        id1, id2 = random.sample(range(0, len(image_names)), 2)
        print(f"Random id {id1,id2}")
        print(f"Similarity for images: {image_names[id1],image_names[id2]}")
        try:
            sim, match = similarity(id1, id2, sim_dic)
            same_class = df.loc[df['file'] == image_names[id1], 'label'].values[0] == df.loc[df['file'] == image_names[id2], 'label'].values[0]
            print(f"Similarity index = {sim}")
            print(f"Accepted = {match}")
            print(f"Same class = {same_class}")
        except:
            pass
        

def test_similarity_2():
    path_img = "flower_images/"
    image_names = list(glob.glob(f'{path_img}*.png'))
    image_names = [name.split("\\")[-1] for name in image_names]
    print('Loading CLIP Model...')
    model = SentenceTransformer('clip-ViT-B-32')  
    encoded_image = model.encode([Image.open(f"{path_img}{filepath}") for filepath in image_names], batch_size=64, convert_to_tensor=True, show_progress_bar=True)
    correct = 0
    incorrect = 0
    for _ in range(20):
        id1, id2 = random.sample(range(0, len(image_names)), 2)
        print(f"Random id {id1,id2}")
        print(f"Similarity for images: {image_names[id1],image_names[id2]}")
        sim, match = similarity(id1,id2,encoded_image)
        same_class = df.loc[df['file'] == image_names[id1], 'label'].values[0] == df.loc[df['file'] == image_names[id2], 'label'].values[0]
        if (same_class and match == 1) or (not same_class and match == 0):
            correct += 1
        else:
            incorrect += 1
        print(f"Similarity index = {sim}")
        print(f"Accepted = {match}")
        print(f"Same class = {same_class}")
    print(f"{correct} correct | {incorrect} incorrect")
        

if __name__ == "__main__":
    # metrics()
    test_similarity_2()
