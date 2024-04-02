from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

script_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_directory)

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

# Load the OpenAI CLIP Model
print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')
image_names = list(glob.glob('flower_images/*.png'))
image_names = [name.split("\\")[-1] for name in image_names]
print("Images:", len(image_names))
encoded_image = model.encode([Image.open(f"flower_images/{filepath}") for filepath in image_names], batch_size=128, convert_to_tensor=True, show_progress_bar=False)
processed_images = util.paraphrase_mining_embeddings(encoded_image)
# NUM_SIMILAR_IMAGES = 10 

# similar = [image for image in processed_images if image[0] < threshold]

# threshold = 0.7
df = pd.read_csv("flower_labels.csv")
#create_csv(image_names, processed_images, df)
