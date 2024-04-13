# Inside this file the distance objects, which will be used inside the
# AntClust class are defined.
# A similarity object should always inherit from the
# informal interface "similarity_interface".
# The similarity function should only return values between 0 and 1
# These similarity objects can then be instantiated and provided to AntClust.

# imports
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
from torch.nn.functional import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from sklearn.metrics import jaccard_score
import torch
from fastdtw import fastdtw
import cv2 as cv



# Informal Interface
class similarity_interface:
    def similarity(self, d_0, d_1):
        """
        Computes the similarity/distance between two data vectors d_0 and d_1.
        This distance/similarity between the two objects should be expressed as
        a range between 0.0 and 1.0, where 1.0 means the objects are similar,
        0.0 is returned if the objects are completely anti-similar.
        I.e. if d_0 == d_1 it should return 1.0
        """
        raise NotImplementedError


class precomputed_similarity_matrix(similarity_interface):
    """
    Assumes that a precomputed similarity matrix is provided to AntClust.
    similarity matrix D such that D_{i, j} is the similarity.
    e.g. 
    [0,1,2] = [ [0,1,2],
                [1,0,1],
                [2,1,0]]
    """

    def similarity(self, d_0, d_1):
        # find the indexes in the similarity matrix
        # i_d0 = np.where(d_0 == 0)[0][0]
        i_d1 = np.where(d_1 == 0)[0][0]

        # get distance, distance matrix is symmetric
        dist = d_0[i_d1]

        # normalize between 0-1
        max_dist = np.max(d_0)
        dist = dist/max_dist

        # return inverted distance
        return 1 - dist


class similarity_1d(similarity_interface):
    """
    Implements the 1d numeric distance measure
    """

    def __init__(self, min, max):
        """
        min: the minimal numeric value an object can have
        max: the maximal numeric value an object can have in the data set
        """
        self.min = min
        self.max = max

    def similarity(self, d_0, d_1):
        """
        Inverted distance between two numbers, normalized between 0 and 1.
        Meaning if two numbers are equal they are completely similar => sim(2,2) = 1
        """
        return 1 - (abs(d_0 - d_1)/abs(self.min - self.max))


class similarity_euclid2d(similarity_interface):
    """
    Implements the euclidean distance for 2D vectors
    """

    def __init__(self, min, max):
        """
        min: the minimal value a vector can have
        max: the maximal value a vector can have
        """
        self.min = min
        self.max = max

    def similarity(self, a, b):
        # Calculate eclidean distance for 2D vectors.
        # Normalize between 0 and 1 and invert.
        return 1 - (math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)/abs(self.min-self.max))


# --------------------------------
#    OPENCV distance classes
# --------------------------------
class opencv_image_orb_similarity(similarity_interface):
    """
    Provides a distance function that will work on opencv images.
    """

    def __init__(self, max_distance=70, image_resize_size=(150, 172)):
        import cv2 as cv
        self.max_distance = max_distance
        self.image_resize_size = image_resize_size

        # Initiate ORB detector for feature extraction
        self.orb = cv.ORB_create()

        # create BFMatcher object
        # self.bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)
        self.bfMatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    def similarity(self, d_0, d_1):
        """
        d_n: opencv images read in via cv.imread(image_file, cv.IMREAD_GRAYSCALE)

        Computes the similarity/distance between two data vectors d_0 and d_1.
        This distance/similarity of the two objects should be expressed as
        a range between 0.0 and 1.0, where 1.0 means the objects are similar,
        0.0 is returned if the objects are completely anti-similar.
        I.e. if d_0 == d_1 it should return 1.0
        """

        # resize the images
        d_0_resized = cv.resize(d_0, self.image_resize_size)
        d_1_resized = cv.resize(d_1, self.image_resize_size)

        # find the keypoints and descriptors with ORB
        kp_d_0, des_d_0 = self.orb.detectAndCompute(d_0_resized, None)
        kp_d_1, des_d_1 = self.orb.detectAndCompute(d_1_resized, None)

        # Match descriptors.
        matches = self.bfMatcher.match(des_d_0, des_d_1)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # just return the best descriptor
        # FIXME: there is a better more complex approach
        # only using the nearest descriptor is not the best approach
        # there should be a rating that covers all the found features
        # into one similarity value.

        # check for zero matches
        if len(matches) == 0:
            return 0.0

        # get the shortest distacne and normalize it between 0 and 1
        shortest_descriptor_distance = 1 - \
            ((matches[0].distance)/self.max_distance)

        # really be sure that the number is between 0 and 1
        return np.clip(shortest_descriptor_distance, a_min=0.0, a_max=1.0)


class opencv_image_flann_similarity(similarity_interface):
    """
    Provides a distance function that will work on opencv images
    using OpenCV's FLANN matcher.
    """

    def __init__(self, max_distance=70, image_resize_size=(150, 172)):
        import cv2 as cv
        self.max_distance = max_distance
        self.image_resize_size = image_resize_size

        # Initiate ORB detector for feature extraction
        self.orb = cv.ORB_create()

        # FLANN matcher
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,     # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

    def similarity(self, d_0, d_1):
        """
        d_n: opencv images read in via cv.imread(image_file, cv.IMREAD_GRAYSCALE)

        Computes the similarity/distance between two data vectors d_0 and d_1.
        This distance/similarity of the two objects should be expressed as
        a range between 0.0 and 1.0, where 1.0 means the objects are similar,
        0.0 is returned if the objects are completely anti-similar.
        I.e. if d_0 == d_1 it should return 1.0
        """

        # resize the images
        d_0 = cv.resize(d_0, self.image_resize_size)
        d_1 = cv.resize(d_1, self.image_resize_size)

        # find the keypoints and descriptors with ORB
        kp_d_0, des_d_0 = self.orb.detectAndCompute(d_0, None)
        kp_d_1, des_d_1 = self.orb.detectAndCompute(d_1, None)

        # Match descriptors.
        matches = self.flann.match(des_d_0, des_d_1)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # check for zero matches
        if len(matches) == 0:
            return 0.0

        # get the shortest distacne and normalize it between 0 and 1
        shortest_descriptor_distance = 1 - \
            (matches[0].distance/self.max_distance)

        # really be sure that the number is between 0 and 1
        return np.clip(shortest_descriptor_distance, a_min=0.0, a_max=1.0)


class opencv_orb_similarity(similarity_interface):
    """
    Will work only on OpenCV features wich need to be precomputed.
    This will speed up clustering compared to if using images as the
    features of the images will only be computed once and not whenever
    a call to the similarity function is made.
    """
    def __init__(self, max_distance=70):
        import cv2 as cv
        self.max_distance = max_distance
        # create BFMatcher object
        # self.bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        

    def similarity(self, d_0, d_1):
        """
        Computes the similarity/distance between two data vectors d_0 and d_1.
        This distance/similarity of the two objects should be expressed as
        a range between 0.0 and 1.0, where 1.0 means the objects are similar,
        0.0 is returned if the objects are completely anti-similar.
        I.e. if d_0 == d_1 it should return 1.0
        """

        # Match descriptors.
        matches = self.bf.match(d_0, d_1)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # check for zero matches
        if len(matches) == 0:
            return 0.0

        # get the shortest distacne and normalize it between 0 and 1
        shortest_descriptor_distance = 1 - \
            (matches[0].distance/self.max_distance)

        # really be sure that the number is between 0 and 1
        return np.clip(shortest_descriptor_distance, a_min=0.0, a_max=1.0)


# IMAGE DATA
class image_cosine_similarity(similarity_interface):
    """
    Implements the similarity function based on cosine similarity given using a pretrained CLIP
    model
    """
    
    def __init__(self, path_to_image=None, image_name_list=None, img_tensor=None):
        """
        Using a clip Vision transformer, encode images in the model.
        """
        self.model = SentenceTransformer('clip-ViT-B-32')  
        if path_to_image and image_name_list:
            self.encoded_image = self.model.encode([Image.open(f"{path_to_image}{filepath}") for filepath in image_name_list], batch_size=256, convert_to_tensor=True, show_progress_bar=True)
        elif img_tensor.all() != None:
            self.encoded_image = self.model.encode([Image.fromarray(img.astype('uint8')) for img in img_tensor], batch_size=256, convert_to_tensor=True, show_progress_bar=True)



    def similarity(self, d_0, d_1):
        """
        Cosine similarity between embeddings.
        """
        indices_to_keep = torch.tensor([d_0, d_1])
        pair_img = self.encoded_image[indices_to_keep]
        sim, _, _ = util.paraphrase_mining_embeddings(pair_img)[0]    
        return sim
    
# TEXT DATA
class text_cosine_similarity(similarity_interface):
    """
    Implements the similarity function based on cosine similarity given using a pretrained CLIP
    model
    """
    
    def __init__(self, text_list):
        """
        Using a clip Vision transformer, encode images in the model.
        """
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  
        self.encoded_text = self.model.encode(text_list, batch_size=256, convert_to_tensor=True, show_progress_bar=True)
        


    def similarity(self, d_0, d_1):
        """
        Cosine similarity between embeddings.
        """
        sim = cosine_similarity(self.encoded_text[d_0], self.encoded_text[d_1],dim=0).item()   
        return sim
    
# NUMERICAL DATA (n dimension)
class similarity_euclid(similarity_interface):
    """
    Implements the euclidean distance for n-D vectors
    """

    def __init__(self, min, max):
        """
        min: the minimal value a vector can have
        max: the maximal value a vector can have
        """
        self.min = min
        self.max = max

    def similarity(self, a, b):
        # Calculate eclidean distance for 2D vectors.
        # Normalize between 0 and 1 and invert.
        dist = np.sqrt(np.sum((a - b)**2))
        norm_dist = (dist - self.min) / (self.max - self.min)
        return 1 - norm_dist

# CATEGORICAL DATA
class similarity_categorical(similarity_interface):
    """
    Implements the Jaccard similarity score for encoded categorical data
    """

    def __init__(self):
        """
        """

    def similarity(self, a, b):
        # Calculate jaccard score for two rows of the dataset.
        return  jaccard_score(a.tolist(), b.tolist(), average='micro')
    
# TIME SERIES DATA
class similarity_time_series(similarity_interface):
    """
    Implements Dynamic Time Warping distance for time series data
    """

    def __init__(self, min, max):
        """
        min: the minimal value a distance can have
        max: the maximal value a distance can have
        """
        self.min = min
        self.max = max

    def similarity(self, a, b):
        # Calculate Dynamic Time Warping distance 
        distance_same, path = fastdtw(a, b)
        norm_dist = (distance_same - self.min) / (self.max - self.min)
        return  1 - norm_dist

# SPATIAL DATA
class similarity_cosine_spatial(similarity_interface):
    """
    Implements the Cosine similarity score for spatial data
    """

    def __init__(self):
        """
        """

    def similarity(self, a, b):
        # Calculate cosine similarity for two rows of the dataset.
        sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
        return max(0,sim)
    
class levenshtein_distance(similarity_interface):
    """
    Implements the Levenshtein Distance score for text data
    """

    def __init__(self, min, max):
        """
        min: the minimal value a distance can have
        max: the maximal value a distance can have
        """
        self.min = min
        self.max = max

    def levenshtein_distance(self, a, b):
        # Create a matrix to store distances
        matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

        # Initialize first row and column of the matrix
        for i in range(len(a) + 1):
            matrix[i][0] = i
        for j in range(len(b) + 1):
            matrix[0][j] = j

        # Fill the matrix
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                matrix[i][j] = min(matrix[i - 1][j] + 1,       # Deletion
                                matrix[i][j - 1] + 1,       # Insertion
                                matrix[i - 1][j - 1] + cost)  # Substitution

        # Return the bottom-right cell of the matrix
        return matrix[len(a)][len(b)]

    def similarity(self, a, b):
        # Calculate Levenshtein distance 
        norm_dist = (self.levenshtein_distance(a,b) - self.min) / (self.max - self.min)
        return  1 - norm_dist