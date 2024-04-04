# Notes from the meeting
- Work on AntClust algorithm.
- Build outline of goals and steps to achieve.
- Main idead: 
  - Look at similarity function, adapt AntClust to multiple problems and to deal with multiple data objects (smilariy between images, text, numbers, etc.)
  - Look into ideas to improve clustering for higher number of clusters.
  - Check possibility of using custom gpt for creating ruleset. (doesn't work)

# Idea 1: apply AntClust to image tensors (like cifar10 or MNIST) and compare with basic methods such as k-means
- Cosine similarity measures the similarity between two vectors of an inner product space, by using CLIP (a vision transformer model pretrained on imageNet) it can encode the image into vector embeddings and then applying the cosine similarity function.
- Using a prtrained transformer, encodes the dataset and then compute cosine similarity between images. Works for small number of images due to quadratic complexity. For MNIST and CIFAR-10 it works very poorly.
- Tried using ORB similarity to compare, which worked fine in the flower datasets since the images were stored, but with MNIST or CIFAR-10 ORB had a difficult dealing with smaller images (32x32). Sometimes it wasn't available to find descriptors, so I had to limit only use descriptors from images that were able to compute them, which reduced the total lenght, so I also kept track of those indexes and removed from the label list for a better comparison. ORB with cifar/mnist images didn't use grayscaled features. 
- For flower dataset (210 flower of 10 classes):
   ARI for AntClust (Cosine Similarity) 0.6389888868288874
   ARI for AntClust (ORB Similarity) 0.011252503833263665
   ARI for K-means (k=10) 0.23139984255616106
- For MNIST dataset (balanced subset of 1000 images, a hundred images per class):
   ARI for AntClust (Cosine Similarity) 0.12179987677390727
   ARI for AntClust (ORB Similarity) 0.059549661048787034
   ARI for K-means (k=10) 0.34392891199081566
- For CIFAR-10 dataset (balanced subset of 1000 images, a hundred images per class):
   ARI for AntClust (Cosine Similarity) 0.00475886692575333
   ARI for AntClust (ORB Similarity) 0.0013394124288566736
   ARI for K-means (k=10) 0.04253002081807093
- For flower dataset used in training example (3 classes, 3 flowers each):
   ARI for AntClust (Cosine Similarity): 1.0
   ARI for AntClust (ORB Similarity): 0.1111111111111111
   ARI for K-means (k=3): 1.0
- 
# Idea 2: apply AntClust to text data
See https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
Compare using 3387 documents in 4 categories, compared to different k-means measurement from the above website. It measures the following metrics.
- homogeneity, which quantifies how much clusters contain only members of a single class;
- completeness, which quantifies how much members of a given class are assigned to the same clusters;
- V-measure, the harmonic mean of completeness and homogeneity;
- Adjusted Rand-Index, a chance-adjusted Rand-Index such that random cluster assignment have an ARI of 0 in expectation.
  
Results: 

| Method                            | Homogeneity | Completeness | V-measure | Adjusted Rand-Index |
|-----------------------------------|-------------|-------------:|----------:|--------------------:|
| KMeans with LSA on hashed vectors |    0.392749 |     0.441747 |  0.415805 |            0.330707 |
| KMeans with LSA on hashed vectors |    0.393501 |     0.435161 |  0.413194 |            0.323662 |
| KMeans on tf-idf vectors          |    0.343134 |     0.404380 |  0.370404 |            0.212785 |
| AntClust Cosine Similarity        |    0.484350 |     0.242738 |  0.323400 |            0.192821 |

AntClust manages to find much higher homogeinity but on the cost of having worse results on the other fields.

**Levenshtein distance**
By using the Levenshtein distance as a similarity function, antClust can succefully group words that are similar to eachother based on the Levenshtein distance idea, which is measuring how many changes(deletions, insertion, change letters) necessary for a world to build another.

Using bag of words: 
| **Words per group**                             |
|-------------------------------------------------|
| apple, apply, apples, applies, appliance        |
| banana, bananas, bannana, bannanas, banane      |
| cat, cats, cot, cots, cut                       |
| dog, dogs, dig, digs, dag                       |
| elephant, elephants, elegant, elegance, element |
| frog, frogs, frag, frags, frug                  |

reuslts:

| Methods                           | Homogeneity | Completeness | V-measure | Adjusted Rand-Index |
|-----------------------------------|-------------|--------------|-----------|---------------------|
| KMeans with LSA on hashed vectors | 0.231582    | 0.295393     | 0.258886  | -0.019937           |
| KMeans with LSA on tf-idf vectors | 0.315513    | 0.328625     | 0.321905  | 0.024780            |
| KMeans on ntf-idf vectors         | 0.175018    | 0.436269     | 0.249817  | 0.000149            |
| AntClust Levenshtein distance     | 0.949707    | 0.953278     | 0.951490  | 0.913605            |

# Idea 3: Apply AntClust to numerical data

**Apply to iris dataset** (src: https://www.kaggle.com/code/khotijahs1/k-means-clustering-of-iris-dataset)
- Added euclidean distance similarity function which is independent of dimensionality (acepts any dimension).
- Is a good method compared to k-mean, although it underpeforms for k close to the true amount of clusters (3).
- Number of features: 4
- Number of data points: 150
- Number of classes: 3

Results:
| Method                        | Homogeneity | Completeness | V-measure | Adjusted Rand-Index |
|-------------------------------|------------:|-------------:|----------:|--------------------:|
| AntClust (euclidean distance) |    0.667307 |     0.664601 |  0.665951 |            0.544441 |
|                 K-means (k=2) |    0.522322 |     0.883514 |  0.656519 |            0.539922 |
|                 K-means (k=3) |    0.751485 |     0.764986 |  0.758176 |            0.730238 |
|                 K-means (k=4) |    0.808314 |     0.652211 |  0.721920 |            0.649818 |
|                 K-means (k=5) |    0.823883 |     0.599287 |  0.693863 |            0.607896 |
|                 K-means (k=6) |    0.823883 |     0.520492 |  0.637954 |            0.447534 |
|                 K-means (k=7) |    0.914483 |     0.524576 |  0.666707 |            0.474661 |
|                 K-means (k=8) |    0.925560 |     0.513151 |  0.660247 |            0.463783 |

**Apply to wine dataset**  (src: https://archive.ics.uci.edu/dataset/109/wine)
- Apply AntClust with n-dimensional euclidean distance.
- Performs better than k-means even with true number of classes (3). Could be due to this dataset having more values.
- Number of features: 13
- Number of data points: 178
- Number of classes: 3

Results:
| Method                        | Homogeneity | Completeness | V-measure | Adjusted Rand-Index |
|-------------------------------|------------:|-------------:|----------:|--------------------:|
| AntClust (euclidean distance) |    0.416515 |     0.413527 |  0.415015 |            0.399096 |
|                 K-means (k=2) |    0.334199 |     0.587027 |  0.425919 |            0.369408 |
|                 K-means (k=3) |    0.428812 |     0.428701 |  0.428757 |            0.371114 |
|                 K-means (k=4) |    0.409747 |     0.336367 |  0.369449 |            0.288788 |
|                 K-means (k=5) |    0.495139 |     0.351292 |  0.410993 |            0.311588 |
|                 K-means (k=6) |    0.495760 |     0.334879 |  0.399739 |            0.290902 |
|                 K-means (k=7) |    0.502329 |     0.299775 |  0.375477 |            0.220960 |
|                 K-means (k=8) |    0.506082 |     0.275232 |  0.356553 |            0.197813 |


**Apply to breast cancer dataset** (src: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- Apply AntClust with n-dimensional euclidean distance.
- Performs better than k-means even with true number of classes (2)
- Number of features: 30
- Number of data points: 569
- Number of classes: 2

Results:
| Method                        | Homogeneity | Completeness | V-measure | Adjusted Rand-Index |
|-------------------------------|------------:|-------------:|----------:|--------------------:|
| AntClust (euclidean distance) |    0.578839 |     0.273374 |  0.371361 |            0.503367 |
|                 K-means (k=2) |    0.422291 |     0.516809 |  0.464793 |            0.491425 |
|                 K-means (k=3) |    0.447857 |     0.451041 |  0.449444 |            0.501563 |
|                 K-means (k=4) |    0.575050 |     0.333277 |  0.421986 |            0.412743 |
|                 K-means (k=5) |    0.601928 |     0.297758 |  0.398425 |            0.341810 |
|                 K-means (k=6) |    0.604317 |     0.274087 |  0.377129 |            0.313490 |
|                 K-means (k=7) |    0.629624 |     0.245605 |  0.353368 |            0.233988 |
|                 K-means (k=8) |    0.635252 |     0.248015 |  0.356748 |            0.237783 |

**Apply to digits dataset** (src: https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html#:~:text=This%20dataset%20is%20made%20up,feature%20vector%20with%20length%2064.)
- Digits dataset containing a 8x8 digit.
- Flatten it and apply euclidean distance.
- Number of features: 64
- Number of data points: 1797
- Number of classes: 10
- Antclust performs worse than k-means

Results:
| Method                        | Homogeneity | Completeness | V-measure | Adjusted Rand-Index |
|-------------------------------|------------:|-------------:|----------:|--------------------:|
| AntClust (euclidean distance) |    0.809031 |     0.640100 |  0.714719 |            0.519198 |
|                 K-means (k=8) |    0.669316 |     0.764487 |  0.713743 |            0.579670 |
|                 K-means (k=9) |    0.692027 |     0.750020 |  0.719858 |            0.597194 |
|                K-means (k=10) |    0.737745 |     0.745229 |  0.741468 |            0.664962 |
|                K-means (k=11) |    0.790131 |     0.763908 |  0.776799 |            0.731170 |
|                K-means (k=12) |    0.793983 |     0.744832 |  0.768623 |            0.703364 |

**Compare performances between AntClust and K-Means (with true k)**
Comparing AntClust with k-means for different number of features, we can see AntClust does sightly better when dealing with more features. but the difference is minimal. 


# Idea 4: Apply AntClust to categorical data
**For Mushroom dataset**
- Number of features: 22
- Number of data points: 1000
- Number of classes: 2
Result: 
| Method                        | Homogeneity | Completeness | V-measure | Adjusted Rand-Index |
|-------------------------------|------------:|-------------:|----------:|--------------------:|
| AntClust (Jaccard similarity) |    0.926437 |     0.114274 |  0.203452 |            0.046784 |
|                 K-means (k=2) |    0.011993 |     0.005950 |  0.007954 |            0.023562 |
|                 K-means (k=3) |    0.195499 |     0.058787 |  0.090393 |            0.033639 |
|                 K-means (k=4) |    0.239852 |     0.058869 |  0.094535 |            0.059449 |

**For Titanic dataset**
- Number of features: 4
- Number of data points: 891
- Number of classes: 2
- AntClust perfoms better than kmeans
Result:
| Method                        | Homogeneity | Completeness | V-measure | Adjusted Rand-Index |
|-------------------------------|------------:|-------------:|----------:|--------------------:|
| AntClust (Jaccard similarity) |    1.000000 |     0.234375 |  0.379747 |            0.199048 |
|                 K-means (k=2) |    0.014472 |     0.013651 |  0.014049 |            0.023880 |
|                 K-means (k=3) |    0.311123 |     0.185400 |  0.232344 |            0.165621 |
|                 K-means (k=4) |    0.166684 |     0.079905 |  0.108025 |            0.109695 |

**For adult income dataset**
- Number of features: 10
- Number of data points: 1000
- Number of classes: 2
- AntClust is worse.
Result:
| Method                        | Homogeneity | Completeness | V-measure | Adjusted Rand-Index |
|-------------------------------|------------:|-------------:|----------:|--------------------:|
| AntClust (Jaccard similarity) |    0.174465 |     0.039372 |  0.064245 |            0.033928 |
|                 K-means (k=2) |    0.062004 |     0.048914 |  0.054687 |            0.073171 |
|                 K-means (k=3) |    0.103540 |     0.053535 |  0.070578 |            0.035250 |
|                 K-means (k=4) |    0.101109 |     0.043136 |  0.060473 |            0.024432 |

# Idea 5: Apply AntClust to time series data
Based on the implementation of: https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering

Utilize the retail time series data set, containing of 23 different time series, since we didn't have the ground truth labels the metric used were the following:
-  **Silhouette Score**: The silhouette score measures how well-separated the clusters are. It ranges from -1 to 1, where a higher value indicates better-defined clusters. It considers both the distance between points within the same cluster and the distance between points in different clusters
-  **Davies-Bouldin Index**: The Davies-Bouldin index measures the compactness and separation between clusters. Lower values indicate better clustering. It considers the average similarity ratio of each cluster with the cluster that is most similar to it.
  
After preprocessing to normalize the data and fill in missing values the three methods were applied:
-  Self Organizing Maps (SOM)
-  K-means with Dynamic Time Warping distance.
-  AntClust with Dynamic Time Warping distance.

Antclust has the highest silhouette score by far and the second smalled Davies-Boulding index but very close to the smallest value (difference of 0.005) 
Results:
|                  Method | Silhouette score | Davies-Bouldin Index |
|------------------------:|-----------------:|---------------------:|
| AntClust (DTW distance) |         0.421470 |             0.792838 |
|           K means (k=5) |         0.217016 |             1.138775 |
|    Self Organizing Maps |         0.186506 |             0.787121 |

# Idea 6: Apply AntClust to spatial data
Based off: https://geographicdata.science/book/notebooks/10_clustering_and_regionalization.html#hierarchical-clustering

Measures: 
- Calinski Harabasz Score (CH): the within-cluster variance divided by the between-cluster variance.
- Silhouette Score: the average standardized distance from each observation to its “next best fit” cluster—the most similar cluster to which the observation is not currently assigned.

Apply Antclust using cosine similarity on the dataset row. It performs very poorly. Worse than all other methods. Having the worst of both metrics.
Results:
|    Method | Silhouette score | Calinski Harabasz Score |
|----------:|-----------------:|------------------------:|
|     k5cls |         0.185683 |              106.907609 |
|     ward5 |         0.162304 |               98.529245 |
|   ward5wq |         0.061462 |               62.518714 |
| ward5wknn |         0.049983 |               54.378576 |
|  antclust |        -0.167726 |               19.338995 |

# Idea 8: testing new rulesets
Ant: 
- age (how many meetings)
- M (succesful meetings estimator)
- M+ (success estimator inside colony)
- Max similairty
- Mean similarity

Labroche ruleset:
- R1: if both no label and accept -> create new colony and assign both ants.
- R2: if one no label and accept -> label of labeled ant is assigned to the other.
- R3: if same label and accept -> increase estimators M and M+ on both ants.
- R4: if same label and not accept -> increase M and decrease M+, ant with smaller M+ loses label.
- R5: if different label and accept -> decrease M from both ants, ant with smaller M is assigned to the other's label.
- R6 nothing happens.

Extra rules:
- R5_merge (merge clusters): if different label and accept and M+ of both ants is higher than-> merge both labels (all ants from both lables become one based on label that has higher M+). 
   **Reasoning**: If two ants from different labels are accepting eachother and have a M+ higher than avarage, it measn that they are similar within their label and also similar with eachoter, which likely means both colonys should be the same.
- R5_new (create clusters): if different label and accept and M+ of both ants is lower than mean -> create new label and assign both ants to new label (reset estimators).
   **Reasoning**: If two ants have different labels but accept, means they should belong to same colony, but if both are poorly accepted within nest (lower than avg M+) they also likely don't belong in that colony, therefore they should create a new colony.
- R5_age: if different label and accept -> decrease M from both ants, ant with smaller age is assigned to the other's label.
   **Reasoning**: Ants with older age had more meetings and their parameters should be better tuned than an younger ant, so we take preference on the older ants.
- R5_stability: Label Stability Reinforcement: Add stability measurement of ant, whenever ant change lables, decrease it. When two ants with the different label accept, the one with less stability get the other's label.
- R3_boost (reinforcement of M+): if same label and accept and M+ from one is higher than mean -> increase estimators M and M+ on both ants with boost on ant with lower estimator.
   **Reasoning**: If one ant is highly accepted within its colony (higher than avg M+) menas they're similar with most its nestamtes, and if that's true and it is similar to the second ant (which has a lower M+) this should mean the second ant is also likely to be accepted well in the colony, so we boost the increase of M+ by doubling alpha.  
- R4_age: if same label and not accept -> increase M and decrease M+, ant with smaller age loses label.
   **Reasoning**: Ants with older age had more meetings and their parameters should be better tuned than an younger ant, so we take preference on the older ants.
- R4_stability: Label Stability Reinforcement: Add stability measurement of ant, whenever ant change lables, decrease it. When two ants with the same label reject, the one with less stability loses label.


- [Applied to AntClust] Ant dropout (based on dropout from machine learning): with a certain probability reset a few ants (remove labels, set estimators and age to zero). Added limitations to dropout (only x times maximum per run, not happen after the X iteration (to give time for ants to re-meet))
   **Reasoning**: By removing some ants a few times in the process it can remove biases from ants that clsutered in the beginning.
-[Applied to AntClust] Threshold Adjustment (dynamic template): Multiply template by a sigmoid of age, so that younger ants have a tempalte that accepts more ants. adjust the similarity threshold based on the age of the ants.


Tests: test new rules features individually
[x] Boosted increase
[x] Cluster avg
[x] Age comparisson
[ ] Dropout

Rule set:
- [x] Labroche (R1,R2,R3,R4,R5,R6).
- [x] Labroche_age_penalty (R1,R2,R3,R4_age,R5_age,R6)
- [x] Labroche_carvalho (R1,R2,R3_boost,R4,R5,R5_new, R5_merge, R6)
- [x] Labroche_carvalho_age_penalty (R1,R2,R3_boost,R4_age,R5_age,R6)
  
- [x] Labroche_dropout (R1,R2,R3,R4,R5,R6)(dropout)
- [x] Labroche_age_penalty_dropout (R1,R2,R3,R4_age,R5_age,R6)(dropout)
- [x] Labroche_carvalho_dropout (R1,R2,R3_boost,R4,R5,R5_new, R5_merge, R6)(dropout)
- [x] Labroche_carvalho_age_penalty_dropout (R1,R2,R3_boost,R4_age,R5_age,R6)(dropout)

Result comparison between all the rulesets (ARI):
|                                    Ruleset | 2 cluster(s) | 3 cluster(s) | 4 cluster(s) | 5 cluster(s) | 6 cluster(s) | 7 cluster(s) | 8 cluster(s) | 9 cluster(s) | 10 cluster(s) |
|-------------------------------------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|--------------:|
|                                   Labroche |     0.974555 |    1.0 |     1.000000 |     0.994751 |     0.918541 |     0.739739 |     0.547493 |     0.441381 |      0.407015 |
|                       Labroche Age Penalty |     0.934459 |    1.0 |     0.995550 |     0.970565 |     0.871186 |     0.660000 |     0.463374 |     0.326845 |      0.338013 |
|                          Labroche Carvalho |     0.954328 |    1.0 |     1.000000 |     0.991781 |     0.855371 |     0.608393 |     0.433169 |     0.330848 |      0.359873 |
|              Labroche Carvalho Age Penalty |     0.887585 |    1.0 |     1.000000 |     0.980150 |     0.818053 |     0.497506 |     0.335551 |     0.272397 |      0.297039 |
|                      Labroche With Dropout |     1.000000 |    1.0 |     1.000000 |     0.995508 |     0.907568 |     0.737671 |     0.480289 |     0.421618 |      0.414740 |
|         Labroche Age Penalty With Dropout  |     1.000000 |    1.0 |     0.989234 |     0.947809 |     0.852314 |     0.686064 |     0.386732 |     0.312168 |      0.333771 |
|             Labroche Carvalho With Dropout |     0.965240 |    1.0 |     1.000000 |     0.987436 |     0.835141 |     0.647994 |     0.414341 |     0.321063 |      0.310649 |
| Labroche Carvalho Age Penalty With Dropout |     0.991585 |    1.0 |     0.995627 |     0.966568 |     0.747813 |     0.541202 |     0.319363 |     0.292444 |      0.255851 |
Labroche Carvalho Age Penalty With Dropout |       0.991585 |    1.0 |     0.995627 |     0.966568 |     0.747813 |     0.541202 |     0.319363 |     0.292444 |      0.255851 |
GA Ruleset	                              |        0.907564	|    0.993639| 0.967010	|     0.939551|	   0.804795	|     0.602781	|     0.423630	|     0.437685	|     0.351389  |

Best result per cluster:
- 2: 1.0 (Labroche_dropout_True, Labroche_Age_Penalty_dropout_True)	
- 3: 1.0 (all)
- 4: 1.0 (Labroche_dropout_False, Labroche_Carvalho_dropout_False, Labroche_Carvalho_Age_Penalty_dropout_False, Labroche_dropout_True, Labroche_Carvalho_dropout_True)
- 5: 0.995508 (Labroche_dropout_True)
- 6: 0.918541 (Labroche_dropout_False)
- 7: 0.739739 (Labroche_dropout_False)
- 8: 0.547493 (Labroche_dropout_False)
- 9: 0.441381 (Labroche_dropout_False)
- 10: 0.414740 (Labroche_dropout_True)


Mean ARI per cluster:

Labroche_dropout_False                         0.780386
Labroche_Age_Penalty_dropout_False             0.728888
Labroche_Carvalho_dropout_False                0.725974
Labroche_Carvalho_Age_Penalty_dropout_False    0.676476
Labroche_dropout_True                          0.773044
Labroche_Age_Penalty_dropout_True              0.723121
Labroche_Carvalho_dropout_True                 0.720207
Labroche_Carvalho_Age_Penalty_dropout_True     0.678939
GA Ruleset                                     0.714227

# Idea 8.2: Apply GA to ruleset (again)
Set up better way to generate rulesets at "random":
-  Design a dictionary where keys are the conditions (R1_condiiton, R2_condition, etc) and values are a list of conseguences (R3,R3_BOOST), and a list of tuple is generated with the Rule and condition pair. 
-  Condition and action name is retrived by the class to call corresponding function.
-  Crossover point from R3 onwards (since R1 and R2 only have one possible consequence).
-  Fitnessfunction measures the mean ARI for different tests, ranging from min and max number of clusters and min and max values per cluster.
-  GA runs for number of generations or until best fintess is equal or higher than stopping criteria.

Running the fitess funtion with the following parameters:
- clusters_min = 2
- clusters_max = 10
- values_per_cluster_min = 3
- values_per_cluster_max = 5
- Pop size = 15
- Generations = 10
- Stopping criteria = 0.8

**GA rule:**
['R1', 'R2', 'R3', 'R4_YOUNG', 'R5_STABILITY'], Dropout: True, Dynamic template: False


The best scenario would be creating a continuous rule search space so that the GA can make good use of this.

# Ideia 9: GA and PSO for hyperparameters
[x] Use GA results from previous work
[x] Run PSO for hyperparameters:
PSO parameters: pop_size=10, dimensions=4, generations=10, w=0.75591797, c1=0.75660476, c2=0.82302211
GA parameters: pop_size=10, generations=10, mutation_rate=0.2

**Result**
AntClust parameters fixed: {"alpha": 500, "betta": 0.9, "shrink": 0.2, "removal": 0.3}
AntClust parameters found by GA: {'alpha': 372, 'betta': 0.951, 'shrink': 0.466, 'removal': 0.320}
AntClust parameters found by PSO: {'alpha': 485, 'betta': 1.135, 'shrink': 0.193, 'removal': 0.285}

| Num clusters | Fixed ari | GA ari   | PSO ari  | Best ari  |
|--------------|-----------|----------|----------|-----------|
| 2            | 0.952589  | 0.975695 | 0.973034 | GA        |
| 3            | 1.000000  | 0.997913 | 1.000000 | Fixed/PSO |
| 4            | 1.000000  | 1.000000 | 1.000000 | Equal     |
| 5            | 0.992561  | 0.993272 | 0.994378 | PSO       |
| 6            | 0.849935  | 0.861555 | 0.893287 | PSO       |
| 7            | 0.643491  | 0.726394 | 0.745864 | PSO       |
| 8            | 0.630180  | 0.550109 | 0.554702 | Fixed     |
| 9            | 0.447469  | 0.431815 | 0.459544 | PSO       |
| 10           | 0.414322  | 0.433749 | 0.437076 | PSO       |

Using bioinspired methods can increase performance of antclust. This works for the hypeparameters in a different way from the rulesets because of the fitness landscape, since we're dealing with only numbers it's easier for the individuals to go trhough the landscape until finding the optimization. As in the ruleset it's much more complex and there is no smooth transition between two individuals since we're dealing with categorical individuals.


# Idea 10: maybe develop some sort of module for visualization of real time clustering.
- Modified rulesets to return name of rule.
- Modified AntClust to update clusters labels and store list of labels and rule applied for every meeting.
- Created a library to visualize it.

**Result**
Able to visualize the rules applied the entire run, except for the shrinking stage. Also allow to see rules applied in individual graphs. 
  
# Idea 11: rerun the opencv notebook for VeRi with cosine similarity function

**Entire dataset**
Run the antclust with cosine similarity for the images of cars in different views (multiple angles) and compare with base results.
visualize some clusters as well.
Results for the ARI are bad, but when visualizing the clusters it shows to be able to cluster different views of a single vehicle, maybe this could be used combined with the ORB visualization rather than completely replacing.

**Limited angles**
Run the antclust with cosine similarity for the images of cars in unique views (only front/back angle)
- Test done by winfried since antclust with ORB similarity had problems with identifiying different angles of the same car as similar. 
- Results: improved for fewer clusters, is worse for larger number of clusters.
- The combination of two or more similarity functions for images could be the key for a better comparison.


# Idea 12: apply antclust for ethics decision
- Antclust could be restructured so it a action is proposed and the ants goal is to decide wheter this action is good or bad.
- Could encode the word/sentence with a sentimental analysis. such as 1 = good, -1 = bad. (words like= kill, hurt, steal, hug, love, help) and normalize so that is between 0 and 1.
- Initalize template at random with gaussian distibution mean 0.5, std deviation 0.4.
- If word below template = bad, if above = good.
- Acceptance = both ants have same label.
- Take majority label. 

**Ethical Ant**
- M becomes the estimator of how successful the ant is
- Is initialized with the label alread (1 = for good, 0 for bad) based on sentimental analysis value of word/sentence.

**Ethical AntClust**
- Removed similairty function, acceptance becomes only if both antes have same label.
- Fit function only has the meeting phase, no need to remove unlabeled ants.

**Ethical Rule**
- If acceptance: increase both ant's M
- if Reject: ant with smaller M gets the other's label and decreased M.

**Results**:
- Ethics is tighly close to sentiment for the human being, but for computers that's a stretch. 
- Sentiment analysis just calculates the weights of labaled "positive" or "negative" words. So if a unetchical sentence is written without words that individually spark negative sentiment, it will end up being label as a good.
- The idea works well when dealing with single words, like kill, hate, steal, hurt will be always classified as bad due to their strong negative sentment. While words like love, kiss, hug, help, will always be classified as good.
- This is not enough to decide if a sentece describing an action is ethical or not, since it's possible to generate unethical actions without using negative words.
- There are datasets with ethical results, but not standardized by any sort of ethical authority. It seems to be handmade.
  - https://github.com/hendrycks/ethics which gives binary values for sentences, 0 being acceptable, 1 being not acceptable.
  - https://paperswithcode.com/dataset/ethics-2 a russian handmade ethics dataset, whihc includes fields like vitue, moral, justice, laws and utilitarianism (always 1 or 0). This is interesting but since it addresses laws and rules it's subjetive to a country, in this case russia, so using this in a different context could yeld biased results.
- There needs to be the appeal of big organizations to define a dataset of ethical regulations, so it could be used by AI community, not only to train the models with it, but to force AI to follow such guidelines.

- https://ieeexplore.ieee.org/document/10316130


