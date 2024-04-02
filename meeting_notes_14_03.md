What's been done:
- Applied antclust to image tensors (cifar10 and MINST using cosine similarity)
- Applied Antclust to text data (cosine similarity as well)
- Applied to numerical data (n dimesnion euclidean dist)
- Apllied to categorical data
- Applied to timeseries data
- Applied to spatial data (didn't work)
- Tested new rulesets
- Used GA to new ruleset combinations.With new Rules:
  - R3_BOOST (same label and accept): if one of the ants have higher M+ than avg in it's colony, the other ant's M+ is increased by double the factor.
  - R4_YOUNG (same label and reject): younger ant loses label. 
  - R4_OLD (same label and reject): older ant loses label.
  - R4_STABILITY (same label and reject): add a stability parameter to ant, that decreases everytime it changes label (except for the first time), ant with smaller stability loses label.
  - R5_YOUNG (different label and accept): younger ant gets older ant's label. 
  - R5_OLD (different label and accept): older ant gets younger ant's label. 
  - R5_MERGE (different label and accept): if both ants have M+ higher than avg in their clsuters, merge entire cluster.
  - R5_NEW (different label and accept): if both ants have lower M+ than avg in their clusters, create new label and assign them both to it.
  - R5_STABILITY (different label and accept): ant with smaller stability gets other ant's label.
- Running VeRi dataset with cosine similarity.
- Thought of visualization somehow but still unclear.

Things that I need advice with
From Winfried:
- From the conclusion, mentioned neural nets or other means to create new rulesets, is there any concrete recommendation that you saw during your research?

From Parisa:
- What does she think of the amount of work?
- What else could i explore?
- Is there a template for report?

GOOD POINT but new things:
- Add GA for hyperparameters from seminar into this project
- test PSO for the hyperparameters
- test data from winfried
- rerun GA because of bug
- visualization
- ANTCLUST FOR ETHICS?