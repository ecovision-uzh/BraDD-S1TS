# DEFORESTATION DETECTION IN THE AMAZON WITH SENTINEL-1 SAR IMAGE TIME SERIES

This is the official implementation of the paper:  
- Deforestation Detection in the Amazon with Sentinel-1 SAR Image Time Series (ISPRS 23)

## Abstract

Deforestation has a significant impact on the environment, accelerating global warming and causing irreversible damage to ecosystems. Large-scale deforestation monitoring techniques still mostly rely on statistical approaches and traditional machine learning models applied to multi-spectral, optical satellite imagery and meta-data like land cover maps. However, clouds often obstruct observations of land in optical satellite imagery, especially in the tropics, which limits their effectiveness. Moreover, statistical approaches and traditional machine learning methods may not capture the wide range of underlying distributions in deforestation data due to limited model capacity. To overcome these drawbacks, we apply an attention-based neural network architecture that learns to detect deforestation end-to-end from time series of synthetic aperture radar (SAR) images. Sentinel-1 C-Band SAR data are mostly independent of the weather conditions and our trained neural network model generalizes across a wide range of deforestation patterns of Amazon forests. We curate a new dataset, called BraDD-S1TS, comprising approximately 25,000 image sequences for deforested and unchanged land throughout the Brazilian Amazon. We experimentally evaluate our method on this dataset and compare it to state-of-the-art approaches. We find it outperforms still-in-use methods by 13.7 percentage points in intersection over union (IoU). We make BraDD-S1TS publicly available along with this publication to serve as a novel testbed for comparing different deforestation detection methods in future studies. 

# Setup
## Conda Environment

## Dataset (BraDD-S1TS)

# Experiments
