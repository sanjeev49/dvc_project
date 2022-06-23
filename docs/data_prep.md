# Data Preparation Stage 

- Convert my data into train.tsv and test.tsv int 70:30 ratio

```
data.xml
    |-train.tsv
    |-test.tsv
```

- We are choosing only three tags in the xml data - 1. row id, 2. title and body , 3. Tags
(stackoverflow tags specific to python)

|Tags|features names|
|-|-|
|row ID|row ID|
|stackoverflow tags|Label - Python|