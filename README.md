## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![318691387-9a445ed3-f79e-46ed-8493-a0138abde135](https://github.com/user-attachments/assets/8bb39013-83bf-4c98-8727-eba69330bc23)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![318692227-c5ae2314-6f2b-4d93-92b3-f44d1b74015a](https://github.com/user-attachments/assets/7bc1cfd3-879b-4ac2-8532-bf99decc10c8)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![318692322-4ae17d2a-aa22-4340-9faf-8567549250f6](https://github.com/user-attachments/assets/8def72b2-ba32-4b90-9d06-fd1798a9024e)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![318692437-2249ccf3-4a16-462b-b745-677312c7fd42](https://github.com/user-attachments/assets/81e380b8-3858-4861-91e2-02aa84300c7c)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![318692763-d2714505-ceae-48c6-b428-fc421aaa735d](https://github.com/user-attachments/assets/aefb68c2-45fd-4452-a21b-6d81f3797ea1)

```
df2=pd.concat([df2,enc],axis=1)
df2
```
![318692827-b4b4c5b2-9bc8-4f41-8649-096999696847](https://github.com/user-attachments/assets/c1149d74-a026-4248-9988-51af45f5aac2)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![318692921-e56e11b0-9489-41a5-973c-e32fca8f9840](https://github.com/user-attachments/assets/f46f2008-d620-421e-9bc7-855457fe6f9a)

```
pip install --upgrade category_encoders
```
![318693032-0711d42f-4456-4222-8334-f183bc7c2385](https://github.com/user-attachments/assets/59346de1-80c7-4dd8-9565-3f4ecfd3f08d)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![318693230-3d2f8b4c-0ffc-4754-8c1b-ad637c727c9b](https://github.com/user-attachments/assets/a128ca80-d348-4b08-9b2b-81e091dd2158)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```

![318897767-781ddd71-1fc6-499b-9234-b83778405580](https://github.com/user-attachments/assets/9716bd8f-254d-4a07-8c3e-baee4290be91)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![318897871-6f1877a4-9ba9-45d6-8df2-38fdc103a0ef](https://github.com/user-attachments/assets/64563060-6647-479e-b77c-52db635939dc)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![318897982-63cbb12a-e9eb-447e-855a-e56c706bbfa9](https://github.com/user-attachments/assets/4e56613c-2244-4ca5-831c-f907f893ded2)

```
df.skew()
```
![318898092-3d04bbce-76dc-4571-8c8d-5aad234c1766](https://github.com/user-attachments/assets/26ce9e2b-5643-478b-a69b-9b649e6b5cf6)

```
np.log(df["Highly Positive Skew"])
```
![318898189-7247340c-6488-4b75-9deb-0ad3f10e03fd](https://github.com/user-attachments/assets/dc1f0980-e5ad-45c3-a141-914a9bdf1c43)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![318898261-71ae0399-a828-406a-93a6-0e36cc31e249](https://github.com/user-attachments/assets/acce6a0f-7b4c-40f1-b7e9-07de8fbb54d5)

```
np.sqrt(df["Highly Positive Skew"])
```
![318898327-9b500fd0-9b55-4397-b1e8-364652aca983](https://github.com/user-attachments/assets/86c4b061-b9ed-4bde-9f42-4808e6226feb)

```
np.square(df["Highly Positive Skew"])
```

![318898423-d243323b-c97e-4c55-a41f-f76d176e6461](https://github.com/user-attachments/assets/2753f56d-ed3e-470d-8441-b431c6088e1e)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![318898927-4945b8c6-e27d-4526-9032-0c0aeb9ab576](https://github.com/user-attachments/assets/e860384e-af17-412e-8b76-5d91da4f8a19)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![318899248-52a7553c-c1bd-4489-a0cb-b13a27684c23](https://github.com/user-attachments/assets/ac877e72-8459-4494-b189-69b51c16792d)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
![318899545-3688ed78-4920-4cd4-9e33-4420fc790b8d](https://github.com/user-attachments/assets/7340464e-5b61-4a25-a090-a0f00f01d38b)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![318899696-9ef5152c-d766-48e1-857c-a7dbfde4e648](https://github.com/user-attachments/assets/aed4b036-5771-4b3f-a4b6-7d47327f2d6d)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![318899799-fde4b296-88ec-46ad-b6f3-2cf2b64a15f2](https://github.com/user-attachments/assets/3c3a03da-2cf3-44e9-9e48-a52de2018d8d)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![318899874-57bae70b-8ee0-4ab1-86bf-733d2597089d](https://github.com/user-attachments/assets/bb9059c7-d4fe-4b2d-a433-1a62ce56e770)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![318900112-3987a28b-3816-41b2-9a9d-6a1cedf8382e](https://github.com/user-attachments/assets/855f057d-9106-4588-ab8a-11aeb8ad440e)

# RESULT:
       
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
