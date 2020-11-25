# Recommender Systems with Python
Let's get started!

# Import Libraries



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
```

# Get the Data


```python
movies = pd.read_csv(r"C:\Users\Sid Saliyam\Downloads\archive\movies.csv")
```


```python
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
tags = pd.read_csv(r"C:\Users\Sid Saliyam\Downloads\archive\tags.csv")
```


```python
tags.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>tag</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>60756</td>
      <td>funny</td>
      <td>1445714994</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>60756</td>
      <td>Highly quotable</td>
      <td>1445714996</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>60756</td>
      <td>will ferrell</td>
      <td>1445714992</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>89774</td>
      <td>Boxing story</td>
      <td>1445715207</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>89774</td>
      <td>MMA</td>
      <td>1445715200</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings = pd.read_csv(r"C:\Users\Sid Saliyam\Downloads\archive\ratings.csv")
```


```python
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>



We can merge them together:


```python
movies = pd.merge(movies,ratings,on='movieId')
```


```python
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>5</td>
      <td>4.0</td>
      <td>847434962</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>7</td>
      <td>4.5</td>
      <td>1106635946</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>15</td>
      <td>2.5</td>
      <td>1510577970</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>17</td>
      <td>4.5</td>
      <td>1305696483</td>
    </tr>
  </tbody>
</table>
</div>



# EDA

Let's explore the data a bit and get a look at some of the best rated movies.

# Visualization Imports


```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline
```

Let's create a ratings dataframe with average rating and number of ratings:


```python
movies.groupby('title')['rating'].mean().sort_values(ascending=False).head()
```




    title
    Karlson Returns (1970)                           5.0
    Winter in Prostokvashino (1984)                  5.0
    My Love (2006)                                   5.0
    Sorority House Massacre II (1990)                5.0
    Winnie the Pooh and the Day of Concern (1972)    5.0
    Name: rating, dtype: float64




```python
movies.groupby('title')['rating'].count().sort_values(ascending=False).head()
```




    title
    Forrest Gump (1994)                 329
    Shawshank Redemption, The (1994)    317
    Pulp Fiction (1994)                 307
    Silence of the Lambs, The (1991)    279
    Matrix, The (1999)                  278
    Name: rating, dtype: int64




```python
ratings = pd.DataFrame(movies.groupby('title')['rating'].mean())
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>'71 (2014)</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>'Hellboy': The Seeds of Creation (2004)</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>'Round Midnight (1986)</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>'Salem's Lot (2004)</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>'Til There Was You (1997)</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



Now set the number of ratings column:


```python
ratings['num of ratings'] = pd.DataFrame(movies.groupby('title')['rating'].count())
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>num of ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>'71 (2014)</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>'Hellboy': The Seeds of Creation (2004)</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>'Round Midnight (1986)</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <td>'Salem's Lot (2004)</td>
      <td>5.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>'Til There Was You (1997)</td>
      <td>4.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Now a few histograms:


```python
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c2bc1443c8>




![png](output_21_1.png)



```python
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c2bc241a08>




![png](output_22_1.png)



```python
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
```




    <seaborn.axisgrid.JointGrid at 0x1c2bc65ed88>




![png](output_23_1.png)


Okay! Now that we have a general idea of what the data looks like, let's move on to creating a simple recommendation system:

# Recommending Similar Movies

Now let's create a matrix that has the user ids on one access and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie. Note there will be a lot of NaN values, because most people have not seen most of the movies.


```python
moviemat = movies.pivot_table(index='userId',columns='title',values='rating')
moviemat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>title</th>
      <th>'71 (2014)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'Tis the Season for Love (2015)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
      <th>*batteries not included (1987)</th>
      <th>...</th>
      <th>Zulu (2013)</th>
      <th>[REC] (2007)</th>
      <th>[REC]² (2009)</th>
      <th>[REC]³ 3 Génesis (2012)</th>
      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>
      <th>eXistenZ (1999)</th>
      <th>xXx (2002)</th>
      <th>xXx: State of the Union (2005)</th>
      <th>¡Three Amigos! (1986)</th>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9719 columns</p>
</div>



Most rated movie:


```python
ratings.sort_values('num of ratings',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>num of ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Forrest Gump (1994)</td>
      <td>4.164134</td>
      <td>329</td>
    </tr>
    <tr>
      <td>Shawshank Redemption, The (1994)</td>
      <td>4.429022</td>
      <td>317</td>
    </tr>
    <tr>
      <td>Pulp Fiction (1994)</td>
      <td>4.197068</td>
      <td>307</td>
    </tr>
    <tr>
      <td>Silence of the Lambs, The (1991)</td>
      <td>4.161290</td>
      <td>279</td>
    </tr>
    <tr>
      <td>Matrix, The (1999)</td>
      <td>4.192446</td>
      <td>278</td>
    </tr>
    <tr>
      <td>Star Wars: Episode IV - A New Hope (1977)</td>
      <td>4.231076</td>
      <td>251</td>
    </tr>
    <tr>
      <td>Jurassic Park (1993)</td>
      <td>3.750000</td>
      <td>238</td>
    </tr>
    <tr>
      <td>Braveheart (1995)</td>
      <td>4.031646</td>
      <td>237</td>
    </tr>
    <tr>
      <td>Terminator 2: Judgment Day (1991)</td>
      <td>3.970982</td>
      <td>224</td>
    </tr>
    <tr>
      <td>Schindler's List (1993)</td>
      <td>4.225000</td>
      <td>220</td>
    </tr>
  </tbody>
</table>
</div>



Let's choose a Movie : Braveheart,a war/drama movie.


```python
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>num of ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>'71 (2014)</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>'Hellboy': The Seeds of Creation (2004)</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>'Round Midnight (1986)</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <td>'Salem's Lot (2004)</td>
      <td>5.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>'Til There Was You (1997)</td>
      <td>4.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Now let's grab the user ratings for that movie.


```python
Braveheart_user_ratings = moviemat['Braveheart (1995)']
Braveheart_user_ratings.head()
```




    userId
    1    4.0
    2    NaN
    3    NaN
    4    NaN
    5    4.0
    Name: Braveheart (1995), dtype: float64




```python
similar_to_Braveheart = moviemat.corrwith(Braveheart_user_ratings)
```

    C:\Users\Sid Saliyam\Anaconda3\lib\site-packages\numpy\lib\function_base.py:2522: RuntimeWarning: Degrees of freedom <= 0 for slice
      c = cov(x, y, rowvar)
    C:\Users\Sid Saliyam\Anaconda3\lib\site-packages\numpy\lib\function_base.py:2451: RuntimeWarning: divide by zero encountered in true_divide
      c *= np.true_divide(1, fact)
    

Let's clean this by removing NaN values and using a DataFrame instead of a series:


```python
corr_Braveheart = pd.DataFrame(similar_to_Braveheart,columns=['Correlation'])
corr_Braveheart.dropna(inplace=True)
corr_Braveheart.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>'burbs, The (1989)</td>
      <td>0.332504</td>
    </tr>
    <tr>
      <td>(500) Days of Summer (2009)</td>
      <td>0.021388</td>
    </tr>
    <tr>
      <td>*batteries not included (1987)</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <td>...And Justice for All (1979)</td>
      <td>0.327327</td>
    </tr>
    <tr>
      <td>10 Cloverfield Lane (2016)</td>
      <td>0.534522</td>
    </tr>
  </tbody>
</table>
</div>



Now if we sort the dataframe by correlation, we should get the most similar movies, however note that we get some results that don't really make sense. This is because there are a lot of movies only watched once by users who also watched Braveheart (it was the most popular movie).


```python
corr_Braveheart.sort_values('Correlation',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sisters (2015)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Class, The (Klass) (2007)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Ulee's Gold (1997)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Say It Isn't So (2001)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Savannah Smiles (1982)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Chasers (1994)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Children of Dune (2003)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Living Out Loud (1998)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Underworld (1996)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Furious 7 (2015)</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Let's fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier).


```python
corr_Braveheart = corr_Braveheart.join(ratings['num of ratings'])
corr_Braveheart.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
      <th>num of ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>'burbs, The (1989)</td>
      <td>0.332504</td>
      <td>17</td>
    </tr>
    <tr>
      <td>(500) Days of Summer (2009)</td>
      <td>0.021388</td>
      <td>42</td>
    </tr>
    <tr>
      <td>*batteries not included (1987)</td>
      <td>-1.000000</td>
      <td>7</td>
    </tr>
    <tr>
      <td>...And Justice for All (1979)</td>
      <td>0.327327</td>
      <td>3</td>
    </tr>
    <tr>
      <td>10 Cloverfield Lane (2016)</td>
      <td>0.534522</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>



Now sort the values and notice how the titles make a lot more sense:


```python
corr_Braveheart[corr_Braveheart['num of ratings']>100].sort_values('Correlation',ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation</th>
      <th>num of ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Braveheart (1995)</td>
      <td>1.000000</td>
      <td>237</td>
    </tr>
    <tr>
      <td>Batman Begins (2005)</td>
      <td>0.610550</td>
      <td>116</td>
    </tr>
    <tr>
      <td>Ocean's Eleven (2001)</td>
      <td>0.575751</td>
      <td>119</td>
    </tr>
    <tr>
      <td>Inception (2010)</td>
      <td>0.555414</td>
      <td>143</td>
    </tr>
    <tr>
      <td>Matrix, The (1999)</td>
      <td>0.496045</td>
      <td>278</td>
    </tr>
  </tbody>
</table>
</div>



# Thank You!
