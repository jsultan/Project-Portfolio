# WordCloud Creator and Text Analysis


This project was one of the first I undertook to further my understand of R, as well as understanding basic text analysis. 
This script creates some user defined functions which form word clouds and histograms of the most used words in any file. In this case, I used the function on the three main canonical literature pieces of the Abrahamic faiths (Old Testament, New Testament, and Quran)
Any type of data can be used as long as the text is in one column, so data preprocessing is necessary in order for the functions to work. The created functions will create an object called a corpus which has certain mapping functions that can clean up the text (i.e remove punctuation, remove capitalization, and trailing/leading whitespace) as well as remove commonly used words known as stop words.
More stop words can be added through to the script to account of cultural or linguistic differences. Since I used the King James version of the Bible (both Old and New Testament), quite a bit of words had to be added to an already large list of stop words. 

Here are the results from the three books:

## Old Testament 
![]({{site.baseurl}}/Old Testament.png)

## New Testament 
![]({{site.baseurl}}/New Testament.png)

## Quran 
![]({{site.baseurl}}/Quran.png)
