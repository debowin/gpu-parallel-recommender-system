# parallel-recommenders
APP Recommender Systems Project

## Compilation Instructions

make

## Usage

`./parallel-recommenders <path-to-rating-file> <path-to-movie-file> <delimiter> <user-id-file> <n>`

* specify path to a valid ratings file.
* specify path to a valid movies file.
* delimiter used by these files.
* sample valid data files [here](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip).
* specify path to file with IDs of the users we need to make recommendations for.
* n is the number of top recommendations to return.

For example, `./parallel_recommenders data/ratings-20k.dat data/movies.dat :: user-ids.txt 10` yields..

```
/*** Console Output ***/
 
 Normalizing Ratings took 0.010721 seconds.

 Computing UU Similarity - Gold
 Took 169.373 seconds.

 Calculating Top-10 Recommendations for 10 users - Gold.
 Took 6.86971 seconds for 10 users.

 Computing UU Similarity - Parallel
 Took 4.26382 seconds.
 Error %age: 0.32
 Similarity Kernel Result Verification: SUCCESS

 Calculating Top-10 Recommendations for 10 users - Parallel.
 Took 0.035726 seconds for 10 users.

 Total Kernel Time: 4.299544

 Similarity Speedup: 39.723209
 Recommendation Speedup: 192.288895
 Total Speedup: 40.990917
 
 Error %age: 0
 Recommendations Kernel Result Verification: SUCCESS
 ```

** Recommendations Output File "kernelRecommendations_3.csv" **

| Movie Id | Movie Title | Score |
|----------|---------------------------------------------------------------------|-------|
| 52694 | Mr. Bean's Holiday (2007) | 6.358 |
| 26246 | Johnny Got His Gun (1971) | 6.179 |
| 6255 | Bible, The (a.k.a. Bible... In the Beginning, The) (1966) | 5.990 |
| 6237 | Glenn Miller Story, The (1953) | 5.990 |
| 5960 | Bad Influence (1990) | 5.932 |
| 5389 | Spirit | 5.917 |
| 27509 | Carolina (2005) | 5.858 |
| 50005 | Curse of the Golden Flower (Man cheng jin dai huang jin jia) (2006) | 5.858 |
| 26122 | Onibaba (1964) | 5.858 |
| 34198 | Russian Dolls (Les PoupĂŠes russes) (2005) | 5.858 |
