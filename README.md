# parallel-recommenders
APP Recommender Systems Project

## Compilation Instructions

`cmake CMakeLists.txt`

OR

Just use [Jetbrains CLion](https://www.jetbrains.com/clion/). One of the best IDEs. Free for students.

## Usage

`parallel-recommenders <path-to-rating-file> <path-to-movie-file> <delimiter> <user-id-file> <n>`

* specify path to a valid ratings file.
* specify path to a valid movies file.
* delimiter used by these files.
* sample valid data files [here](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip).
* specify path to file with IDs of the users we need to make recommendations for.
* n is the number of top recommendations to return.

For example, `parallel_recommenders data/ratings-20k.dat data/movies.dat :: user-ids.txt 10` yields..

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


/*** Output file "kernelRecommendations_8.csv" which stores recommendations for user 8 ***/
| MovieID |             Movie Name              | Rating|
|------------|--------------------------------------- |----------|
|   5022    |       Servant, The (1963)        | 6.034  |
|  26495   |        Yellowbeard (1983)        | 6.034  |
|   3601    | Castaway Cowboy, The (1974)| 5.596|
|   3642    |     In Old California (1942)     |  5.596 |
|   3073    |     Sandpiper, The (1965)      |  5.596 |
|   2820    |            Hamlet (1964)            |   5.596 |
|    975     | Something to Sing About (1937)  |    5.596   |
|    790     | Unforgettable Summer, An (Un été inoubliable) (1994)| 5.596    |
|   3154    |    Blood on the Sun (1945)    |    5.596....|
|    967     |    Outlaw, The (1943)            |   5.596   |
