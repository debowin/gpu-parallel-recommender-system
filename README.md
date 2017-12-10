# parallel-recommenders
APP Recommender Systems Project

## Compilation Instructions

`cmake CMakeLists.txt`

OR

Just use [Jetbrains CLion](https://www.jetbrains.com/clion/). One of the best IDEs. Free for students.

## Usage

`parallel-recommenders <path-to-rating-file> <path-to-movie-file> <delimiter> <user-id> <n>`

* specify path to a valid ratings file.
* specify path to a valid movies file.
* delimiter used by these files.
* sample valid data files [here](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip).
* user-id is the ID of the user we're making recommendations for.
* n is the number of top recommendations to return.

For example, `parallel_recommenders data/ratings.csv data/movies.csv , user-ids.txt 10` yields..

| MovieID | Movie Name | Rating|
|--------|-------------------------------|-------------|
|116     |  Anne Frank Remembered (1995) |       4.990 |
|121231  |  It Follows (2014)            |       4.972 |
|116897  |  Wild Tales (2014)            |       4.954 |
|3989    |  One Day in September (1999)  |       4.875 |
|6732    |  Hello, Dolly! (1969)         |       4.838 |
|7566    | 28 Up (1985)                  |       4.836 |
|52767   | 21 Up (1977)                  |       4.836 |
|8264    | Grey Gardens (1975)           |       4.789 |
|146656  | Creed (2015)                  |       4.780 |
|3943    | Bamboozled (2000)             |       4.745 |