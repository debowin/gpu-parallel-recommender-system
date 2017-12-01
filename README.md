# parallel-recommenders
APP Recommender Systems Project

## Compilation Instructions

`cmake CMakeLists.txt`

OR

Just use [Jetbrains CLion](https://www.jetbrains.com/clion/). One of the best IDEs. Free for students.

## Usage

`parallel-recommenders <path-to-rating-csv> <path-to-movie-csv> <user-id> <n>`

* specify path to a valid ratings CSV file.
* specify path to a valid movies CSV file.
* sample valid CSV files [here](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip).
* user-id is the ID of the user we're making recommendations for.
* n is the number of top recommendations to return.
