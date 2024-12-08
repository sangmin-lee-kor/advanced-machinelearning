import json
import numpy as np
import pandas as pd
import fire


def preprocess(input_path="data/ml-latest-small", output_path="data/train", instruction_list=None) :
    ratings_file_path = input_path + "/ratings.csv"
    movies_file_path = input_path + "/movies.csv"
    tags_file_path = input_path + "/tags.csv"


    ratings_df = pd.read_csv(ratings_file_path)
    movies_df = pd.read_csv(movies_file_path)
    tags_df = pd.read_csv(tags_file_path)

    merged_df = pd.merge(ratings_df, movies_df, on="movieId")

    tmp = tags_df.groupby("movieId", as_index=False).agg({"tag": " | ".join})
    merged_df = pd.merge(merged_df, tmp, how="left" , on="movieId")

    merged_df = merged_df.sort_values(["userId", "timestamp"])

    def exclude_last_title(titles):
        return titles[:-1]

    def _last_title(titles):
        return titles[-1]
    
    tmp = merged_df.groupby("userId")["title"].apply(list).apply(exclude_last_title).apply(lambda x: ", ".join(x)).reset_index()
    input_df = tmp[tmp["title"].apply(len) > 0]
    input_df.rename(columns={"title":"input"},inplace=True)

    output_df = merged_df.groupby("userId")["title"].apply(list).apply(_last_title).reset_index()
    output_df.rename(columns={"title":"output"},inplace=True)

    if instruction_list == None :
        instruction_list = [
            "Given the sequence of movies watched, can you suggest the next movie the user might enjoy?",
            "Based on the listed movies, recommend the next logical movie for the user to watch.",
            "What would be a great follow-up movie for the user to continue their journey?",
            "Considering the user's movie history, what is the best next movie to watch?",
            "From the watched movies below, predict the next movie the user might like.",
            "What should be the user's next movie to watch, based on the previous titles?",
            "Given these movie titles, suggest a movie that fits the user's preferences.",
            "Using the movies listed, recommend the most fitting next movie for the user.",
            "Looking at the movies the user has seen, what is the next must-watch film?",
            "Based on this sequence of movies, what would be the next movie to recommend?",
            ]
        
    train_df = input_df.merge(output_df, how="inner")
    train_df["instruction"] = np.random.choice(instruction_list, size=len(train_df))

    train_df.drop(columns=["userId"], inplace=True)

    with open(output_path+"/train_data.json", "w") as f:
        json.dump(train_df.to_dict(orient="records"), f)


if __name__ == "__main__":
    fire.Fire(preprocess)