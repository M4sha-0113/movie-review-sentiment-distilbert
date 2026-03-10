You can use the system to determine the sentiment of any review - to understand whether it is positive or negative.
1) Go to the “distilBERT/scripts” directory and run the following command:
python3 use_model.py
(make sure that all requirements are installed: pip install -r ./config/requirements.txt). 
2) The program will then ask you to enter the file name of a review (including its extension). You can provide any review file that you want to analyze 
(as an example you can use film_review.txt and film_review2.txt, which are already in this folder).
The result will appear in the console, indicating whether the review is positive or negative.
------

In the 'scripts' folder, you can also find files that contain the results of the model’s evaluation on different datasets that I previously ran - test_results (for stanford dataset, imdb, rotten tomatoes and amazon). 

I’ve excluded the datasets from GitHub because they exceed the file size limit. 
You can access and download them here: https://drive.google.com/drive/folders/1nCmRpyAtbnwV0l-rj9gsu4uIMJYp6tf9?usp=drive_link
