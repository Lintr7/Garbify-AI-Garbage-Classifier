# Welcome to Garbify! ♻️📸🌱
---
## Garbify is an AI powered garbage classifier that takes a picture input (png, jpg, or jpeg) and classifies it into one of 6 categories: 
1. Plastic
2. Metal
3. Glass
4. Paper
5. Cardboard
6. Trash

## Tech Stack 🛠️
---
Using a dataset from Kaggle, I was able to find hundreds of labeled images of trash. I used the ResNet50 neural network along with PyTorch to train the model. Here is the accuracy of the model vs. the number of epochs graph:
<img width="588" alt="Screenshot 2025-03-03 at 8 32 37 PM" src="https://github.com/user-attachments/assets/da74c85e-86bf-4b91-b111-c42d14e5cc7a" />

I implemented Flask to connect the model to the frontend.
The frontend consists of HTML/CSS and Bootstrap. The UI feature pictures and videos of nature to fit with the goal of Garbify: To ensure trash ends up where it is supposed to be and protect the environment.
<img width="700" alt="Screenshot 2025-03-03 at 8 50 42 PM" src="https://github.com/user-attachments/assets/4067930a-22f4-4f9e-b30e-e0f9fc01d390" />
<img width="700" alt="Screenshot 2025-03-03 at 8 47 09 PM" src="https://github.com/user-attachments/assets/6862fe97-f411-41bb-9a4d-796cc316ae32" />

## Challenges Faced 😰
---
The machine learning model took a very long time to run, so I had to reduce the RAM it was using, as well as make the size of the images and batch size smaller while ensuring the model is accurate. 

## Try it yourself 😃
---
Simply clone the repository, go to the main.py file and run it, enter a picture and Garbify will tell you what it is!







