<img width="769" height="91" alt="image" src="https://github.com/user-attachments/assets/5783f7b4-c8a9-424a-a37f-243625f24aaf" />

This repository contains the implementation used for the experimental evaluation reported in the paper.

**1: Install the required dependencies:**  
pip install torch torchvision torchaudio  <br>
pip install torch-geometric  <br>
pip install numpy pandas scikit-learn tqdm transformers<br>

**2. Project Structure**<br>
├── config/        # Configuration files<br>
├── data/          # Data preprocessing scripts<br>
├── dataset/       # Dataset loader<br>
├── graph/         # Graph construction modules<br>
├── model/         # Model definitions<br>
├── train/         # Training and evaluation scripts<br>
├── utils/         # Utility functions<br>
├── outputs/       # Saved models and logs<br>
├── Tweets.xlsx    # Dataset file<br>
└── main.ipynb     # Main execution notebook<br>

**3. Dataset Preparation**<br>
Download Data Set (https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)<br>

**4. Running the Experiments**<br>
Open and run **main.ipynb**

**5. Outputs**<br>
Model evaluation results are saved in **outputs** folder:

