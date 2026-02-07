# Aircraft Damage Captioning System

This project aims to automate the detection and captioning of aircraft damage. It utilizes deep learning models to classify damage types (such as dents and cracks) and generates descriptive captions for the damage using a pre-trained Transformer model.

## Project Overview

The system performs two main tasks:
1.  **Damage Classification**: Classifies aircraft damage into two categories: "dent" and "crack" using a pre-trained **VGG16** model as a feature extractor.
2.  **Image Captioning**: Generates descriptive captions and summaries for the damage images using the **BLIP** (Bootstrapping Language-Image Pre-training) model from Hugging Face Transformers.

## Dataset

The project uses the **Roboflow Aircraft Dataset**.
-   **Source**: [Roboflow Aircraft Damage Detection](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk)
-   **Classes**: `dent`, `crack`
-   **Structure**: The dataset is organized into `train`, `valid`, and `test` directories.

## Prerequisites

To run this project, you need Python installed along with the following libraries:

-   `pandas`
-   `tensorflow` (CPU version specified in notebook: `tensorflow_cpu==2.17.1`)
-   `pillow`
-   `matplotlib`
-   `transformers`
-   `torch`
-   `torchvision`
-   `torchaudio`

You can install the dependencies using the following command (based on the notebook configuration):

```bash
pip install pandas==2.2.3 tensorflow_cpu==2.17.1 pillow==11.1.0 matplotlib==3.9.2 transformers==4.38.2
pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

## Usage

1.  **Clone/Download the repository**.
2.  **Dataset**: Ensure the `aircraft_damage_dataset_v1` folder is present in the project root with the correct structure (`train`, `valid`, `test`).
3.  **Run the Notebook**: Open `Final_Project_Classification_and_Captioning.ipynb` in Jupyter Notebook or VS Code.
4.  **Execution**: Run the cells sequentially to:
    -   Preprocess the data.
    -   Train the VGG16 classifier.
    -   Evaluate the classification performance.
    -   Generate captions using the BLIP model.

## Models Used

-   **VGG16**: A convolutional neural network model pre-trained on ImageNet, used here for feature extraction and fine-tuned for binary classification of aircraft damage.
-   **BLIP**: A Transformer-based model capable of understanding and generating content for images, used here to create context-aware captions for the damage.

## License

The dataset is provided by a Roboflow user under the CC BY 4.0 license.
