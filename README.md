# explore-ColPali

## Prerequisite
### 1. Prepare HuggingFace Hub Token
0. Generate HuggingFace Hub Token if you dont already have it
1. Create a `.env` file in the project dir and put the token in `HUGGING_FACE_HUB_TOKEN` field's value
```env
HUGGING_FACE_HUB_TOKEN={{TOKEN}}
```

### 2. Request Access to `paligemma-3b-mix-448` model [here](https://huggingface.co/google/paligemma-3b-mix-448)


### 3. Install poppler
For ubuntu, run: 
```sh
sudo apt-get install poppler-utils
```
For macOS, run:
```sh
brew install poppler
```
Make sure that the poppler binary is added to the PATH.

### 4. Install Python Libraries
```sh
pip install -r requirements.txt
```
