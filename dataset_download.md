# Dataset Download Guide

Follow the steps below to download and prepare the dataset.

## 1. Install Dependencies

Run the following command from the project root directory:

```bash
pip install -r requirements.txt
```

---

## 2. Create `kaggle.json`

You need a Kaggle API token to download datasets.

### Generate the API Token

1. Go to your Kaggle account settings:
   https://www.kaggle.com/settings

2. Scroll to the **API** section.

3. Click **Create New Token**.

### Create the File

Create the following file in the project root directory:

```text
kaggle.json
```

With the following content:
```json
{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}
```

---
## 3. Download the Dataset

Run the dataset download command:

```bash
python3 ./download_dataset.py
```
