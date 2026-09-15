# Intel Image Classifier — Free Deployment

This is a Streamlit version of the Gradio app from the supplied Colab notebook.

## Required model file

Copy your trained model:

`intel_efficientnet_v2.pth`

into this same folder, next to `app.py`.

The notebook saved the model at:

`/content/drive/MyDrive/ml-journey/intel_efficientnet_v2.pth`

## Local test

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Free deployment

Use Streamlit Community Cloud:

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `intel_efficientnet_v2.pth`.
3. Go to https://share.streamlit.io/
4. Sign in with GitHub.
5. Create app.
6. Select your repository and `app.py`.
7. Deploy.

Your app will receive a public `streamlit.app` URL.
