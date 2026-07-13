import streamlit as st
import numpy as np
import cv2
import tempfile

from keras import layers, ops
from keras.models import load_model

# ==================================================
# CONFIG
# ==================================================
INPUT_SHAPE = (32,112,112,3)

# ==================================================
# CUSTOM LAYER
# ==================================================
class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)

        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )

        self.flatten = layers.Reshape(
            target_shape=(-1, embed_dim)
        )

    def call(self, videos):
        projected_patches = self.projection(videos)
        return self.flatten(projected_patches)


class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape

        self.position_embedding = layers.Embedding(
            input_dim=num_tokens,
            output_dim=self.embed_dim,
        )

        self.positions = ops.arange(
            start=0,
            stop=num_tokens,
            step=1,
        )

    def call(self, encoded_tokens):
        encoded_positions = self.position_embedding(
            self.positions
        )

        return encoded_tokens + encoded_positions


# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_vivit():

    model = load_model(
        "model_vivit_categorical_frame_32.keras",
        custom_objects={
            "TubeletEmbedding": TubeletEmbedding,
            "PositionalEncoder": PositionalEncoder,
        },
    )

    return model


# ==================================================
# LABEL
# ==================================================
@st.cache_data
def load_labels():

    with open("labels.txt","r",encoding="utf-8") as f:
        labels = [x.strip() for x in f.readlines()]

    return labels


# ==================================================
# PREPROCESS VIDEO
# ==================================================
def read_video_frames(
    path,
    num_frames=32,
    frame_size=(112,112)
):

    cap = cv2.VideoCapture(path)

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    idx = np.linspace(
        0,
        len(frames)-1,
        num_frames
    ).astype(int)

    sampled_frames = np.array(
        [frames[i] for i in idx],
        dtype=np.float32
    )

    sampled_frames = sampled_frames / 255.0

    sampled_frames = np.expand_dims(
        sampled_frames,
        axis=0
    )

    return sampled_frames


# ==================================================
# STREAMLIT UI
# ==================================================
st.set_page_config(
    page_title="BISINDO Recognition",
    layout="wide"
)

st.title("🤟 BISINDO Word Recognition")

st.write(
    "Upload video bahasa isyarat BISINDO untuk melakukan prediksi menggunakan model ViViT."
)

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4","avi","mov","mkv"]
)

if uploaded_file:

    st.video(uploaded_file)

    if st.button("Prediksi"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp:

            tmp.write(uploaded_file.read())

            temp_video_path = tmp.name

        with st.spinner("Sedang memproses..."):

            video = read_video_frames(
                temp_video_path
            )

            model = load_vivit()

            labels = load_labels()

            pred = model.predict(
                video,
                verbose=0
            )

            pred_class = np.argmax(pred)

            confidence = float(
                np.max(pred)
            )

            st.success(
                f"Hasil Prediksi : {labels[pred_class]}"
            )

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.subheader(
                "Top 5 Prediction"
            )

            top5 = np.argsort(
                pred[0]
            )[::-1][:5]

            for idx in top5:

                st.write(
                    f"{labels[idx]} : {pred[0][idx]*100:.2f}%"
                )