import numpy as np
import PIL
from PIL import Image
import torch
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

def preprocess(image: PIL.Image.Image):        
    im_size = (378, 378)
    return Compose(
        [
            Resize(size=im_size, interpolation=InterpolationMode.BICUBIC),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )(image)

def image_preprocessing(img_path):
    image = Image.open(img_path)
    image.show()
    im = image
    im = preprocess(im.convert("RGB"))
    resized_image = F.interpolate(im.unsqueeze(0), size=(378, 378), mode="bilinear")
    combined_image = resized_image
    combined_image_np = combined_image.cpu().numpy()
    return combined_image_np

def text_emb(text, img_emb, text_emb):
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def _tokenize(txt):
        return tokenizer(
            txt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(torch.device('cpu'))
    
    text_emb = torch.from_numpy(text_emb).to(torch.device('cpu'))
    # Use the passed `text_emb` instead of loading from the file
    bos_token_id = tokenizer.bos_token_id
    embedding_vector = text_emb[bos_token_id].reshape(1,1,2048)

    t = _tokenize(text)
    token_embeddings = text_emb[t]

    img_emb_tensor = torch.from_numpy(img_emb).to(torch.device('cpu'))
    embeds = []
    embeds.append(embedding_vector)
    embeds.append(img_emb_tensor)
    embeds.append(token_embeddings)

    return torch.cat(embeds, dim=1).cpu().numpy()

def embedding_function(input_ids, weight_matrix):
    # Convert the weight matrix (if it's a numpy array) to a PyTorch tensor
    weight_matrix = torch.from_numpy(weight_matrix).to(torch.device('cpu'))
    input_ids = torch.from_numpy(input_ids).long().to(torch.device('cpu'))

    # Perform the embedding lookup by using the input_ids to index the weight matrix
    embeddings = torch.nn.functional.embedding(input_ids, weight_matrix)

    return embeddings.cpu().numpy()

def cos_sin(layer):
    state_dict = torch.load(r'./cos_sin_cached.pth', map_location="cpu",  weights_only=True)
    return state_dict[layer].detach().numpy()


def layer_weights_text(layer):
    state_dict = torch.load(r'./text_model_weights.pth', map_location="cpu",  weights_only=True)
    return state_dict[layer].detach().numpy()

def h7_state_dict():
    x = np.load(r'./text_model_weights_h0_h7.npz')
    return x

def h8_state_dict():
    x = np.load(r'./text_model_weights_h8_h17.npz')
    return x

def h18_state_dict():
    x = np.load(r'./text_model_weights_h18_h23.npz')
    return x

def argmax_index(logits):
    return np.argmax(logits, axis=-1)

def decode(output_ids):
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    output_ids_tensor = torch.tensor(output_ids)
    return tokenizer.batch_decode(output_ids_tensor, skip_special_tokens=True)
