import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from functools import lru_cache
import re

def classify_age_group(age: int) -> str:
    if age < 18:
        return "child's"
    elif age < 40:
        return "young adult's"
    elif age < 65:
        return "middle-aged adult's"
    return "elderly person's"

def get_gender_adjective(gender: str) -> str:
    gender = gender.lower()
    if gender in {"male", "man", "boy"}:
        return "masculine"
    elif gender in {"female", "woman", "girl"}:
        return "feminine"
    return "gender-neutral"

def map_emotion_to_style(emotion: str) -> tuple[str, str]:
    emotion = emotion.lower()
    color_map = {
        "happy": "bright and warm colors like yellow and orange",
        "sad": "cool and calming colors like blue and gray",
        "neutral": "neutral tones like beige and white",
        "angry": "bold colors like red and black",
        "calm": "soft pastel colors and minimalistic design",
        "energetic": "vibrant colors and dynamic patterns",
        "tired": "soft muted colors and cozy textures"
    }

    style_map = {
        "happy": "cheerful and lively",
        "sad": "serene and peaceful",
        "neutral": "simple and elegant",
        "angry": "dramatic and intense",
        "calm": "relaxing and cozy",
        "energetic": "dynamic and playful",
        "tired": "warm and comforting"
    }

    return (
        color_map.get(emotion, "neutral tones like beige and white"),
        style_map.get(emotion, "simple and elegant")
    )

def generate_prompt(emotion: str, age: str, gender: str) -> str:
    try:
        age_group = classify_age_group(int(age))
    except ValueError:
        age_group = "adult's"

    gender_adjective = get_gender_adjective(gender)
    colors, style = map_emotion_to_style(emotion)

    return (
        f"A {gender_adjective} {age_group} bedroom interior, {style} style, "
        f"decorated with {colors}, with modern furniture, soft lighting, plants, and cozy textures. "
        f"Photorealistic, high-resolution interior design"
    )

@lru_cache(maxsize=1)
def load_pipeline(device: str) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    return pipe.to(device)

def sanitize_filename(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '', s.replace(" ", "_").lower())

def generate_room_image(prompt: str, device: str) -> Image.Image:
    print(f"\n📝 Using prompt:\n{prompt}\n")
    pipe = load_pipeline(device)
    result = pipe(prompt, guidance_scale=7.5)
    return result.images[0]

def main():
    print("Welcome to the EmoRoom Design Generator!")
    emotion = input("Enter your emotion (e.g. happy, sad, calm): ").strip()
    age = input("Enter your age: ").strip()
    gender = input("Enter your gender (e.g. male, female, non-binary): ").strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Running on device: {device}")

    prompt = generate_prompt(emotion, age, gender)
    image = generate_room_image(prompt, device)

    image.show()

    # Construct and sanitize filename
    filename = f"{sanitize_filename(emotion)}_{sanitize_filename(age)}_{sanitize_filename(gender)}.png"
    image.save(filename)
    print(f"Room design image saved as '{filename}'.")

if __name__ == "__main__":
    main()
