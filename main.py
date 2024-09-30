import torch
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка модели Mask R-CNN
def load_detection_model():
    detection_model = maskrcnn_resnet50_fpn(pretrained=True)
    detection_model.eval()
    return detection_model

# Предсказание маски и класса объекта
def get_mask_and_class(image, detection_model, device):
    image_transforms = transforms.Compose([transforms.ToTensor()])
    image_tensor = image_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = detection_model(image_tensor)[0]

    if len(predictions['masks']) == 0:
        print("Объект не найден")
        return None, None
    
    object_mask = predictions['masks'][0, 0].cpu().numpy()
    object_class_id = predictions['labels'][0].item()

    return object_mask, object_class_id

# Обработка маски изображения
def refine_mask(mask, target_size):
    resized_mask = cv2.resize(mask, target_size)
    binary_mask = (resized_mask > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    blurred_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (15, 15), 0)
    return np.clip(blurred_mask, 0, 1)

# Применение маски для удаления фона
def mask_image(image, mask):
    image_array = np.array(image)
    mask_3_channels = np.stack([mask]*3, axis=-1)
    resized_mask = cv2.resize(mask_3_channels, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_NEAREST)
    masked_image_result = image_array * resized_mask
    return masked_image_result

# Замена фона на заданный тип (сплошной цвет или изображение)
def change_background(image, mask, bg_type="solid", bg_color=(255, 255, 255), bg_image=None):
    image_array = np.array(image)
    mask_3_channels = np.stack([mask]*3, axis=-1)

    if bg_type == "solid":
        new_background = np.full_like(image_array, bg_color, dtype=np.uint8)
    elif bg_type == "image" and bg_image is not None:
        new_background = cv2.resize(bg_image, (image_array.shape[1], image_array.shape[0]))
    else:
        raise ValueError("Неверный тип фона или отсутствует изображение фона")
    
    inverted_mask = 1 - mask_3_channels
    result_image = (image_array * mask_3_channels + new_background * inverted_mask).astype(np.uint8)
    return result_image

# Генерация описания товара на основе класса
def create_description(class_name, gpt_model, tokenizer):
    text_prompt = f"Этот товар {class_name} обладает следующими характеристиками:"
    input_tensor = tokenizer.encode(text_prompt, return_tensors="pt")
    output_tensor = gpt_model.generate(input_tensor, max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(output_tensor[0], skip_special_tokens=True)
    return generated_text

# Список классов COCO
COCO_CLASSES = [ ... ]  # Здесь должен быть добавлен список классов COCO

# Основной процесс обработки изображений
def process_image_folder(input_dir, output_dir, bg_type="solid", bg_color=(255, 255, 255), bg_image_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detection_model = load_detection_model().to(device)
    
    text_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx in range(500):  # Обрабатываем изображения с именами 0.jpg по 499.jpg
        image_filename = f"{idx}.jpg"
        image_filepath = os.path.join(input_dir, image_filename)

        if not os.path.exists(image_filepath):  # Пропускаем файлы, если их нет
            print(f"Файл {image_filename} не найден, пропускаем.")
            continue

        image = Image.open(image_filepath).convert('RGB')
        mask, class_id = get_mask_and_class(image, detection_model, device)

        if mask is None:
            print(f"Объект не найден в файле {image_filename}")
            continue
        
        refined_mask = refine_mask(mask, (image.width, image.height))
        masked_img = mask_image(image, refined_mask)

        bg_img = None
        if bg_image_path:
            bg_img = cv2.imread(bg_image_path)
        
        final_img = change_background(masked_img, refined_mask, bg_type, bg_color, bg_img)

        output_image_filepath = os.path.join(output_dir, f"processed_{image_filename}")
        cv2.imwrite(output_image_filepath, final_img)
        print(f"Файл сохранен: {output_image_filepath}")
        
        # Проверка класса перед генерацией описания
        if 1 <= class_id <= len(COCO_CLASSES):
            class_name = COCO_CLASSES[class_id - 1]  # Получаем имя класса из COCO_CLASSES
            description_text = create_description(class_name, text_model, tokenizer)
        else:
            print(f"Некорректный class_id {class_id} для файла {image_filename}. Пропуск генерации описания.")
            continue  # Пропускаем генерацию описания для некорректных class_id
        
        description_filepath = os.path.join(output_dir, f"description_{image_filename}.txt")
        with open(description_filepath, 'w') as description_file:
            description_file.write(description_text)
        print(f"Описание сохранено: {description_filepath}")

# Пример вызова функции
if __name__ == "__main__":
    input_dir = "sirius_data/sirius_data"  # Директория с исходными изображениями
    output_dir = "output_images"  # Директория для сохранения результатов
    
    process_image_folder(input_dir, output_dir, bg_type="solid", bg_color=(220, 220, 220))
